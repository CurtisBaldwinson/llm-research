"""Lambda Labs GPU Cloud plugin for the multi-cloud dispatcher.

Wraps the Lambda Cloud v1 REST API (https://cloud.lambdalabs.com/api/v1/docs)
behind the :class:`cloud_providers.CloudProvider` interface.

Notes specific to Lambda Labs
-----------------------------

* Authentication: HTTP Basic with the API key as the username and an empty
  password.  The plugin reads ``LAMBDA_API_KEY`` from the environment.
* Pod create: ``POST /api/v1/instance-operations/launch`` with a list of
  ``instance_type_name``, ``region_name``, ``ssh_key_names`` and
  ``file_system_names`` (we leave file systems empty; the dispatcher
  re-pushes code every run).
* SSH endpoint discovery: ``GET /api/v1/instances/<id>`` returns an
  ``ip`` once the instance is ``active``.  No port translation, no
  proxy SSH — port 22 on the public IP, same as Vultr.
* Pricing is plan-pinned and exposed via ``GET /api/v1/instance-types``;
  the plugin caches the lookup at construction time so we have a cost
  rate to put in ``pod_provisioning.json``.
* SSH key management: Lambda expects pre-uploaded SSH keys referenced by
  name.  The plugin ensures the operator's local public key is uploaded
  under the label ``sm-dispatcher`` (uploading on first call, reusing on
  subsequent calls).
* Lambda's inventory is region-locked.  When the operator's preferred
  region is out of capacity the API surfaces a ``insufficient-capacity``
  error code that the plugin maps to :class:`CapacityError`.
* Storage: Lambda instances use ephemeral local storage on the GPU node.
  The ``container_disk_gb`` and ``volume_gb`` flags from the orchestrator
  CLI are accepted for API compatibility but ignored; Lambda's plan ID
  pins the disk size.

The abstraction this is the third instance of made adding the plugin
roughly four method bodies long, which is the point.
"""
from __future__ import annotations

import base64
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

from . import (
    AuthError,
    BalanceError,
    CapacityError,
    CloudInstance,
    CloudProvider,
    ProvisioningError,
    UnknownGpuError,
)
from ._ssh import SSH_OPTS


API_BASE = "https://cloud.lambdalabs.com/api/v1"


# ---------------------------------------------------------------------------
# Canonical → Lambda instance_type_name mapping
# ---------------------------------------------------------------------------
# Lambda names plans by GPU + count + RAM, e.g. ``gpu_1x_a100_sxm4`` for a
# single-GPU 80GB SXM4 A100 instance, ``gpu_1x_a10`` for a single-GPU A10,
# and so on.  This map covers the single-GPU plans for GPUs the dispatcher
# might want; multi-GPU plans are out of scope for the per-benchmark workloads
# we currently dispatch.

_CANONICAL_TO_PLAN: Dict[str, str] = {
    # Ampere
    "A100-40GB":   "gpu_1x_a100",
    "A100-80GB":   "gpu_1x_a100_sxm4",
    "A6000":       "gpu_1x_a6000",
    "A10":         "gpu_1x_a10",  # not in canonical GPU_CANONICAL yet but harmless
    # Hopper
    "H100-80GB":   "gpu_1x_h100_pcie",
    # Lambda doesn't currently expose A40, RTX-4090, L40, L40S, Blackwell.
}


# ---------------------------------------------------------------------------
# Instance handle
# ---------------------------------------------------------------------------

class LambdaInstance(CloudInstance):
    """Lambda Labs instance handle returned by :meth:`LambdaProvider.create`."""

    def __init__(
        self,
        *,
        instance_id: str,
        canonical_gpu: str,
        plan_id: str,
        region: str,
        cost_per_hour_usd: Optional[float],
        created_utc: str,
    ) -> None:
        self.instance_id = instance_id
        self.canonical_gpu = canonical_gpu
        self.plan_id = plan_id
        self.region = region
        self.cost_per_hour_usd = cost_per_hour_usd
        self.created_utc = created_utc
        self.ssh_ip: Optional[str] = None
        self.ssh_port: int = 22

    def ssh_target(self) -> List[str]:
        if not self.ssh_ip:
            raise RuntimeError(
                f"Lambda instance {self.instance_id} has no SSH endpoint yet; "
                "wait_for_ready must succeed first."
            )
        return ["-p", str(self.ssh_port), f"ubuntu@{self.ssh_ip}"]

    def label(self) -> str:
        if not self.ssh_ip:
            return f"lambda instance {self.instance_id} (no SSH yet)"
        return f"ssh ubuntu@{self.ssh_ip}"

    def provenance(self) -> Dict[str, Any]:
        return {
            "provider":        "lambda_labs",
            "instance_id":     self.instance_id,
            "plan_id":         self.plan_id,
            "canonical_gpu":   self.canonical_gpu,
            "region":          self.region,
            "cost_per_hr_usd": self.cost_per_hour_usd,
            "created_utc":     self.created_utc,
        }


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class LambdaProvider(CloudProvider):
    """Lambda Labs GPU Cloud control-plane plugin."""

    name = "lambda_labs"
    # Lambda's stock image is Ubuntu 22.04 with CUDA + PyTorch preinstalled.
    # The dispatcher's bash wrapper installs benchmark-specific deps on top.
    default_image = "lambda-stack-22.04"
    default_gpu = "A100-80GB"
    approval_gated_gpus: Set[str] = frozenset()  # Lambda doesn't gate GPUs
    cloud_type_choices = []  # Lambda has only on-demand
    # Default remote user (Lambda images authenticate as `ubuntu`, not `root`).
    REMOTE_USER = "ubuntu"
    # Tag applied to the dispatcher-uploaded SSH key for reuse.
    SSH_KEY_LABEL = "sm-dispatcher"

    def __init__(
        self,
        *,
        api_key_env: str = "LAMBDA_API_KEY",
        **_unused: Any,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise AuthError(
                f"environment variable {api_key_env} is not set. "
                "Generate one at https://cloud.lambdalabs.com/api-keys"
            )
        self._api_key = api_key
        # Lambda uses HTTP Basic with the API key as the username, no
        # password.  Compute the header once.
        self._auth_header = "Basic " + base64.b64encode(
            f"{api_key}:".encode("utf-8")
        ).decode("ascii")
        # Cache the plan -> hourly cost map.  Populated lazily on first
        # create() call.
        self._plan_costs_cache: Optional[Dict[str, float]] = None

    # ---- ABC implementation ------------------------------------------------

    def gpu_sku(self, canonical: str) -> str:
        plan = _CANONICAL_TO_PLAN.get(canonical)
        if plan is None:
            raise UnknownGpuError(
                f"Lambda Labs does not offer canonical GPU {canonical!r}. "
                f"Known: {sorted(_CANONICAL_TO_PLAN.keys())}"
            )
        return plan

    def create(
        self,
        *,
        canonical_gpu: str,
        image: str,
        name: str,
        gpu_count: int = 1,
        container_disk_gb: int = 50,  # ignored; plan controls disk
        volume_gb: int = 20,           # ignored; plan controls disk
        regions: Optional[List[str]] = None,
    ) -> LambdaInstance:
        if gpu_count != 1:
            raise ProvisioningError(
                "Lambda plans pin GPU count; multi-GPU dispatch is not "
                "supported via this plugin."
            )
        plan_id = self.gpu_sku(canonical_gpu)
        ssh_key_name = self._ensure_ssh_key()

        # Lambda requires a region.  If none is supplied we ask the API for
        # the first region that currently has stock for this plan.
        region = (regions[0] if regions else None) or self._find_region_with_stock(plan_id)

        if region is None:
            raise CapacityError(
                f"Lambda has no region with stock for plan {plan_id!r} right now."
            )

        payload: Dict[str, Any] = {
            "region_name": region,
            "instance_type_name": plan_id,
            "ssh_key_names": [ssh_key_name],
            "name": name,
            "quantity": 1,
        }
        try:
            r = requests.post(
                f"{API_BASE}/instance-operations/launch",
                headers=self._headers(),
                json=payload,
                timeout=60,
            )
        except requests.RequestException as exc:
            raise ProvisioningError(
                f"Lambda POST /instance-operations/launch network error: {exc}"
            ) from exc
        if r.status_code >= 400:
            self._classify_and_raise(r.status_code, r.text or "", plan_id)
        data = (r.json() or {}).get("data") or {}
        instance_ids = data.get("instance_ids") or []
        if not instance_ids:
            raise ProvisioningError(
                f"Lambda launch returned no instance_ids: {r.text!r}"
            )
        instance_id = instance_ids[0]
        cost_per_hr = self._plan_cost(plan_id)
        return LambdaInstance(
            instance_id=instance_id,
            canonical_gpu=canonical_gpu,
            plan_id=plan_id,
            region=region,
            cost_per_hour_usd=cost_per_hr,
            created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def wait_for_ready(
        self, instance: CloudInstance, timeout_s: int = 900
    ) -> LambdaInstance:
        assert isinstance(instance, LambdaInstance)
        start = time.time()
        last_status: Optional[str] = None
        while time.time() - start < timeout_s:
            r = requests.get(
                f"{API_BASE}/instances/{instance.instance_id}",
                headers=self._headers(),
                timeout=30,
            )
            if r.status_code >= 400:
                print(
                    f"  [lambda] GET /instances failed rc={r.status_code}: "
                    f"{r.text[:200]!r}",
                    flush=True,
                )
                time.sleep(10)
                continue
            inst = (r.json() or {}).get("data") or {}
            status = inst.get("status") or "?"
            ip = inst.get("ip")
            if status != last_status:
                print(f"  [lambda] status={status} ip={ip!r}", flush=True)
                last_status = status
            if status == "active" and ip:
                if self._ssh_probe(ip, 22):
                    instance.ssh_ip = ip
                    return instance
            time.sleep(10)
        raise TimeoutError(
            f"Lambda instance {instance.instance_id} did not become "
            f"SSH-ready within {timeout_s}s"
        )

    def destroy(self, instance: CloudInstance) -> None:
        assert isinstance(instance, LambdaInstance)
        try:
            r = requests.post(
                f"{API_BASE}/instance-operations/terminate",
                headers=self._headers(),
                json={"instance_ids": [instance.instance_id]},
                timeout=30,
            )
            if r.status_code >= 400:
                print(
                    f"!! Failed to terminate Lambda instance "
                    f"{instance.instance_id}: HTTP {r.status_code}: "
                    f"{r.text[:200]!r}",
                    flush=True,
                )
        except requests.RequestException as exc:
            print(
                f"!! Failed to terminate Lambda instance "
                f"{instance.instance_id}: {exc}",
                flush=True,
            )

    # ---- Internal helpers ---------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": self._auth_header,
            "Content-Type": "application/json",
        }

    def _ensure_ssh_key(self) -> str:
        """Return the *name* of an SSH key on the Lambda account that
        corresponds to the operator's local public key, uploading it if
        not present.

        Lambda's API expects key *names* (not IDs) in the launch payload.
        """
        candidates = [
            Path.home() / ".ssh" / "id_ed25519.pub",
            Path.home() / ".ssh" / "id_rsa.pub",
        ]
        pub_path = next((p for p in candidates if p.exists()), None)
        if pub_path is None:
            raise ProvisioningError(
                "Could not find ~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub. "
                "Generate one with `ssh-keygen -t ed25519` and re-run."
            )
        pub_body = pub_path.read_text().strip()

        r = requests.get(
            f"{API_BASE}/ssh-keys", headers=self._headers(), timeout=30
        )
        if r.status_code >= 400:
            raise ProvisioningError(
                f"Lambda GET /ssh-keys failed: HTTP {r.status_code}: {r.text!r}"
            )
        for entry in (r.json() or {}).get("data", []):
            if (entry.get("public_key") or "").strip() == pub_body:
                return entry["name"]

        # Not registered yet — upload it.
        r = requests.post(
            f"{API_BASE}/ssh-keys",
            headers=self._headers(),
            json={"name": self.SSH_KEY_LABEL, "public_key": pub_body},
            timeout=30,
        )
        if r.status_code >= 400:
            raise ProvisioningError(
                f"Lambda POST /ssh-keys failed: HTTP {r.status_code}: {r.text!r}"
            )
        return self.SSH_KEY_LABEL

    def _find_region_with_stock(self, plan_id: str) -> Optional[str]:
        """Return the first region with non-zero stock for ``plan_id``.

        Lambda's ``/instance-types`` endpoint lists each plan with the
        regions that currently have capacity for it.  We pick the first
        one; the operator can override by passing ``--regions``.
        """
        try:
            r = requests.get(
                f"{API_BASE}/instance-types",
                headers=self._headers(),
                timeout=30,
            )
            if r.status_code != 200:
                return None
            payload = (r.json() or {}).get("data") or {}
            entry = payload.get(plan_id) or {}
            regions = entry.get("regions_with_capacity_available") or []
            if regions:
                return regions[0].get("name")
        except requests.RequestException:
            return None
        return None

    def _plan_cost(self, plan_id: str) -> Optional[float]:
        """Look up the per-hour cost of a plan from /instance-types.  Cached."""
        if self._plan_costs_cache is None:
            try:
                r = requests.get(
                    f"{API_BASE}/instance-types",
                    headers=self._headers(),
                    timeout=30,
                )
                if r.status_code != 200:
                    self._plan_costs_cache = {}
                    return None
                payload = (r.json() or {}).get("data") or {}
                costs: Dict[str, float] = {}
                for name, entry in payload.items():
                    spec = (entry or {}).get("instance_type") or {}
                    price_cents = spec.get("price_cents_per_hour")
                    if isinstance(price_cents, (int, float)):
                        costs[name] = round(price_cents / 100.0, 4)
                self._plan_costs_cache = costs
            except requests.RequestException:
                self._plan_costs_cache = {}
                return None
        return self._plan_costs_cache.get(plan_id)

    def _ssh_probe(self, ip: str, port: int) -> bool:
        r = subprocess.run(
            ["ssh"] + SSH_OPTS + [
                "-o", "ConnectTimeout=5",
                "-o", "BatchMode=yes",
                "-p", str(port),
                f"{self.REMOTE_USER}@{ip}",
                "true",
            ],
            capture_output=True, timeout=20,
        )
        return r.returncode == 0

    @staticmethod
    def _classify_and_raise(
        status_code: int, body: str, plan_id: str
    ) -> None:
        """Map a Lambda HTTP error to the right typed CloudError.

        Order of checks matters: Lambda returns HTTP 403 for both genuine
        authentication failures (bad API key) AND for account-state issues
        like an invalid billing address.  Both surface as 403, but the
        operator-recovery path is different - billing fixes are a console
        action, auth fixes are a key rotation.  Check billing-keyword and
        capacity-keyword patterns BEFORE the 401/403 -> AuthError fallback
        so the more-specific signal wins.
        """
        bl = body.lower()
        if (
            "billing" in bl
            or "balance" in bl
            or "payment" in bl
            or "invalid-address" in bl   # observed 2026-05-18 on a 403
        ):
            raise BalanceError(
                f"Lambda billing problem (HTTP {status_code}): {body}"
            )
        if "insufficient-capacity" in bl or "no capacity" in bl:
            raise CapacityError(
                f"Lambda has no capacity for plan {plan_id!r}: {body}"
            )
        if status_code in (401, 403):
            raise AuthError(
                f"Lambda auth failed (HTTP {status_code}): {body}"
            )
        raise ProvisioningError(
            f"Lambda launch failed ({status_code}): {body}"
        )


# Registry hook ---------------------------------------------------------------
PROVIDER_CLASS = LambdaProvider
