"""Vultr (https://vultr.com) Cloud GPU plugin for the multi-cloud dispatcher.

Wraps Vultr's v2 REST API (https://www.vultr.com/api/) behind the
:class:`cloud_providers.CloudProvider` interface.

Notes specific to Vultr
-----------------------

* Authentication: Bearer token from ``VULTR_API_KEY``.  Vultr currently issues
  only full-permission keys (no per-resource scoping); the operator is
  responsible for rotating the key after every machine they SSH from, and
  for setting an account-level billing alert.
* Pod create: ``POST /v2/instances`` returns an instance dict with an ``id``,
  ``main_ip`` (usually 0.0.0.0 until the instance is fully provisioned),
  and the chosen plan/region.  Unlike RunPod there is no port translation;
  every Vultr instance gets a direct public IPv4 and SSH is on port 22.
* No proxy SSH.  No GraphQL.  No marketplace tier.  The control plane is
  the simplest of the three providers in this directory.
* Plans pin a specific GPU; the dispatcher cycles through canonical GPU
  candidates and asks for the matching plan ID on each iteration.
* Regions: Vultr Cloud GPU is currently in NJ, ATL, FRA.  We don't pin a
  region by default; Vultr places the instance in whichever region has
  capacity.
* SSH key management: Vultr expects an ``sshkey_id`` in the create payload.
  The plugin fetches the operator's ``~/.ssh/id_ed25519.pub`` (or
  ``id_rsa.pub``), uploads it as a one-shot key labeled ``sm-dispatcher``,
  and references it when creating instances.  The key is left on the account
  for reuse across runs.

The canonical-to-Vultr-plan map below covers Vultr's current Cloud GPU plan
inventory.  Plan IDs occasionally change when Vultr restructures pricing;
update the map below when that happens.
"""
from __future__ import annotations

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


API_BASE = "https://api.vultr.com/v2"


# ---------------------------------------------------------------------------
# Canonical → Vultr plan-ID mapping
# ---------------------------------------------------------------------------
# Vultr Cloud GPU plans are named ``vcg-<gpu>-<count>c-<ram>gb-<disk>gb``.
# The canonical map here uses the single-GPU on-demand plan for each card.
# Fractional plans (1/8, 1/4 of an A100) are intentionally omitted; the
# dispatcher targets full-GPU instances.
_CANONICAL_TO_PLAN: Dict[str, str] = {
    # Single-A100 80GB instance (full GPU).
    "A100-80GB":   "vcg-a100-1c-15gb-80gb",
    # A40 / A6000 / A16 not currently offered as full-GPU plans on Vultr;
    # commented out so UnknownGpuError fires loudly.
    # "A40":       "...",
    # "A6000":     "...",
    "L40S":        "vcg-l40s-1c-12gb-48gb",
    # H100 80GB (limited regions).
    "H100-80GB":   "vcg-h100-1c-30gb-80gb",
    # GH200 not yet exposed via REST API as of the most recent Vultr docs;
    # operator can add it here once the plan-ID is known.
}


# ---------------------------------------------------------------------------
# Instance handle
# ---------------------------------------------------------------------------

class VultrInstance(CloudInstance):
    """Vultr instance handle returned by :meth:`VultrProvider.create`."""

    def __init__(
        self,
        *,
        instance_id: str,
        canonical_gpu: str,
        plan_id: str,
        region: str,
        os_id: int,
        cost_per_hour_usd: Optional[float],
        created_utc: str,
    ) -> None:
        self.instance_id = instance_id
        self.canonical_gpu = canonical_gpu
        self.plan_id = plan_id
        self.region = region
        self.os_id = os_id
        self.cost_per_hour_usd = cost_per_hour_usd
        self.created_utc = created_utc
        # Set by wait_for_ready once the instance reports an SSH-reachable IP.
        self.ssh_ip: Optional[str] = None
        self.ssh_port: int = 22

    def ssh_target(self) -> List[str]:
        if not self.ssh_ip:
            raise RuntimeError(
                f"Vultr instance {self.instance_id} has no SSH endpoint yet; "
                "wait_for_ready must succeed first."
            )
        return ["-p", str(self.ssh_port), f"root@{self.ssh_ip}"]

    def label(self) -> str:
        if not self.ssh_ip:
            return f"vultr instance {self.instance_id} (no SSH yet)"
        return f"ssh root@{self.ssh_ip}"

    def provenance(self) -> Dict[str, Any]:
        return {
            "provider":        "vultr",
            "instance_id":     self.instance_id,
            "plan_id":         self.plan_id,
            "canonical_gpu":   self.canonical_gpu,
            "region":          self.region,
            "os_id":           self.os_id,
            "cost_per_hr_usd": self.cost_per_hour_usd,
            "created_utc":     self.created_utc,
        }


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class VultrProvider(CloudProvider):
    """Vultr Cloud GPU control-plane plugin."""

    name = "vultr"
    # Vultr offers Ubuntu 22.04 LTS with NVIDIA drivers + Docker preinstalled.
    # The dispatcher's bash wrapper installs PyTorch via pip on top.  Using
    # the OS image directly (not a Vultr Marketplace app) keeps the surface
    # smaller and the boot time faster.
    default_image = "ubuntu-22.04-cuda"  # nominal label; see os_id below
    default_gpu = "A100-80GB"
    approval_gated_gpus: Set[str] = frozenset({"H100-80GB"})
    cloud_type_choices = []  # Vultr has only on-demand

    # Vultr OS id 1743 == "Ubuntu 22.04 LTS x64" (verified via /v2/os).
    # Cloud GPU plans support cloud-init via Vultr's user_data field, which
    # the operator can use to preinstall extra packages.  We rely on the
    # dispatcher wrapper's `pip install` step instead.
    OS_ID_UBUNTU_22_04 = 1743

    # Tag applied to every dispatcher-created SSH key so subsequent runs
    # can find / reuse the same key without re-uploading.
    SSH_KEY_LABEL = "sm-dispatcher"

    def __init__(
        self,
        *,
        api_key_env: str = "VULTR_API_KEY",
        **_unused: Any,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise AuthError(
                f"environment variable {api_key_env} is not set. "
                "Generate one at https://my.vultr.com/settings/#settingsapi"
            )
        self._api_key = api_key

    # ---- ABC implementation ------------------------------------------------

    def gpu_sku(self, canonical: str) -> str:
        plan = _CANONICAL_TO_PLAN.get(canonical)
        if plan is None:
            raise UnknownGpuError(
                f"Vultr does not offer canonical GPU {canonical!r}. "
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
        container_disk_gb: int = 50,  # informational; plan controls disk size
        volume_gb: int = 20,           # informational; plan controls disk size
        regions: Optional[List[str]] = None,
    ) -> VultrInstance:
        if gpu_count != 1:
            raise ProvisioningError(
                "Vultr plans pin GPU count; multi-GPU dispatch is not "
                "supported via this plugin.  Use a multi-GPU plan ID instead."
            )
        plan_id = self.gpu_sku(canonical_gpu)
        sshkey_id = self._ensure_ssh_key()

        # Vultr requires a single ``region`` string, not a list.  We pick the
        # first entry of the operator-supplied list (if any); otherwise the
        # API places the instance wherever the plan has capacity.
        region = regions[0] if regions else None

        payload: Dict[str, Any] = {
            "plan": plan_id,
            "label": name,
            "hostname": name,
            "os_id": self.OS_ID_UBUNTU_22_04,
            "sshkey_id": [sshkey_id],
            "enable_ipv6": False,
            "backups": "disabled",
            "activation_email": False,
        }
        if region:
            payload["region"] = region

        try:
            r = requests.post(
                f"{API_BASE}/instances",
                headers=self._headers(),
                json=payload,
                timeout=60,
            )
        except requests.RequestException as exc:
            raise ProvisioningError(
                f"Vultr POST /instances network error: {exc}"
            ) from exc

        if r.status_code >= 400:
            self._classify_and_raise(r.status_code, r.text or "", plan_id)

        data = (r.json() or {}).get("instance") or {}
        instance_id = data.get("id")
        if not instance_id:
            raise ProvisioningError(
                f"Vultr create-instance response missing 'id': {r.text!r}"
            )
        cost_per_hr = self._plan_hourly_cost(plan_id)
        return VultrInstance(
            instance_id=instance_id,
            canonical_gpu=canonical_gpu,
            plan_id=plan_id,
            region=data.get("region") or (region or ""),
            os_id=self.OS_ID_UBUNTU_22_04,
            cost_per_hour_usd=cost_per_hr,
            created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def wait_for_ready(
        self, instance: CloudInstance, timeout_s: int = 900
    ) -> VultrInstance:
        assert isinstance(instance, VultrInstance)
        start = time.time()
        last_status: Optional[str] = None
        while time.time() - start < timeout_s:
            r = requests.get(
                f"{API_BASE}/instances/{instance.instance_id}",
                headers=self._headers(),
                timeout=30,
            )
            if r.status_code >= 400:
                # Transient API failures shouldn't kill the wait; surface once
                # and keep polling.
                print(
                    f"  [vultr] GET /instances failed rc={r.status_code}: "
                    f"{r.text[:200]!r}",
                    flush=True,
                )
                time.sleep(10)
                continue
            inst = (r.json() or {}).get("instance") or {}
            status = inst.get("server_status") or inst.get("status") or "?"
            main_ip = inst.get("main_ip")
            if status != last_status:
                print(
                    f"  [vultr] status={status} ip={main_ip!r}",
                    flush=True,
                )
                last_status = status
            # Vultr considers the instance "ready" when server_status ==
            # 'ok' AND a non-zero main_ip is present.
            if main_ip and main_ip != "0.0.0.0" and status in ("ok", "active"):
                if self._ssh_probe(main_ip, 22):
                    instance.ssh_ip = main_ip
                    return instance
            time.sleep(10)
        raise TimeoutError(
            f"Vultr instance {instance.instance_id} did not become "
            f"SSH-ready within {timeout_s}s"
        )

    def destroy(self, instance: CloudInstance) -> None:
        assert isinstance(instance, VultrInstance)
        try:
            r = requests.delete(
                f"{API_BASE}/instances/{instance.instance_id}",
                headers=self._headers(),
                timeout=30,
            )
            if r.status_code not in (200, 202, 204):
                print(
                    f"!! Failed to delete Vultr instance "
                    f"{instance.instance_id}: HTTP {r.status_code}: "
                    f"{r.text[:200]!r}",
                    flush=True,
                )
        except requests.RequestException as exc:
            print(
                f"!! Failed to delete Vultr instance "
                f"{instance.instance_id}: {exc}",
                flush=True,
            )

    # ---- Internal helpers ---------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _ensure_ssh_key(self) -> str:
        """Return a Vultr ``sshkey_id`` for the operator's public key,
        uploading the key if it isn't already registered.

        Reads ``~/.ssh/id_ed25519.pub`` first; falls back to
        ``~/.ssh/id_rsa.pub`` if the first is absent.  Raises
        :class:`ProvisioningError` if neither exists.
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

        # Is the key already registered?  List keys and search by body
        # (Vultr stores the openssh line verbatim).
        r = requests.get(
            f"{API_BASE}/ssh-keys",
            headers=self._headers(),
            timeout=30,
        )
        if r.status_code >= 400:
            raise ProvisioningError(
                f"Vultr GET /ssh-keys failed: HTTP {r.status_code}: {r.text!r}"
            )
        for entry in (r.json() or {}).get("ssh_keys", []):
            if (entry.get("ssh_key") or "").strip() == pub_body:
                return entry["id"]

        # Not registered yet — upload it.
        r = requests.post(
            f"{API_BASE}/ssh-keys",
            headers=self._headers(),
            json={"name": self.SSH_KEY_LABEL, "ssh_key": pub_body},
            timeout=30,
        )
        if r.status_code >= 400:
            raise ProvisioningError(
                f"Vultr POST /ssh-keys failed: HTTP {r.status_code}: {r.text!r}"
            )
        new_id = ((r.json() or {}).get("ssh_key") or {}).get("id")
        if not new_id:
            raise ProvisioningError(
                f"Vultr POST /ssh-keys returned no id: {r.text!r}"
            )
        return new_id

    def _plan_hourly_cost(self, plan_id: str) -> Optional[float]:
        """Look up the per-hour cost of a plan from Vultr's plans endpoint."""
        try:
            r = requests.get(
                f"{API_BASE}/plans?type=vcg",
                headers=self._headers(),
                timeout=30,
            )
            if r.status_code != 200:
                return None
            for plan in (r.json() or {}).get("plans", []):
                if plan.get("id") == plan_id:
                    monthly = plan.get("monthly_cost")
                    if isinstance(monthly, (int, float)):
                        # Vultr publishes monthly cost; convert to hourly.
                        return round(monthly / 730.0, 4)
        except Exception:
            return None
        return None

    @staticmethod
    def _ssh_probe(ip: str, port: int) -> bool:
        r = subprocess.run(
            ["ssh"] + SSH_OPTS + [
                "-o", "ConnectTimeout=5",
                "-o", "BatchMode=yes",
                "-p", str(port),
                f"root@{ip}",
                "true",
            ],
            capture_output=True, timeout=20,
        )
        return r.returncode == 0

    @staticmethod
    def _classify_and_raise(
        status_code: int, body: str, plan_id: str
    ) -> None:
        """Map a Vultr HTTP error to the right typed CloudError.

        Order of checks matters: a 403 with a billing-related body should
        surface as BalanceError, not AuthError, because the operator's
        recovery action is a billing-console fix not a key rotation.
        Check billing-keyword and capacity-keyword patterns BEFORE the
        401/403 -> AuthError fallback so the more-specific signal wins.
        """
        bl = body.lower()
        if "balance" in bl or "billing" in bl or "payment" in bl:
            raise BalanceError(
                f"Vultr balance / billing problem (HTTP {status_code}): {body}"
            )
        # Vultr returns 503 / 400 with various capacity-out messages depending
        # on the region; matching loosely on the common substrings.
        if (
            "out of stock" in bl
            or "no resources" in bl
            or "unavailable" in bl
            or status_code == 503
        ):
            raise CapacityError(
                f"Vultr has no capacity for plan {plan_id!r}: {body}"
            )
        if status_code in (401, 403):
            raise AuthError(
                f"Vultr auth failed (HTTP {status_code}): {body}"
            )
        raise ProvisioningError(
            f"Vultr create-instance failed ({status_code}): {body}"
        )


# Registry hook ---------------------------------------------------------------
PROVIDER_CLASS = VultrProvider
