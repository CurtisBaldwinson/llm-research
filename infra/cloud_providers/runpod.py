"""RunPod (https://runpod.io) plugin for the multi-cloud dispatcher.

Wraps RunPod's REST + GraphQL control plane behind the
:class:`cloud_providers.CloudProvider` interface.

Notes specific to RunPod
------------------------

* Authentication: Bearer token from ``RUNPOD_API_KEY``.  Generate at
  https://www.console.runpod.io/user/settings (API Keys section).
* Pod create: ``POST /v1/pods`` (REST) returns a pod dict with an ``id``
  and (sometimes) a ``costPerHr`` rate.
* SSH endpoint discovery: REST returns the pod's ``desiredStatus`` and
  partial port info, but the authoritative port mapping and the proxy SSH
  username live on GraphQL (``runtime.ports[]`` + ``machine.podHostId``).
  We poll REST to detect ``RUNNING`` and then enrich with one GraphQL call.
* Two SSH access methods:
    - Direct: ``ssh root@<publicIp> -p <mappedPort>``.  Supports SCP and
      the dispatcher's tar-over-SSH transport.
    - Proxy: ``ssh <podId>-<hostHash>@ssh.runpod.io``.  Does NOT accept
      non-interactive ``ssh ... -T`` commands cleanly.  We refuse to use
      the proxy and only return a pod once a direct endpoint is up.
* Two cloud tiers: ``SECURE`` (operator datacenters, higher reliability,
  slightly more expensive) and ``COMMUNITY`` (peer-to-peer marketplace,
  cheaper, less reliable).  Selected via ``cloud_type=`` opt.
* Blackwell GPUs (sm_120) are exposed for sale but the default PyTorch
  2.4.0 image lacks sm_120 kernels.  The orchestrator filters them out
  using :func:`cloud_providers.gpu_supports_pytorch_2_4`; this plugin
  surfaces the SKUs anyway via ``gpu_sku`` so a future PyTorch 2.5+ image
  can use them by updating one constant.
"""
from __future__ import annotations

import json as _json
import os
import subprocess
import time
from datetime import datetime, timezone
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

REST_API = "https://rest.runpod.io/v1"
GRAPHQL_API = "https://api.runpod.io/graphql"


# ---------------------------------------------------------------------------
# Canonical → RunPod SKU mapping
# ---------------------------------------------------------------------------
# Maps the orchestrator-visible canonical GPU names (see
# `cloud_providers.GPU_CANONICAL`) to RunPod's `gpuTypeId` strings.  RunPod
# uses the marketing-style name with the "NVIDIA" prefix in most cases.

_CANONICAL_TO_SKU: Dict[str, str] = {
    "A100-40GB":    "NVIDIA A100 40GB PCIe",
    "A100-80GB":    "NVIDIA A100 80GB PCIe",
    "A40":          "NVIDIA A40",
    "A6000":        "NVIDIA RTX A6000",
    "A5000":        "NVIDIA RTX A5000",
    "A4000":        "NVIDIA RTX A4000",
    "RTX-3090":     "NVIDIA GeForce RTX 3090",
    "L40":          "NVIDIA L40",
    "L40S":         "NVIDIA L40S",
    "L4":           "NVIDIA L4",
    "RTX-4090":     "NVIDIA GeForce RTX 4090",
    "RTX-6000-ADA": "NVIDIA RTX 6000 Ada Generation",
    "H100-80GB":    "NVIDIA H100 80GB HBM3",
    "H100-NVL":     "NVIDIA H100 NVL",
    "H200":         "NVIDIA H200",
    "RTX-5090":     "NVIDIA GeForce RTX 5090",
    "RTX-5080":     "NVIDIA GeForce RTX 5080",
    "B200":         "NVIDIA B200",
}


# ---------------------------------------------------------------------------
# Instance handle
# ---------------------------------------------------------------------------

class RunPodInstance(CloudInstance):
    """RunPod pod handle returned by :meth:`RunPodProvider.create`."""

    def __init__(
        self,
        *,
        pod_id: str,
        canonical_gpu: str,
        sku: str,
        cloud_type: str,
        image: str,
        gpu_count: int,
        container_disk_gb: int,
        volume_gb: int,
        data_centers: Optional[List[str]],
        cost_per_hour_usd: Optional[float],
        created_utc: str,
    ) -> None:
        self.pod_id = pod_id
        self.canonical_gpu = canonical_gpu
        self.sku = sku
        self.cloud_type = cloud_type
        self.image = image
        self.gpu_count = gpu_count
        self.container_disk_gb = container_disk_gb
        self.volume_gb = volume_gb
        self.data_centers = data_centers
        self.cost_per_hour_usd = cost_per_hour_usd
        self.created_utc = created_utc

        # Populated by wait_for_ready once a direct SSH endpoint is up.
        self.ssh_ip: Optional[str] = None
        self.ssh_port: Optional[int] = None
        self.proxy_user: Optional[str] = None

    def ssh_target(self) -> List[str]:
        if self.ssh_ip is None or self.ssh_port is None:
            raise RuntimeError(
                f"RunPod pod {self.pod_id} has no direct SSH endpoint yet; "
                "wait_for_ready must succeed before ssh_target is callable."
            )
        return ["-p", str(self.ssh_port), f"root@{self.ssh_ip}"]

    def label(self) -> str:
        if self.ssh_ip is None or self.ssh_port is None:
            return f"runpod pod {self.pod_id} (no SSH yet)"
        return f"ssh root@{self.ssh_ip} -p {self.ssh_port}"

    def provenance(self) -> Dict[str, Any]:
        return {
            "provider":          "runpod",
            "pod_id":            self.pod_id,
            "gpu_type":          self.sku,
            "canonical_gpu":     self.canonical_gpu,
            "cloud_type":        self.cloud_type,
            "gpu_count":         self.gpu_count,
            "image":             self.image,
            "cost_per_hr_usd":   self.cost_per_hour_usd,
            "container_disk_gb": self.container_disk_gb,
            "volume_gb":         self.volume_gb,
            "data_centers":      self.data_centers,
            "created_utc":       self.created_utc,
        }


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class RunPodProvider(CloudProvider):
    """RunPod control-plane plugin."""

    name = "runpod"
    default_image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    default_gpu = "A40"
    # H100 / B200 / H200 typically require capacity approval.
    approval_gated_gpus: Set[str] = frozenset({
        "H100-80GB", "H100-NVL", "H200", "B200",
    })
    cloud_type_choices = ["SECURE", "COMMUNITY"]

    def __init__(
        self,
        *,
        cloud_type: str = "SECURE",
        api_key_env: str = "RUNPOD_API_KEY",
        **_unused: Any,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise AuthError(
                f"environment variable {api_key_env} is not set. "
                "Generate one at https://www.console.runpod.io/user/settings"
            )
        if cloud_type not in self.cloud_type_choices:
            raise ValueError(
                f"cloud_type must be one of {self.cloud_type_choices}, "
                f"got {cloud_type!r}"
            )
        self._api_key = api_key
        self._cloud_type = cloud_type

    # ---- ABC implementation ------------------------------------------------

    def gpu_sku(self, canonical: str) -> str:
        sku = _CANONICAL_TO_SKU.get(canonical)
        if sku is None:
            raise UnknownGpuError(
                f"RunPod does not offer canonical GPU {canonical!r}. "
                f"Known: {sorted(_CANONICAL_TO_SKU.keys())}"
            )
        return sku

    def create(
        self,
        *,
        canonical_gpu: str,
        image: str,
        name: str,
        gpu_count: int = 1,
        container_disk_gb: int = 50,
        volume_gb: int = 20,
        regions: Optional[List[str]] = None,
    ) -> RunPodInstance:
        sku = self.gpu_sku(canonical_gpu)
        payload: Dict[str, Any] = {
            "name": name,
            "imageName": image,
            "gpuTypeIds": [sku],
            "gpuCount": gpu_count,
            "cloudType": self._cloud_type,
            "containerDiskInGb": container_disk_gb,
            "volumeInGb": volume_gb,
            "volumeMountPath": "/workspace",
            "ports": ["22/tcp"],
        }
        if regions:
            payload["dataCenterIds"] = regions
            payload["dataCenterPriority"] = "custom"

        try:
            pod = self._post("/pods", payload, timeout=60)
        except _RunPodApiError as exc:
            self._classify_and_raise(exc, sku)
        pod_id = pod.get("id")
        if not pod_id:
            raise ProvisioningError(
                f"RunPod create-pod response missing 'id': {pod!r}"
            )
        cost = pod.get("costPerHr")
        if cost is None:
            machine = pod.get("machine") or {}
            cost = (
                machine.get("costPerHr")
                or (pod.get("pricing") or {}).get("costPerHr")
            )
        return RunPodInstance(
            pod_id=pod_id,
            canonical_gpu=canonical_gpu,
            sku=sku,
            cloud_type=self._cloud_type,
            image=image,
            gpu_count=gpu_count,
            container_disk_gb=container_disk_gb,
            volume_gb=volume_gb,
            data_centers=regions,
            cost_per_hour_usd=cost,
            created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def wait_for_ready(
        self, instance: CloudInstance, timeout_s: int = 900
    ) -> RunPodInstance:
        assert isinstance(instance, RunPodInstance)
        start = time.time()
        last_status: Optional[str] = None
        gql_fetched = False
        while time.time() - start < timeout_s:
            pod = self._get(f"/pods/{instance.pod_id}", timeout=30)
            status = pod.get("desiredStatus") or pod.get("status") or "?"

            # Once RUNNING, ask GraphQL for the authoritative port mapping +
            # proxy SSH username.  REST may report partial info for a few
            # seconds, so we loop until runtime.ports is populated.
            if status == "RUNNING" and not gql_fetched:
                gql = self._graphql_pod(instance.pod_id)
                if gql:
                    machine = gql.get("machine") or {}
                    host_id = machine.get("podHostId")
                    if host_id:
                        instance.proxy_user = host_id
                    gql_runtime = gql.get("runtime")
                    if gql_runtime and gql_runtime.get("ports"):
                        pod["runtime"] = gql_runtime
                        gql_fetched = True
                        print(
                            f"  [graphql] {_json.dumps(gql, indent=2, default=str)}",
                            flush=True,
                        )
                    else:
                        print(
                            "  [graphql] runtime not populated yet, retrying...",
                            flush=True,
                        )

            ssh_ip, ssh_port = self._extract_direct_ssh(pod)
            if status != last_status:
                label = (
                    f"ssh root@{ssh_ip} -p {ssh_port}"
                    if ssh_ip and ssh_port
                    else "(not available)"
                )
                print(
                    f"  status={status}  ssh={label}",
                    flush=True,
                )
                last_status = status

            if status == "RUNNING" and ssh_ip and ssh_port:
                # Probe for a real SSH handshake before declaring ready.
                # Direct SSH is sometimes accepted by the API tier but the
                # underlying machine takes a few seconds longer to spin up
                # its sshd.
                if self._ssh_probe(ssh_ip, ssh_port):
                    instance.ssh_ip = ssh_ip
                    instance.ssh_port = ssh_port
                    return instance

            time.sleep(10)
        raise TimeoutError(
            f"Pod {instance.pod_id} did not become SSH-ready within {timeout_s}s"
        )

    def destroy(self, instance: CloudInstance) -> bool:
        assert isinstance(instance, RunPodInstance)
        try:
            self._delete(f"/pods/{instance.pod_id}", timeout=30)
            return True
        except _RunPodApiError as exc:
            # 404 = already gone (e.g. pod_finalize self-terminated). Success.
            if getattr(exc, "status_code", None) == 404:
                return True
            print(
                f"!! Failed to delete RunPod pod {instance.pod_id}: {exc}",
                flush=True,
            )
            return False

    def attach_from_provisioning(self, prov: Dict[str, Any]) -> RunPodInstance:
        """Reconstruct a live RunPodInstance from a persisted provisioning dict.

        Reverses :meth:`RunPodInstance.provenance` to rebuild the dataclass
        fields, then calls :meth:`wait_for_ready` to refresh the (transient)
        SSH endpoint from the current API state.  ``pod_provisioning.json``
        does not store ``ssh_ip``/``ssh_port`` because they are populated
        only after :meth:`wait_for_ready` returns; the JSON is written
        immediately after :meth:`create`, before SSH is up.

        Verifies the pod is still queryable (raises :class:`ProvisioningError`
        if RunPod's API can't find the pod_id, e.g. it was destroyed manually).
        """
        pod_id = prov.get("pod_id")
        if not pod_id:
            raise ProvisioningError(
                f"pod_provisioning dict is missing 'pod_id'; "
                f"cannot reattach.  prov={prov!r}"
            )
        # Existence check up front so we get a clean error instead of a
        # 404 mid-wait_for_ready.
        try:
            self._get(f"/pods/{pod_id}", timeout=30)
        except _RunPodApiError as exc:
            raise ProvisioningError(
                f"RunPod cannot find pod_id={pod_id!r}: {exc}.  "
                "It may have been destroyed already; reattach is not possible."
            ) from exc

        instance = RunPodInstance(
            pod_id=pod_id,
            canonical_gpu=prov.get("canonical_gpu", "") or "",
            sku=prov.get("gpu_type", "") or "",
            cloud_type=prov.get("cloud_type", "") or "",
            image=prov.get("image", "") or "",
            gpu_count=int(prov.get("gpu_count", 1) or 1),
            container_disk_gb=int(prov.get("container_disk_gb", 0) or 0),
            volume_gb=int(prov.get("volume_gb", 0) or 0),
            data_centers=prov.get("data_centers"),
            cost_per_hour_usd=prov.get("cost_per_hr_usd"),
            created_utc=prov.get("created_utc", "") or "",
        )
        # Re-populate ssh_ip/ssh_port/proxy_user from current API state.
        # wait_for_ready is idempotent on a RUNNING pod (returns immediately
        # once GraphQL ports are populated, which is the steady state).
        return self.wait_for_ready(instance, timeout_s=180)

    # ---- Internal: REST + GraphQL helpers ----------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _post(
        self, path: str, payload: Dict[str, Any], timeout: int = 60
    ) -> Dict[str, Any]:
        r = requests.post(
            f"{REST_API}{path}",
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )
        if r.status_code >= 400:
            raise _RunPodApiError(
                status_code=r.status_code, body=r.text or "", path=path
            )
        return r.json() if r.text else {}

    def _get(self, path: str, timeout: int = 30) -> Dict[str, Any]:
        r = requests.get(
            f"{REST_API}{path}", headers=self._headers(), timeout=timeout,
        )
        if r.status_code >= 400:
            raise _RunPodApiError(
                status_code=r.status_code, body=r.text or "", path=path
            )
        return r.json() if r.text else {}

    def _delete(self, path: str, timeout: int = 30) -> None:
        r = requests.delete(
            f"{REST_API}{path}", headers=self._headers(), timeout=timeout,
        )
        # 404 = already gone (pod_finalize self-terminate race); treat as OK.
        if r.status_code not in (200, 202, 204, 404):
            raise _RunPodApiError(
                status_code=r.status_code, body=r.text or "", path=path
            )

    def _graphql_pod(self, pod_id: str) -> Dict[str, Any]:
        """One-shot GraphQL fetch for the fields REST doesn't expose:
        ``machine.podHostId`` (proxy SSH user) and ``runtime.ports[]``."""
        query = (
            'query { pod(input: {podId: "%s"}) { id machineId '
            'machine { podHostId } '
            'runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } '
            "} }" % pod_id
        )
        try:
            # Prefer Authorization header (REST already does this). Putting
            # api_key in the query string leaks into proxy/APM access logs.
            r = requests.post(
                GRAPHQL_API,
                json={"query": query},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                timeout=30,
            )
            if r.status_code != 200:
                return {}
            data = r.json()
            return data.get("data", {}).get("pod") or {}
        except Exception:
            return {}

    @staticmethod
    def _extract_direct_ssh(pod: Dict[str, Any]) -> tuple:
        """Return ``(ip, port)`` for the direct SSH endpoint, or ``(None, None)``.

        The dispatcher only accepts direct SSH because the proxy SSH
        endpoint refuses non-interactive commands (the dispatcher uses
        non-interactive ``ssh -T`` for tar-over-SSH).
        """
        top_ip = pod.get("publicIp") or pod.get("public_ip")
        runtime = pod.get("runtime") or {}
        for entry in (runtime.get("ports") or []):
            if not isinstance(entry, dict):
                continue
            priv = entry.get("privatePort")
            pub = entry.get("publicPort")
            if priv == 22 and pub and int(pub) > 0:
                ip = top_ip or entry.get("ip")
                is_public = entry.get("isIpPublic", True)
                if ip and is_public:
                    return ip, int(pub)
        return None, None

    @staticmethod
    def _ssh_probe(ip: str, port: int) -> bool:
        """Run ``ssh ... true`` with a 5s timeout; return True on success."""
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
    def _classify_and_raise(exc: "_RunPodApiError", sku: str) -> "RunPodInstance":
        """Map a RunPod HTTP error to the right typed CloudError."""
        body = exc.body.lower()
        if "balance" in body or "funds" in body:
            raise BalanceError(
                f"RunPod balance too low (POST /pods -> {exc.status_code}). "
                f"Add credits at https://www.console.runpod.io/billing"
            ) from exc
        if exc.status_code in (401, 403):
            raise AuthError(
                f"RunPod auth failed (POST /pods -> {exc.status_code}): {exc.body}"
            ) from exc
        # RunPod returns 500 with the literal message "no instances currently
        # available" on capacity-out.  Also "does not have the resources to
        # deploy your pod" is a per-machine resource error that's effectively
        # the same situation from the dispatcher's perspective.
        if "no instances" in body or "does not have the resources" in body:
            raise CapacityError(
                f"RunPod has no capacity for SKU {sku!r} right now: {exc.body}"
            ) from exc
        raise ProvisioningError(
            f"RunPod create-pod failed ({exc.status_code}): {exc.body}"
        ) from exc


# ---------------------------------------------------------------------------
# Internal exception used to carry the HTTP details up to _classify_and_raise.
# ---------------------------------------------------------------------------

class _RunPodApiError(Exception):
    def __init__(self, *, status_code: int, body: str, path: str) -> None:
        super().__init__(f"{path} -> {status_code}: {body}")
        self.status_code = status_code
        self.body = body
        self.path = path


# Registry hook ---------------------------------------------------------------
PROVIDER_CLASS = RunPodProvider
