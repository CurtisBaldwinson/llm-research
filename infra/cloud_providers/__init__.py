"""Cloud-provider plugin framework for the synthetic-myelination dispatcher.

The orchestrator (``scripts/run_on_cloud.py``) is provider-agnostic: it stages
a launch directory, opens a tee'd ``dispatcher.log``, pushes code to a remote,
runs a benchmark there, polls for completion, pulls results, and (optionally)
finalizes a lab report.  The cloud-specific control plane (create instance,
poll for ready, get SSH endpoint, destroy) lives behind the
:class:`CloudProvider` interface defined here.

To add a new cloud:

  1. Create ``scripts/cloud_providers/<name>.py`` with a subclass of
     :class:`CloudProvider`.
  2. Register it by name in :data:`_REGISTRY` below.
  3. Define the canonical-to-SKU map (see :data:`GPU_CANONICAL` for the set
     of canonical names the orchestrator recognises).
  4. Document the API-key env var (e.g., ``LAMBDA_API_KEY``).

That's it.  The orchestrator and SSH helpers in :mod:`._ssh` do not need to
change.

Design notes
------------

The provider methods all take and return a ``provider``-typed ``Instance``
object whose concrete shape is the plugin's business.  The orchestrator only
needs five things from an instance (``ssh_target``, ``cost_per_hour_usd``,
``provisioning_dict``, ``destroy``, ``wait_for_ready``); everything else is
the provider's responsibility.  This is the principal extensibility seam.

Errors are typed (:class:`CloudError` and its children) so the orchestrator
can route them correctly: :class:`CapacityError` is a soft failure (try the
next candidate GPU); :class:`AuthError` / :class:`BalanceError` are
hard-aborts with operator-visible messages; :class:`UnknownGpuError` means
the plugin doesn't support this canonical name and the orchestrator can
skip it without alarm.
"""
from __future__ import annotations

import abc
import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Typed error hierarchy
# ---------------------------------------------------------------------------

class CloudError(Exception):
    """Base class for all errors raised by a :class:`CloudProvider`."""


class AuthError(CloudError):
    """API key is missing, invalid, or revoked."""


class BalanceError(CloudError):
    """Account is out of funds or hit a billing limit."""


class CapacityError(CloudError):
    """No instances available for the requested SKU.  Soft failure;
    the dispatcher should try the next candidate GPU/region."""


class UnknownGpuError(CloudError):
    """The provider does not offer this canonical GPU name.  Soft failure;
    the dispatcher should skip this candidate."""


class ProvisioningError(CloudError):
    """Anything else from the create-instance / wait-for-ready path."""


# ---------------------------------------------------------------------------
# Canonical GPU naming
# ---------------------------------------------------------------------------
# Short, vendor-agnostic GPU identifiers used in the orchestrator CLI and in
# ``bench_caps.json``.  Each provider plugin maps these to its own SKU.  The
# orchestrator uses the metadata here (VRAM and compute capability) to filter
# candidates against a benchmark's declared minimum requirements regardless
# of provider.

@dataclass(frozen=True)
class GpuSpec:
    """Hardware envelope for a canonical GPU.

    `vram_gb` is the on-card VRAM in gigabytes (advertised, not effective
    after PyTorch's caching allocator reserves).  `compute_capability` is
    the CUDA compute-capability identifier (sm_XY) for image-compatibility
    filtering: a PyTorch wheel compiled for sm_50..sm_90 cannot launch
    kernels on an sm_120 (Blackwell) card.
    """
    vram_gb: int
    compute_capability: str  # e.g. "sm_86", "sm_120"


GPU_CANONICAL: Dict[str, GpuSpec] = {
    # Ampere generation (sm_80 / sm_86)
    "A100-40GB":   GpuSpec(vram_gb=40,  compute_capability="sm_80"),
    "A100-80GB":   GpuSpec(vram_gb=80,  compute_capability="sm_80"),
    "A40":         GpuSpec(vram_gb=48,  compute_capability="sm_86"),
    "A6000":       GpuSpec(vram_gb=48,  compute_capability="sm_86"),
    "A5000":       GpuSpec(vram_gb=24,  compute_capability="sm_86"),
    "A4000":       GpuSpec(vram_gb=16,  compute_capability="sm_86"),
    "RTX-3090":    GpuSpec(vram_gb=24,  compute_capability="sm_86"),
    # Ada generation (sm_89)
    "L40":         GpuSpec(vram_gb=48,  compute_capability="sm_89"),
    "L40S":        GpuSpec(vram_gb=48,  compute_capability="sm_89"),
    "L4":          GpuSpec(vram_gb=24,  compute_capability="sm_89"),
    "RTX-4090":    GpuSpec(vram_gb=24,  compute_capability="sm_89"),
    "RTX-6000-ADA":GpuSpec(vram_gb=48,  compute_capability="sm_89"),
    # Hopper generation (sm_90)
    "H100-80GB":   GpuSpec(vram_gb=80,  compute_capability="sm_90"),
    "H100-NVL":    GpuSpec(vram_gb=94,  compute_capability="sm_90"),
    "H200":        GpuSpec(vram_gb=141, compute_capability="sm_90"),
    # Blackwell generation (sm_120) — NOT YET compatible with the default
    # PyTorch 2.4.0 image used by RunPod's runpod/pytorch:2.4.0-cuda12.4.1.
    # Upgrade to a PyTorch 2.5+ image before requesting these.
    "RTX-5090":    GpuSpec(vram_gb=32,  compute_capability="sm_120"),
    "RTX-5080":    GpuSpec(vram_gb=16,  compute_capability="sm_120"),
    "B200":        GpuSpec(vram_gb=192, compute_capability="sm_120"),
}


# Compute capabilities the default PyTorch 2.4.0 image supports.  The
# orchestrator uses this to warn early when an operator requests a card the
# image's kernel set can't drive.
PYTORCH_2_4_SM_SUPPORTED: Set[str] = {
    "sm_50", "sm_60", "sm_70", "sm_75",
    "sm_80", "sm_86", "sm_89", "sm_90",
}


def gpu_supports_pytorch_2_4(canonical: str) -> bool:
    """True iff the canonical GPU's compute capability is in
    :data:`PYTORCH_2_4_SM_SUPPORTED`.  Used by the orchestrator to filter
    incompatible candidates before paying for a pod."""
    spec = GPU_CANONICAL.get(canonical)
    if spec is None:
        return True  # unknown to us — defer to the plugin's gpu_sku()
    return spec.compute_capability in PYTORCH_2_4_SM_SUPPORTED


# ---------------------------------------------------------------------------
# Provider ABC
# ---------------------------------------------------------------------------

class CloudProvider(abc.ABC):
    """Abstract base for a cloud-GPU provider plugin.

    Implementations carry their own credentials, their own SKU naming, and
    their own create/poll/destroy implementations.  The orchestrator only
    sees the methods declared here.

    Subclasses MUST set ``name``, ``default_image``, and ``default_gpu``
    as class attributes; the orchestrator uses these for CLI defaults and
    diagnostic output.
    """

    # --- Class-level attributes (subclasses must set) -------------------
    name: str = ""                  # e.g. "runpod", "vultr", "lambda_labs"
    default_image: str = ""         # Docker image or VM template id
    default_gpu: str = "A40"        # canonical GPU name from GPU_CANONICAL
    approval_gated_gpus: Set[str] = frozenset()  # canonical names
    cloud_type_choices: List[str] = []  # e.g. ["SECURE", "COMMUNITY"] for RunPod;
                                        # ["on-demand"] for Vultr; empty if not applicable

    # --- Required methods (subclasses must implement) --------------------

    @abc.abstractmethod
    def __init__(self, **opts: Any) -> None:
        """Construct the provider from environment/options.

        Implementations should read their provider-specific API-key env var
        here (e.g. ``RUNPOD_API_KEY``) and raise :class:`AuthError` if it
        is missing or invalid.  Additional opts may be passed via the
        orchestrator's CLI (e.g. ``--cloud SECURE`` for RunPod).
        """

    @abc.abstractmethod
    def gpu_sku(self, canonical: str) -> str:
        """Map a canonical GPU name (e.g. ``"A40"``) to this provider's SKU.

        Raises :class:`UnknownGpuError` if the provider does not offer this
        canonical GPU.
        """

    @abc.abstractmethod
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
    ) -> "CloudInstance":
        """Create a single GPU instance synchronously.

        Returns a :class:`CloudInstance` (the provider's instance handle).
        Raises :class:`CapacityError` on capacity-out (soft, orchestrator
        will try the next candidate), :class:`AuthError` /
        :class:`BalanceError` on credential / funding issues (hard abort),
        :class:`ProvisioningError` otherwise.
        """

    @abc.abstractmethod
    def wait_for_ready(
        self, instance: "CloudInstance", timeout_s: int = 900
    ) -> "CloudInstance":
        """Poll until the instance is SSH-reachable, then return it.

        Implementations may mutate the instance to record the SSH endpoint.
        Raises :class:`TimeoutError` if the instance does not become
        SSH-ready within ``timeout_s`` seconds.
        """

    @abc.abstractmethod
    def destroy(self, instance: "CloudInstance") -> None:
        """Best-effort delete.  Errors should be logged, not raised."""

    # --- Re-attach (used by run_on_cloud.py --reattach) ------------------

    def attach_from_provisioning(self, prov: Dict[str, Any]) -> "CloudInstance":
        """Reconstruct a live :class:`CloudInstance` from a persisted
        ``pod_provisioning.json`` dict, suitable for the reattach path
        (:mod:`scripts.run_on_cloud` ``--reattach``).

        ``prov`` is the dict written by the dispatcher at provisioning time
        (provider's :meth:`provisioning_dict` plus a few orchestrator-added
        fields like ``benchmark`` and ``smoke``).  Implementations MUST:

          1. Reconstruct the provider-specific instance handle from
             ``prov`` (only fields the provider itself wrote via
             :meth:`CloudInstance.provenance` are guaranteed to be present).
          2. Refresh any fields that may have changed since provisioning
             (e.g. RunPod's SSH host/port can rotate across pod restarts) so
             the returned handle is immediately usable for SSH and destroy.
          3. Return the live handle.

        Implementations may call :meth:`wait_for_ready` internally to
        re-populate transient fields, but should treat a non-RUNNING pod as
        a hard error (we are reattaching to an already-launched run, so the
        pod must already be up).

        The default implementation raises :class:`NotImplementedError`;
        providers that don't yet support reattach inherit this default so
        the failure mode is loud rather than silent.
        """
        raise NotImplementedError(
            f"Provider {self.name!r} does not implement attach_from_provisioning. "
            "Re-attach is currently only supported on RunPod; "
            "open a PR adding the method to this plugin to enable it."
        )

    # --- Optional methods (default to no-op / None) ----------------------

    def cost_per_hour_usd(self, instance: "CloudInstance") -> Optional[float]:
        """Per-hour billing rate for this instance, in USD.

        Default implementation returns the instance's
        :attr:`CloudInstance.cost_per_hour_usd` if set, else None.
        """
        return getattr(instance, "cost_per_hour_usd", None)

    def provisioning_dict(self, instance: "CloudInstance") -> Dict[str, Any]:
        """Return a JSON-serializable provenance dict for ``pod_provisioning.json``.

        The orchestrator merges the dispatcher's own fields (benchmark name,
        smoke flag, created_utc) on top; this method only needs to supply
        provider-specific fields (instance_id, gpu SKU, region, etc.).
        """
        return instance.provenance()


# ---------------------------------------------------------------------------
# CloudInstance — opaque provider-specific instance handle
# ---------------------------------------------------------------------------

class CloudInstance(abc.ABC):
    """Opaque per-instance handle returned by :meth:`CloudProvider.create`.

    Each provider subclasses this with its own fields (REST IDs, IP+port
    pairs, machine metadata).  The orchestrator only uses the four methods
    declared here.
    """

    @abc.abstractmethod
    def ssh_target(self) -> List[str]:
        """Return the SSH-arg list, e.g. ``["-p", "17445", "root@1.2.3.4"]``.

        The orchestrator splices this into ``["ssh", *SSH_OPTS, *target]``
        when shelling out.
        """

    @abc.abstractmethod
    def label(self) -> str:
        """Human-readable identifier for logs, e.g. ``"ssh root@1.2.3.4 -p 17445"``."""

    @abc.abstractmethod
    def provenance(self) -> Dict[str, Any]:
        """JSON-serializable per-instance snapshot for ``pod_provisioning.json``."""

    # Subclasses may set this to surface billing rate; default None.
    cost_per_hour_usd: Optional[float] = None


# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------
#
# Providers self-register by being importable from
# ``scripts/cloud_providers/<name>.py`` and exposing a top-level
# ``PROVIDER_CLASS = <ConcreteProvider>``.  The orchestrator calls
# :func:`get_provider` with the operator-supplied name to instantiate.

_REGISTRY: Dict[str, str] = {
    # canonical name → module path under scripts/cloud_providers/
    "runpod":      "cloud_providers.runpod",
    "vultr":       "cloud_providers.vultr",
    "lambda_labs": "cloud_providers.lambda_labs",
}


def list_providers() -> List[str]:
    """Return the registered provider names."""
    return sorted(_REGISTRY.keys())


def get_provider(name: str, **opts: Any) -> CloudProvider:
    """Instantiate and return the named provider.

    Raises ``KeyError`` if the name is unknown.  Provider-specific opts
    (``cloud_type``, ``regions``, etc.) are passed through to the provider's
    ``__init__``.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown cloud provider {name!r}. "
            f"Registered: {list_providers()}"
        )
    module = importlib.import_module(_REGISTRY[name])
    cls = getattr(module, "PROVIDER_CLASS", None)
    if cls is None:
        raise RuntimeError(
            f"Provider module {_REGISTRY[name]!r} does not expose "
            f"`PROVIDER_CLASS`; cannot instantiate {name!r}."
        )
    return cls(**opts)
