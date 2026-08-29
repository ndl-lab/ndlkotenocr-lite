"""Shared ONNX Runtime device and session handling.

The project deliberately keeps CPU execution as the default.  Accelerator
providers are opt-in and are checked before a session is created so that a
CPU-only installation cannot silently report a GPU run.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Optional

import onnxruntime


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
DIRECTML_PROVIDER = "DmlExecutionProvider"

_DEVICE_ALIASES = {
    "cpu": "cpu",
    "cuda": "cuda",
    "dml": "directml",
    "directml": "directml",
}
_DEVICE_PROVIDERS = {
    "cpu": CPU_PROVIDER,
    "cuda": CUDA_PROVIDER,
    "directml": DIRECTML_PROVIDER,
}


class ProviderUnavailableError(RuntimeError):
    """Raised when the explicitly requested execution provider is unavailable."""


def normalize_device(device: str) -> str:
    """Return the canonical device name used by the OCR models.

    ``dml`` is retained as a short command-line alias for DirectML.  Unknown
    values are rejected instead of being treated as CPU, which prevents a
    misspelled accelerator request from looking like a successful CPU run.
    """

    if not isinstance(device, str):
        raise ValueError(f"Unsupported device {device!r}; choose cpu, cuda, dml, or directml")
    normalized = device.strip().casefold()
    try:
        return _DEVICE_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported device {device!r}; choose cpu, cuda, dml, or directml"
        ) from exc


def parse_device_argument(value: str) -> str:
    """Argparse-compatible wrapper that preserves the useful error message."""

    try:
        return normalize_device(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def execution_providers(
    device: str, available_providers: Optional[Iterable[str]] = None
) -> list[str]:
    """Resolve an explicit device to an ordered ONNX Runtime provider list.

    The requested accelerator is checked against the runtime's available
    providers.  Explicit accelerator requests fail clearly when unavailable;
    they do not fall back to CPU.  CPU remains available as the second
    provider for accelerator sessions, allowing supported operators to use
    CPU fallback after the requested provider has been successfully loaded.
    """

    canonical_device = normalize_device(device)
    provider = _DEVICE_PROVIDERS[canonical_device]
    if available_providers is None:
        available = tuple(onnxruntime.get_available_providers())
    else:
        available = tuple(available_providers)

    if CPU_PROVIDER not in available:
        raise ProviderUnavailableError(
            "CPUExecutionProvider is not available in this ONNX Runtime installation; "
            f"available providers: {list(available)!r}"
        )
    if provider not in available:
        raise ProviderUnavailableError(
            f"Requested device {canonical_device!r} requires {provider}, but it is not "
            f"available; available providers: {list(available)!r}"
        )

    if canonical_device == "cpu":
        return [CPU_PROVIDER]
    return [provider, CPU_PROVIDER]


def create_inference_session(
    model_path: str,
    device: str = "cpu",
    *,
    session_options=None,
):
    """Create an ONNX Runtime session using the shared provider policy.

    ``session_options`` is intentionally accepted and forwarded as
    ``sess_options`` so model-specific optimization settings are not silently
    discarded.  When an accelerator session reports that the requested
    provider was not activated, an explicit error is raised instead of
    returning a CPU-only session.
    """

    canonical_device = normalize_device(device)
    providers = execution_providers(canonical_device)
    if session_options is None:
        session_options = onnxruntime.SessionOptions()
    if canonical_device == "directml":
        session_options.enable_mem_pattern = False
        session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    try:
        session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers,
        )
    except Exception as exc:
        if canonical_device == "cpu":
            raise
        requested_provider = providers[0]
        raise ProviderUnavailableError(
            f"Could not initialize ONNX Runtime for device {canonical_device!r} "
            f"using {requested_provider}: {exc}"
        ) from exc

    if canonical_device != "cpu":
        get_providers = getattr(session, "get_providers", None)
        if callable(get_providers):
            active_providers = tuple(get_providers())
            if providers[0] not in active_providers:
                raise ProviderUnavailableError(
                    f"Requested device {canonical_device!r} was not activated; "
                    f"active providers: {list(active_providers)!r}"
                )
    return session


def add_device_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the common device option to a CLI or GUI argument parser."""

    parser.add_argument(
        "--device",
        type=parse_device_argument,
        default="cpu",
        metavar="{cpu,cuda,dml,directml}",
        help=(
            "Execution device (cpu by default; cuda or dml/directml require the "
            "matching ONNX Runtime provider)"
        ),
    )
    return parser
