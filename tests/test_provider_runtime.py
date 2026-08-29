import argparse
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import onnxruntime_utils as runtime
from parseq import PARSEQ
from rtmdet import RTMDet


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("cpu", "cpu"),
        ("CPU", "cpu"),
        (" cuda ", "cuda"),
        ("dml", "directml"),
        ("DiReCtMl", "directml"),
    ],
)
def test_normalize_device_accepts_case_and_directml_aliases(requested, expected):
    assert runtime.normalize_device(requested) == expected


def test_normalize_device_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported device"):
        runtime.normalize_device("rocm")


def test_execution_provider_order_is_accelerator_then_cpu(monkeypatch):
    monkeypatch.setattr(
        runtime.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"],
    )

    assert runtime.execution_providers("cpu") == ["CPUExecutionProvider"]
    assert runtime.execution_providers("CUDA") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert runtime.execution_providers("dml") == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]


@pytest.mark.parametrize("requested", ["cuda", "directml"])
def test_unavailable_accelerator_fails_explicitly(monkeypatch, requested):
    monkeypatch.setattr(
        runtime.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    with pytest.raises(runtime.ProviderUnavailableError, match="not available"):
        runtime.execution_providers(requested)


def test_session_options_are_forwarded_to_inference_session(monkeypatch):
    calls = []

    class FakeSession:
        pass

    def fake_inference_session(model_path, *, sess_options, providers):
        calls.append((model_path, sess_options, providers))
        return FakeSession()

    monkeypatch.setattr(
        runtime.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    monkeypatch.setattr(runtime.onnxruntime, "InferenceSession", fake_inference_session)
    options = object()

    session = runtime.create_inference_session(
        "model.onnx", device="cuda", session_options=options
    )

    assert isinstance(session, FakeSession)
    assert calls == [
        ("model.onnx", options, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    ]


def test_accelerator_session_must_report_requested_provider(monkeypatch):
    class CpuOnlySession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(
        runtime.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    monkeypatch.setattr(
        runtime.onnxruntime,
        "InferenceSession",
        lambda *args, **kwargs: CpuOnlySession(),
    )

    with pytest.raises(runtime.ProviderUnavailableError, match="was not activated"):
        runtime.create_inference_session("model.onnx", device="cuda")


def test_session_initialization_failure_is_reported_with_requested_device(monkeypatch):
    monkeypatch.setattr(
        runtime.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider", "DmlExecutionProvider"],
    )

    def fail_to_create_session(*args, **kwargs):
        raise RuntimeError("DirectML adapter could not be initialized")

    monkeypatch.setattr(runtime.onnxruntime, "InferenceSession", fail_to_create_session)

    with pytest.raises(runtime.ProviderUnavailableError, match="directml"):
        runtime.create_inference_session("model.onnx", device="directml")


def test_cpu_session_initialization_preserves_original_exception(monkeypatch):
    monkeypatch.setattr(
        runtime.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    original_error = ValueError("invalid model graph")

    def fail_to_create_session(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(runtime.onnxruntime, "InferenceSession", fail_to_create_session)

    with pytest.raises(ValueError) as exc_info:
        runtime.create_inference_session("model.onnx", device="cpu")

    assert exc_info.value is original_error


class _FakeSession:
    calls = []

    def __init__(self, model_path, *, sess_options, providers):
        self.model_path = model_path
        self.sess_options = sess_options
        self.providers = providers
        self.inputs = [SimpleNamespace(name="input", shape=[1, 3, 32, 384])]
        self.outputs = [SimpleNamespace(name="output")]
        _FakeSession.calls.append(self)

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs


def test_parseq_uses_shared_directml_session_path(monkeypatch):
    _FakeSession.calls.clear()
    monkeypatch.setattr(
        runtime.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider", "DmlExecutionProvider"],
    )
    monkeypatch.setattr(runtime.onnxruntime, "InferenceSession", _FakeSession)

    PARSEQ("model.onnx", ["a"], device="DiReCtMl")

    session = _FakeSession.calls[-1]
    assert session.providers == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert session.sess_options.graph_optimization_level == (
        runtime.onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    assert session.sess_options.enable_mem_pattern is False
    assert session.sess_options.execution_mode == (
        runtime.onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    )


def test_rtmdet_uses_shared_cuda_session_path(monkeypatch, tmp_path):
    _FakeSession.calls.clear()
    monkeypatch.setattr(
        runtime.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    monkeypatch.setattr(runtime.onnxruntime, "InferenceSession", _FakeSession)
    classes_path = tmp_path / "classes.yaml"
    classes_path.write_text("names:\n  - line_main\n", encoding="utf-8")

    RTMDet("model.onnx", str(classes_path), device="CUDA")

    session = _FakeSession.calls[-1]
    assert session.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert session.sess_options.graph_optimization_level == (
        runtime.onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    assert session.sess_options.enable_mem_pattern is True


def test_models_keep_cpu_as_the_default_session_path(monkeypatch, tmp_path):
    _FakeSession.calls.clear()
    monkeypatch.setattr(
        runtime.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(runtime.onnxruntime, "InferenceSession", _FakeSession)
    classes_path = tmp_path / "classes.yaml"
    classes_path.write_text("names:\n  - line_main\n", encoding="utf-8")

    PARSEQ("parseq.onnx", ["a"])
    RTMDet("rtmdet.onnx", str(classes_path))

    assert [session.providers for session in _FakeSession.calls] == [
        ["CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]


def test_cli_device_argument_returns_canonical_value():
    parser = argparse.ArgumentParser()
    runtime.add_device_argument(parser)

    assert parser.parse_args(["--device", "dMl"]).device == "directml"


def test_cli_device_argument_explains_unknown_value(capsys):
    parser = argparse.ArgumentParser()
    runtime.add_device_argument(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "rocm"])

    assert "choose cpu, cuda, dml, or directml" in capsys.readouterr().err


def test_ocr_cli_uses_common_device_argument():
    source = (SRC_DIR / "ocr.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "add_device_argument"
    ]

    assert calls
    assert "build_arg_parser" in {
        parent.name
        for parent in tree.body
        if isinstance(parent, ast.FunctionDef)
        for child in ast.walk(parent)
        if child in calls
    }


def test_gui_entry_uses_common_device_argument():
    gui_main = Path(__file__).resolve().parents[1] / "ndlkotenocr-lite-gui" / "main.py"

    assert "ocr.add_device_argument(parser)" in gui_main.read_text(encoding="utf-8")


def test_readme_documents_optional_provider_installation_without_mixing_packages():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )

    assert "onnxruntime-gpu" in readme
    assert "onnxruntime-directml" in readme
    assert "Do not install these provider packages together" in readme
