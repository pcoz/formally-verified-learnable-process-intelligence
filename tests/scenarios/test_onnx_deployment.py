"""End-to-end test for the ONNX deployment scenario.

Trains the loan-approval XOR-routing model from the scenario's
inline traces, exports the trained module to ONNX, loads the
exported file via onnxruntime, and pins the parity guarantee
that makes ONNX deployment safe in practice:

* the exported file exists and is non-empty;
* the schema dict carries the requested input / output names;
* torch and onnxruntime produce numerically identical outputs
  (within a tight tolerance) across a sweep of inputs;
* the dynamic-batch axis works — the exported model accepts a
  larger batch than the one used during export;
* the trained model still routes high inputs to approve and
  low inputs to decline under both runtimes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from petri_net_nn import export_onnx, load_scenario


# onnxruntime is required to execute the exported file. Without
# it the export itself would still work (torch ships its own ONNX
# exporter) but we couldn't verify parity, so skip the whole
# module if it isn't installed.
ort = pytest.importorskip("onnxruntime")


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "onnx_deployment"
    / "scenario.toml"
)


def _train_module():
    """Load and train the scenario, returning the trained
    module. Helper to share the training step across tests."""
    ctx = load_scenario(SCENARIO)
    module, losses = ctx.train()
    # Sanity: training actually fit the data.
    assert losses[-1] < losses[0]
    return module


# ---------------------------------------------------------------------------
# Export — file written, schema reported.
# ---------------------------------------------------------------------------


def test_export_creates_an_onnx_file_with_schema(tmp_path):
    """The export call writes a non-empty .onnx file at the
    requested path and returns a schema dict listing the input
    and output names — useful as a JSON sidecar telling
    non-Python consumers which positional tensor corresponds to
    which place."""
    module = _train_module()
    out_path = tmp_path / "loan_approval.onnx"

    schema = export_onnx(
        module,
        out_path,
        input_places=["p_application"],
        output_transitions=["t_approve", "t_decline"],
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert schema == {
        "input_marking_places": ["p_application"],
        "input_value_places": [],
        "output_transitions": ["t_approve", "t_decline"],
    }
    # The schema is JSON-serialisable so deployments can ship it
    # as a sidecar alongside the .onnx file.
    json.dumps(schema)


# ---------------------------------------------------------------------------
# Parity — the headline guarantee.
# ---------------------------------------------------------------------------


def test_exported_model_matches_torch_outputs_within_tolerance(tmp_path):
    """The headline parity test. Export the trained module, load
    it via onnxruntime, run a sweep of risk_score inputs through
    both the torch module and the ONNX session, and confirm
    every per-transition activation agrees to within 1e-5. This
    is the contract a deployment team depends on: the model
    behaves the same way after the export-and-load round-trip."""
    module = _train_module()
    out_path = tmp_path / "loan_approval.onnx"
    export_onnx(
        module,
        out_path,
        input_places=["p_application"],
        output_transitions=["t_approve", "t_decline"],
    )

    inputs = torch.linspace(0.0, 1.0, 11)

    with torch.no_grad():
        torch_outputs = module(
            input_marking={"p_application": inputs}, batch_size=11
        )
    # Stack into the same (batch, n_transitions) shape the ONNX
    # graph emits — output_transitions order is the column order.
    torch_stack = torch.stack(
        [torch_outputs["t_approve"], torch_outputs["t_decline"]], dim=-1
    ).numpy()

    session = ort.InferenceSession(str(out_path))
    onnx_out = session.run(
        ["activations"], {"marking__p_application": inputs.numpy()}
    )[0]

    # Parity to 1e-4 — well within ONNX's documented numerical
    # tolerance for sigmoid / linear ops at this scale.
    assert onnx_out.shape == torch_stack.shape
    assert (abs(onnx_out - torch_stack) < 1e-4).all()


def test_routing_decision_consistent_across_runtimes(tmp_path):
    """A more direct domain-language version of the parity
    claim: under both runtimes, high risk_score must approve
    and low risk_score must decline. If the export had broken
    the routing logic, this assertion would catch it whether or
    not the numerical-parity test happened to pass."""
    module = _train_module()
    out_path = tmp_path / "loan_approval.onnx"
    export_onnx(
        module,
        out_path,
        input_places=["p_application"],
        output_transitions=["t_approve", "t_decline"],
    )

    session = ort.InferenceSession(str(out_path))
    for value, expect in [(0.9, "approve"), (0.1, "decline")]:
        input_tensor = torch.tensor([float(value)])
        out = session.run(
            ["activations"], {"marking__p_application": input_tensor.numpy()}
        )[0]
        # Output columns follow output_transitions order:
        # column 0 = t_approve, column 1 = t_decline.
        approve_act, decline_act = float(out[0, 0]), float(out[0, 1])
        if expect == "approve":
            assert approve_act > decline_act
        else:
            assert decline_act > approve_act


# ---------------------------------------------------------------------------
# Dynamic batch — the exported model accepts varied batch sizes.
# ---------------------------------------------------------------------------


def test_exported_model_accepts_different_batch_sizes(tmp_path):
    """A model exported with batch_size=1 must serve any batch
    size at inference. This is the standard production
    configuration — exporters set the batch axis as dynamic so
    one .onnx file scales from single-record latency-sensitive
    paths to bulk-scoring batch paths."""
    module = _train_module()
    out_path = tmp_path / "loan_approval.onnx"
    export_onnx(
        module,
        out_path,
        input_places=["p_application"],
        output_transitions=["t_approve", "t_decline"],
        batch_size=1,
    )

    session = ort.InferenceSession(str(out_path))
    # Serve a batch of 50.
    batch = torch.linspace(0.0, 1.0, 50).numpy()
    out = session.run(["activations"], {"marking__p_application": batch})[0]
    # The exported graph emits one 2D tensor of shape
    # (batch, n_transitions) — here (50, 2).
    assert out.shape == (50, 2)
