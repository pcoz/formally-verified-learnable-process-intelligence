"""Tests for the ONNX export path.

We require onnxruntime to be installed (it lives in the
``[onnx]`` optional-deps group in pyproject.toml). When it isn't,
the entire test module is skipped — the export code itself only
needs torch's built-in ONNX support, but the *parity* tests need
a runtime that can load and execute the exported file.

The load-bearing assertions:

  * the exported file loads in onnxruntime;
  * the exported model's outputs match the original torch
    module's forward pass to within a numerical tolerance, on
    representative inputs;
  * the exported model accepts a different batch size than the
    one used during export (dynamic-batch axis works);
  * value-channel exports carry the coloured-token value channel
    through correctly;
  * bad place / transition names raise a useful ValueError
    rather than failing inside torch.onnx.export with a cryptic
    message.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from petri_net_nn import PetriNet, PetriNetModule, export_onnx, load_scenario


# Skip the whole module when onnxruntime isn't installed — the
# parity tests need it to load and execute the exported file.
ort = pytest.importorskip("onnxruntime")


def _xor_module():
    """A 2-transition XOR-shape net with hand-set thresholds so
    the exported model has predictable activations to compare
    against. Both transitions share input p_in and produce
    different output places."""
    net = PetriNet()
    net.add_place("p_in")
    net.add_place("p_a")
    net.add_place("p_b")
    net.add_transition("t_a", label="Path A")
    net.add_transition("t_b", label="Path B")
    net.add_arc("p_in", "t_a")
    net.add_arc("t_a", "p_a")
    net.add_arc("p_in", "t_b")
    net.add_arc("t_b", "p_b")
    module = PetriNetModule(net, sharpness=4.0)
    # Pin weights / thresholds so the test is deterministic.
    module.arc_weights[module._arc_key[("p_in", "t_a")]].data = torch.tensor(1.0)
    module.arc_weights[module._arc_key[("p_in", "t_b")]].data = torch.tensor(1.0)
    module.transition_thresholds[module._threshold_key["t_a"]].data = torch.tensor(0.3)
    module.transition_thresholds[module._threshold_key["t_b"]].data = torch.tensor(0.7)
    return module


def test_export_onnx_writes_a_file_and_returns_schema(tmp_path):
    """A round-trip: export to a fresh path, verify the file
    exists and is non-empty, and confirm the returned schema
    dict carries the requested input and output names."""
    module = _xor_module()
    out_path = tmp_path / "xor.onnx"

    schema = export_onnx(
        module,
        out_path,
        input_places=["p_in"],
        output_transitions=["t_a", "t_b"],
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert schema == {
        "input_marking_places": ["p_in"],
        "input_value_places": [],
        "output_transitions": ["t_a", "t_b"],
    }


def test_exported_model_outputs_match_torch_within_tolerance(tmp_path):
    """The headline parity test. Export the XOR module, load it
    via onnxruntime, run the same inputs through both the torch
    module and the ONNX session, and confirm the outputs agree."""
    module = _xor_module()
    out_path = tmp_path / "xor.onnx"
    export_onnx(
        module,
        out_path,
        input_places=["p_in"],
        output_transitions=["t_a", "t_b"],
    )

    test_inputs = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    with torch.no_grad():
        torch_out = module(
            input_marking={"p_in": test_inputs},
            batch_size=5,
        )
    torch_stack = torch.stack(
        [torch_out["t_a"], torch_out["t_b"]], dim=-1
    ).numpy()

    session = ort.InferenceSession(str(out_path))
    onnx_out = session.run(
        ["activations"],
        {"marking__p_in": test_inputs.numpy()},
    )[0]

    # Allow a small tolerance; ONNX tracing can introduce minor
    # numerical drift in the sigmoid path on some opsets.
    assert (abs(onnx_out - torch_stack) < 1e-4).all()


def test_exported_model_accepts_different_batch_size(tmp_path):
    """The exported graph is built with batch_size=1 by default
    but marked dynamic on axis 0. Running it with batch_size=8
    at inference time should work without re-exporting."""
    module = _xor_module()
    out_path = tmp_path / "xor.onnx"
    export_onnx(
        module,
        out_path,
        input_places=["p_in"],
        output_transitions=["t_a", "t_b"],
        batch_size=1,
    )

    session = ort.InferenceSession(str(out_path))
    batch_input = torch.linspace(0.0, 1.0, 8).numpy()
    out = session.run(
        ["activations"],
        {"marking__p_in": batch_input},
    )[0]
    assert out.shape == (8, 2)


def test_value_channel_export_preserves_routing(tmp_path):
    """The CPN credit-approval scenario's trained guard threshold
    sits in the empirical decision band 900-1500. Export with
    the value channel as an input and verify the ONNX model's
    routing matches the torch module's on high / low amounts."""
    ctx = load_scenario(
        Path(__file__).parent.parent
        / "examples"
        / "credit_approval_coloured"
        / "scenario.toml"
    )
    module, _ = ctx.train()

    out_path = tmp_path / "credit.onnx"
    export_onnx(
        module,
        out_path,
        input_places=["p_submitted"],
        input_value_places=["p_submitted"],
        output_transitions=["t_approve", "t_decline"],
    )

    # High and low amounts at the same activation.
    marking = torch.tensor([1.0, 1.0])
    values = torch.tensor([5000.0, 300.0])

    with torch.no_grad():
        torch_out = module(
            input_marking={"p_submitted": marking},
            input_values={"p_submitted": values},
            batch_size=2,
        )
    torch_stack = torch.stack(
        [torch_out["t_approve"], torch_out["t_decline"]], dim=-1
    ).numpy()

    session = ort.InferenceSession(str(out_path))
    onnx_out = session.run(
        ["activations"],
        {
            "marking__p_submitted": marking.numpy(),
            "value__p_submitted": values.numpy(),
        },
    )[0]
    assert (abs(onnx_out - torch_stack) < 1e-3).all()


def test_export_rejects_unknown_input_place(tmp_path):
    """Passing a place id that doesn't exist on the net is a
    misconfiguration — the function should raise with a clear
    message before calling torch.onnx.export and producing a
    cryptic trace."""
    module = _xor_module()
    with pytest.raises(ValueError, match="not in the net's place set"):
        export_onnx(
            module,
            tmp_path / "bad.onnx",
            input_places=["p_does_not_exist"],
        )


def test_export_rejects_unknown_output_transition(tmp_path):
    """Same for transition ids — bad outputs should raise here,
    not in the runtime."""
    module = _xor_module()
    with pytest.raises(ValueError, match="not in the net's transition set"):
        export_onnx(
            module,
            tmp_path / "bad.onnx",
            input_places=["p_in"],
            output_transitions=["t_does_not_exist"],
        )


def test_schema_dict_is_json_serialisable(tmp_path):
    """The schema dict is meant to ship alongside the .onnx file
    as a sidecar (JSON, TOML, whatever the consumer prefers).
    Verify it round-trips through json.dumps / json.loads."""
    module = _xor_module()
    schema = export_onnx(
        module,
        tmp_path / "xor.onnx",
        input_places=["p_in"],
        output_transitions=["t_a", "t_b"],
    )
    s = json.dumps(schema)
    restored = json.loads(s)
    assert restored == schema
