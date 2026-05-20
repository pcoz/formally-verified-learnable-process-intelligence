"""Export a trained :class:`PetriNetModule` to ONNX.

ONNX is the cross-runtime tensor-graph format. A PETRA module
exported here runs unchanged in **any** ONNX runtime — Python's
``onnxruntime``, C++, Java, browser (via onnxruntime-web), mobile,
and most accelerator stacks. That covers the
"deploy outside Python" half of the productionisation story
without forcing PETRA to ship platform-specific bindings.

The wrinkle PETRA introduces over an off-the-shelf
``torch.onnx.export`` call is the **dict-of-tensors input**:
:meth:`PetriNetModule.forward` takes
``input_marking={place_id: tensor}`` (and optionally
``input_values={place_id: tensor}``), not positional tensors.
ONNX's tracing model expects positional inputs with named axes.
The bridge:

1. The caller declares a **schema** — the ordered list of place
   ids that will become positional inputs at inference time.
2. We wrap the module in a small ``nn.Module`` whose forward
   takes positional tensors, re-keys them into the dict shapes
   the underlying module expects, runs the forward pass, and
   returns a stack of the requested output transitions'
   activations.
3. We call ``torch.onnx.export`` on the wrapper.

What this lets you do: ship a trained PETRA model as an ``.onnx``
file with a small JSON sidecar describing the schema (which input
tensor names map to which places). Consumers in any language can
load the ONNX file, pass tensors in the documented order, and get
out the per-transition activations.

**Limitations carried forward:**

* Time-unrolled mode with non-uniform transition durations uses
  per-transition Python lists for the in-flight queue (see
  :mod:`petri_net_nn.compiler`). Those lists don't survive ONNX
  tracing. Acyclic mode (``num_steps == 0``) and time-unrolled
  mode where every transition has duration 1 export cleanly.
* Softmax routing exports correctly. STE firing exports the
  forward path (the autograd detach trick is only relevant for
  training).
* Inhibitor arcs and structural-guard sigmoid gates export
  cleanly as standard tensor ops.
* ``torch_guard`` / ``torch_output_value`` callables export iff
  their bodies use only ONNX-exportable torch ops (most do).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from petri_net_nn.compiler import PetriNetModule


class _PositionalWrapper(nn.Module):
    """Adapter that turns the dict-input forward into a
    positional-tensor-input forward, so ``torch.onnx.export`` can
    trace it cleanly.

    Construction takes the underlying module plus the schema —
    the order in which positional inputs map to place ids on the
    marking and (optionally) value channels, plus the transition
    ids whose activations we want as outputs.

    Forward accepts the positional tensors in the documented
    order, re-keys them into the dict shapes the module's forward
    expects, runs it, and returns the stacked output-transition
    activations as a single ``(batch, n_outputs)`` tensor.
    """

    def __init__(
        self,
        module: PetriNetModule,
        input_places: tuple[str, ...],
        input_value_places: tuple[str, ...],
        output_transitions: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.inner = module
        # Tuples (not lists) so they're treated as static attributes
        # by torch's tracer — the schema can't change between
        # invocations of the exported graph.
        self.input_places = input_places
        self.input_value_places = input_value_places
        self.output_transitions = output_transitions

    def forward(self, *positional_tensors: torch.Tensor) -> torch.Tensor:
        # The positional inputs are interleaved as
        # [marking_0, marking_1, …, marking_{m-1},
        #  value_0,   value_1,   …, value_{v-1}].
        # We re-key them into the dict form the wrapped module's
        # forward expects.
        n_marking = len(self.input_places)
        marking_tensors = positional_tensors[:n_marking]
        value_tensors = positional_tensors[n_marking:]

        input_marking = {
            place: tensor
            for place, tensor in zip(self.input_places, marking_tensors)
        }
        input_values: dict[str, torch.Tensor] | None
        if self.input_value_places:
            input_values = {
                place: tensor
                for place, tensor in zip(self.input_value_places, value_tensors)
            }
        else:
            input_values = None

        out = self.inner(
            input_marking=input_marking,
            input_values=input_values,
            # Infer batch_size from the first input tensor — torch
            # tracing carries this through symbolically when
            # dynamic_axes is set, so the exported model accepts
            # any batch size at runtime.
            batch_size=marking_tensors[0].shape[0],
        )
        # Stack the requested output activations along a new last
        # axis: shape (batch, n_outputs). Single-output graphs
        # still get a (batch, 1) tensor, keeping consumer code
        # uniform.
        stacked = torch.stack(
            [out[t] for t in self.output_transitions], dim=-1
        )
        return stacked


def export_onnx(
    module: PetriNetModule,
    output_path: str | Path,
    *,
    input_places: list[str],
    input_value_places: list[str] | None = None,
    output_transitions: list[str] | None = None,
    batch_size: int = 1,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> dict[str, list[str]]:
    """Export a trained module to ONNX.

    Parameters
    ----------
    module
        The trained :class:`PetriNetModule` to export. ``eval()``
        is called on it before tracing so any training-mode
        randomness is disabled.
    output_path
        Where to write the ``.onnx`` file. Parent directory must
        exist.
    input_places
        Ordered list of place ids whose activations are inputs at
        inference time. The exported graph's first ``len(input_places)``
        positional tensors correspond to these places, in this
        order. Place ids must exist on the module's net.
    input_value_places
        Optional ordered list of place ids whose **values** are
        inputs (the coloured-Petri-net value channel). Omit for
        pure marking-channel models. When supplied, the exported
        graph's next positional tensors after the markings carry
        these values.
    output_transitions
        Ordered list of transition ids whose firing activations
        are returned. Defaults to all transitions whose label is
        not an auto-generated gateway label (matches the
        ``train_on_traces`` / ``anomaly_score`` convention).
    batch_size
        Batch size to use for the tracing example tensors. The
        exported graph supports arbitrary batch sizes at runtime
        when ``dynamic_batch=True`` (the default).
    opset_version
        ONNX opset to target. 17 is the modern baseline supported
        by all currently-shipping ONNX runtimes.
    dynamic_batch
        When ``True`` (default), mark the batch dimension as
        dynamic on every input and output. The exported graph
        then accepts any batch size at inference time. Set to
        ``False`` only for very space-constrained deployments
        where shape inference is faster with a fixed batch.

    Returns
    -------
    dict[str, list[str]]
        A schema dict the caller can serialise (JSON, TOML, etc.)
        alongside the .onnx file. Keys:

        * ``"input_marking_places"`` — names of the marking
          inputs in the exported graph (``marking__<place_id>``);
        * ``"input_value_places"`` — names of the value inputs;
        * ``"output_transitions"`` — names of the output
          transitions in the order they're stacked along the
          last axis of the returned tensor.

        Consumers in other languages use this schema to feed
        tensors in the right order.
    """
    output_path = Path(output_path)

    # Validate the requested schema against the module's net —
    # bad place / transition names are easier to debug here than
    # in a downstream ONNX runtime error.
    net = module.net
    for place in input_places:
        if place not in net.places:
            raise ValueError(
                f"export_onnx: input_place {place!r} is not in the "
                f"net's place set"
            )
    if input_value_places is not None:
        for place in input_value_places:
            if place not in net.places:
                raise ValueError(
                    f"export_onnx: input_value_place {place!r} is "
                    f"not in the net's place set"
                )

    # Default the output set to "all scoreable transitions" —
    # matches the train_on_traces / anomaly_score convention of
    # skipping auto-generated gateway labels (those containing
    # "->" in their label).
    if output_transitions is None:
        output_transitions = sorted(
            t for t in net.transitions
            if "->" not in net.transition_labels.get(t, t)
        )
    for t in output_transitions:
        if t not in net.transitions:
            raise ValueError(
                f"export_onnx: output_transition {t!r} is not in "
                f"the net's transition set"
            )

    # Build the positional-input wrapper around the trained
    # module. Note: the wrapper holds a reference to the same
    # underlying nn.Parameter set, so we're not copying weights.
    wrapper = _PositionalWrapper(
        module=module,
        input_places=tuple(input_places),
        input_value_places=tuple(input_value_places or ()),
        output_transitions=tuple(output_transitions),
    )
    wrapper.eval()

    # Example inputs for tracing — one tensor per (marking) place
    # plus one per (value) place. The actual values don't matter
    # for tracing; what matters is the shapes.
    example_marking = [
        torch.full((batch_size,), 0.5) for _ in input_places
    ]
    example_values = [
        torch.full((batch_size,), 1.0)
        for _ in (input_value_places or [])
    ]
    example_inputs = tuple(example_marking + example_values)

    # Name the ONNX graph's inputs and outputs after the place /
    # transition ids so downstream consumers can index them by
    # the same names PETRA used.
    input_names = (
        [f"marking__{p}" for p in input_places]
        + [f"value__{p}" for p in (input_value_places or [])]
    )
    output_names = ["activations"]

    # Mark the batch axis as dynamic so the exported graph isn't
    # locked to a single batch size.
    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in input_names}
        # The output tensor has shape (batch, n_outputs); batch
        # is the dynamic axis. n_outputs is static (equal to the
        # length of output_transitions at export time).
        dynamic_axes["activations"] = {0: "batch"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use the legacy (tracing-based) exporter rather than the new
    # dynamo-based one. The legacy path has no extra dependency
    # beyond torch itself; the dynamo path requires onnxscript.
    # The output ONNX file is functionally equivalent for the
    # tensor-op surface PETRA uses.
    torch.onnx.export(
        wrapper,
        example_inputs,
        str(output_path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        # do_constant_folding shrinks the graph by precomputing
        # nodes whose inputs are constants. Safe to leave on for
        # inference-only exports.
        do_constant_folding=True,
        dynamo=False,
    )

    return {
        "input_marking_places": list(input_places),
        "input_value_places": list(input_value_places or []),
        "output_transitions": list(output_transitions),
    }
