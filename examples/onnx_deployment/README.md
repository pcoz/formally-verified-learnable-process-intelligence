# ONNX deployment — exported torch model, identical inference under onnxruntime

PETRA models are torch modules. For deployment outside Python
— or for inference in a non-Python service — torch ships an
**ONNX exporter** that emits an interchange-format `.onnx`
file. The exported file runs unchanged in any ONNX-aware
runtime: Python (onnxruntime), C++, Java, .NET, mobile,
browser (onnxruntime-web), plus accelerator stacks (TensorRT,
OpenVINO, DirectML).

This scenario demonstrates the export end to end: train a
small loan-approval XOR-routing model from inline traces,
call `export_onnx`, load the resulting file via
`onnxruntime`, and verify that the exported model gives the
same answers as the torch module across a sweep of inputs.

## What this scenario demonstrates

End-to-end through the `petri_net_nn.onnx_export` module:

1. **Train a small model** via the standard `load_scenario`
   path. The trained model routes on a single input
   (`risk_score`) — high values approve, low values decline.

2. **Export to ONNX.** `export_onnx(module, path,
   input_places=[...], output_transitions=[...])` writes an
   `.onnx` file and returns a schema dict describing the
   input / output names — useful as a JSON sidecar that tells
   non-Python consumers which positional tensor corresponds
   to which place.

3. **Load via onnxruntime** and run inference on the exported
   file. The same tensor shapes the torch module accepts feed
   straight into the ONNX session.

4. **Verify parity.** The headline assertion: the torch
   module and the ONNX session produce numerically identical
   outputs (to within `1e-5` tolerance) on a sweep of test
   inputs. This is the contract a deployment team relies on
   — the model they trained in Python is the same model
   they're serving in production, with no behavioural drift
   from the export step.

5. **Dynamic batch size.** The exported model accepts batch
   sizes different from the one used at export time. A model
   exported with batch_size=1 still serves a batch of 100 at
   inference. This is the standard production shape.

## Why this matters

The most common deployment story for a trained PETRA model
crosses a language boundary at some point:

- **JVM services** (Camunda / Activiti / Flowable production
  deployments) can load the `.onnx` via ONNX Runtime for
  Java and call inference without a Python interpreter in the
  loop.
- **C++ inference servers** for low-latency / high-throughput
  paths.
- **Browser-based decisioning UIs** via onnxruntime-web,
  where the model runs entirely client-side.
- **Mobile** via onnxruntime-mobile.
- **Accelerator stacks** (NVIDIA TensorRT, Intel OpenVINO,
  Windows DirectML) for hardware-accelerated inference.

All of these consume the same `.onnx` file PETRA produces.
The scenario test pins the parity guarantee that makes the
hand-off safe; the deployment target is the consumer's
choice.

## Limitations carried forward

ONNX export handles the acyclic forward pass and uniform-
duration time-unrolled mode. Non-uniform transition durations
(the in-flight queue uses Python lists) don't survive tracing
— those scenarios stay on the torch path. See
[`docs/api/onnx_export.md`](../../docs/api/onnx_export.md) for
the exhaustive constraint list.

## Files

- `scenario.toml` — net, training traces, training
  hyperparameters.
- `../../tests/scenarios/test_onnx_deployment.py` — trains
  the module, exports to ONNX in a temp directory, verifies
  parity with the torch forward pass under onnxruntime, and
  confirms the dynamic-batch axis works.

## Running

```
python -m pytest tests/scenarios/test_onnx_deployment.py
```

The test requires the `[onnx]` optional extra installed
(`pip install petra-nn[onnx]`) — onnxruntime is what loads
and executes the exported file. Without it the test is
skipped.
