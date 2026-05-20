"""Command-line entry points for the PETRA toolkit.

Three commands, all subcommands of a single ``petra`` parser
(plus the convenience aliases declared in ``pyproject.toml``'s
``[project.scripts]`` block so ``petra-train`` / ``petra-score``
/ ``petra-serve`` work without the subcommand word):

  * **petra-train** — load a scenario from a TOML config, train
    the compiled module on the scenario's trace set, and save
    the trained module to disk as a two-file bundle (a
    ``.pt`` pickle of the module plus a JSON metadata sidecar
    that records the scenario's input_marking and input_values
    specs along with the petra-nn version that produced it).

  * **petra-score** — load a saved bundle plus a trace file
    (XES / CSV / JSON), compute :func:`anomaly_score` for each
    trace using the metadata's input_marking spec, and emit
    a JSON document with the per-trace scores. Suitable for
    a cron job or a batch pipeline.

  * **petra-serve** — load a saved bundle and run the FastAPI
    REST app under uvicorn on a port. A workflow engine or
    monitoring system can then talk to PETRA over HTTP without
    any Python wiring on its side.

The design target is **engine integration**: a workflow team
that wants to plug PETRA into Camunda / Activiti / Flowable
should be able to install ``petra-nn``, train from a scenario
file, and stand up either a REST endpoint or a batch scoring
job without writing any Python beyond the scenario TOML. The
heavyweight JVM-side plugins (execution listeners, command
interceptors, process delegates) are out of scope for this
package — see ``docs/INTEGRATION_PATTERNS.md`` for the wiring
recipes that bridge an engine's audit log or event stream to
these CLI commands or to the REST API.

Bundle format
-------------

A trained-model bundle is two files written side-by-side:

* ``<name>.pt`` — pickled :class:`PetriNetModule`. Use
  :func:`torch.load` to reconstruct. **Don't load bundles from
  untrusted sources** — pickle deserialises arbitrary Python
  objects.
* ``<name>.meta.json`` — JSON sidecar with:

      {
        "scenario_name": "...",
        "scenario_description": "...",
        "input_marking_spec": { ... },     # from [training.input_marking]
        "input_values_spec":  { ... },     # from [training.input_values]
        "petra_version":      "0.1.0",
        "saved_at":           "2026-05-20T...Z"
      }

  Read by :command:`petra-score` to know which trace attributes
  to map onto which places at inference time.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from petri_net_nn import __version__
from petri_net_nn.adapter import _resolve_attribute_spec, load_scenario
from petri_net_nn.traces import anomaly_score
from petri_net_nn.xes import XESEvent, XESTrace, parse_xes


# --------------------------------------------------------------------------
# Bundle save / load helpers — kept in one place so the train and score
# commands stay symmetric and the format is documented once.
# --------------------------------------------------------------------------


def _meta_path_for(model_path: Path) -> Path:
    """Conventional sidecar filename. We append ``.meta.json``
    rather than substituting the extension so the user can tell
    at a glance which sidecar goes with which model."""
    return model_path.with_suffix(model_path.suffix + ".meta.json")


def _save_bundle(
    module: Any,
    output_path: Path,
    *,
    scenario_name: str,
    scenario_description: str,
    input_marking_spec: dict[str, dict],
    input_values_spec: dict[str, dict],
) -> Path:
    """Write the bundle pair atomically-ish (parent dir created
    if needed). Returns the path of the .pt file.

    Strips un-picklable callable fields from the net before
    pickling and restores them after, so the caller's in-memory
    module isn't mutated. The stripped fields
    (``transition_guards`` callables and any ``arc_output_values``
    callables) are **token-game-only** — they're not read by the
    compiler's forward pass, so removing them is correctness-safe
    for the inference use case the CLI is built for. Refuses to
    save when ``torch_guard`` / ``torch_output_value`` callables
    are present, since those *are* compiler-relevant but pickle
    only when defined as named module-level functions; the
    caller should save the state_dict separately and reconstruct
    via the scenario TOML instead."""
    net = module.net

    # Compiler-relevant callables we can't safely strip. Surface
    # cleanly rather than fail deep inside torch.save with a
    # cryptic pickle error.
    if net.transition_torch_guards or net.arc_torch_output_values:
        raise ValueError(
            "petra-train cannot save a module that uses "
            "torch_guard or arc_torch_output_value callables — "
            "Python lambdas don't pickle. Define them as named "
            "module-level functions, or save the module's "
            "state_dict separately and reconstruct via the "
            "scenario TOML at load time."
        )

    # Snapshot the un-picklable token-game callables and pop
    # them from the net for the duration of the save.
    saved_guards = net.transition_guards
    saved_output_value_callables = {
        k: v for k, v in net.arc_output_values.items() if callable(v)
    }
    net.transition_guards = {}
    for k in list(saved_output_value_callables):
        del net.arc_output_values[k]

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(module, output_path)
    finally:
        # Restore so the in-memory net is unchanged from the
        # caller's perspective whether the save succeeded or
        # raised.
        net.transition_guards = saved_guards
        for k, v in saved_output_value_callables.items():
            net.arc_output_values[k] = v

    meta = {
        "scenario_name": scenario_name,
        "scenario_description": scenario_description,
        "input_marking_spec": input_marking_spec,
        "input_values_spec": input_values_spec,
        "petra_version": __version__,
        "saved_at": datetime.now(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _meta_path_for(output_path).write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def _load_bundle(model_path: Path) -> tuple[Any, dict[str, Any]]:
    """Load a bundle. Returns ``(module, metadata)``.

    Uses ``weights_only=False`` because the bundle is a full
    pickled module, not just a state_dict; the calling convention
    for petra-score / petra-serve is "you saved this with
    petra-train and you trust it." Warn the user not to load
    untrusted bundles in the surrounding docs."""
    module = torch.load(
        model_path, weights_only=False, map_location="cpu"
    )
    meta_path = _meta_path_for(model_path)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata sidecar not found: {meta_path}. The bundle "
            f"is a (.pt, .pt.meta.json) pair; produce it with "
            f"`petra-train`."
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return module, metadata


# --------------------------------------------------------------------------
# Subcommand handlers. Each takes the parsed argparse Namespace and
# returns an exit code (0 = success, non-zero = failure).
# --------------------------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> int:
    """petra-train <scenario.toml> [--output ...] [--steps N] [--lr LR]"""
    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        print(
            f"error: scenario file not found: {scenario_path}",
            file=sys.stderr,
        )
        return 2

    ctx = load_scenario(scenario_path)

    # Allow CLI overrides for training params. If unset on the
    # command line we honour what the TOML config declared.
    if args.steps is not None:
        ctx.training.steps = args.steps
    if args.lr is not None:
        ctx.training.lr = args.lr
    if args.seed is not None:
        ctx.training.seed = args.seed

    if not ctx.traces:
        print(
            "error: scenario has no [traces] section; petra-train "
            "needs trace data to fit weights against",
            file=sys.stderr,
        )
        return 2

    print(
        f"training '{ctx.name}' on {len(ctx.traces)} trace(s) "
        f"for {ctx.training.steps} step(s) at lr={ctx.training.lr}...",
        file=sys.stderr,
    )
    module, losses = ctx.train()

    output_path = Path(
        args.output if args.output is not None
        else scenario_path.with_suffix(".pt")
    )
    _save_bundle(
        module,
        output_path,
        scenario_name=ctx.name,
        scenario_description=ctx.description,
        input_marking_spec=ctx.input_marking_spec,
        input_values_spec=ctx.input_values_spec,
    )
    print(
        f"saved model bundle: {output_path}, "
        f"{_meta_path_for(output_path).name} "
        f"(final loss {losses[-1]:.4f})",
        file=sys.stderr,
    )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    """petra-score <model.pt> --traces <file> [--format auto|xes|csv|json]"""
    model_path = Path(args.model)
    traces_path = Path(args.traces)
    if not model_path.exists():
        print(f"error: model not found: {model_path}", file=sys.stderr)
        return 2
    if not traces_path.exists():
        print(f"error: traces file not found: {traces_path}", file=sys.stderr)
        return 2

    module, metadata = _load_bundle(model_path)

    fmt = (
        args.format
        if args.format and args.format != "auto"
        else _guess_format(traces_path)
    )
    traces = _load_traces(traces_path, fmt, args)
    if not traces:
        print(
            "error: no traces parsed from the input file",
            file=sys.stderr,
        )
        return 2

    input_marking_spec = metadata.get("input_marking_spec", {}) or {}
    input_values_spec = metadata.get("input_values_spec", {}) or {}

    def attribute_to_marking(trace: XESTrace) -> dict[str, float]:
        return _resolve_attribute_spec(
            trace, input_marking_spec, "input_marking"
        )

    attribute_to_values = None
    if input_values_spec:
        def attribute_to_values(trace: XESTrace) -> dict[str, float]:  # noqa: F811
            return _resolve_attribute_spec(
                trace, input_values_spec, "input_values"
            )

    # Emit one JSON document on stdout — a list of per-trace
    # objects. Suitable for piping into jq, a monitoring system,
    # or a downstream batch process.
    results: list[dict[str, Any]] = []
    for i, trace in enumerate(traces):
        try:
            residuals = anomaly_score(
                module,
                trace,
                attribute_to_marking=attribute_to_marking,
                attribute_to_values=attribute_to_values,
            )
        except KeyError as e:
            # A missing attribute the input_marking_spec
            # required — report the trace but flag the failure.
            results.append({
                "trace_index": i,
                "attributes": dict(trace.attributes),
                "error": f"missing attribute: {e}",
            })
            continue
        results.append({
            "trace_index": i,
            "attributes": dict(trace.attributes),
            "trace_score": sum(residuals.values()),
            "per_transition_residuals": residuals,
        })

    output_path: Path | None = (
        Path(args.output) if args.output is not None else None
    )
    payload = json.dumps(results, indent=2, sort_keys=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        print(
            f"scored {len(traces)} trace(s) → {output_path}",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(payload + "\n")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """petra-serve <model.pt> [--host ...] [--port N]"""
    try:
        import uvicorn
    except ImportError:
        print(
            "error: uvicorn is required for petra-serve. Install "
            "with: pip install 'petra-nn[rest]'",
            file=sys.stderr,
        )
        return 2
    from petri_net_nn.rest import build_app

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"error: model not found: {model_path}", file=sys.stderr)
        return 2

    module, metadata = _load_bundle(model_path)
    app = build_app(
        module,
        title=f"PETRA: {metadata.get('scenario_name', 'model')}",
        version=metadata.get("petra_version", __version__),
        description=metadata.get("scenario_description") or None,
    )
    print(
        f"serving '{metadata.get('scenario_name', 'model')}' "
        f"on http://{args.host}:{args.port} (Swagger UI at /docs)",
        file=sys.stderr,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


# --------------------------------------------------------------------------
# Format detection + trace loading for the score command.
# --------------------------------------------------------------------------


def _guess_format(path: Path) -> str:
    """Auto-pick a trace format from the file's extension. We
    accept ``.xes`` / ``.xes.gz`` / ``.csv`` / ``.json``;
    anything else gets a usable error rather than a silent
    misparse."""
    name = path.name.lower()
    if name.endswith(".xes") or name.endswith(".xes.gz"):
        return "xes"
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".json"):
        return "json"
    raise ValueError(
        f"cannot auto-detect trace format for {path.name}; "
        f"pass --format xes|csv|json explicitly"
    )


def _load_traces(
    path: Path, fmt: str, args: argparse.Namespace
) -> list[XESTrace]:
    """Slim trace loader for the score command. Mirrors the
    adapter's behaviour but is invoked directly so a saved
    bundle isn't tied to its original scenario.toml at score
    time."""
    if fmt == "xes":
        return parse_xes(path)
    if fmt == "csv":
        return _csv_to_traces(
            path,
            case_column=args.case_column,
            activity_column=args.activity_column,
        )
    if fmt == "json":
        return _json_to_traces(path)
    raise ValueError(
        f"unknown format {fmt!r}; supported: xes, csv, json"
    )


def _csv_to_traces(
    path: Path, *, case_column: str, activity_column: str
) -> list[XESTrace]:
    """One-shot CSV → XESTrace converter for the score command.
    Identical convention to the adapter's _load_csv_traces: each
    row is one event; rows with the same case_column value
    accumulate into a single trace; the first row's other
    columns lift to trace level."""
    from collections import OrderedDict

    traces_by_case: OrderedDict[str, XESTrace] = OrderedDict()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or case_column not in reader.fieldnames:
            raise ValueError(
                f"CSV at {path} must have a {case_column!r} column "
                f"(got headers: {reader.fieldnames})"
            )
        if activity_column not in reader.fieldnames:
            raise ValueError(
                f"CSV at {path} must have an {activity_column!r} column "
                f"(got headers: {reader.fieldnames})"
            )
        for row in reader:
            case_id = row[case_column]
            activity = row[activity_column]
            event_attrs = {
                k: v for k, v in row.items()
                if k not in (case_column, activity_column) and v is not None
            }
            trace = traces_by_case.get(case_id)
            if trace is None:
                trace = XESTrace(
                    attributes={case_column: case_id, **event_attrs},
                    events=[],
                )
                traces_by_case[case_id] = trace
            trace.events.append(
                XESEvent(name=activity, attributes=event_attrs)
            )
    return list(traces_by_case.values())


def _json_to_traces(path: Path) -> list[XESTrace]:
    """JSON trace loader for the score command. Same shape as
    the adapter's _load_json_traces: a list of
    ``{"attributes": ..., "events": [...]}`` objects."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"JSON at {path} must be a list of trace objects at "
            f"the top level"
        )
    out: list[XESTrace] = []
    for entry in data:
        attrs = {
            k: str(v) for k, v in (entry.get("attributes") or {}).items()
        }
        events: list[XESEvent] = []
        for ev in entry.get("events", []):
            if isinstance(ev, str):
                events.append(XESEvent(name=ev))
            elif isinstance(ev, dict):
                name = ev.get("name")
                if not name:
                    raise ValueError(
                        f"JSON event object missing 'name': {ev}"
                    )
                ev_attrs = {
                    k: str(v)
                    for k, v in (ev.get("attributes") or {}).items()
                }
                events.append(
                    XESEvent(name=str(name), attributes=ev_attrs)
                )
            else:
                raise ValueError(
                    f"JSON events must be strings or objects; got "
                    f"{type(ev).__name__}"
                )
        out.append(XESTrace(attributes=attrs, events=events))
    return out


# --------------------------------------------------------------------------
# argparse wiring. We expose a single `petra` parser plus the three
# convenience commands declared in pyproject.toml [project.scripts]
# (petra-train / petra-score / petra-serve) which call back into the
# same handlers with the subcommand fixed.
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petra",
        description=(
            "Command-line toolkit for PETRA. Three subcommands: "
            "train a model from a TOML scenario; score traces with "
            "a saved model; or serve a saved model behind a REST "
            "API for engine integration. See "
            "docs/INTEGRATION_PATTERNS.md for end-to-end wiring "
            "recipes."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"petra-nn {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser(
        "train",
        help="train from a TOML scenario, save the bundle",
    )
    train.add_argument(
        "scenario", help="path to the scenario.toml"
    )
    train.add_argument(
        "--output", "-o",
        help=(
            "output model path (default: <scenario_stem>.pt next "
            "to the scenario file)"
        ),
    )
    train.add_argument(
        "--steps", type=int,
        help="override [training].steps from the scenario",
    )
    train.add_argument(
        "--lr", type=float,
        help="override [training].lr from the scenario",
    )
    train.add_argument(
        "--seed", type=int,
        help="override [training].seed from the scenario",
    )
    train.set_defaults(func=_cmd_train)

    score = subparsers.add_parser(
        "score",
        help="score traces against a saved model bundle",
    )
    score.add_argument("model", help="path to the .pt bundle file")
    score.add_argument(
        "--traces", required=True,
        help="path to the trace file (XES, CSV, or JSON)",
    )
    score.add_argument(
        "--format", choices=["auto", "xes", "csv", "json"],
        default="auto",
        help="trace file format (default: auto-detect from extension)",
    )
    score.add_argument(
        "--case-column", default="case_id",
        help="CSV case-id column name (default: case_id)",
    )
    score.add_argument(
        "--activity-column", default="activity",
        help="CSV activity column name (default: activity)",
    )
    score.add_argument(
        "--output", "-o",
        help=(
            "write the JSON result to this path (default: stdout)"
        ),
    )
    score.set_defaults(func=_cmd_score)

    serve = subparsers.add_parser(
        "serve",
        help="serve a saved model bundle behind the REST API",
    )
    serve.add_argument("model", help="path to the .pt bundle file")
    serve.add_argument(
        "--host", default="127.0.0.1",
        help="host to bind (default: 127.0.0.1)",
    )
    serve.add_argument(
        "--port", type=int, default=8000,
        help="port to bind (default: 8000)",
    )
    serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Single ``petra`` entry point. Dispatches to the chosen
    subcommand's handler."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help(sys.stderr)
        return 2
    return args.func(args)


# Three thin wrappers that pyproject.toml's [project.scripts] block
# points at, so ``petra-train`` etc. work without typing the
# subcommand. Each forces the subcommand and forwards everything
# else through main().


def train_entry() -> int:
    return main(["train", *sys.argv[1:]])


def score_entry() -> int:
    return main(["score", *sys.argv[1:]])


def serve_entry() -> int:
    return main(["serve", *sys.argv[1:]])
