"""Config-driven adapter from external data into the framework.

Each scenario — BPMN process, biological cascade, network protocol,
manufacturing line — needs roughly the same pipeline: load a Petri
net, load traces, train, optionally extract rules, optionally score
anomalies. The boilerplate is identical across scenarios, only the
data differs.

This module reads a TOML config file describing the data sources and
training parameters, and returns a ``ScenarioContext`` that exposes
the standard pipeline as methods. A scenario then becomes one config
file plus a data file (BPMN or XES), with no per-scenario glue code.

Config schema (TOML):

    [scenario]
    name = "..."
    description = "..."

    # Net source — exactly one of:
    [net]
    source = "bpmn_file"
    path = "process.bpmn"
    # OR
    [net]
    source = "inline"
    # then declare places/transitions/arcs as TOML tables below

    [[net.places]]
    id = "p_x"
    tokens = 1            # optional, default 0
    label = "..."         # optional

    [[net.transitions]]
    id = "t_x"
    label = "..."

    [[net.arcs]]
    src = "p_x"
    dst = "t_x"

    # Trace source — exactly one of:
    [traces]
    source = "xes_file"
    path = "traces.xes"
    # OR
    [traces]
    source = "inline"
    # with [[traces.inline]] entries below

    [[traces.inline]]
    attributes = { signal_strength = "0.9" }
    events = ["t_a", "t_b"]

    # Map per-trace attribute (or constant) onto a place's input
    # activation. Each key is a place id; each value is either
    # { attribute = "name" } or { constant = 0.5 }.
    [training.input_marking]
    p_signal = { attribute = "signal_strength" }

    [training]
    steps = 1500
    lr = 0.1
    sharpness = 1.0
    firing = "sigmoid"          # or "ste"
    routing = "independent"     # or "softmax"
    num_steps = 0
    seed = 0

Missing optional sections fall back to library defaults.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from petri_net_nn.bpmn import parse_bpmn
from petri_net_nn.compiler import PetriNetModule
from petri_net_nn.interpretability import (
    extract_and_join_rules,
    extract_routing_rules,
)
from petri_net_nn.petri_net import PetriNet
from petri_net_nn.pnml import parse_pnml
from petri_net_nn.traces import anomaly_score, train_on_traces
from petri_net_nn.xes import XESEvent, XESTrace, parse_xes


if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class TrainingConfig:
    steps: int = 500
    lr: float = 0.1
    sharpness: float = 1.0
    firing: str = "sigmoid"
    routing: str = "independent"
    num_steps: int = 0
    seed: int | None = None


@dataclass
class ScenarioContext:
    """Self-contained, config-driven scenario.

    Use ``compile()`` + ``train()`` for the standard pipeline, or
    ``attribute_to_marking`` / ``anomaly_score`` if you want to mix in
    your own scoring code. Carrying the original ``config`` dict lets
    consumers read scenario-specific extras (e.g. anomaly-generator
    settings) without re-parsing the file."""

    name: str
    description: str
    net: PetriNet
    traces: list[XESTrace]
    training: TrainingConfig
    input_marking_spec: dict[str, dict]
    config: dict[str, Any] = field(default_factory=dict)
    config_dir: Path = field(default_factory=Path)

    def attribute_to_marking(self, trace: XESTrace) -> dict[str, float]:
        marking: dict[str, float] = {}
        for place, spec in self.input_marking_spec.items():
            if "constant" in spec:
                marking[place] = float(spec["constant"])
            elif "attribute" in spec:
                attr_name = spec["attribute"]
                if attr_name not in trace.attributes:
                    raise KeyError(
                        f"trace missing attribute {attr_name!r} required "
                        f"by input_marking for place {place!r}"
                    )
                marking[place] = float(trace.attributes[attr_name])
            else:
                raise ValueError(
                    f"input_marking entry for {place!r} must declare "
                    f"either 'attribute' or 'constant'; got {spec}"
                )
        return marking

    def compile(self) -> PetriNetModule:
        if self.training.seed is not None:
            torch.manual_seed(self.training.seed)
        return PetriNetModule(
            self.net,
            sharpness=self.training.sharpness,
            num_steps=self.training.num_steps,
            firing=self.training.firing,  # type: ignore[arg-type]
            routing=self.training.routing,  # type: ignore[arg-type]
        )

    def train(
        self, module: PetriNetModule | None = None
    ) -> tuple[PetriNetModule, list[float]]:
        if module is None:
            module = self.compile()
        losses = train_on_traces(
            module,
            self.traces,
            attribute_to_marking=self.attribute_to_marking,
            steps=self.training.steps,
            lr=self.training.lr,
        )
        return module, losses

    def extract_rules(self, module: PetriNetModule) -> dict[str, list]:
        cfg = self.config.get("interpretability", {})
        result: dict[str, list] = {}
        if cfg.get("extract_xor_rules", False):
            result["xor"] = extract_routing_rules(module)
        if cfg.get("extract_and_join_rules", False):
            result["and_join"] = extract_and_join_rules(module)
        return result

    def anomaly_score(
        self, module: PetriNetModule, trace: XESTrace
    ) -> dict[str, float]:
        return anomaly_score(
            module, trace, attribute_to_marking=self.attribute_to_marking
        )


def load_scenario(config_path: str | Path) -> ScenarioContext:
    """Read a TOML scenario config and produce a ``ScenarioContext``.

    Relative ``path`` entries inside the config are resolved against
    the config file's directory, so configs can sit next to their
    data files without absolute paths."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        config = tomllib.load(f)

    config_dir = path.parent

    scenario = config.get("scenario", {})
    name = scenario.get("name", path.stem)
    description = scenario.get("description", "")

    net = _load_net(config.get("net", {}), config_dir)
    traces = _load_traces(config.get("traces", {}), config_dir)
    training_dict = config.get("training", {}) or {}
    input_marking_spec = training_dict.pop("input_marking", {}) or {}
    training = TrainingConfig(**{k: v for k, v in training_dict.items() if k in TrainingConfig.__dataclass_fields__})

    return ScenarioContext(
        name=name,
        description=description,
        net=net,
        traces=traces,
        training=training,
        input_marking_spec=input_marking_spec,
        config=config,
        config_dir=config_dir,
    )


def _resolve(path_value: str, config_dir: Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else config_dir / p


# Map declarative guard operators (TOML-friendly strings) to actual
# Python comparisons. Lets a scenario.toml carry a guard without
# embedding code.
_GUARD_OPS: dict[str, Any] = {
    ">":  lambda a, b: a >  b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a <  b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _build_guard(spec: dict[str, Any] | None):
    """Turn a ``{place, op, value}`` declarative guard spec into a
    callable suitable for ``PetriNet.add_transition(guard=...)``.

    Returns ``None`` when no guard is declared, so the caller can
    pass the result straight through to the API."""
    if spec is None:
        return None
    place = spec.get("place")
    op = spec.get("op")
    value = spec.get("value")
    if place is None or op is None or value is None:
        raise ValueError(
            f"guard spec needs 'place', 'op', and 'value' fields; got {spec}"
        )
    if op not in _GUARD_OPS:
        raise ValueError(
            f"guard op {op!r} is unsupported; use one of {sorted(_GUARD_OPS)}"
        )
    compare = _GUARD_OPS[op]
    threshold = float(value)

    def guard(input_values: dict[str, float]) -> bool:
        if place not in input_values:
            return False
        return bool(compare(float(input_values[place]), threshold))

    return guard


def _load_net(spec: dict[str, Any], config_dir: Path) -> PetriNet:
    source = spec.get("source")
    if source == "bpmn_file":
        path_value = spec.get("path")
        if not path_value:
            raise ValueError("net.source='bpmn_file' requires 'path'")
        return parse_bpmn(_resolve(path_value, config_dir))
    if source == "pnml_file":
        # PNML is the standard interchange format for Petri nets;
        # this bridge lets PETRA consume nets emitted by CPN Tools,
        # GreatSPN, TINA, ProM, Snoopy, and any other PNML-aware
        # modeller without going through BPMN first.
        path_value = spec.get("path")
        if not path_value:
            raise ValueError("net.source='pnml_file' requires 'path'")
        return parse_pnml(_resolve(path_value, config_dir))
    if source == "inline":
        net = PetriNet()
        for place in spec.get("places", []):
            net.add_place(
                place["id"],
                tokens=int(place.get("tokens", 0)),
                label=place.get("label"),
            )
        for transition in spec.get("transitions", []):
            # Optional `duration` (default 1) lets scenarios annotate
            # how many time-unrolled steps a transition takes to
            # produce its output once fired. Surfaces purely in the
            # time-unrolled compiler; ignored in acyclic mode.
            # Optional `rate` (default 1.0) is a per-transition
            # firing-rate multiplier on the pre-activation. High
            # rate = transition fires more eagerly on the same
            # inputs; useful for encoding prior knowledge about
            # transition propensity.
            # Optional `guard` is a declarative spec
            # ``{place, op, value}`` that builds a coloured-Petri-net
            # guard callable (e.g. ``{place="p_in", op=">=",
            # value=1000.0}`` fires only when the token at p_in
            # carries a value at least 1000.0). Used by the
            # coloured token-game; transparent to the compiler.
            net.add_transition(
                transition["id"],
                label=transition.get("label"),
                duration=int(transition.get("duration", 1)),
                rate=float(transition.get("rate", 1.0)),
                guard=_build_guard(transition.get("guard")),
            )
        for arc in spec.get("arcs", []):
            weight = int(arc.get("weight", 1))
            # Optional `output_value` on transition -> place arcs sets
            # the value carried by tokens this arc produces in the
            # coloured token-game. Constant scalar only via TOML;
            # callable transforms must be supplied through the
            # Python API.
            kwargs: dict[str, Any] = {"weight": weight}
            if "output_value" in arc:
                kwargs["output_value"] = float(arc["output_value"])
            net.add_arc(arc["src"], arc["dst"], **kwargs)
        # Inhibitor arcs: [[net.inhibitor_arcs]] place = "...", transition = "..."
        # These declare a structural guard — the transition cannot fire
        # while the place holds a token. They are not part of the flow
        # relation, do not have weights, and are not consumed when the
        # guarded transition fires.
        for inh in spec.get("inhibitor_arcs", []):
            net.add_inhibitor_arc(inh["place"], inh["transition"])
        return net
    raise ValueError(
        f"net.source must be 'bpmn_file', 'pnml_file', or 'inline', "
        f"got {source!r}"
    )


def _load_traces(spec: dict[str, Any], config_dir: Path) -> list[XESTrace]:
    source = spec.get("source")
    if source is None:
        return []
    if source == "xes_file":
        path_value = spec.get("path")
        if not path_value:
            raise ValueError("traces.source='xes_file' requires 'path'")
        return parse_xes(_resolve(path_value, config_dir))
    if source == "inline":
        out: list[XESTrace] = []
        for entry in spec.get("inline", []):
            attrs = entry.get("attributes", {}) or {}
            events = entry.get("events", []) or []
            out.append(
                XESTrace(
                    attributes={k: str(v) for k, v in attrs.items()},
                    events=[XESEvent(name=str(e)) for e in events],
                )
            )
        return out
    raise ValueError(
        f"traces.source must be 'xes_file' or 'inline', got {source!r}"
    )
