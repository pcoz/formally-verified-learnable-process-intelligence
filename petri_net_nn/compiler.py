"""Compile a PetriNet into a differentiable nn.Module.

Implements §4 of the architecture spec. The compiled module instantiates
the continuous relaxation from §4.2 directly over the net's flow
relation:

    activation(t) = sigmoid( sharpness * (sum_p w(p,t)*a(p) - theta(t)) )
    a(p)          = sum_{t: (t,p) in F} activation(t) * w(t,p)

The structural constraint from §4.3 — "weights outside this structure
are zero by construction and cannot be learned away from zero" — holds
because the module allocates exactly one learnable scalar per arc in F
and one threshold per transition in T. There is no global weight matrix
with a mask; the parameters that don't exist literally don't exist.

Two forward-pass modes:

* ``num_steps == 0`` (default) — acyclic mode. The constructor
  topologically sorts (P ∪ T, F) and forward does a single propagation
  pass in that order. The §4.2 equations are evaluated exactly once
  per node. Rejects cyclic nets at construction.

* ``num_steps > 0`` — time-unrolled mode. The constructor skips the
  topological sort so cyclic nets are accepted. Forward initialises
  place activations from the input marking / M_0 and then performs
  ``num_steps`` synchronous updates (each step: refresh every
  transition's activation from current place activations, then refresh
  every non-source place's activation from the new transition
  activations). Source places — those with empty preset — clamp to
  their input value at every step, so they behave as a persistent
  input layer.

Coloured-Petri-net layer
------------------------

When a transition has a structural guard (declarative
``{place, op, value}`` form), the compiler builds a *learnable*
soft-guard alongside the standard firing equation: an
``nn.Parameter`` threshold initialised at ``value`` and a sigmoid
gate that multiplies the transition's firing strength by
``sigmoid(s * sign * (value(place) - threshold))`` (sign = +1 for
``>``/``>=``, −1 for ``<``/``<=``). The threshold trains end-to-end
with the rest of the network, so the model can refine the
declared boundary from execution traces. Guards declared as opaque
callables stay transparent to the compiler (the token-game still
uses them — they don't take part in training).

To make value-conditioned routing trainable, the forward pass
carries a per-place *value* channel alongside the activation
channel. Source-place values come from the optional ``input_values``
argument (default 1.0). Non-source places combine the values
arriving on their incoming arcs into an activation-weighted average
— the natural soft-token analogue of "what value would this place
hold right now if a token were here." Output-arc values may be
declared on the net (``arc_output_values``); constant scalars are
honoured by the compiler, callable transforms are evaluated only
in the discrete coloured token-game and treated as the default
value 1.0 in the differentiable forward pass.

The guard sigmoid scales sharpness by ``1 / max(|theta_init|, 1.0)``
so the initial gradient at the boundary is in O(1) regardless of
the units the modeller used. The same ``SharpnessScheduler`` from
Phase 6 sharpens guards alongside firing transitions during
training.
"""
from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from petri_net_nn.petri_net import PetriNet


FiringMode = Literal["sigmoid", "ste"]
RoutingMode = Literal["independent", "softmax"]


def _fire_sigmoid(pre: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(pre)


def _fire_ste(pre: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward returns 0 or 1 (hard step at
    sigmoid output 0.5), backward returns the sigmoid gradient. The
    detach trick — ``sigm + (hard - sigm).detach()`` — gives a forward
    value of ``hard`` while only the ``sigm`` term contributes to the
    autograd graph, so ``d/dx = sigmoid'(pre)`` flows back to the
    parameters."""
    sigm = torch.sigmoid(pre)
    hard = (sigm > 0.5).to(sigm.dtype)
    return sigm + (hard - sigm).detach()


def _xor_groups(net: PetriNet) -> list[tuple[str, list[str]]]:
    """Structural XOR-shape detector — local copy of the function in
    petri_net_nn.interpretability, inlined to avoid an import cycle.

    Catches both single-input XOR (BPMN-style) and shared-preset XOR
    (N transitions with identical multi-place input sets, e.g.
    competing alternatives in a 2PC-style protocol)."""
    by_preset: dict[frozenset[str], list[str]] = {}
    for t in sorted(net.transitions):
        preset = frozenset(net.preset(t))
        if preset:
            by_preset.setdefault(preset, []).append(t)

    groups: list[tuple[str, list[str]]] = []
    for preset, consumers in by_preset.items():
        if len(consumers) < 2:
            continue
        groups.append((sorted(preset)[0], sorted(consumers)))
    return sorted(groups, key=lambda g: g[0])


class PetriNetModule(nn.Module):
    """Differentiable neural network whose topology is exactly a PetriNet.

    Parameters
    ----------
    net :
        A well-formed PetriNet. Validation errors are rejected at
        construction time; cycles are rejected only when
        ``num_steps == 0``.
    sharpness :
        Multiplier inside the sigmoid (§4.2 has no such factor; this is a
        training aid for AND-join–shaped transitions where a near-step
        activation is needed — see §5 Subnet 4). Default 1.0 keeps the
        forward pass faithful to §4.2 verbatim.
    num_steps :
        ``0`` (default) selects acyclic single-pass mode; any positive
        integer selects time-unrolled mode and accepts cyclic nets.
    """

    def __init__(
        self,
        net: PetriNet,
        *,
        sharpness: float = 1.0,
        num_steps: int = 0,
        firing: FiringMode = "sigmoid",
        routing: RoutingMode = "independent",
    ) -> None:
        super().__init__()
        issues = net.validate()
        if issues:
            raise ValueError(f"net is not well-formed: {issues}")
        if num_steps < 0:
            raise ValueError(f"num_steps must be non-negative, got {num_steps}")
        if firing not in ("sigmoid", "ste"):
            raise ValueError(
                f"firing must be 'sigmoid' or 'ste', got {firing!r}"
            )
        if routing not in ("independent", "softmax"):
            raise ValueError(
                f"routing must be 'independent' or 'softmax', got {routing!r}"
            )

        self.net = net
        self.sharpness = sharpness
        self.num_steps = num_steps
        self.firing = firing
        self.routing = routing
        self._fire_fn = _fire_ste if firing == "ste" else _fire_sigmoid

        self._softmax_groups: dict[str, list[str]] = {}
        if routing == "softmax":
            for _, group in _xor_groups(net):
                for t in group:
                    self._softmax_groups[t] = group

        if num_steps == 0:
            self._order: tuple[str, ...] | None = self._toposort(net)
        else:
            self._order = None

        self.arc_weights = nn.ParameterDict()
        self._arc_key: dict[tuple[str, str], str] = {}
        for i, edge in enumerate(sorted(net.flow)):
            key = f"arc_{i}"
            self._arc_key[edge] = key
            self.arc_weights[key] = nn.Parameter(
                torch.normal(mean=1.0, std=0.1, size=())
            )

        self.transition_thresholds = nn.ParameterDict()
        self._threshold_key: dict[str, str] = {}
        for i, t in enumerate(sorted(net.transitions)):
            key = f"theta_{i}"
            self._threshold_key[t] = key
            n_inputs = len(net.preset(t))
            theta_init = max(0.0, (n_inputs - 1) * 0.5)
            self.transition_thresholds[key] = nn.Parameter(torch.tensor(theta_init))

        # Learnable soft-guard thresholds — one per transition with a
        # structural guard. The TOML value seeds the parameter; training
        # refines it. The per-guard sharpness scale (kept fixed at
        # construction) keeps the sigmoid gradient at O(1) at the
        # boundary regardless of the value units the modeller used.
        self.guard_thresholds = nn.ParameterDict()
        self._guard_meta: dict[str, dict] = {}
        for i, t in enumerate(sorted(net.transitions)):
            spec = net.transition_structural_guards.get(t)
            if spec is None:
                continue
            op = spec["op"]
            if op not in (">", ">=", "<", "<="):
                raise ValueError(
                    f"transition {t!r}: compiler only supports structural "
                    f"guard ops in {{>, >=, <, <=}}; got {op!r}. "
                    f"Equality / inequality guards must be expressed "
                    f"as opaque callables (token-game only)."
                )
            place = spec["place"]
            if place not in net.places:
                raise ValueError(
                    f"transition {t!r}: structural guard references "
                    f"unknown place {place!r}"
                )
            init = float(spec["value"])
            key = f"guard_theta_{i}"
            self.guard_thresholds[key] = nn.Parameter(torch.tensor(init))
            self._guard_meta[t] = {
                "place": place,
                "op": op,
                "key": key,
                # Auto-scale the sigmoid steepness so that learning is
                # well-conditioned for guard thresholds in any unit
                # (loan amounts in £, signal strengths in [0,1], etc).
                "scale": 1.0 / max(abs(init), 1.0),
            }

    @staticmethod
    def _toposort(net: PetriNet) -> tuple[str, ...]:
        ts: TopologicalSorter[str] = TopologicalSorter()
        for node in net.places | net.transitions:
            ts.add(node)
        for src, dst in net.flow:
            ts.add(dst, src)
        try:
            return tuple(ts.static_order())
        except CycleError as e:
            cycle = e.args[1] if len(e.args) > 1 else e.args
            raise ValueError(
                f"Petri net has a cycle; pass num_steps>0 to enable the "
                f"time-unrolled forward pass (cycle: {cycle})"
            ) from None

    def _device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def _source_activation(
        self,
        place: str,
        input_marking: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if place in input_marking:
            return input_marking[place]
        if place in self.net.initial_marking:
            return torch.ones(batch_size, device=device)
        return torch.zeros(batch_size, device=device)

    def forward(
        self,
        input_marking: dict[str, torch.Tensor] | None = None,
        *,
        input_values: dict[str, torch.Tensor] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass.

        In acyclic mode (``num_steps == 0``), produces a single
        topological propagation. In time-unrolled mode, returns the
        activations after ``self.num_steps`` synchronous updates.

        ``input_marking`` overrides any place's activation. In
        time-unrolled mode the override is re-applied at every step,
        which is how you clamp a "persistent input" through the
        unrolled dynamics — equivalent to the §7.1 "predict next
        activations from a partial execution" use case.

        ``input_values`` feeds the per-place value channel that the
        coloured-Petri-net layer reads. Each entry is a 1D tensor of
        shape ``(batch_size,)`` giving the scalar value carried by
        the token at that source place — loan amount, signal
        strength, sensor reading, whatever the modeller chose. Any
        source place absent from this dict defaults to value 1.0
        (the value-carrying-no-information case, equivalent to a
        plain unannotated token).
        """
        if input_marking is None:
            input_marking = {}
        if input_values is None:
            input_values = {}

        if batch_size is None:
            if input_marking:
                batch_size = next(iter(input_marking.values())).shape[0]
            elif input_values:
                batch_size = next(iter(input_values.values())).shape[0]
            else:
                batch_size = 1

        device = self._device()

        if self.num_steps == 0:
            return self._forward_acyclic(
                input_marking, input_values, batch_size, device
            )
        return self._forward_unrolled(
            input_marking, input_values, batch_size, device
        )

    def _forward_acyclic(
        self,
        input_marking: dict[str, torch.Tensor],
        input_values: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        assert self._order is not None
        activations: dict[str, torch.Tensor] = {}
        # Parallel value channel — one scalar tensor per place. Source
        # places read from input_values (default 1.0). Non-source
        # places combine arriving values as an activation-weighted
        # average of each contributing transition's output-arc value.
        place_values: dict[str, torch.Tensor] = {}
        softmax_cache: dict[str, torch.Tensor] = {}

        eps = 1e-6

        for node in self._order:
            if node in input_marking:
                activations[node] = input_marking[node]
                if node in self.net.places:
                    place_values[node] = input_values.get(
                        node, torch.ones(batch_size, device=device)
                    )
                continue

            if node in self.net.places:
                preset = self.net.preset(node)
                if not preset:
                    activations[node] = self._source_activation(
                        node, input_marking, batch_size, device
                    )
                    place_values[node] = input_values.get(
                        node, torch.ones(batch_size, device=device)
                    )
                else:
                    contribs = [
                        self.arc_weights[self._arc_key[(t, node)]] * activations[t]
                        for t in preset
                    ]
                    activations[node] = sum(contribs)
                    value_numerator = sum(
                        self.arc_weights[self._arc_key[(t, node)]]
                        * activations[t]
                        * self._output_value(
                            t, node, place_values, batch_size, device
                        )
                        for t in preset
                    )
                    place_values[node] = value_numerator / (activations[node] + eps)
            else:
                if node in softmax_cache:
                    activations[node] = softmax_cache.pop(node)
                    continue
                pre = self._pre_activation(node, activations)
                if self.routing == "softmax" and node in self._softmax_groups:
                    group = self._softmax_groups[node]
                    pres = [
                        pre if member == node
                        else self._pre_activation(member, activations)
                        for member in group
                    ]
                    softmaxed = F.softmax(torch.stack(pres, dim=0), dim=0)
                    for member, soft in zip(group, softmaxed):
                        gate = self._inhibitor_gate(member, activations)
                        if gate is not None:
                            soft = soft * gate
                        gguard = self._guard_gate(member, place_values)
                        if gguard is not None:
                            soft = soft * gguard
                        if member == node:
                            activations[node] = soft
                        else:
                            softmax_cache[member] = soft
                else:
                    fired = self._fire_fn(pre)
                    gate = self._inhibitor_gate(node, activations)
                    if gate is not None:
                        fired = fired * gate
                    gguard = self._guard_gate(node, place_values)
                    if gguard is not None:
                        fired = fired * gguard
                    activations[node] = fired

        return activations

    def _pre_activation(
        self, transition: str, activations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Sum the weighted input-place activations for this transition.
        # Only arcs in the flow relation contribute — there is no
        # learnable weight outside F (§4.3 of the spec).
        weighted = sum(
            self.arc_weights[self._arc_key[(p, transition)]] * activations[p]
            for p in self.net.preset(transition)
        )
        theta = self.transition_thresholds[self._threshold_key[transition]]
        # rate is a per-transition multiplier (default 1.0); high rate
        # = steeper firing curve. sharpness is the global multiplier.
        # Together they scale the (weighted - threshold) gap before
        # the sigmoid / STE.
        rate = self.net.rate(transition)
        return rate * self.sharpness * (weighted - theta)

    def _output_value(
        self,
        transition: str,
        place: str,
        place_values: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Value carried by tokens this transition produces at ``place``.

        Precedence: ``arc_torch_output_values`` (callable on bound input
        value tensors) → constant entries in ``arc_output_values`` →
        default 1.0. Float-callable entries in ``arc_output_values``
        live in the discrete-token-game world and are not evaluated
        here — declare a ``torch_output_value`` on the arc if you want
        a non-constant transform during training."""
        torch_fn = self.net.arc_torch_output_values.get((transition, place))
        if torch_fn is not None:
            bound = {
                p: place_values[p]
                for p in self.net.preset(transition)
                if p in place_values
            }
            return torch_fn(bound)
        spec = self.net.arc_output_values.get((transition, place))
        if spec is None or callable(spec):
            return torch.ones(batch_size, device=device)
        return torch.full((batch_size,), float(spec), device=device)

    def _guard_gate(
        self,
        transition: str,
        place_values: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Soft-guard multiplier on a transition's firing strength.

        Precedence: a user-supplied ``torch_guard`` on the transition
        wins (it's strictly more expressive — multi-input, compound,
        custom learnable sub-networks). Otherwise the structural
        ``{place, op, value}`` form contributes a learnable-threshold
        sigmoid. Returns ``None`` for transitions with no guard at
        all so the caller skips the multiplication and keeps autograd
        clean in the common case. When the gate's input value hasn't
        been computed yet (e.g. the first step of a time-unrolled run
        where an upstream transition hasn't fired) we also return
        ``None`` — the activation gate carries the load until a
        value is available.
        """
        torch_fn = self.net.transition_torch_guards.get(transition)
        if torch_fn is not None:
            bound = {
                p: place_values[p]
                for p in self.net.preset(transition)
                if p in place_values
            }
            if not bound:
                return None
            return torch_fn(bound)

        meta = self._guard_meta.get(transition)
        if meta is None:
            return None
        v = place_values.get(meta["place"])
        if v is None:
            return None
        theta = self.guard_thresholds[meta["key"]]
        scaled = self.sharpness * meta["scale"] * (v - theta)
        if meta["op"] in (">", ">="):
            return torch.sigmoid(scaled)
        return torch.sigmoid(-scaled)

    def _inhibitor_gate(
        self, transition: str, place_acts: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        """Compute the multiplicative inhibitor gate for one transition.

        In the discrete Petri-net world, an inhibitor arc from place p
        to transition t means: t can fire only if p is empty. The
        continuous relaxation we use here interprets place activations
        as soft token-presence indicators in [0, 1]: a(p) = 1 means
        "definitely a token here", a(p) = 0 means "definitely empty".
        The corresponding soft inhibitor rule is therefore to multiply
        the transition's firing strength by (1 - a(p)) — strong
        activation in the inhibitor place drives the multiplier to
        zero, suppressing the firing; weak activation leaves it
        essentially untouched.

        When a transition has multiple inhibitor places we multiply
        the per-place gates together — semantically, *every* inhibitor
        must be near-empty for the transition to fire, which matches
        a product of (1 - a(p)) terms going to one.

        Inhibitor arcs deliberately have no learnable weight: they are
        a *structural* guard the modeller has declared, not a soft
        relationship the network is asked to discover. Adding a
        learnable weight would let training erode the guard, which
        defeats the point of declaring it.

        Returns ``None`` when the transition has no inhibitor arcs,
        so the caller can skip the multiplication entirely (avoids
        building unnecessary autograd ops in the common case).
        """
        inhibitors = self.net.inhibitor_preset(transition)
        if not inhibitors:
            return None
        gate = None
        for p in inhibitors:
            # In acyclic single-pass mode, the topological order can
            # visit a transition BEFORE its inhibitor place is computed
            # (the inhibitor place may be filled by some OTHER
            # downstream firing — typical of mutex patterns where two
            # transitions guard each other against a shared
            # "occupied" place that one of them will produce). For
            # those cases the inhibitor place hasn't been computed
            # yet, so we treat it as empty (no inhibition). The
            # time-unrolled forward pass populates all places at
            # every step, so on cyclic / multi-step runs the gate
            # behaves as the standard Petri-net semantics requires.
            if p not in place_acts:
                continue
            # (1 - a(p)) is the soft "place p is empty" indicator.
            factor = 1.0 - place_acts[p]
            gate = factor if gate is None else gate * factor
        return gate

    def _forward_unrolled(
        self,
        input_marking: dict[str, torch.Tensor],
        input_values: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        place_acts: dict[str, torch.Tensor] = {
            p: self._source_activation(p, input_marking, batch_size, device)
            if not self.net.preset(p)
            else input_marking[p]
            if p in input_marking
            else torch.zeros(batch_size, device=device)
            for p in self.net.places
        }
        # Initial place values: source places get their input_values
        # (default 1.0); non-source places start at zero. Both update
        # each step alongside place activations.
        place_values: dict[str, torch.Tensor] = {}
        for p in self.net.places:
            if p in input_values:
                place_values[p] = input_values[p]
            elif not self.net.preset(p):
                place_values[p] = torch.ones(batch_size, device=device)
            else:
                place_values[p] = torch.zeros(batch_size, device=device)
        trans_acts: dict[str, torch.Tensor] = {
            t: torch.zeros(batch_size, device=device) for t in self.net.transitions
        }
        # Per-transition in-flight queue for duration > 1 transitions.
        # When a transition with duration D fires at step n, its
        # firing activation is pushed onto in_flight[t]. The effective
        # output that contributes to place updates at step n is the
        # firing from D-1 steps ago — i.e. the head of the queue when
        # it has reached length D. Until then the transition has not
        # yet produced any output and contributes zero.
        # Transitions with duration 1 (the default) bypass this queue.
        in_flight: dict[str, list[torch.Tensor]] = {
            t: [] for t in self.net.transitions
        }

        eps = 1e-6

        for _ in range(self.num_steps):
            pre_acts: dict[str, torch.Tensor] = {
                t: self._pre_activation(t, place_acts) for t in self.net.transitions
            }
            new_trans: dict[str, torch.Tensor] = {}
            handled: set[str] = set()
            if self.routing == "softmax":
                for t in self.net.transitions:
                    if t in handled:
                        continue
                    if t in self._softmax_groups:
                        group = self._softmax_groups[t]
                        stacked = torch.stack([pre_acts[m] for m in group], dim=0)
                        softmaxed = F.softmax(stacked, dim=0)
                        for member, soft in zip(group, softmaxed):
                            gate = self._inhibitor_gate(member, place_acts)
                            if gate is not None:
                                soft = soft * gate
                            gguard = self._guard_gate(member, place_values)
                            if gguard is not None:
                                soft = soft * gguard
                            new_trans[member] = soft
                        handled.update(group)
                    else:
                        fired = self._fire_fn(pre_acts[t])
                        gate = self._inhibitor_gate(t, place_acts)
                        if gate is not None:
                            fired = fired * gate
                        gguard = self._guard_gate(t, place_values)
                        if gguard is not None:
                            fired = fired * gguard
                        new_trans[t] = fired
                        handled.add(t)
            else:
                for t, pre in pre_acts.items():
                    fired = self._fire_fn(pre)
                    gate = self._inhibitor_gate(t, place_acts)
                    if gate is not None:
                        fired = fired * gate
                    gguard = self._guard_gate(t, place_values)
                    if gguard is not None:
                        fired = fired * gguard
                    new_trans[t] = fired

            # Apply per-transition firing duration. For each transition
            # with duration D > 1, the firing computed *this* step
            # only contributes to place updates D-1 steps later. We
            # push the just-computed firing onto an in-flight queue
            # and emit whichever earlier firing is now mature.
            # Transitions with duration 1 pass through unchanged.
            trans_acts = {}
            for t, fired in new_trans.items():
                d = self.net.duration(t)
                if d == 1:
                    trans_acts[t] = fired
                else:
                    in_flight[t].append(fired)
                    if len(in_flight[t]) >= d:
                        # The firing at the head of the queue is exactly
                        # D-1 steps old now — its time has come.
                        trans_acts[t] = in_flight[t].pop(0)
                    else:
                        # Buffer not yet full: the transition is still
                        # "in progress" and has not produced anything
                        # observable yet.
                        trans_acts[t] = torch.zeros_like(fired)

            new_places: dict[str, torch.Tensor] = {}
            new_place_values: dict[str, torch.Tensor] = {}
            for p in self.net.places:
                if p in input_marking:
                    new_places[p] = input_marking[p]
                    # Persistent clamping also re-asserts the value
                    # channel — same role as input_marking for
                    # activations, just for values.
                    new_place_values[p] = input_values.get(
                        p, torch.ones(batch_size, device=device)
                    )
                    continue
                preset = self.net.preset(p)
                if not preset:
                    new_places[p] = self._source_activation(
                        p, input_marking, batch_size, device
                    )
                    new_place_values[p] = input_values.get(
                        p, torch.ones(batch_size, device=device)
                    )
                else:
                    new_places[p] = sum(
                        self.arc_weights[self._arc_key[(t, p)]] * trans_acts[t]
                        for t in preset
                    )
                    # Torch output transforms read bound input values
                    # from the *previous* step's place_values — the
                    # values that were standing at the transition's
                    # input places at the moment it fired.
                    value_numerator = sum(
                        self.arc_weights[self._arc_key[(t, p)]]
                        * trans_acts[t]
                        * self._output_value(
                            t, p, place_values, batch_size, device
                        )
                        for t in preset
                    )
                    new_place_values[p] = value_numerator / (new_places[p] + eps)
            place_acts = new_places
            place_values = new_place_values

        return {**place_acts, **trans_acts}
