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
        """
        if input_marking is None:
            input_marking = {}

        if batch_size is None:
            if input_marking:
                batch_size = next(iter(input_marking.values())).shape[0]
            else:
                batch_size = 1

        device = self._device()

        if self.num_steps == 0:
            return self._forward_acyclic(input_marking, batch_size, device)
        return self._forward_unrolled(input_marking, batch_size, device)

    def _forward_acyclic(
        self,
        input_marking: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        assert self._order is not None
        activations: dict[str, torch.Tensor] = {}
        softmax_cache: dict[str, torch.Tensor] = {}

        for node in self._order:
            if node in input_marking:
                activations[node] = input_marking[node]
                continue

            if node in self.net.places:
                preset = self.net.preset(node)
                if not preset:
                    activations[node] = self._source_activation(
                        node, input_marking, batch_size, device
                    )
                else:
                    contribs = [
                        self.arc_weights[self._arc_key[(t, node)]] * activations[t]
                        for t in preset
                    ]
                    activations[node] = sum(contribs)
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
                        if member == node:
                            activations[node] = soft
                        else:
                            softmax_cache[member] = soft
                else:
                    fired = self._fire_fn(pre)
                    gate = self._inhibitor_gate(node, activations)
                    if gate is not None:
                        fired = fired * gate
                    activations[node] = fired

        return activations

    def _pre_activation(
        self, transition: str, activations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        weighted = sum(
            self.arc_weights[self._arc_key[(p, transition)]] * activations[p]
            for p in self.net.preset(transition)
        )
        theta = self.transition_thresholds[self._threshold_key[transition]]
        return self.sharpness * (weighted - theta)

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
        trans_acts: dict[str, torch.Tensor] = {
            t: torch.zeros(batch_size, device=device) for t in self.net.transitions
        }

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
                            new_trans[member] = soft if gate is None else soft * gate
                        handled.update(group)
                    else:
                        fired = self._fire_fn(pre_acts[t])
                        gate = self._inhibitor_gate(t, place_acts)
                        new_trans[t] = fired if gate is None else fired * gate
                        handled.add(t)
            else:
                for t, pre in pre_acts.items():
                    fired = self._fire_fn(pre)
                    gate = self._inhibitor_gate(t, place_acts)
                    new_trans[t] = fired if gate is None else fired * gate
            trans_acts = new_trans

            new_places: dict[str, torch.Tensor] = {}
            for p in self.net.places:
                if p in input_marking:
                    new_places[p] = input_marking[p]
                    continue
                preset = self.net.preset(p)
                if not preset:
                    new_places[p] = self._source_activation(
                        p, input_marking, batch_size, device
                    )
                else:
                    new_places[p] = sum(
                        self.arc_weights[self._arc_key[(t, p)]] * trans_acts[t]
                        for t in preset
                    )
            place_acts = new_places

        return {**place_acts, **trans_acts}
