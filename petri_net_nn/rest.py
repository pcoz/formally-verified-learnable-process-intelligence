"""HTTP wrapper around a trained :class:`PetriNetModule`.

A REST API is the simplest "talk to PETRA from any language"
surface. Non-Python consumers — Java services, Go workers,
React dashboards, Postman scratchpads — speak JSON over HTTP
and don't want to bother with model loading, torch weights, or
the dict-of-tensor interface PETRA uses natively.

This module exposes a :func:`build_app` factory that takes a
trained module and returns a FastAPI application. The returned
app exposes six endpoints, listed in the order an analyst would
typically hit them:

  * ``GET /healthz``       — liveness probe (always 200).
  * ``GET /schema``        — net structure (places, transitions,
                              labels, initial marking).
  * ``POST /forward``      — inference: marking + optional value
                              channel → per-transition activations.
  * ``POST /anomaly``      — trace scoring: events + marking →
                              per-transition residuals + trace
                              score.
  * ``POST /counterfactual`` — minimal change to flip one target
                                transition's firing decision.
  * ``POST /sensitivity``  — per-input gradients of one target
                              transition at a base point.

FastAPI auto-generates OpenAPI 3.x schemas and Swagger UI from
the Pydantic models below — point a browser at ``/docs`` to
exercise the API interactively. Request and response bodies are
plain JSON with no PETRA-specific encoding; place ids and
transition ids are just strings.

The fastapi / pydantic imports are guarded so that
``import petri_net_nn`` works without the ``[rest]`` extra
installed — :func:`build_app` will raise a clear ImportError
when called if the dependencies are missing. The Pydantic
models below are only defined when the imports succeed; this
avoids the Pydantic-v2 forward-reference issue that hits when
models referencing each other are defined inside a function
scope.

What's deliberately NOT included in this slice:

  * Multi-model registries (one app == one trained module).
    Spin up multiple FastAPI processes if you need multiple
    models served from one host.
  * Training endpoints — this is an **inference** API.
    Training mutates the module's parameters and is expensive
    enough that it doesn't belong on a synchronous HTTP request
    path.
  * Authentication / rate limiting / CORS — wire these in at
    the caller's app layer using FastAPI's standard middleware.
"""
from __future__ import annotations

from typing import Any, Literal

import torch

from petri_net_nn.compiler import PetriNetModule
from petri_net_nn.interpretability import (
    find_counterfactual,
    transition_sensitivity,
)
from petri_net_nn.traces import anomaly_score
from petri_net_nn.xes import XESEvent, XESTrace


# Try the optional fastapi / pydantic imports at module load.
# If they're missing the module still loads (so importing
# petri_net_nn doesn't break for users without the [rest]
# extra); build_app raises a clear ImportError when called.
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


# Pydantic models live at module scope when FastAPI is
# available. Nesting them inside build_app trips up Pydantic
# v2's forward-reference resolver (classes referencing each
# other from a function-local scope don't resolve cleanly).
if _FASTAPI_AVAILABLE:

    class HealthResponse(BaseModel):
        """Liveness probe payload."""

        status: Literal["ok"] = "ok"
        version: str
        n_places: int
        n_transitions: int

    class SchemaResponse(BaseModel):
        """Static structural information about the served net.
        Returned by ``GET /schema``; useful for discovering the
        place and transition vocabulary the other endpoints
        accept and emit."""

        places: list[str]
        transitions: dict[str, str] = Field(
            description="transition id → label"
        )
        initial_marking: dict[str, int]
        has_silent_transitions: bool
        has_structural_guards: bool

    class ForwardRequest(BaseModel):
        """Inputs to the forward pass.

        ``input_marking`` keys are place ids; values are
        activations in [0, 1]. Places not present in the dict
        default to their initial-marking activation (1.0 if
        flagged in the net's initial marking, 0.0 otherwise).
        ``input_values`` is the optional coloured-token value
        channel — same shape, value scale chosen by the
        modeller's domain."""

        input_marking: dict[str, float]
        input_values: dict[str, float] | None = None

    class ForwardResponse(BaseModel):
        """Per-transition firing activations from one forward
        pass, plus the per-place activations the net carries at
        the end of the pass."""

        transition_activations: dict[str, float]
        place_activations: dict[str, float]

    class TraceEvent(BaseModel):
        """One event in a trace.

        ``name`` is matched against the net's transition labels
        — the same convention the offline ``parse_xes`` /
        ``anomaly_score`` pipeline uses. Unknown event names
        are silently ignored (consistent with offline
        semantics)."""

        name: str
        attributes: dict[str, str] = Field(default_factory=dict)

    class AnomalyRequest(BaseModel):
        """Score a trace under the trained module.

        The request bundles the input marking the trace was
        observed under (so the model can compute its expected
        firings) and the list of events the trace actually
        contained."""

        input_marking: dict[str, float]
        input_values: dict[str, float] | None = None
        events: list[TraceEvent]
        # Trace attributes carry domain context that
        # downstream attribute-driven helpers might use. They're
        # passed through to the assembled XESTrace.attributes
        # for completeness.
        attributes: dict[str, str] = Field(default_factory=dict)

    class AnomalyResponse(BaseModel):
        """Per-transition residuals plus the trace-level
        scalar (sum of residuals). Higher score = more
        anomalous; per-transition entries identify *which* parts
        of the process diverged."""

        trace_score: float
        per_transition_residuals: dict[str, float]

    class CounterfactualRequest(BaseModel):
        """Find the minimum change to one input that flips the
        target transition's firing decision."""

        input_marking: dict[str, float]
        input_values: dict[str, float] | None = None
        flip_place: str
        target_transition: str
        flip_channel: Literal["marking", "value"] = "marking"
        search_range: tuple[float, float] = (0.0, 1.0)
        tolerance: float = 1e-3

    class CounterfactualResponse(BaseModel):
        """Counterfactual analysis result. ``found`` is False
        when no crossing of the firing-decision boundary exists
        in the supplied search range; the other fields are then
        absent."""

        found: bool
        target_transition: str | None = None
        target_label: str | None = None
        flipped_place: str | None = None
        flipped_channel: str | None = None
        original_input: float | None = None
        counterfactual_input: float | None = None
        original_activation: float | None = None
        counterfactual_activation: float | None = None

    class SensitivityRequest(BaseModel):
        """Per-input gradient analysis of one transition's
        firing activation at the supplied base point."""

        input_marking: dict[str, float]
        input_values: dict[str, float] | None = None
        target_transition: str

    class SensitivityResponse(BaseModel):
        """Per-place gradient magnitudes for marking and value
        channels separately. Larger absolute values = the
        firing decision pivots more on that input *locally*."""

        target_transition: str
        target_label: str
        base_activation: float
        marking_gradients: dict[str, float]
        value_gradients: dict[str, float]


def _build_tensors(
    input_marking: dict[str, float],
    input_values: dict[str, float] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """Build the (batch=1) tensor dicts the forward pass
    expects from the JSON-serialisable float dicts the API
    receives. Centralised so each endpoint stays short."""
    marking_t = {
        p: torch.tensor([float(v)]) for p, v in input_marking.items()
    }
    values_t: dict[str, torch.Tensor] | None
    if input_values is not None:
        values_t = {
            p: torch.tensor([float(v)]) for p, v in input_values.items()
        }
    else:
        values_t = None
    return marking_t, values_t


def build_app(
    module: PetriNetModule,
    *,
    title: str = "PETRA REST API",
    version: str = "0.1.0",
    description: str | None = None,
) -> Any:
    """Build a FastAPI application that wraps a trained module.

    The returned object is a ``fastapi.FastAPI`` instance; the
    return annotation is ``Any`` to keep the fastapi dependency
    optional at the type level.

    Parameters
    ----------
    module
        The trained :class:`PetriNetModule` to serve. The
        module's net structure determines what places and
        transitions the API endpoints accept and emit by name.
    title, version, description
        Forwarded to ``FastAPI(...)`` and surface in the
        auto-generated OpenAPI / Swagger UI at ``/docs``.

    Returns
    -------
    fastapi.FastAPI
        Mount with uvicorn or any ASGI server. A minimal
        production runner::

            import uvicorn
            from petri_net_nn import load_scenario, build_app

            ctx = load_scenario("my_scenario.toml")
            module, _ = ctx.train()
            app = build_app(module)

            uvicorn.run(app, host="0.0.0.0", port=8000)

    Raises
    ------
    ImportError
        If fastapi or pydantic isn't available. Install with
        ``pip install 'petra-nn[rest]'``.
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "petri_net_nn.rest.build_app requires fastapi and "
            "pydantic. Install with: pip install 'petra-nn[rest]'"
        )

    app = FastAPI(
        title=title,
        version=version,
        description=description or (
            "REST inference API for a PETRA-trained Petri-net model. "
            "Six endpoints — schema discovery, forward pass, anomaly "
            "scoring, counterfactual analysis, and per-input "
            "sensitivity. See /docs for the interactive Swagger UI."
        ),
    )

    @app.get(
        "/healthz",
        response_model=HealthResponse,
        summary="Liveness probe",
    )
    def health() -> HealthResponse:
        """Always returns 200 once the server has loaded a
        module. Useful for Kubernetes / load-balancer health
        checks."""
        return HealthResponse(
            version=version,
            n_places=len(module.net.places),
            n_transitions=len(module.net.transitions),
        )

    @app.get(
        "/schema",
        response_model=SchemaResponse,
        summary="Net structure (places, transitions, initial marking)",
    )
    def schema() -> SchemaResponse:
        """Return the static structure of the served net so a
        client can discover what place ids and transition ids
        the other endpoints will accept and emit."""
        net = module.net
        return SchemaResponse(
            places=sorted(net.places),
            transitions={
                t: net.transition_labels.get(t, t)
                for t in sorted(net.transitions)
            },
            initial_marking=dict(net.initial_marking),
            has_silent_transitions=bool(net.silent_transitions),
            has_structural_guards=bool(net.transition_structural_guards),
        )

    @app.post(
        "/forward",
        response_model=ForwardResponse,
        summary="One forward pass through the trained module",
    )
    def forward(request: ForwardRequest) -> ForwardResponse:
        """Run the trained module on the supplied input marking
        (and optional value channel). Returns per-transition
        firing activations and the per-place activations after
        the pass."""
        marking_t, values_t = _build_tensors(
            request.input_marking, request.input_values
        )
        module.eval()
        with torch.no_grad():
            out = module(
                input_marking=marking_t,
                input_values=values_t,
                batch_size=1,
            )
        # The output dict contains both places and transitions
        # in time-unrolled mode (acyclic mode populates both via
        # the topological pass). Split them by kind so the
        # caller gets a clean two-bucket response.
        net = module.net
        transition_activations = {
            t: float(out[t].item()) for t in net.transitions if t in out
        }
        place_activations = {
            p: float(out[p].item()) for p in net.places if p in out
        }
        return ForwardResponse(
            transition_activations=transition_activations,
            place_activations=place_activations,
        )

    @app.post(
        "/anomaly",
        response_model=AnomalyResponse,
        summary="Score a trace under the trained module",
    )
    def anomaly(request: AnomalyRequest) -> AnomalyResponse:
        """Build a :class:`XESTrace` from the supplied events
        and attributes; score it against the trained module
        using the offline ``anomaly_score`` function. Returns
        per-transition residuals (absolute difference between
        predicted and observed firing) and the trace-level
        scalar score (sum of those residuals)."""
        trace = XESTrace(
            attributes=dict(request.attributes),
            events=[
                XESEvent(name=e.name, attributes=dict(e.attributes))
                for e in request.events
            ],
        )
        # The offline anomaly_score function expects an
        # attribute_to_marking callable to derive the input
        # marking from the trace. Over HTTP, the marking is
        # supplied directly in the request — so we plug in a
        # closure that ignores the trace and returns the
        # request's marking.
        request_marking = dict(request.input_marking)
        request_values = (
            dict(request.input_values)
            if request.input_values is not None
            else None
        )

        def attribute_to_marking(_trace: XESTrace) -> dict[str, float]:
            return dict(request_marking)

        attribute_to_values = None
        if request_values is not None:
            def attribute_to_values(_trace: XESTrace) -> dict[str, float]:  # noqa: F811
                return dict(request_values)

        residuals = anomaly_score(
            module,
            trace,
            attribute_to_marking=attribute_to_marking,
            attribute_to_values=attribute_to_values,
        )
        return AnomalyResponse(
            trace_score=sum(residuals.values()),
            per_transition_residuals=residuals,
        )

    @app.post(
        "/counterfactual",
        response_model=CounterfactualResponse,
        summary="Find the minimum input change that flips a transition's firing",
    )
    def counterfactual(
        request: CounterfactualRequest,
    ) -> CounterfactualResponse:
        """Binary-search the input at ``flip_place`` (along
        ``flip_channel``) to find the boundary at which
        ``target_transition``'s firing activation crosses 0.5.
        Returns ``found=False`` when no crossing exists in the
        supplied search range."""
        net = module.net
        if request.flip_place not in net.places:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"flip_place {request.flip_place!r} is not in "
                    f"the net's place set"
                ),
            )
        if request.target_transition not in net.transitions:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"target_transition {request.target_transition!r} "
                    f"is not in the net's transition set"
                ),
            )
        cf = find_counterfactual(
            module,
            dict(request.input_marking),
            flip_place=request.flip_place,
            target_transition=request.target_transition,
            base_values=(
                dict(request.input_values)
                if request.input_values is not None
                else None
            ),
            flip_channel=request.flip_channel,
            search_range=request.search_range,
            tolerance=request.tolerance,
        )
        if cf is None:
            return CounterfactualResponse(
                found=False,
                target_transition=request.target_transition,
            )
        return CounterfactualResponse(
            found=True,
            target_transition=cf.target_transition,
            target_label=cf.target_label,
            flipped_place=cf.flipped_place,
            flipped_channel=cf.flipped_channel,
            original_input=cf.original_input,
            counterfactual_input=cf.counterfactual_input,
            original_activation=cf.original_activation,
            counterfactual_activation=cf.counterfactual_activation,
        )

    @app.post(
        "/sensitivity",
        response_model=SensitivityResponse,
        summary="Per-input gradient magnitudes at one base point",
    )
    def sensitivity(request: SensitivityRequest) -> SensitivityResponse:
        """Compute the per-input gradient of
        ``target_transition``'s firing activation with respect
        to every input at the supplied base point. Larger
        magnitude = the firing decision pivots more on that
        input locally."""
        net = module.net
        if request.target_transition not in net.transitions:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"target_transition {request.target_transition!r} "
                    f"is not in the net's transition set"
                ),
            )
        report = transition_sensitivity(
            module,
            dict(request.input_marking),
            request.target_transition,
            base_values=(
                dict(request.input_values)
                if request.input_values is not None
                else None
            ),
        )
        return SensitivityResponse(
            target_transition=report.target_transition,
            target_label=report.target_label,
            base_activation=report.base_activation,
            marking_gradients=dict(report.marking_gradients),
            value_gradients=dict(report.value_gradients),
        )

    return app
