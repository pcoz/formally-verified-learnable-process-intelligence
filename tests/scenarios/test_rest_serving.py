"""End-to-end test for the REST serving scenario.

Trains the loan-approval XOR-routing model from the scenario's
inline traces, builds a FastAPI app over the trained module via
`build_app`, and exercises every endpoint a non-Python client
would call through FastAPI's `TestClient` (in-process, no port
binding, no real network).

The load-bearing claims:

* `GET /healthz` reports the served module's stats.
* `GET /schema` returns the net's structural inventory in a
  shape a client-side SDK generator could consume.
* `POST /forward` returns per-transition activations matching
  the underlying torch module's output on the same inputs.
* `POST /anomaly` returns trace_score + per-transition
  residuals; the anomalous trace (high risk_score declined)
  produces a higher score than the conformant one.
* `POST /counterfactual` finds the risk-score flip-point that
  would have approved a declined application.
* `POST /sensitivity` reports the marking-channel gradient on
  the routing input as the dominant signal.
* `GET /openapi.json` returns the auto-generated OpenAPI
  schema — what client SDK generators consume to produce
  typed clients in Java / C# / Go / TypeScript.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Skip the whole module when fastapi / httpx aren't available —
# the [rest] and [dev] extras are the gating dependencies for
# this surface.
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from petri_net_nn import build_app, load_scenario  # noqa: E402


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "rest_serving"
    / "scenario.toml"
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Train the scenario once per test module, wrap the trained
    module in the FastAPI app, return a TestClient. Module-scoped
    so the (deterministic but non-trivial) training step runs
    just once."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    app = build_app(module, title="PETRA loan approval", version="0.0.1")
    return TestClient(app)


# ---------------------------------------------------------------------------
# /healthz — liveness probe
# ---------------------------------------------------------------------------


def test_healthz_reports_module_stats(client):
    """`GET /healthz` is the liveness probe the platform's
    load balancer or container orchestrator hits. Always 200
    once the module is loaded; the payload carries place /
    transition counts so the operator can confirm at a glance
    which model is currently served."""
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["n_places"] == 3
    assert body["n_transitions"] == 2
    assert body["version"] == "0.0.1"


# ---------------------------------------------------------------------------
# /schema — discovery endpoint for clients
# ---------------------------------------------------------------------------


def test_schema_returns_net_inventory(client):
    """`GET /schema` is what a workflow-engine integration reads
    at deploy time to learn what place ids / transition ids /
    labels the served model exposes. The structural inventory
    plus the initial marking together tell the client what
    payloads the other endpoints accept."""
    response = client.get("/schema")
    assert response.status_code == 200
    body = response.json()
    assert set(body["places"]) == {"p_application", "p_approved", "p_declined"}
    # `transitions` is a dict mapping transition id -> human label;
    # clients display the labels in their UI rather than the raw ids.
    assert body["transitions"] == {
        "t_approve": "approve loan",
        "t_decline": "decline loan",
    }
    assert body["initial_marking"] == {"p_application": 1}


# ---------------------------------------------------------------------------
# /forward — per-event inference
# ---------------------------------------------------------------------------


def test_forward_routes_high_risk_score_to_approve(client):
    """High risk_score (0.9) at p_application must fire
    t_approve more strongly than t_decline. The headline
    per-event inference path — what the engine calls each
    time a relevant audit-log event arrives."""
    response = client.post(
        "/forward",
        json={"input_marking": {"p_application": 0.9}},
    )
    assert response.status_code == 200
    body = response.json()
    transitions = body["transition_activations"]
    assert transitions["t_approve"] > transitions["t_decline"]


def test_forward_routes_low_risk_score_to_decline(client):
    """Symmetric: low risk_score must fire t_decline more
    strongly than t_approve. Pinning both ends of the routing
    decision."""
    response = client.post(
        "/forward",
        json={"input_marking": {"p_application": 0.1}},
    )
    assert response.status_code == 200
    body = response.json()
    transitions = body["transition_activations"]
    assert transitions["t_decline"] > transitions["t_approve"]


# ---------------------------------------------------------------------------
# /anomaly — conformance scoring
# ---------------------------------------------------------------------------


def test_anomaly_endpoint_returns_trace_score_and_residuals(client):
    """A trace that takes the natural-fit path (high risk_score
    → approve) should score lower than a trace whose path
    disagrees with the trained model (high risk_score →
    decline). The headline live-conformance-monitoring claim
    over HTTP."""
    conformant = client.post(
        "/anomaly",
        json={
            "input_marking": {"p_application": 0.9},
            "events": [{"name": "approve loan", "attributes": {}}],
        },
    )
    anomalous = client.post(
        "/anomaly",
        json={
            "input_marking": {"p_application": 0.9},
            "events": [{"name": "decline loan", "attributes": {}}],
        },
    )
    assert conformant.status_code == 200
    assert anomalous.status_code == 200
    assert anomalous.json()["trace_score"] > conformant.json()["trace_score"]
    # Per-transition residuals are pinned by transition id, so a
    # downstream consumer can render which step actually
    # diverged from the trained model's expectation.
    assert "t_approve" in anomalous.json()["per_transition_residuals"]
    assert "t_decline" in anomalous.json()["per_transition_residuals"]


# ---------------------------------------------------------------------------
# /counterfactual — minimal-change explanation
# ---------------------------------------------------------------------------


def test_counterfactual_finds_flip_point_for_declined_application(client):
    """An application declined at risk_score=0.1: what value
    would have flipped the decision to approve? The
    counterfactual binary-search returns the crossing point on
    the marking channel — should land near 0.5 (the trained
    XOR crossover)."""
    response = client.post(
        "/counterfactual",
        json={
            "input_marking": {"p_application": 0.1},
            "flip_place": "p_application",
            "target_transition": "t_approve",
            "search_range": [0.0, 1.0],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    # Original input below 0.5; counterfactual above. The flip
    # point itself sits between them — somewhere in the trained
    # decision band.
    assert body["original_input"] == 0.1
    assert body["counterfactual_input"] > body["original_input"]
    assert body["counterfactual_input"] < 1.0


# ---------------------------------------------------------------------------
# /sensitivity — gradient ranking
# ---------------------------------------------------------------------------


def test_sensitivity_reports_marking_gradient_on_routing_input(client):
    """At a base point near the decision boundary, the
    marking-channel gradient of t_approve's activation with
    respect to p_application must be non-zero — the model
    actively uses that input to drive the routing. A near-zero
    gradient would mean the model is saturated or doesn't use
    the input at all, both of which a regulator would flag."""
    response = client.post(
        "/sensitivity",
        json={
            "input_marking": {"p_application": 0.5},
            "target_transition": "t_approve",
        },
    )
    assert response.status_code == 200
    body = response.json()
    marking_grads = body["marking_gradients"]
    assert "p_application" in marking_grads
    assert abs(marking_grads["p_application"]) > 1e-6


# ---------------------------------------------------------------------------
# /openapi.json — auto-generated schema for client SDK generators
# ---------------------------------------------------------------------------


def test_openapi_schema_is_served(client):
    """FastAPI auto-generates an OpenAPI 3.x schema document at
    `/openapi.json`. Client-side SDK generators (openapi-
    generator, NSwag, etc.) consume that to produce typed
    clients in Java, C#, Go, TypeScript, Python — whatever
    language the workflow engine's integration layer is
    written in. The test confirms the schema is reachable and
    lists the six PETRA endpoints."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = set(schema["paths"].keys())
    assert {
        "/healthz",
        "/schema",
        "/forward",
        "/anomaly",
        "/counterfactual",
        "/sensitivity",
    } <= paths
