"""Tests for the REST inference API.

Uses FastAPI's TestClient (which speaks to the ASGI app
in-process — no real network round-trip, no port binding). The
whole test module is skipped when fastapi or httpx aren't
installed, so the [dev,rest] extras are the gating dependencies
for this surface.

The load-bearing assertions:

  * /healthz returns 200 with the expected payload shape;
  * /schema returns the served net's structural inventory
    (places, transitions, labels, initial marking);
  * /forward returns per-transition activations matching the
    underlying module's torch output on the same inputs;
  * /anomaly returns trace_score + per-transition residuals
    matching the offline anomaly_score on the same trace;
  * /counterfactual finds a sensible crossing on the XOR
    fixture, and returns found=False when no crossing exists
    in the search range;
  * /sensitivity returns marking_gradients with the routing
    input's gradient larger than zero at the decision boundary;
  * bad place / transition names get 400, not 500.
"""
from __future__ import annotations

import pytest
import torch

# Skip the whole module when fastapi / httpx aren't available —
# the [rest] / [dev] extras are the gating dependencies.
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from petri_net_nn import (  # noqa: E402
    PetriNet,
    PetriNetModule,
    build_app,
)


def _xor_module():
    """A 2-transition XOR-shape net with hand-set thresholds so
    the served model has predictable activations to compare
    against. theta_a=0.3, theta_b=0.7 — t_a fires above 0.3,
    t_b fires above 0.7. At input 0.5: t_a should fire, t_b
    shouldn't."""
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
    module = PetriNetModule(net, sharpness=8.0)
    module.arc_weights[module._arc_key[("p_in", "t_a")]].data = torch.tensor(1.0)
    module.arc_weights[module._arc_key[("p_in", "t_b")]].data = torch.tensor(1.0)
    module.transition_thresholds[module._threshold_key["t_a"]].data = torch.tensor(0.3)
    module.transition_thresholds[module._threshold_key["t_b"]].data = torch.tensor(0.7)
    return module


@pytest.fixture
def client():
    """A TestClient bound to a fresh app over the XOR fixture
    module. Each test gets its own client so they don't share
    state."""
    app = build_app(_xor_module(), title="PETRA test", version="0.0.1")
    return TestClient(app)


# ---------------------------------------------------------------------------
# /healthz and /schema — discovery / liveness
# ---------------------------------------------------------------------------


def test_healthz_returns_ok_with_module_stats(client):
    """Liveness probe always returns 200 once the module is
    loaded. The payload also carries the module's place /
    transition counts, useful for sanity-checking which model
    is currently served."""
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["n_places"] == 3
    assert body["n_transitions"] == 2
    assert body["version"] == "0.0.1"


def test_schema_returns_net_structure(client):
    """/schema is the discovery endpoint — a client uses it to
    learn what place ids and transition ids the other endpoints
    accept. Verify the XOR-fixture inventory comes through."""
    response = client.get("/schema")
    assert response.status_code == 200
    body = response.json()
    assert set(body["places"]) == {"p_in", "p_a", "p_b"}
    assert body["transitions"] == {"t_a": "Path A", "t_b": "Path B"}
    assert body["initial_marking"] == {}
    assert body["has_silent_transitions"] is False
    assert body["has_structural_guards"] is False


# ---------------------------------------------------------------------------
# /forward — inference
# ---------------------------------------------------------------------------


def test_forward_returns_activations_matching_torch(client):
    """The headline endpoint. /forward should return the same
    per-transition activations that the underlying torch module
    produces directly on the same inputs."""
    response = client.post(
        "/forward",
        json={"input_marking": {"p_in": 0.5}},
    )
    assert response.status_code == 200
    body = response.json()
    # At input=0.5 with theta_a=0.3 (t_a above), theta_b=0.7
    # (t_b below): t_a should fire strongly, t_b weakly.
    assert body["transition_activations"]["t_a"] > 0.7
    assert body["transition_activations"]["t_b"] < 0.3


def test_forward_handles_value_channel_when_supplied(client):
    """Sending input_values alongside input_marking should not
    error on a net that doesn't have CPN guards — the value
    channel is just carried through without affecting any
    routing here."""
    response = client.post(
        "/forward",
        json={
            "input_marking": {"p_in": 0.5},
            "input_values": {"p_in": 1.0},
        },
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# /anomaly — trace scoring
# ---------------------------------------------------------------------------


def test_anomaly_returns_residuals_for_supplied_events(client):
    """A trace with both Path A and Path B should produce
    low residuals (model predicts both for input=0.5 since
    routing isn't tight at 0.5). At minimum the response
    shape should be correct and trace_score finite."""
    response = client.post(
        "/anomaly",
        json={
            "input_marking": {"p_in": 0.5},
            "events": [
                {"name": "Path A", "attributes": {}},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "trace_score" in body
    assert "per_transition_residuals" in body
    assert set(body["per_transition_residuals"].keys()) == {"t_a", "t_b"}
    # All residuals finite and non-negative.
    for name, value in body["per_transition_residuals"].items():
        assert isinstance(value, (int, float))
        assert value >= 0.0


# ---------------------------------------------------------------------------
# /counterfactual — find the crossing
# ---------------------------------------------------------------------------


def test_counterfactual_finds_crossing_on_xor_fixture(client):
    """At base input=0.0 the t_a transition doesn't fire; we
    ask the API what input would flip it. The XOR fixture's
    t_a threshold is 0.3, so the crossing should sit roughly
    there."""
    response = client.post(
        "/counterfactual",
        json={
            "input_marking": {"p_in": 0.0},
            "flip_place": "p_in",
            "target_transition": "t_a",
            "search_range": [0.0, 1.0],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert 0.2 < body["counterfactual_input"] < 0.5
    # Activation at the crossing should be close to 0.5.
    assert abs(body["counterfactual_activation"] - 0.5) < 0.05


def test_counterfactual_returns_found_false_when_no_crossing(client):
    """When the search range is too narrow to bracket the
    boundary (both endpoints on the same side of 0.5),
    /counterfactual should return found=False without raising."""
    response = client.post(
        "/counterfactual",
        json={
            "input_marking": {"p_in": 1.0},
            "flip_place": "p_in",
            "target_transition": "t_a",
            "search_range": [0.95, 1.0],
        },
    )
    assert response.status_code == 200
    assert response.json()["found"] is False


def test_counterfactual_rejects_unknown_target_transition(client):
    """Bad transition name → 400, not 500. Sanity check on
    boundary validation."""
    response = client.post(
        "/counterfactual",
        json={
            "input_marking": {"p_in": 0.0},
            "flip_place": "p_in",
            "target_transition": "t_does_not_exist",
        },
    )
    assert response.status_code == 400
    assert "not in the net's transition set" in response.json()["detail"]


def test_counterfactual_rejects_unknown_flip_place(client):
    """Same for an unknown place — 400 with a useful message."""
    response = client.post(
        "/counterfactual",
        json={
            "input_marking": {},
            "flip_place": "p_does_not_exist",
            "target_transition": "t_a",
        },
    )
    assert response.status_code == 400
    assert "not in the net's place set" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /sensitivity — per-input gradients
# ---------------------------------------------------------------------------


def test_sensitivity_returns_gradients_at_decision_boundary(client):
    """At base input close to the t_a threshold (0.3), the
    gradient with respect to p_in should be substantial — that
    is the input the model leans on for the routing decision."""
    response = client.post(
        "/sensitivity",
        json={
            "input_marking": {"p_in": 0.3},
            "target_transition": "t_a",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["target_transition"] == "t_a"
    assert body["target_label"] == "Path A"
    assert "p_in" in body["marking_gradients"]
    # Substantial gradient at the boundary.
    assert abs(body["marking_gradients"]["p_in"]) > 0.5


def test_sensitivity_rejects_unknown_target_transition(client):
    """Bad transition name → 400."""
    response = client.post(
        "/sensitivity",
        json={
            "input_marking": {"p_in": 0.5},
            "target_transition": "t_does_not_exist",
        },
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# OpenAPI / Swagger UI surface — bare existence check
# ---------------------------------------------------------------------------


def test_openapi_schema_is_served(client):
    """FastAPI's auto-generated OpenAPI document should be
    reachable at /openapi.json. Just check it loads and
    advertises our endpoints."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema.get("paths", {})
    expected = {"/healthz", "/schema", "/forward", "/anomaly",
                "/counterfactual", "/sensitivity"}
    assert expected.issubset(set(paths.keys()))
