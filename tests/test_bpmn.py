"""Tests for the BPMN -> Petri net extractor.

Each fixture is a small BPMN 2.0 document exercising one of the
translation rules in §3 of the architecture spec. The tests assert
both structural properties of the produced Petri net (place / transition
counts, presence of expected arcs) and dynamic properties (token-game
execution from the initial marking reaches the sink place).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from petri_net_nn import PetriNet, parse_bpmn


FIXTURES = Path(__file__).parent / "fixtures"


def _fire_to_completion(net: PetriNet, sink: str, max_steps: int = 100) -> bool:
    """Greedy token-game simulation: at each step fire any enabled
    transition until the sink place holds a token (success) or no
    transition is enabled (deadlock). Returns whether the sink was
    reached. Suitable only for the small acyclic fixtures here."""
    marking = dict(net.initial_marking)
    for _ in range(max_steps):
        if marking.get(sink, 0) >= 1:
            return True
        enabled = sorted(net.enabled_transitions(marking))
        if not enabled:
            return False
        marking = net.fire(enabled[0], marking)
    return False


# ---------------------------------------------------------------------------
# Subnet 1 — sequential
# ---------------------------------------------------------------------------


def test_simple_sequence_structure():
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    assert net.places == {"p_f1", "p_f2"}
    assert net.transitions == {"t_do_work"}
    assert net.flow == {("p_f1", "t_do_work"), ("t_do_work", "p_f2")}
    assert net.initial_marking == {"p_f1": 1}
    assert net.transition_labels["t_do_work"] == "Do work"
    assert net.validate() == []


def test_simple_sequence_reaches_sink():
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    assert _fire_to_completion(net, sink="p_f2")


# ---------------------------------------------------------------------------
# Subnet 2 — XOR
# ---------------------------------------------------------------------------


def test_xor_branch_has_split_transitions_sharing_input():
    net = parse_bpmn(FIXTURES / "xor_branch.bpmn")
    split_in = "p_f0"
    split_transitions = {t for t in net.transitions if t.startswith("t_xor_split_")}
    assert len(split_transitions) == 2
    for t in split_transitions:
        assert net.preset(t) == {split_in}
    outs = {next(iter(net.postset(t))) for t in split_transitions}
    assert outs == {"p_fA1", "p_fB1"}


def test_xor_branch_has_join_transitions_sharing_output():
    net = parse_bpmn(FIXTURES / "xor_branch.bpmn")
    join_out = "p_fEnd"
    join_transitions = {t for t in net.transitions if t.startswith("t_xor_join_")}
    assert len(join_transitions) == 2
    for t in join_transitions:
        assert net.postset(t) == {join_out}
    ins = {next(iter(net.preset(t))) for t in join_transitions}
    assert ins == {"p_fA2", "p_fB2"}


def test_xor_branch_well_formed_and_completes():
    net = parse_bpmn(FIXTURES / "xor_branch.bpmn")
    assert net.validate() == []
    assert _fire_to_completion(net, sink="p_fEnd")


# ---------------------------------------------------------------------------
# Subnet 3 / Subnet 4 — AND-split and AND-join
# ---------------------------------------------------------------------------


def test_and_branch_split_is_one_transition_with_multiple_outputs():
    net = parse_bpmn(FIXTURES / "and_branch.bpmn")
    t_split = "t_and_split"
    assert t_split in net.transitions
    assert net.preset(t_split) == {"p_f0"}
    assert net.postset(t_split) == {"p_fA1", "p_fB1"}


def test_and_branch_join_is_one_transition_with_multiple_inputs():
    net = parse_bpmn(FIXTURES / "and_branch.bpmn")
    t_join = "t_and_join"
    assert t_join in net.transitions
    assert net.preset(t_join) == {"p_fA2", "p_fB2"}
    assert net.postset(t_join) == {"p_fEnd"}


def test_and_branch_join_blocks_until_both_branches_complete():
    """Token-game check that AND-join semantics are preserved: the join
    transition is not enabled when only one branch has finished."""
    net = parse_bpmn(FIXTURES / "and_branch.bpmn")
    partial = {"p_fA2": 1}
    assert not net.is_enabled("t_and_join", partial)
    full = {"p_fA2": 1, "p_fB2": 1}
    assert net.is_enabled("t_and_join", full)


def test_and_branch_completes_via_token_game():
    net = parse_bpmn(FIXTURES / "and_branch.bpmn")
    assert net.validate() == []
    assert _fire_to_completion(net, sink="p_fEnd")


# ---------------------------------------------------------------------------
# Composition — the approval process sketched in §6
# ---------------------------------------------------------------------------


def test_approval_process_contains_all_four_supported_subnet_patterns():
    net = parse_bpmn(FIXTURES / "approval.bpmn")
    assert net.validate() == []

    seq_transitions = {
        t for t in net.transitions
        if len(net.preset(t)) == 1 and len(net.postset(t)) == 1
    }
    assert any(t == "t_triage" for t in seq_transitions)

    and_splits = {
        t for t in net.transitions
        if len(net.preset(t)) == 1 and len(net.postset(t)) > 1
    }
    assert "t_expedite_split" in and_splits

    and_joins = {
        t for t in net.transitions
        if len(net.preset(t)) > 1 and len(net.postset(t)) == 1
    }
    assert "t_expedite_join" in and_joins

    xor_split_input = "p_f_to_route"
    xor_split_ts = [t for t in net.transitions if net.preset(t) == {xor_split_input}]
    assert len(xor_split_ts) == 2

    xor_join_output = "p_f_merge_decide"
    xor_join_ts = [
        t for t in net.transitions if net.postset(t) == {xor_join_output}
    ]
    assert len(xor_join_ts) == 2


def test_approval_process_initial_marking_is_unique_and_on_start():
    net = parse_bpmn(FIXTURES / "approval.bpmn")
    assert net.initial_marking == {"p_f_submit": 1}


def test_approval_process_completes_for_either_route():
    """Run the token game twice. We force the XOR branches by removing
    the unwanted branch's transition before simulating, so each run
    exercises exactly one of the routing alternatives."""
    for forced_branch in ("standard", "expedited"):
        net = parse_bpmn(FIXTURES / "approval.bpmn")

        unwanted = "p_f_route_exp" if forced_branch == "standard" else "p_f_route_std"
        net.transitions = {
            t for t in net.transitions if unwanted not in net.postset(t)
        }
        net.flow = {
            (s, d) for s, d in net.flow
            if s in net.transitions or d in net.transitions
        }

        assert _fire_to_completion(net, sink="p_f_done"), (
            f"approval did not complete for branch {forced_branch!r}"
        )


# ---------------------------------------------------------------------------
# Retry loop — a cyclic BPMN that produces a cyclic Petri net
# ---------------------------------------------------------------------------


def test_retry_loop_parses_into_cyclic_net():
    """parse_bpmn does not enforce acyclicity; a retry pattern (XOR
    back-edge to an earlier gateway) yields a cyclic Petri net that is
    perfectly well-formed structurally."""
    net = parse_bpmn(FIXTURES / "retry_loop.bpmn")
    assert net.validate() == []
    assert net.initial_marking == {"p_f_start": 1}

    backedge_transitions = [
        t for t in net.transitions
        if "p_f_retry" in net.preset(t)
        and "p_f_join_attempt" in net.postset(t)
    ]
    assert len(backedge_transitions) == 1


def test_retry_loop_token_game_can_complete_in_one_pass():
    """One attempt that succeeds: fire join, attempt, then the success
    branch of the check gateway. The token should end on p_f_ok."""
    net = parse_bpmn(FIXTURES / "retry_loop.bpmn")
    marking = dict(net.initial_marking)

    enabled_join = [
        t for t in net.enabled_transitions(marking) if t.startswith("t_join_")
    ]
    marking = net.fire(enabled_join[0], marking)
    marking = net.fire("t_attempt", marking)

    success = next(
        t for t in net.transitions
        if "p_f_attempt_check" in net.preset(t)
        and "p_f_ok" in net.postset(t)
    )
    marking = net.fire(success, marking)
    assert marking == {"p_f_ok": 1}


def test_retry_loop_token_game_can_loop_back():
    """Fire the retry branch of the check gateway and verify the token
    lands on p_f_retry, from which the join gateway re-enables the
    attempt task."""
    net = parse_bpmn(FIXTURES / "retry_loop.bpmn")
    marking = dict(net.initial_marking)
    enabled_join = [
        t for t in net.enabled_transitions(marking) if t.startswith("t_join_")
    ]
    marking = net.fire(enabled_join[0], marking)
    marking = net.fire("t_attempt", marking)

    retry = next(
        t for t in net.transitions
        if "p_f_attempt_check" in net.preset(t)
        and "p_f_retry" in net.postset(t)
    )
    marking = net.fire(retry, marking)
    assert marking == {"p_f_retry": 1}

    rejoin = next(
        t for t in net.transitions
        if "p_f_retry" in net.preset(t)
        and "p_f_join_attempt" in net.postset(t)
    )
    marking = net.fire(rejoin, marking)
    assert marking == {"p_f_join_attempt": 1}


# ---------------------------------------------------------------------------
# Subnet 5 — saga compensation via boundary event + association
# ---------------------------------------------------------------------------


def test_saga_structure_matches_section_5_subnet():
    net = parse_bpmn(FIXTURES / "saga.bpmn")
    assert net.places == {
        "p_f_start",
        "p_f_ok",
        "p_f_refunded",
        "p_compensating_bnd_charge",
    }
    assert net.transitions == {"t_charge_card", "t_bnd_charge", "t_refund_card"}
    assert net.initial_marking == {"p_f_start": 1}
    assert net.validate() == []


def test_saga_competing_transitions_share_p_active():
    """T_succeed and T_fail both consume from P_active, exactly as §5
    Subnet 5 draws it. Either can fire from the initial marking."""
    net = parse_bpmn(FIXTURES / "saga.bpmn")
    assert net.preset("t_charge_card") == {"p_f_start"}
    assert net.preset("t_bnd_charge") == {"p_f_start"}
    assert net.is_enabled("t_charge_card", net.initial_marking)
    assert net.is_enabled("t_bnd_charge", net.initial_marking)


def test_saga_compensation_chains_to_recovery_state():
    net = parse_bpmn(FIXTURES / "saga.bpmn")
    assert net.preset("t_refund_card") == {"p_compensating_bnd_charge"}
    assert net.postset("t_refund_card") == {"p_f_refunded"}


def test_saga_both_outcomes_reachable_via_token_game():
    net = parse_bpmn(FIXTURES / "saga.bpmn")

    success_marking = net.fire("t_charge_card", net.initial_marking)
    assert success_marking == {"p_f_ok": 1}

    after_fail = net.fire("t_bnd_charge", net.initial_marking)
    assert after_fail == {"p_compensating_bnd_charge": 1}
    after_compensate = net.fire("t_refund_card", after_fail)
    assert after_compensate == {"p_f_refunded": 1}


def test_saga_boundary_event_without_association_raises():
    xml = """<?xml version="1.0"?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <process id="p">
            <startEvent id="s"/><task id="t" name="x"/><endEvent id="e"/>
            <boundaryEvent id="b" attachedToRef="t">
              <compensateEventDefinition/>
            </boundaryEvent>
            <sequenceFlow id="a" sourceRef="s" targetRef="t"/>
            <sequenceFlow id="c" sourceRef="t" targetRef="e"/>
          </process>
        </definitions>"""
    with pytest.raises(ValueError, match="no <association>"):
        parse_bpmn(xml)


def test_saga_handler_without_isForCompensation_raises():
    xml = """<?xml version="1.0"?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <process id="p">
            <startEvent id="s"/>
            <task id="t" name="x"/>
            <task id="h" name="handler"/>
            <boundaryEvent id="b" attachedToRef="t">
              <compensateEventDefinition/>
            </boundaryEvent>
            <endEvent id="e1"/><endEvent id="e2"/>
            <sequenceFlow id="a" sourceRef="s" targetRef="t"/>
            <sequenceFlow id="c" sourceRef="t" targetRef="e1"/>
            <sequenceFlow id="d" sourceRef="h" targetRef="e2"/>
            <association sourceRef="b" targetRef="h"/>
          </process>
        </definitions>"""
    with pytest.raises(ValueError, match="isForCompensation"):
        parse_bpmn(xml)


def test_saga_with_throw_structure():
    """The throw-model interpretation: t_charge_card now has TWO output
    arcs (the normal outflow and a completion marker), and a new
    t_trigger_compensate transition AND-joins the throw event's
    incoming flow with that completion marker before producing the
    compensating place."""
    net = parse_bpmn(FIXTURES / "saga_with_throw.bpmn")
    assert net.validate() == []
    assert "p_completed_charge_card" in net.places
    assert "t_trigger_compensate" in net.transitions

    assert net.postset("t_charge_card") == {
        "p_f_charge_check",
        "p_completed_charge_card",
    }
    assert net.preset("t_trigger_compensate") == {
        "p_f_to_throw",
        "p_completed_charge_card",
    }
    assert net.postset("t_trigger_compensate") == {"p_compensating_bnd_charge"}


def test_saga_with_throw_no_t_fail_alternative_created():
    """In throw mode the boundary event is a passive marker; the
    t_{bnd_id} fail-alternative transition that exists in the plain
    saga.bpmn must NOT be created."""
    net = parse_bpmn(FIXTURES / "saga_with_throw.bpmn")
    assert "t_bnd_charge" not in net.transitions


def test_saga_with_throw_ok_path_token_game():
    """Token game for the normal-success path: charge_card produces both
    p_f_charge_check and p_completed_charge_card; the check gateway
    routes to end_ok; the completion marker is left as a dangling
    token (no further transitions enabled)."""
    net = parse_bpmn(FIXTURES / "saga_with_throw.bpmn")
    marking = dict(net.initial_marking)
    marking = net.fire("t_charge_card", marking)
    assert marking == {"p_f_charge_check": 1, "p_completed_charge_card": 1}

    ok_t = next(
        t for t in net.transitions
        if "p_f_charge_check" in net.preset(t)
        and "p_f_ok" in net.postset(t)
    )
    marking = net.fire(ok_t, marking)
    assert marking == {"p_f_ok": 1, "p_completed_charge_card": 1}
    assert not net.enabled_transitions(marking)


def test_saga_with_throw_compensation_path_token_game():
    net = parse_bpmn(FIXTURES / "saga_with_throw.bpmn")
    marking = dict(net.initial_marking)
    marking = net.fire("t_charge_card", marking)

    compensate_t = next(
        t for t in net.transitions
        if "p_f_charge_check" in net.preset(t)
        and "p_f_to_throw" in net.postset(t)
    )
    marking = net.fire(compensate_t, marking)
    assert marking == {"p_f_to_throw": 1, "p_completed_charge_card": 1}

    assert net.is_enabled("t_trigger_compensate", marking)
    marking = net.fire("t_trigger_compensate", marking)
    assert marking == {"p_compensating_bnd_charge": 1}

    marking = net.fire("t_refund_card", marking)
    assert marking == {"p_f_refund_done": 1}


def test_error_boundary_event_creates_alternative_transition():
    net = parse_bpmn(FIXTURES / "error_boundary.bpmn")
    assert net.validate() == []
    assert net.initial_marking == {"p_f_start": 1}

    succeed_t = "t_risky"
    error_t = "t_bnd_err"
    assert succeed_t in net.transitions
    assert error_t in net.transitions

    assert net.preset(succeed_t) == {"p_f_start"}
    assert net.preset(error_t) == {"p_f_start"}
    assert net.postset(succeed_t) == {"p_f_ok"}
    assert net.postset(error_t) == {"p_f_err"}


def test_error_boundary_event_both_paths_reachable():
    net = parse_bpmn(FIXTURES / "error_boundary.bpmn")
    marking_ok = net.fire("t_risky", net.initial_marking)
    assert marking_ok == {"p_f_ok": 1}

    marking_err = net.fire("t_bnd_err", net.initial_marking)
    assert marking_err == {"p_f_err": 1}
    marking_handled = net.fire("t_error_handler", marking_err)
    assert marking_handled == {"p_f_handled": 1}


def test_timer_boundary_event_uses_same_alternative_pattern():
    xml = """<?xml version="1.0"?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <process id="p">
            <startEvent id="s"/>
            <task id="wait" name="Wait for input"/>
            <task id="timeout_handler" name="Handle timeout"/>
            <boundaryEvent id="bnd_timeout" attachedToRef="wait">
              <timerEventDefinition/>
            </boundaryEvent>
            <endEvent id="e_ok"/>
            <endEvent id="e_timeout"/>
            <sequenceFlow id="a" sourceRef="s" targetRef="wait"/>
            <sequenceFlow id="b" sourceRef="wait" targetRef="e_ok"/>
            <sequenceFlow id="c" sourceRef="bnd_timeout" targetRef="timeout_handler"/>
            <sequenceFlow id="d" sourceRef="timeout_handler" targetRef="e_timeout"/>
          </process>
        </definitions>"""
    net = parse_bpmn(xml)
    assert "t_bnd_timeout" in net.transitions
    assert net.preset("t_bnd_timeout") == {"p_a"}
    assert net.postset("t_bnd_timeout") == {"p_c"}


def test_non_interrupting_boundary_event_forks_task_output():
    """A non-interrupting boundary event (cancelActivity="false") does
    not compete with the task — when the task fires, BOTH the normal
    outflow and the boundary's handler path receive tokens. In Petri
    net terms the task transition gets a second output arc."""
    xml = """<?xml version="1.0"?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <process id="p">
            <startEvent id="s"/>
            <task id="t" name="x"/>
            <task id="notifier" name="Notify"/>
            <boundaryEvent id="b" attachedToRef="t" cancelActivity="false">
              <signalEventDefinition/>
            </boundaryEvent>
            <endEvent id="e_main"/>
            <endEvent id="e_notified"/>
            <sequenceFlow id="a" sourceRef="s" targetRef="t"/>
            <sequenceFlow id="c" sourceRef="t" targetRef="e_main"/>
            <sequenceFlow id="d" sourceRef="b" targetRef="notifier"/>
            <sequenceFlow id="e" sourceRef="notifier" targetRef="e_notified"/>
          </process>
        </definitions>"""
    net = parse_bpmn(xml)
    assert net.postset("t_t") == {"p_c", "p_d"}
    assert "t_b" not in net.transitions

    marking = net.fire("t_t", net.initial_marking)
    assert marking == {"p_c": 1, "p_d": 1}


def test_boundary_event_without_event_definition_rejected():
    xml = """<?xml version="1.0"?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <process id="p">
            <startEvent id="s"/>
            <task id="t" name="x"/>
            <boundaryEvent id="b" attachedToRef="t"/>
            <endEvent id="e"/>
            <sequenceFlow id="a" sourceRef="s" targetRef="t"/>
            <sequenceFlow id="c" sourceRef="t" targetRef="e"/>
          </process>
        </definitions>"""
    with pytest.raises(ValueError, match="no event definition"):
        parse_bpmn(xml)


def test_multiple_boundary_events_on_one_task():
    """A task with both an error AND a timer boundary should produce
    three competing transitions out of the task's input place: succeed,
    error, timeout."""
    xml = """<?xml version="1.0"?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <process id="p">
            <startEvent id="s"/>
            <task id="risky" name="Risky"/>
            <task id="err_h" name="Error handler"/>
            <task id="timeout_h" name="Timeout handler"/>
            <boundaryEvent id="bnd_err" attachedToRef="risky">
              <errorEventDefinition/>
            </boundaryEvent>
            <boundaryEvent id="bnd_to" attachedToRef="risky">
              <timerEventDefinition/>
            </boundaryEvent>
            <endEvent id="e_ok"/>
            <endEvent id="e_err"/>
            <endEvent id="e_to"/>
            <sequenceFlow id="a" sourceRef="s" targetRef="risky"/>
            <sequenceFlow id="b" sourceRef="risky" targetRef="e_ok"/>
            <sequenceFlow id="c" sourceRef="bnd_err" targetRef="err_h"/>
            <sequenceFlow id="d" sourceRef="err_h" targetRef="e_err"/>
            <sequenceFlow id="e" sourceRef="bnd_to" targetRef="timeout_h"/>
            <sequenceFlow id="f" sourceRef="timeout_h" targetRef="e_to"/>
          </process>
        </definitions>"""
    net = parse_bpmn(xml)
    competing = [t for t in net.transitions if "p_a" in net.preset(t)]
    assert set(competing) == {"t_risky", "t_bnd_err", "t_bnd_to"}


# ---------------------------------------------------------------------------
# Intermediate events
# ---------------------------------------------------------------------------


def test_intermediate_event_is_pass_through_transition():
    net = parse_bpmn(FIXTURES / "intermediate_event.bpmn")
    assert net.validate() == []
    assert "t_checkpoint" in net.transitions
    assert net.preset("t_checkpoint") == {"p_f2"}
    assert net.postset("t_checkpoint") == {"p_f3"}


def test_intermediate_event_with_event_definition_rejected():
    xml = """<?xml version="1.0"?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <process id="p">
            <startEvent id="s"/>
            <intermediateCatchEvent id="wait_for_msg">
              <messageEventDefinition/>
            </intermediateCatchEvent>
            <endEvent id="e"/>
            <sequenceFlow id="a" sourceRef="s" targetRef="wait_for_msg"/>
            <sequenceFlow id="b" sourceRef="wait_for_msg" targetRef="e"/>
          </process>
        </definitions>"""
    with pytest.raises(ValueError, match="only plain pass-through"):
        parse_bpmn(xml)


# ---------------------------------------------------------------------------
# Lanes (informational) and subProcess (unsupported)
# ---------------------------------------------------------------------------


def test_lanes_are_silently_ignored():
    """The Petri net produced from xor_with_lanes.bpmn must match the
    plain xor_branch.bpmn exactly — lanes have no control-flow
    semantics."""
    with_lanes = parse_bpmn(FIXTURES / "xor_with_lanes.bpmn")
    without_lanes = parse_bpmn(FIXTURES / "xor_branch.bpmn")
    assert with_lanes.places == without_lanes.places
    assert with_lanes.transitions == without_lanes.transitions
    assert with_lanes.flow == without_lanes.flow
    assert with_lanes.initial_marking == without_lanes.initial_marking


def test_subprocess_rejected_with_pointer_to_roadmap():
    xml = """<?xml version="1.0"?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <process id="p">
            <startEvent id="s"/>
            <subProcess id="sub" name="Inner"/>
            <endEvent id="e"/>
            <sequenceFlow id="a" sourceRef="s" targetRef="sub"/>
            <sequenceFlow id="b" sourceRef="sub" targetRef="e"/>
          </process>
        </definitions>"""
    with pytest.raises(ValueError, match="subProcess.*not supported"):
        parse_bpmn(xml)


def test_saga_throw_requires_compensation_boundary():
    """A compensate end event without any matching compensation
    boundary in the process is rejected."""
    xml = """<?xml version="1.0"?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <process id="p">
            <startEvent id="s"/>
            <task id="t" name="x"/>
            <endEvent id="e_throw">
              <compensateEventDefinition/>
            </endEvent>
            <sequenceFlow id="a" sourceRef="s" targetRef="t"/>
            <sequenceFlow id="b" sourceRef="t" targetRef="e_throw"/>
          </process>
        </definitions>"""
    with pytest.raises(ValueError, match="no compensation boundary"):
        parse_bpmn(xml)


def test_error_boundary_without_outgoing_flow_rejected():
    """An error (or any interrupting) boundary event must have an
    outgoing sequenceFlow to its handler. The old "no non-compensation
    boundaries" rejection was tightened to this in Phase 4."""
    xml = """<?xml version="1.0"?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <process id="p">
            <startEvent id="s"/><task id="t" name="x"/><endEvent id="e"/>
            <boundaryEvent id="b" attachedToRef="t">
              <errorEventDefinition/>
            </boundaryEvent>
            <sequenceFlow id="a" sourceRef="s" targetRef="t"/>
            <sequenceFlow id="c" sourceRef="t" targetRef="e"/>
          </process>
        </definitions>"""
    with pytest.raises(ValueError, match="exactly one outgoing"):
        parse_bpmn(xml)


# ---------------------------------------------------------------------------
# Parser surface
# ---------------------------------------------------------------------------


def test_parse_bpmn_accepts_xml_string():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <process id="p"><startEvent id="s"/><task id="t" name="x"/>
          <endEvent id="e"/>
          <sequenceFlow id="a" sourceRef="s" targetRef="t"/>
          <sequenceFlow id="b" sourceRef="t" targetRef="e"/>
          </process>
        </definitions>"""
    net = parse_bpmn(xml)
    assert net.transitions == {"t_t"}
    assert net.initial_marking == {"p_a": 1}


def test_parse_bpmn_rejects_gateway_that_is_both_split_and_join():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <process id="p">
            <startEvent id="s1"/><startEvent id="s2"/>
            <exclusiveGateway id="g"/>
            <endEvent id="e1"/><endEvent id="e2"/>
            <sequenceFlow id="a" sourceRef="s1" targetRef="g"/>
            <sequenceFlow id="b" sourceRef="s2" targetRef="g"/>
            <sequenceFlow id="c" sourceRef="g" targetRef="e1"/>
            <sequenceFlow id="d" sourceRef="g" targetRef="e2"/>
          </process>
        </definitions>"""
    with pytest.raises(ValueError, match="both a split and a join"):
        parse_bpmn(xml)


def test_parse_bpmn_rejects_missing_process_element():
    xml = """<?xml version="1.0"?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL"/>"""
    with pytest.raises(ValueError, match="no <process> element"):
        parse_bpmn(xml)
