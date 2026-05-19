"""Tests for cross-pool BPMN composition (Phase 5).

Each pool of a `<collaboration>` is parsed into the same Petri net with
its participant ID prefixed to every place / transition. `<messageFlow>`
elements add a shared message place between two pools, threading token
flow across the pool boundary.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import PetriNet, PetriNetModule, parse_bpmn


FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Structural shape
# ---------------------------------------------------------------------------


def test_collaboration_namespaces_each_pool():
    net = parse_bpmn(FIXTURES / "order_collaboration.bpmn")
    assert net.validate() == []

    customer_places = {p for p in net.places if p.startswith("customer:")}
    vendor_places = {p for p in net.places if p.startswith("vendor:")}
    msg_places = {p for p in net.places if p.startswith("msg_")}

    assert len(customer_places) == 3
    assert len(vendor_places) == 4
    assert msg_places == {"msg_order", "msg_confirmation"}

    customer_transitions = {
        t for t in net.transitions if t.startswith("customer:")
    }
    vendor_transitions = {t for t in net.transitions if t.startswith("vendor:")}
    assert "customer:t_send_order" in customer_transitions
    assert "customer:t_receive_confirmation" in customer_transitions
    assert "vendor:t_receive_order" in vendor_transitions
    assert "vendor:t_process_order" in vendor_transitions
    assert "vendor:t_send_confirmation" in vendor_transitions


def test_collaboration_initial_marking_lights_both_pools():
    net = parse_bpmn(FIXTURES / "order_collaboration.bpmn")
    assert net.initial_marking == {
        "customer:p_c_f1": 1,
        "vendor:p_v_f1": 1,
    }


def test_message_flow_wires_sender_to_receiver_via_shared_place():
    net = parse_bpmn(FIXTURES / "order_collaboration.bpmn")
    assert "msg_order" in net.postset("customer:t_send_order")
    assert "msg_order" in net.preset("vendor:t_receive_order")
    assert "msg_confirmation" in net.postset("vendor:t_send_confirmation")
    assert "msg_confirmation" in net.preset("customer:t_receive_confirmation")


# ---------------------------------------------------------------------------
# Dynamic semantics — token game across the pool boundary
# ---------------------------------------------------------------------------


def test_receiver_blocks_until_sender_fires():
    """vendor:t_receive_order requires both vendor:p_v_f1 (its incoming
    sequenceFlow place) AND msg_order (filled only after the customer
    sends). At M_0 only the vendor's start place is present so the
    receiver is NOT yet enabled."""
    net = parse_bpmn(FIXTURES / "order_collaboration.bpmn")
    assert not net.is_enabled("vendor:t_receive_order", net.initial_marking)
    assert net.is_enabled("customer:t_send_order", net.initial_marking)


def test_full_collaboration_token_game_completes_both_pools():
    net = parse_bpmn(FIXTURES / "order_collaboration.bpmn")
    marking = dict(net.initial_marking)

    marking = net.fire("customer:t_send_order", marking)
    assert "msg_order" in marking
    assert "customer:p_c_f2" in marking

    marking = net.fire("vendor:t_receive_order", marking)
    assert "msg_order" not in marking
    assert "vendor:p_v_f2" in marking

    marking = net.fire("vendor:t_process_order", marking)
    marking = net.fire("vendor:t_send_confirmation", marking)
    assert "msg_confirmation" in marking
    assert "vendor:p_v_f4" in marking

    marking = net.fire("customer:t_receive_confirmation", marking)
    assert "customer:p_c_f3" in marking
    assert "msg_confirmation" not in marking

    assert marking == {"customer:p_c_f3": 1, "vendor:p_v_f4": 1}


# ---------------------------------------------------------------------------
# Compilation — composed nets feed PetriNetModule cleanly
# ---------------------------------------------------------------------------


def test_collaboration_compiles_to_acyclic_module():
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "order_collaboration.bpmn")
    module = PetriNetModule(net)
    out = module()
    assert "customer:p_c_f3" in out
    assert "vendor:p_v_f4" in out
    assert torch.all(torch.isfinite(out["customer:p_c_f3"]))


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_participant_with_unknown_processref_raises():
    xml = """<?xml version='1.0'?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <collaboration>
            <participant id="ghost" processRef="does_not_exist"/>
          </collaboration>
        </definitions>"""
    with pytest.raises(ValueError, match="unknown process"):
        parse_bpmn(xml)


def test_message_flow_with_unknown_endpoint_raises():
    xml = """<?xml version='1.0'?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <collaboration>
            <participant id="a" processRef="p_a"/>
            <participant id="b" processRef="p_b"/>
            <messageFlow id="mf" sourceRef="ghost_node" targetRef="task_b"/>
          </collaboration>
          <process id="p_a">
            <startEvent id="sa"/><task id="task_a" name="A"/><endEvent id="ea"/>
            <sequenceFlow id="fa1" sourceRef="sa" targetRef="task_a"/>
            <sequenceFlow id="fa2" sourceRef="task_a" targetRef="ea"/>
          </process>
          <process id="p_b">
            <startEvent id="sb"/><task id="task_b" name="B"/><endEvent id="eb"/>
            <sequenceFlow id="fb1" sourceRef="sb" targetRef="task_b"/>
            <sequenceFlow id="fb2" sourceRef="task_b" targetRef="eb"/>
          </process>
        </definitions>"""
    with pytest.raises(ValueError, match="not a known node"):
        parse_bpmn(xml)


def test_message_flow_within_one_pool_raises():
    xml = """<?xml version='1.0'?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
          <collaboration>
            <participant id="a" processRef="p_a"/>
            <participant id="b" processRef="p_b"/>
            <messageFlow id="mf" sourceRef="task_a1" targetRef="task_a2"/>
          </collaboration>
          <process id="p_a">
            <startEvent id="sa"/>
            <task id="task_a1" name="A1"/>
            <task id="task_a2" name="A2"/>
            <endEvent id="ea"/>
            <sequenceFlow id="fa1" sourceRef="sa" targetRef="task_a1"/>
            <sequenceFlow id="fa2" sourceRef="task_a1" targetRef="task_a2"/>
            <sequenceFlow id="fa3" sourceRef="task_a2" targetRef="ea"/>
          </process>
          <process id="p_b">
            <startEvent id="sb"/><task id="task_b" name="B"/><endEvent id="eb"/>
            <sequenceFlow id="fb1" sourceRef="sb" targetRef="task_b"/>
            <sequenceFlow id="fb2" sourceRef="task_b" targetRef="eb"/>
          </process>
        </definitions>"""
    with pytest.raises(ValueError, match="same pool"):
        parse_bpmn(xml)
