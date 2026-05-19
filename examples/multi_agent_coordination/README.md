# Multi-agent coordination — contract-net protocol

Three agents — one manager, two contractors — coordinate to assign
and execute a task via the contract-net protocol. PETRA running on a
multi-agent coordination problem: three pools, six shared message
places, an AND-join over both bids at the manager, and a bid-driven
XOR routing to the winning contractor.

## What this scenario shows

- Three-pool cross-pool composition (one manager, two contractors)
  with six shared message places (announce-A, announce-B, bid-A,
  bid-B, award-A, award-B) — extending the two-pool 2PC pattern to
  three agents with broadcast and selection.
- The manager's evaluate-bids step is an AND-join over both bid
  messages plus the manager's awaiting state — three input places,
  one transition, demonstrating the AND-join interpretability rules
  on a real coordination primitive.
- The award decision is a binary XOR over the discriminative `a_advantage`
  attribute. The trained network recovers the routing rule and the
  interpretability layer reports it as "if a_advantage > X → award A,
  else → award B".
- §7.2 anomaly detection flags coordination violations: an award
  issued before both bids arrived; an award to a contractor that
  didn't bid; a performance event without a corresponding award.

## Advantage over alternatives

- **vs. FIPA-compliant agent platforms** (JADE, Jason): existing
  agent platforms execute the protocol but cannot detect when an
  agent deviates from it. The structural prior + anomaly detection
  here flags any departure from the validated coordination
  pattern — useful for monitoring open multi-agent systems where
  agent compliance cannot be assumed.
- **vs. reinforcement learning for multi-agent coordination**: RL
  approaches learn policies but provide no equivalence guarantees
  across implementations. Phase 2 bisimulation lets two RL agents
  trained for the same protocol be compared formally before
  deployment.
- **vs. game-theoretic mechanism design**: mechanism design proves
  properties of the protocol under rational play; this framework
  *observes* the deployed protocol's actual execution and flags
  deviations from the verified pattern — complementary.
- **Cost-ranked agent allocation**: by attaching per-transition
  cost weights (using the Phase 6 `expected_cost` helper) you get
  the expected coordination cost of running each contractor under
  the trained award-distribution — directly useful for capacity
  planning in distributed task systems.

## Real-world source

- Smith, "The Contract Net Protocol: High-Level Communication and
  Control in a Distributed Problem Solver" (IEEE Transactions on
  Computers, 1980) — the foundational paper.
- FIPA Contract Net Interaction Protocol Specification (2002) —
  the canonical agent-protocol standard built on Smith's pattern,
  still in use in robotics and distributed task-allocation systems.
- Multi-agent reinforcement learning task-assignment work routinely
  uses contract-net as the coordination skeleton beneath learned
  bidding policies.

## Files

- `scenario.toml` — full specification
- `../../tests/scenarios/test_multi_agent_coordination.py` —
  end-to-end test covering token-game completion, bid-driven routing,
  AND-join rule extraction, and coordination-violation anomalies

## Running

```
python -m pytest tests/scenarios/test_multi_agent_coordination.py
```
