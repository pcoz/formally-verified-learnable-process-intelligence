# Resource lock — mutex via inhibitor arcs

Two clients compete for a single shared resource (database connection,
print head, robot arm, …). Showcases the **Phase 9 inhibitor-arc**
feature: a structural guard that says *"do not fire while this place
holds a token"*.

## What this scenario shows that earlier phases could not

Without inhibitor arcs, declaring "two transitions are mutually
exclusive on a shared resource" required ugly workarounds:

- A chain of explicit pass-the-baton places (one transition produces
  a token that the other consumes), which only works for two
  transitions and breaks for N>2.
- An explicit "available" place that both transitions consume and
  then re-produce, which still doesn't prevent simultaneous firing
  in the continuous relaxation.

With inhibitor arcs the modeller writes exactly the constraint:
*"`serve_a` is inhibited by `resource_busy`. `serve_b` is inhibited
by `resource_busy`."* The framework enforces it both in the discrete
token-game (via `is_enabled`) and in the trained network (via the
multiplicative `(1 - a(p))` gate in the forward pass).

## Why the scenario uses time-unrolled mode

Inhibitor arcs only do useful work *across time*. In a single
acyclic forward pass, the inhibitor place's activation hasn't been
populated yet at the moment the guarded transition is evaluated, so
the gate is inactive. The mutex constraint becomes meaningful when
the forward pass runs across multiple time steps: step 1 fires the
winning transition and lights up `p_resource_busy`; step 2 finds the
place occupied and the gate suppresses any further firing of either
transition.

That is precisely the discrete Petri-net semantics — only one
transition can claim the resource — surfaced into the continuous
relaxation.

## Advantage over alternatives

- **vs. ad-hoc mutex coding in ML pipelines**: declarative; the
  constraint is captured in the topology, not buried in an imperative
  guard.
- **vs. classical Petri-net simulators**: those handle the discrete
  semantics but cannot *learn* the routing decision (which client
  should win on which kind of request) from observed traces.
- **vs. constraint-programming approaches** (e.g. CP-SAT): solvers
  enforce constraints but don't train a continuous model whose
  decisions can be inspected, distilled into rules, or compared
  across variants.

## Real-world source

Mutex / critical-section / single-resource modelling has used Petri
nets with inhibitor arcs since the 1970s. Canonical references:

- Peterson, *Petri Net Theory and the Modeling of Systems* (1981) —
  the standard treatment.
- Reisig, *Petri Nets and Algebraic Specifications* (1991) — the
  mathematics of inhibitor extensions.

Operating-system mutexes, database row locks, network-protocol
channel arbitration, and physical-resource scheduling (CNC machines,
robot arms, printer heads) all share this structural pattern.

## Files

- `scenario.toml` — net + traces + multi-step training config,
  including the two `[[net.inhibitor_arcs]]` entries
- `../../tests/scenarios/test_resource_lock.py` — adapter-driven
  test exercising both the discrete mutex semantics (token-game)
  and the compiled module's inhibitor gate (in time-unrolled mode)

## Running

```
python -m pytest tests/scenarios/test_resource_lock.py
```
