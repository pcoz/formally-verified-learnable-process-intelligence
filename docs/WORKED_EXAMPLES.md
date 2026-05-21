# Worked examples

PETRA ships with **17 end-to-end scenarios** under [`examples/`](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples).
Each is a self-contained TOML configuration plus a paired test
that drives the full pipeline — *load the net, load the traces,
compile, train, extract rules, score anomalies.* They span
deliberately different domains to make the point that the
substrate isn't just for business processes — a process is a
process whether it lives in a loan-approval workflow, a
distributed-consensus protocol, a manufacturing line, or a
biological signalling cascade.

Each scenario links to its own README with the long-form
explanation, the data source, the framework features it
exercises, and the load-bearing claims in its test. The table
below is sorted by the *use case* the scenario illustrates
rather than by the chronology of when it was added — newcomers
can navigate to whichever use case matches their domain.

| Scenario | What it demonstrates | Use case it represents |
|---|---|---|
| [**`cost_ranked_refactoring`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/cost_ranked_refactoring) | Two BPMN variants of the same approval process, proved equivalent by bisimulation, trained on a shared trace distribution, ranked by realised cost. **Variant B comes out ~6× cheaper** while doing provably the same thing. | *Provably-safe process refactoring* — pick redesigns with formal guarantees instead of guesswork. |
| [**`safe_refactoring`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/safe_refactoring) | Two structurally-different variants of the loan-approval process (Variant B adds a silent audit-log step). **Strong bisim rejects them, weak bisim accepts them** — the τ collapse case. Cross-variant comparison shows hard-agreement across the credit-score domain; bootstrap CIs on each variant's distilled rule overlap. | *Refactor-with-confidence on changes that actually alter the shape* — the load-bearing case for "structurally different, behaviourally equivalent." |
| [**`compliance_audit`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/compliance_audit) | Loan-approval net with regulatory invariants stated as CTL formulae (*"every approved loan eventually fires the audit-log step"*, *"decline cannot fire before credit-check"*). The deliberately-broken variant that bypasses the audit step fails the CTL invariant; `check_ctl` returns a counterexample marking. Also pins Aalst soundness and deadlock localisation. | *Regulatory compliance verification at code-review speed* — turn the regulator's rules into mechanical pre-deployment checks instead of post-hoc audits. |
| [**`credit_approval_coloured`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/credit_approval_coloured) | Coloured tokens carry the loan amount; the compiled network **learns the approve/decline threshold from data** rather than taking the modeller's declared 1,000 as given. Learned thresholds land in the empirical band 900–1,500. | *Data-driven decision rules* — when the right threshold is in the data, not in someone's head. |
| [**`discover_and_train_pipeline`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/discover_and_train_pipeline) | Native log-to-net discovery via the basic Inductive Miner — `net.source = "discover"` mines a sound Petri net from a four-trace loan-approval log (sequence + parallel + exclusive-choice cuts), then trains on the same log end to end. | *Log-only entry path* — when you have history but no model. |
| [**`incident_management`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/incident_management) | Trains on the **real BPI Challenge 2013 incidents log** — 7,554 Volvo IT tickets, 65k events, shipped in the repo as a 1.3 MB gzipped XES file. Flags traces that skip the *Resolved* step before *Closing*. | *Real-world, large-scale, public business-process data* — proof that the framework scales beyond synthetic fixtures. |
| [**`distributed_consensus`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/distributed_consensus) | **Two-phase commit (2PC)** modelled as three composed pools (coordinator + two cohorts) with shared message places. Detects *Byzantine commit-after-low-vote* anomalies. | *Distributed-protocol verification* — flagging deviations against the spec from production traces. |
| [**`network_protocol`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/network_protocol) | **TCP three-way handshake** compiled from the RFC's state machine. After training on legitimate traces, flags **SYN-flood** and **half-open-connection** attacks as anomalies pinned to specific transitions. | *Security monitoring on protocol state machines* — attack-pattern detection grounded in the protocol's structural spec. |
| [**`multi_agent_coordination`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/multi_agent_coordination) | **Three-pool contract-net** protocol with bid-driven contractor selection. The AND-join rule extractor recovers the synchronisation rule over three input contributors; *pre-bid award* attempts are flagged as protocol violations. | *Coordination protocols between autonomous agents* — catching out-of-order coordination events. |
| [**`manufacturing_cell`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/manufacturing_cell) | Multi-station production line with **quality-gated ship-or-rework routing**. PETRA distils the quality-driven ship rule from production data; mis-shipped low-quality items are flagged as anomalies. | *Manufacturing and supply-chain analysis* — quality-conditional routing rules recovered from production data. |
| [**`paint_shop`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/paint_shop) | A cure step with declared **duration 3** — parts spend three time-steps in the cure transition before reaching inspection. Exercises the time-unrolled compiler's per-transition in-flight queue. | *Workflows with explicit step durations* — cure times, wait times, batched processing windows. |
| [**`batch_packaging`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/batch_packaging) | A bottle-to-crate transition with **input arc weight 6** — six bottles accumulate before the crate transition fires. Exercises multi-token arc multiplicities. | *Batching and aggregation patterns* — packaging lines, micro-batch processing, N-into-1 combination steps. |
| [**`priority_dispatch`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/priority_dispatch) | Three handlers with **declared firing-rate priors (3.0, 1.0, 0.5)** — high-rate fires more eagerly for the same input. Training refines the priors against the observed dispatch distribution. | *Priority-aware task dispatch* — modeller priors carried through to training, then refined from data. |
| [**`resource_lock`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/resource_lock) | Two clients competing for a shared resource, with **inhibitor arcs enforcing the mutex** — lock-acquire fires only when lock-held is empty. Exercises the soft inhibitor gate `(1 − a(p))`. | *Mutex, semaphore, and other negative-precondition patterns* — exclusive access modelled cleanly into the dynamics. |
| [**`scientific_workflow`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/scientific_workflow) | **PCR (polymerase chain reaction)** modelled with a quality-gate transition that routes pass/fail. PETRA learns the gate from trace data and flags traces that skip it. | *Laboratory and clinical protocol conformance* — deviation analysis on scientific procedures. |
| [**`biological_signalling`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/biological_signalling) | A **kinase cascade** with signal-strength-conditioned fast/slow pathway routing; the XOR rule is distilled in the pathway components' vocabulary, not internal framework labels. | *Cell-biology pathway analysis* — Reactome-style pathways are essentially Petri nets; the same primitives that handle business processes model signalling networks too. |
| [**`mapk_pathway`**](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/mapk_pathway) | Loads the canonical **EGF → MAPK1/3 (ERK1/2) signalling cascade** from a Pathway Commons-style SIF file (real HGNC symbols, standard PC interaction types), compiles, and propagates activation through the full receptor → adapter → small GTPase → MAP3K → MAP2K → MAPK → transcription-factor chain. | *Real biology format on real entities* — Phase 10 ecosystem citizenship: any of the ~3,000 Reactome pathways is one Pathway Commons download away. |

## Running the scenarios

Run any individual scenario with
`python -m pytest tests/scenarios/test_<scenario_name>.py`, or
the whole set with `python -m pytest tests/scenarios/`. The
adapter loads the scenario's TOML, builds the net, parses the
traces, compiles the module, and runs the test's load-bearing
assertions — typically *the model trained, the routing rule
came out where it should, and the anomalies were flagged.*

## Authoring a new scenario

See the *Adding a new scenario* section in
[`DEV_MANUAL.md`](DEV_MANUAL.md) for the step-by-step recipe.
The short version: one TOML config plus one paired test, both
following the patterns above.
