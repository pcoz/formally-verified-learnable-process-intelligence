# PETRA Product Roadmap

**PETRA** is Petri-Net Trained Architecture — a substrate for
learning, verifying, explaining and cost-ranking the dynamics of
discrete-event systems.

---

## Framing — scope beyond business processes

The architecture spec presents this work in the language of BPMN
process reasoning, and most of the phase deliverables below use BPMN
fixtures. That undersells what is actually being built. The structural
constraint isn't a limitation — it is what makes dynamic modelling of
complex systems tractable. Most ML approaches to complex systems fail
because the hypothesis space is too large; this approach inverts that
by making the topology fixed and the dynamics learned within it. That
is a stronger position than the BPMN framing implies.

Five points worth holding in mind while reading the phases below:

1. **Compositionality scales.** Phase 5 showed two pools composing
   cleanly via shared message places. Nothing in the architecture
   breaks at N pools — you can model an entire enterprise's
   interacting processes, or a multi-tier supply chain, or a
   federated multi-agent system, with the same primitives. Each pool
   keeps its local learned weights; the message-place dynamics are
   emergent across the composition. That is a genuinely different
   shape from monolithic models.

2. **The cyclic compiler is bigger than BPMN.** Once you have
   time-unrolled dynamics over a Petri net (Phase 3), you are not
   just modelling BPMN — you are modelling any discrete-event
   dynamical system that can be expressed as a place/transition
   graph. That includes:
   - distributed consensus protocols (Paxos, Raft phases as
     transitions);
   - manufacturing cell dynamics (machines, buffers, parts);
   - biological signalling pathways (pathway databases like Reactome
     are essentially Petri nets);
   - multi-agent coordination protocols;
   - network protocol state machines under load.

3. **Bisimulation is not a nice-to-have — it is a precondition for
   BPMN ML.** Two business processes can be drawn differently,
   labelled differently, decomposed across different gateways, and
   still implement the same workflow. Without a way to verify that
   two trained models are operating over equivalent process
   structures, comparing their learned weights — or transferring
   learnings between them, or aggregating across an organisation's
   process variants — is meaningless: you do not know whether you
   are comparing like with like. Most ML systems cannot answer
   "is this learned model equivalent to that one"; Phase 2 can —
   structurally, before training, with a proof. That makes BPMN ML
   actually possible rather than merely plausible, and it is the
   formal-verification-meets-ML story almost nobody can claim with
   running code.

4. **Anomaly detection generalises beyond business processes.** §7.2
   works for any trace over a known structure: attack-pattern
   detection in protocol traces, fault localisation in distributed
   systems, deviation analysis in scientific workflows.

5. **The discrete-continuous bridge is the real research
   contribution.** STE + sharpness annealing (Phase 6) on a
   Petri-net-constrained network is a tractable approach to a problem
   the spec calls open — and this architecture is one of very few
   substrates where the question is even well-posed, because the
   topology is fixed.

6. **Bisimulation plus learned weights enables cost-ranked variant
   search — equivalently, *provably-safe business process
   refactoring*.** The traditional analogue is code refactoring.
   You can confidently restructure code because the compiler catches
   type errors, tests catch behaviour changes, and sometimes formal
   methods prove invariants. For business processes today you have
   none of those tools — a proposed redesign is either approved on
   intuition (risky), rolled out via expensive shadow-running (slow),
   or never attempted (status quo wins by default).

   What cost-ranking via bisimulation + learned weights gives you,
   in three steps:

   - Refactor a process and *mechanically verify you haven't changed
     what it does* (Phase 2 bisimulation). Pre-bisimulation, you
     cannot meaningfully claim two BPMN variants are "the same
     process, drawn differently."
   - Rank the verified-equivalent variants by realised-execution
     cost — attach per-transition cost weights (wall-clock time,
     monetary cost, resource usage, energy), train each variant on
     the same XES trace distribution, and read off
     expected-cost-to-completion from the resulting firing patterns.
     The cost model is *fitted to actual data*, not stipulated by
     hand.
   - Choose the cheaper variant with formal guarantees that the
     only difference is cost, not behaviour. "Variant B is provably
     equivalent to variant A in behaviour, and 23% cheaper on the
     observed workload."

   Concretely, that sequence enables things that are currently very
   hard:

   - **Cross-organisation process benchmarking.** Two companies run
     different BPMN variants achieving the same outcome. You can
     prove they are equivalent and then compare their cost
     efficiency on each company's workload. Currently impossible
     because you cannot verify behavioural equivalence across
     organisations.
   - **A/B testing of business processes with real guarantees.**
     Run two equivalent variants in production, compare measured
     costs, attribute differences to design rather than behaviour
     drift.
   - **Process refactoring at code-refactoring confidence.**
     Semantic-preserving redesigns ranked quantitatively without
     rollout risk. That converts process redesign from a
     high-stakes decision into a low-stakes iteration loop — the
     same shift that made software engineering's
     continuous-refactoring culture possible.

   The comparison is impossible in monolithic ML approaches because
   they cannot prove equivalence in the first place — and it is
   impossible in classical workflow analysis because the cost model
   is not fitted to realised execution data.

   The scale of this shift is worth naming directly. Software's
   productivity transformation since the 1990s — continuous
   integration, microservices, agile methods, DevOps, the whole
   iterate-fast-and-learn culture — was downstream of refactoring
   becoming *safe*. Once code could change without fear, the
   organisational cost of trying a new design collapsed, and design
   iteration cycles compressed from years to days. Business
   processes are stuck where software was before that shift:
   change-aversion is rational because changes are risky, and
   that risk-aversion shows up everywhere — in multi-year ERP
   implementations, in the billion-dollar process-reengineering
   consulting industry, in "we've always done it this way" being a
   coherent answer. Processes *are* the operating model of every
   organisation. If they become as malleable as code under the same
   safety guarantees, the impact is at the scale of the operating
   model of the economy. That is what this substrate, taken
   seriously, points at.

### Scenarios validated

Each framing claim above has an end-to-end demonstration in
`examples/<scenario>/` with a paired test in
`tests/scenarios/test_<scenario>.py`. Driven entirely from TOML
configs via the `load_scenario` adapter (no per-scenario Python
boilerplate).

| Scenario | Framing claim(s) validated | Key result |
|---|---|---|
| `biological_signalling/` | Point 2 (non-BPMN substrate), Point 3 (bisimulation across variants) | Strength-dependent fast/slow pathway routing learned; rule distilled in pathway vocabulary. |
| `distributed_consensus/` | Point 2 (consensus protocols), Point 4 (fault localisation) | 2PC token-game validated; Byzantine commit-after-low-vote anomaly flagged. |
| `manufacturing_cell/` | Point 2 (manufacturing dynamics) | Quality-driven ship/rework rule recovered; mis-ship anomaly detected. |
| `network_protocol/` | Point 2 (protocol state machines), Point 4 (attack patterns) | TCP handshake compiles; SYN-flood and half-open attack patterns detected. |
| `scientific_workflow/` | Point 4 (deviation analysis in scientific workflows) | PCR quality gate learned; skipped-step deviation flagged. |
| `multi_agent_coordination/` | Point 2 (multi-agent coordination protocols), Point 4 (coordination-violation detection) | Contract-net with 3 pools / 6 message places; bid-driven award routing learned; AND-join rule distilled over 3 inputs; pre-bid award flagged as anomalous. |
| `cost_ranked_refactoring/` | Point 3 (bisimulation), Point 6 (cost-ranked refactoring) | Two cost variants verified equivalent; variant B confirmed ~6× cheaper across input distribution. |
| `batch_packaging/` | Phase 9 multi-token markings | Arc weight 6 batches bottles into crates; six tokens accumulate before the crate transition fires. |
| `resource_lock/` | Phase 9 inhibitor arcs | Two clients race for a shared resource; mutex enforced by the soft `(1 − a(p))` inhibitor gate. |
| `paint_shop/` | Phase 9 transition durations | A duration-3 cure step delays output by two extra time-unrolled steps. |
| `priority_dispatch/` | Phase 9 firing rates | Three handlers at rates 3.0 / 1.0 / 0.5; rate-3 handler fires roughly three times as eagerly for the same input. |
| `credit_approval_coloured/` | Phase 9 coloured tokens + Phase 9 follow-up CPN-aware compiler | Token-game routing on application amount; trained guard threshold lands in the empirical decision band 900–1500 from the hand-set 1000. |
| `incident_management/` | Phase 10 — real BPI Challenge 2013 incidents log | 7,554 Volvo IT tickets, 65k events; PETRA flags traces that skip the Resolved step before Closing. |
| `mapk_pathway/` | Phase 10 — Pathway Commons SIF import | Canonical EGF → MAPK1/3 cascade loaded as SIF; forward pass propagates activation through the full receptor → adapter → small GTPase → MAP3K → MAP2K → MAPK → transcription-factor chain. |
| `discover_and_train_pipeline/` | Phase 12 — basic Inductive Miner + the new `net.source = "discover"` adapter keyword | Synthetic four-trace loan-approval log with sequence / parallel / exclusive-choice cuts is mined to a sound Petri net via `load_scenario`; replay invariant + training-loss convergence both pinned by the test. |
| `safe_refactoring/` | Phase 11 weak bisimulation + Phase 13 cross-variant comparison + Phase 13 bootstrap CIs | Two structurally-different variants of a loan-approval process (Variant A; Variant B with a silent audit-log step) are proved weakly bisimilar despite differing in shape; cross-variant comparison shows hard-agreement across the credit-score domain; bootstrap CIs on each variant's distilled rule overlap. |
| `compliance_audit/` | Phase 11 Aalst soundness + deadlock localisation + CTL model checking | Loan-approval process with two regulatory CTL invariants (`AG (approve → AF audit_logged)`, `AG (decline_enabled → credit_checked)`); the deliberately-broken variant that bypasses the audit step fails the CTL invariant with a counterexample marking witnessing the violation. |
| `regulator_ready_credit_approval/` | Phase 13 full diagnostic toolkit — bootstrap CIs + counterfactuals + sensitivity + prose | Coloured-token loan-approval net instrumented with all four Phase 13 outputs; bootstrap CI brackets the empirical decision band, counterfactual flips a declined application at the right amount, sensitivity confirms the model leans on the amount input, prose helpers substitute domain labels for raw place ids. |
| `realtime_monitoring/` | Phase 14 streaming evaluator | Simplified incident-management net trained on the happy-path lifecycle; an interleaved live stream of three cases (two normal, one anomalous that skips *Resolved*) goes through `StreamingEvaluator`; anomalous case scores strictly higher and the residual pins to `t_resolved` — the actionable alert signal. |
| `onnx_deployment/` | Phase 14 ONNX export | Loan-approval XOR-routing net trained from inline traces, exported to `.onnx`, parity-checked against the torch forward pass under `onnxruntime` — 1e-4 numerical agreement, dynamic batch axis, routing decisions identical across runtimes. |
| `rest_serving/` | Phase 14 REST inference API + CLI | Loan-approval XOR-routing net trained, wrapped in a FastAPI app via `build_app`, every endpoint exercised through `TestClient`: liveness, schema, forward, anomaly, counterfactual, sensitivity, OpenAPI auto-doc. The engine-integration walkthrough that closes the deployment story without needing a real workflow engine on the box. |

**Framework shortcoming discovered and fixed during scenario
validation:** `find_xor_groups` originally only detected single-input
XOR shapes (BPMN-style). Real protocols like 2PC have *shared-preset*
XOR groups — multiple transitions competing for the same multi-place
precondition. The detector was generalised to recognise both patterns,
and the rule extractors gained an automatic "pick the discriminative
input" step that examines learned weight gaps across the preset.

### PETRA as the analytical core of a BPMN optimisation engine

Wiring the delivered phases together gives the analytical substrate
of what a "native BPMN optimisation engine" would do — every step of
the pipeline is in place with running tests:

| Engine capability | PETRA delivers it via |
|---|---|
| Read a BPMN process | `parse_bpmn` (Phase 1) — tasks, gateways, boundaries, compensation, cross-pool collaborations |
| Read external Petri-net or pathway formats | `parse_pnml` / `parse_sif` (Phase 10) — CPN Tools, GreatSPN, TINA, ProM, Pathway Commons |
| Learn how the process actually runs from logs | `train_on_traces` over XES / CSV / JSON (Phase 1, §10 Step 3) |
| Verify two variants are behaviourally equivalent | `are_bisimilar` (Phase 2, strong), `are_weakly_bisimilar` (Phase 11, ignores internal silent steps) |
| Verify the model is structurally sound | `check_soundness` and `find_deadlocks` (Phase 11) — option to complete, proper completion, no dead transitions, deadlock localisation |
| Distill the trained model's decisions into readable rules | `extract_routing_rules` / `extract_and_join_rules` (Phase 8) |
| Detect deviations in production | `anomaly_score` with residuals pinned to BPMN element names (Phase 7) |
| Rank refactorings by realised cost | `expected_cost` (Phase 6 + cost-ranked-refactoring scenario) |
| Capacity, batching, mutex, durations, retry loops, saga compensation | Phase 3 (time-unrolling), Phase 4 (compensation), Phase 9 (multi-token, inhibitor, durations, rates) |
| Routing decisions on per-token values | Phase 9 coloured tokens + the CPN-aware compiler (structural guards, torch guards, torch output transforms) |

The `cost_ranked_refactoring` scenario already runs the full
optimisation pipeline end to end: two BPMN variants → bisimulation
proves them equivalent → train both on the same XES trace
distribution → rank by realised cost. Variant B comes out roughly
six times cheaper while doing provably the same thing.

What's missing on top of PETRA before this is a productionised
engine — and *only* the wrapper layers are missing, not the
substrate:

- **A candidate-generator** that proposes refactorings (today they
  are hand-authored). Could be a rule-based transformation engine,
  an LLM, or heuristic search. PETRA verifies and ranks whatever it
  produces.
- **Connectors to live BPMN runtimes** (Camunda, Activiti, Flowable).
  Phase 14 on this roadmap — pulling production traces, pushing
  trained models / anomaly signals back.
- **A UI** — explicitly out of scope here; the consuming application
  puts a UI on the analytical core PETRA provides.
- **Automated structure discovery from logs** (Phase 12) — for users
  who have logs but not a BPMN model to start with.

A few hundred lines of integration code on top of the current PETRA
would turn it into a deployable engine. Every claim in the table
above is tested today, not aspirational.

The honest framing: this isn't a "business process tool with
interpretability." It is a general framework for learning the
dynamics of any system whose structure can be expressed as a sound
workflow net, with formal equivalence checking and
structurally-grounded anomaly detection as side effects. The phases
below build BPMN fixtures because that is where the spec landed, but
the substrate covers a substantially larger surface.

---

## Phase 1 — Scaffold *(done)*

**Goal:** stand up the full extract → compile → train pipeline against
synthetic data for the tractable BPMN subset.

Delivered:

- [x] §3 BPMN → Petri net translation (`parse_bpmn`): tasks, gateways,
  start/end events, compensation boundary events (acyclic shape).
- [x] §4 continuous relaxation with the §4.3 structural constraint by
  construction (`PetriNetModule`: one weight per arc, one threshold
  per transition, no parameters outside F).
- [x] §5 five elemental subnets — both hand-built (`subnets.py`) and
  emergent under the general compiler.
- [x] §6 composition: the approval process compiles and runs forward.
- [x] §7.1 process execution prediction via training on traces.
- [x] §7.2 anomaly detection with interpretable per-transition
  residuals.
- [x] §10 Step 1 BPMN extraction, Step 2 differentiable subnets,
  Step 3 XES log training (sequential + XOR — the spec's "most common
  and tractable cases").

---

## Phase 2 — Formal/empirical bridge: bisimulation *(done)*

**Goal:** close the §7.3 claim — *two subnets bisimulation equivalent
in the Petri net sense learn identical weight distributions on the same
training data.* The bisimulation checker identifies them structurally;
training confirms them behaviourally.

Delivered:

- [x] `bisimulation.py`: reachability graph + partition-refinement
  strong-bisimulation checker for bounded Petri nets.
- [x] Public API: `are_bisimilar(net1, net2)`,
  `bisimulation_equivalence_classes(net)`, `reachability_graph(net)`.
- [x] Tests covering: identical nets, label-isomorphic renamings,
  structurally distinct but behaviourally equivalent nets
  (redundant parallel transitions with same label), and
  non-equivalent nets.
- [x] Integration test: two structurally isomorphic but disjointly
  named XOR nets, trained on the same XES log under the same seed,
  converge to forward-pass outputs that agree to within 0.05 across
  the input domain.

---

## Phase 3 — Time-unrolled compiler: cycles *(done)*

**Goal:** model genuinely cyclic processes — real compensation
semantics (compensate a task *after* it completed, in response to a
later thrown compensation event), retry loops, repetition.

Delivered:

- [x] Time-step unroll of `PetriNetModule` via a `num_steps` kwarg.
  Default 0 keeps acyclic single-pass behaviour; any positive value
  switches to synchronous time-step updates that accept cyclic nets.
  Source places clamp to their input value at every step (persistent
  input layer); non-source places evolve through §4.2.
- [x] Compensation throw/end events in the BPMN parser, with the real
  "compensate after completion" pattern: an extra completion-marker
  place produced by the task transition, and a throw transition that
  AND-joins the throw event's incoming flow with the completion
  marker before producing the compensating place.
- [x] Retry-loop fixture (`retry_loop.bpmn`) and saga-with-throw
  fixture (`saga_with_throw.bpmn`), each with structural and
  token-game tests, and a compile-and-forward test for the retry loop
  in time-unrolled mode.

Known limitations carried forward:

- The time-unrolled forward is synchronous (all transitions update from
  the previous step's place activations, then all places update from
  the new transition activations). Token conservation is approximate —
  the continuous relaxation lets activation mass grow or shrink along
  cycles. For sound workflow nets without explicit decay this is
  usually fine; for free-running loops the user picks `num_steps` to
  bound how far the dynamics run.
- Compensation throw is restricted to one throw event and one paired
  compensation handler per process. Multi-handler / `activityRef`-targeted
  throws are not supported.

---

## Phase 4 — BPMN coverage expansion *(done)*

**Goal:** extend the parser to the rest of common BPMN.

Delivered:

- [x] Non-compensation interrupting boundary events: error, timer,
  signal, escalation, message. All share one translation pattern —
  the boundary becomes an alternative transition out of the attached
  task's input place, producing tokens at the place corresponding to
  the boundary's outgoing sequenceFlow. Distinguished from the
  compensation pattern (which uses associations).
- [x] Intermediate events (`intermediateThrowEvent`,
  `intermediateCatchEvent`) as plain pass-through transitions. Event
  definitions on intermediate events are rejected with a clear
  message — they need per-type handling (timers/messages/signals) and
  are deferred.
- [x] Lanes (`<laneSet>` / `<lane>` / `<flowNodeRef>`) are silently
  ignored — they have no control-flow semantics. Verified by parsing
  a lane-wrapped BPMN and asserting the resulting Petri net is
  identical to the lane-less version.
- [x] Multiple boundary events on one task: verified (no code change
  was needed — the existing loop handles each boundary independently,
  producing N+1 competing transitions for N boundaries plus the
  task's own success transition).
- [x] `<subProcess>` raises a clear "not supported in this scaffold"
  error pointing at this roadmap. Inline flattening is deferred.

Tightened later:

- [x] Non-interrupting boundary events (`cancelActivity="false"`). The
  task transition gains an extra output arc to the boundary's
  handler-path place, so when the task fires both the normal outflow
  AND the handler path receive tokens — the parallel-fork semantics
  the BPMN spec calls for. No T_fail alternative is created.

Carried forward:

- Intermediate event definitions (timer/message/signal catches as
  intermediate, not boundary, events) need separate translation. They
  would be a natural pair with cross-pool message flows in Phase 5.
- Inline `<subProcess>` flattening — recursive translation with scope
  tracking. Out of scope here; see Phase 4.5 below if reopened.

---

## Phase 5 — Cross-pool composition *(done)*

**Goal:** address §8's "cross-pool composition" open problem.

Delivered:

- [x] `<collaboration>` / `<participant>` / `<messageFlow>` parsing.
  The parser now dispatches at the top level: a collaboration document
  produces a composed multi-pool net, a plain process document
  produces a single-pool net (backward-compatible).
- [x] Multi-pool Petri net composition. The per-process translation
  logic was extracted into `_parse_process(elem, *, prefix, net, ...)`
  which adds to a shared `PetriNet` with every place / transition ID
  prefixed by the participant's BPMN ID. The collaboration handler
  calls it once per participant so internal IDs from different pools
  don't collide.
- [x] Shared message places per `<messageFlow>`. Each flow inserts a
  `msg_{flow_id}` place between the sender's transition (extra output
  arc) and the receiver's transition (extra input arc) — natural
  Aalst-style inter-organisational workflow-net composition.
- [x] Token conservation across pool boundaries is *implied* by the
  structure: a send transition consumes 1 sequenceFlow-place token
  and produces 1 sequenceFlow + 1 message-place token (net +1); the
  receive transition consumes the message-place token plus its own
  sequenceFlow place and produces its outflow place (net −1). The
  full system conserves tokens across the pair, and the test
  `test_full_collaboration_token_game_completes_both_pools` walks the
  marking through every cross-pool transition to verify it.

Carried forward:

- Distributed training across pool nets — per-pool XES logs trained
  independently rather than the whole composed net trained on
  combined traces. Belongs in Phase 6 once the training methodology
  work starts.
- Message flows to/from non-task nodes (events, gateways). Currently
  the message flow endpoint must be a node that produced a
  transition, which excludes start/end events and gateways. Adding
  this is mechanical but needs a careful think about which
  transition to attach the arc to for multi-transition gateways.

---

## Phase 6 — Training methodology *(done)*

**Goal:** address §8's "discrete-continuous interface" and "training
data requirements" open problems.

Delivered:

- [x] Straight-through estimator variant. `PetriNetModule(firing="ste")`
  uses a binary forward (hard step at sigmoid 0.5) with a sigmoid
  backward via the detach-trick. Forward produces exact 0.0 / 1.0
  outputs while gradients still flow to the arc weights and
  thresholds; verified by training the AND truth table to *exact*
  0/1 targets at every truth-table row.
- [x] Sharpness annealing. `SharpnessScheduler(module, start, end,
  num_steps, kind)` mutates `module.sharpness` over training, linear
  or exponential. Comparison test confirms an annealed schedule
  reaches a lower AND-join training loss than fixed `sharpness=1.0`.
- [x] Training-data-requirement sweep. `sweep_trace_count(factory,
  traces, ..., sample_sizes)` trains a fresh module on the first N
  traces for each N, returning per-size final loss. Test confirms
  the XOR routing problem needs more than 2 traces (one of each
  routing direction) to learn opposite routing.

Tightened later:

- [x] Softmax routing for XOR. `PetriNetModule(routing="softmax")`
  detects XOR-shape transitions structurally (single shared input
  place with all consumers having only that one input) and applies
  softmax over their pre-activations. AND-joins and non-XOR
  transitions continue to use the configured firing function. The
  load-bearing test confirms activations sum to 1 across the XOR
  group and AND-joins are untouched. Gumbel-softmax (adding
  reparameterised noise for stochastic discrete sampling) is a
  smaller follow-up on the same scaffold.

---

## Phase 7 — Anomaly detection evaluation *(done)*

**Goal:** §10 Step 4 — quantify §7.2 on realistic anomalous-trace
generators.

Delivered:

- [x] Anomaly generators in `anomalies.py`: `drop_event`,
  `insert_event`, `swap_event_labels` (branch-flipping), and
  `shuffle_events` (reordering). Each returns a corrupted copy of
  the input trace; the input is not mutated.
- [x] Trace-level anomaly score (`trace_anomaly_score`) and an AUC
  helper (`auc`) for ranking traces and measuring detector quality.
- [x] AUC characterisation per anomaly type on the XOR fixture:
  branch-flipping gets AUC > 0.9, inserted unseen events get
  AUC > 0.5, dropped events produce strictly higher scores than
  any in-distribution trace.
- [x] Baseline comparison via `FrequencyBaseline` — a marginal-
  frequency detector that scores traces by negative log probability
  under the training event distribution. The load-bearing test
  confirms that the structured Petri-net detector beats the
  frequency baseline on branch-flip by ≥ 0.3 AUC: branch flipping
  is invisible to a detector that doesn't see attributes, but
  caught cleanly by one that conditions on the input marking.

Carried forward:

- LSTM autoencoder baseline. The frequency baseline already
  isolates the structural-prior contribution for branch-flipping;
  an LSTM autoencoder is the natural next baseline for *sequential*
  anomalies (out-of-order events), where the frequency baseline is
  also blind. Belongs in a follow-up evaluation phase alongside
  longer-trace fixtures (e.g. the approval process) where order
  carries information.

---

## Phase 8 — Interpretability *(done)*

**Goal:** §8's fourth open problem — distill learned weights back into
readable decision rules.

Delivered:

- [x] `find_xor_groups(net)` for structural XOR-shape detection (N
  transitions sharing a single input place, each with only one input).
- [x] `extract_xor_rule(module, place, t_a, t_b)` for per-pair rule
  extraction: derives the crossover threshold and direction from the
  trained weight gap (w_A - w_B) and threshold gap (θ_A - θ_B); the
  weight-gap magnitude doubles as a confidence score.
- [x] `extract_routing_rules(module)` for whole-net rule extraction.
- [x] Smart label rewriting (`_downstream_label`) so distilled rules
  reference the downstream BPMN task name ("Path A") rather than the
  auto-generated gateway transition label ("Route -> fA1"). This is
  what makes the rules *business readable* rather than internal
  implementation labels.
- [x] `explain_anomaly(module, trace, ...)` produces a human-readable
  narrative pinned to specific BPMN labels, with expected vs observed
  activation and residual magnitude per diverging transition. The
  test pins the §7.2 promise: the explanation must not leak internal
  transition IDs like `t_xor_split_0`.

Load-bearing test: the XOR fixture trained on the 12-trace XES log
(routing at risk_score = 0.5) yields a distilled rule with
**crossover ≈ 0.486** and the right direction — high input → Path A,
low input → Path B. The neural net's learned weights have been
distilled back into a single readable business rule:
*"if risk_score > 0.486 → Path A, otherwise → Path B."* That closes
the loop from §4.2's continuous-relaxation training back to
§3-translation-table-readable business logic.

Tightened later:

- [x] N-way XOR rule extraction. `extract_xor_partition(module,
  place, transitions)` computes the upper envelope of the N linear
  pre-activations over the input axis and returns contiguous
  `XORRegion` intervals each mapped to a winning transition. The
  3-way load-bearing test trains on synthetic data routing at 1/3
  and 2/3 and recovers a 3-region partition with the right
  boundaries and labels. A whole-net variant
  (`extract_routing_partitions`) handles binary and N-way uniformly.
- [x] AND-join rule distillation. `extract_and_join_rule(module, t)`
  reads the learned input weights and threshold for a synchronisation
  transition, detects whether the weights are roughly uniform, and
  renders the rule either as a quorum ("fires when all N inputs are
  active" / "fires when at least k of N inputs are active") or as an
  explicit weighted vote with per-input weights. The test trains an
  AND-join on its truth table and confirms the summary contains
  "all" and "2" (i.e. "fires when all 2 inputs are active").

---

# Phase 9 and beyond — where PETRA goes next

Phases 1–8 closed the spec. Everything from §3 through §10 Step 4
runs, with tests. The six phases below aren't about finishing the
spec — they're about pushing PETRA into territory the spec didn't
cover, because real users keep needing more than the spec asked for.

One thing none of these phases include: any kind of visualisation,
dashboard, or UI. Drawing the net, painting activations onto it,
animating traces — that's the job of whatever application is
consuming PETRA, not of PETRA itself. We stop at producing the
numbers and the structure; rendering is somebody else's problem.

---

## Phase 9 — Richer models *(later)*

Right now every place holds at most one token, tokens carry no data,
firing has no duration, and there are no probabilities. That's
enough for the BPMN-flavoured demos but excludes a lot of real
systems. This phase fixes that.

- [x] **Multi-token markings.** Arc multiplicities — one firing can
  consume or produce N tokens per arc. The `batch_packaging` example
  demonstrates a bottle-to-crate transition with input weight 6: the
  transition waits for six tokens to accumulate before firing,
  exactly as a real packaging line works.
- [x] **Coloured Petri nets** *(CPN-lite — structural layer + CPN-aware compiler)*.
  Tokens carry a scalar value; transitions can have guards that
  read input-token values; output arcs can specify the value the
  produced token carries (constant or callable). The
  `credit_approval_coloured` scenario routes loan applications on
  the *amount* carried by the token, with `t_approve` and
  `t_decline` guards declared in TOML (`{place, op, value}` form).
  The compiler now reads these guards too: each structural guard
  contributes a learnable `nn.Parameter` threshold seeded at the
  TOML value, and a sigmoid soft-gate multiplies the transition's
  firing strength by `sign(op) · (value(place) − threshold)` with
  per-guard sharpness auto-scaled by `1 / max(|θ_init|, 1.0)` so
  gradients at the boundary stay O(1) in any unit. The forward
  pass carries a parallel per-place value channel through both
  acyclic and time-unrolled modes; `train_on_traces` /
  `anomaly_score` gain `attribute_to_values` for the value channel,
  and `ScenarioContext` reads it from `[training.input_values]`
  in TOML. The credit-approval scenario trains both seeded
  thresholds into the empirical decision band (900–1500) from the
  hand-set 1000 and routes held-out values correctly under the
  soft guards. Torch-friendly callable guards (`torch_guard` on a
  transition, called as `dict[place_id, Tensor] -> Tensor[batch]`)
  and torch-friendly arc output-value transforms (`torch_output_value`
  on an arc) are also supported as escape hatches for routing logic
  the structural form can't express — multi-input comparisons,
  compound predicates, custom learnable sub-networks. The torch
  guard overrides the structural guard when both are declared on
  the same transition. Bool-returning callable guards and
  float-returning callable arc-output-values continue to live at the
  token-game level only.
- [x] **Inhibitor arcs.** Declarative "fire only when this place is
  empty" guards. The compiler implements them as a multiplicative
  `(1 - a(p))` gate after firing. The `resource_lock` scenario uses
  them to enforce a two-client mutex on a shared resource; in
  time-unrolled mode the inhibitor place fills after one step and the
  gate suppresses further firings.
- [x] **Stochastic firing rates.** Per-transition rate multiplier
  on the pre-activation (default 1.0). High rate fires more eagerly;
  low rate more conservatively. Lets the modeller carry priors about
  transition propensity through to training; the trained weights and
  thresholds then refine on top of the prior. The `priority_dispatch`
  scenario uses rates 3.0, 1.0, and 0.5 to declare three handlers'
  relative dispatch priorities before training.
- [x] **Transition durations.** A transition with duration D fired
  at step *n* produces its output at step *n+D-1*. The compiler
  maintains a per-transition in-flight queue in the time-unrolled
  forward pass; the discrete token-game treats firings as atomic.
  The `paint_shop` scenario demonstrates a cure step with duration
  3: parts spend three time-units in the cure transition before
  reaching the inspection place.

## Phase 10 — Connect to the rest of the world *(later)*

There's a 25-year-old Petri-net ecosystem out there — CPN Tools,
GreatSPN, TINA, ProM, Reactome, BPI Challenge — and PETRA currently
ignores all of it. That's lazy. This phase makes PETRA a citizen of
that ecosystem.

- [x] **PNML import and export.** `petri_net_nn.pnml` provides
  `parse_pnml` and `to_pnml` over the PNML 2009 P/T net subset:
  places, transitions, arcs, initial markings, arc inscriptions
  (weights), names, multi-page documents flattened to one net, and
  the de-facto inhibitor-arc extension (`<arctype>inhibitor`).
  Round-trip preserves the structural core; PETRA-specific
  extensions (durations, rates, guards, output values) without a
  standard PNML representation are dropped on export and ignored
  on import. Adapter accepts `source = "pnml_file"`.
- [x] **SIF (Simple Interaction Format).** Pathway Commons publishes
  Reactome and the other major pathway databases (BioCyc, PID, NCI
  Nature, Panther, HumanCyc, KEGG) in SIF — tab-separated
  ``entity_a [tab] interaction_type [tab] entity_b`` triples.
  `petri_net_nn.sif.parse_sif` maps each entity to a place and
  each interaction to a directed transition; the adapter accepts
  ``source = "sif_file"`` for symmetry with the BPMN/PNML
  branches. The new `mapk_pathway` scenario ships an HGNC-symbol
  curated slice of the canonical EGF → MAPK1/3 cascade — real
  format, real interaction-type vocabulary, structurally faithful
  to the biology, swappable for any Pathway Commons SIF download
  without code changes. **BioPAX and SBGN remain on the
  follow-up list** — BioPAX needs a Level-3 RDF/OWL parser
  (~1,000 LOC of XML traversal) and SBGN is mid-complexity XML.
  SIF closes the spirit of this carry-forward — PETRA can now
  consume real curated pathway content — without the heavy lift,
  and the typed-edge richness of BioPAX is the natural Phase 13
  / Phase 14 follow-up if a user has a specific need.
- [x] **Real BPI Challenge logs.** The `incident_management`
  scenario ships the actual BPI Challenge 2013 incidents log
  (Volvo IT, 7,554 real ITIL tickets, 65k events) committed to
  the repo as a 1.3 MB gzipped XES file. `parse_xes` reads gzip
  transparently; the adapter gained `promote_event_attrs` (lift
  event-level attributes like `impact` to trace level) and
  `event_name_attr` (override event names from a different XES
  attribute, e.g. `lifecycle:transition`). PETRA trains on the
  full corpus in ~20 seconds; anomaly detection flags traces that
  skip the Resolved step before Closing.
- [x] **CSV / JSON trace adapters.** Adapter accepts
  `traces.source = "csv_file"` (standard process-mining flat
  table with configurable `case_column` / `activity_column`) and
  `traces.source = "json_file"` (a list of `{attributes, events}`
  objects, with events either as bare strings or as full
  `{name, attributes}` objects). Both produce the same XESTrace
  list the rest of the pipeline consumes, with full support for
  `promote_event_attrs` and the other adapter options.

## Phase 11 — Deeper formal methods *(later)*

Strong bisimulation is the bare minimum. Real formal-methods workflows
ask for more, and PETRA's claim to be the formal-verification-meets-ML
story depends on having more to offer than the floor.

- [x] **Weak bisimulation.** Transitions marked ``silent=True``
  on the net are collapsed via a τ-saturation step before
  partition refinement, so refactorings that introduce internal
  logging hooks, no-op routing gates, internal handoffs, or any
  other structural artefact that nobody outside the implementation
  cares about get correctly recognised as behaviourally equivalent
  to the reference. Implemented as ``are_weakly_bisimilar(net1,
  net2)`` and ``weak_bisimulation_equivalence_classes(net)`` in
  ``bisimulation.py``; the saturation step makes every state
  τ-self-reachable, extends τ-paths into direct τ-edges, and
  flanks every visible-action edge with τ-prefixes and τ-suffixes
  through all τ-reachable neighbours. Strong bisimulation over
  the saturated LTS is exactly weak bisimulation over the
  original. Branching bisimulation (the strictly finer variant
  that respects the choice points along τ-paths) remains a
  follow-up.
- [x] **CTL property checking.** Temporal-logic invariants
  over the reachability graph. `petri_net_nn.ctl` exposes a
  six-primitive AST (`Atom`, `Not`, `And`, `Or`, `EX`, `EU`, `EG`)
  plus the derived constructors (`AX`, `AG`, `AF`, `EF`, `AU`)
  that expand to the primitives via the standard CTL
  equivalences. `check_ctl(net, formula)` returns a `CTLResult`
  with the full set of states satisfying the formula, whether the
  initial marking is one of them, and a sample counterexample
  marking if the formula fails somewhere. `satisfies(net, formula)`
  is the boolean convenience wrapper. Atomic-proposition helpers
  cover the common marking predicates — `place_has_token`,
  `place_empty`, `place_count_eq`, `place_count_ge`,
  `transition_enabled` — and arbitrary custom predicates over the
  marking are accepted as `Atom(callable, label)`. Implicit
  self-loops are added at deadlock states (the standard Kripke
  convention) so AG / AF / EG behave intuitively on terminating
  workflow nets. The full headline use cases compile cleanly:
  `AG (request → AF response)` (every request is eventually
  followed by a response), `AG ¬deadlock` (no reachable state is
  a deadlock), `A[¬decline U credit_check_done]` (decline cannot
  fire before the credit check). **LTL** (linear-time semantics
  evaluated trace-by-trace) remains a follow-up — most temporal
  questions a process analyst asks are CTL-shaped, and an LTL
  checker needs either single-trace evaluation against logs
  (which the existing anomaly path already partly covers) or full
  Büchi-automaton support.
- [x] **Aalst soundness verification.** `petri_net_nn.soundness`
  exposes `check_soundness(net)` returning a `SoundnessReport`
  with three pinned outcomes: ``incomplete_markings`` (reachable
  markings from which the intended final marking is *not*
  reachable — option-to-complete failures), ``lingering_token_markings``
  (markings where the sink has reached its completion count but
  tokens still hang around at other places — proper-completion
  failures), and ``dead_transitions`` (transitions never enabled
  in any reachable marking). The default final marking is
  one-token-at-each-sink, with explicit override available. A
  net is sound iff all three lists are empty; ``report.is_sound``
  and ``report.summary()`` give the boolean and a one-line digest.
- [x] **Deadlock localisation.** Companion `find_deadlocks(net)`
  returns the *specific markings* the net can reach but can't
  progress from, excluding the intended final marking. Pins the
  root-cause state of an option-to-complete failure — gives a
  process owner a specific named token configuration to look at
  instead of just "this net doesn't complete cleanly".
- [x] **Coverability analysis.** ``petri_net_nn.coverability``
  exposes ``coverability_graph(net)`` and the one-line
  ``is_bounded(net)`` wrapper. The Karp-Miller construction
  builds the coverability tree, replacing token counts with the
  ω sentinel at every place that strictly exceeds an ancestor
  on the path from the root. A ``CoverabilityReport`` returns
  ``is_bounded``, the list of unbounded place names, per-place
  upper bounds (integer for bounded places, ω for unbounded
  ones), and the ω-marking witnesses that pin where the
  unboundedness shows up. Sound and exact on nets without
  inhibitor arcs; conservative (may overreport unboundedness)
  on nets that use them — the inhibitor-arc + plain-Petri-net
  combination is Turing-complete (Minsky 1967), so exact
  coverability would solve the halting problem (Turing 1936).
  ``BUSINESS_ANALYST_GUIDE.md`` §17 carries a long-form
  treatment of why that boundary is mathematical rather than a
  missing feature, with three industry examples of where the
  unbounded-place pattern shows up in practice
  (payment-reconciliation queues, specialty-referral waiting
  lists, manufacturing WIP at bottlenecks).

## Phase 12 — Discovering the net from traces *(later)*

PETRA currently demands a hand-coded or BPMN-supplied structure
before it can do anything. This is a serious barrier — most users
have logs but not models. Process-mining has solved structure
discovery from logs; PETRA should plug into that.

- [x] **Inductive miner.** `petri_net_nn.discovery.discover_inductive(
  traces)` mines a process tree from an event log (any list of
  XESTrace or raw activity-name tuples) and translates the tree
  into a Petri net using the standard block-structured
  construction. Four cut shapes detected, applied in priority
  order: exclusive choice (connected components of the
  undirected DFG), sequence (level-based antichain decomposition
  of the SCC DAG with explicit "every-Σᵢ-reaches-every-Σⱼ"
  validation), parallel (non-parallel-graph components with
  start/end coverage on each), loop (minimal body =
  start_activities ∪ end_activities, redo components in the
  remaining activities, validated against the body↔redo edge
  constraints). Base cases: empty log → τ; single-activity log
  → leaf (wrapped in `Loop(activity, τ)` if the activity repeats
  within any trace). Pathological logs that fit none of the
  cuts fall through to a flower model. **Output is sound by
  construction**; verified by the test suite asserting
  `check_soundness(net).is_sound` on every mined net plus a
  replay-invariant BFS that confirms every input trace is
  replayable on the resulting net modulo τ collapse. This
  closes the spec's biggest "who can use PETRA" gap — log-only
  users no longer need ProM + PNML; everything is one library
  call.

- [x] **Discover → verify → train, in one call.**
  `discover_and_train(traces, *, attribute_to_marking, …)`
  chains `discover_inductive` → `check_soundness` (sanity, since
  IM is sound by construction) → `PetriNetModule` compile →
  `train_on_traces`. Returns `(net, module, losses)`. Single
  entry point for users who have only a log.

- [ ] **Alpha miner.** Classical 2004-era algorithm; covered in
  spirit by the inductive miner above, which is its modern
  successor. Skipped deliberately — alpha-miner has known
  limitations (no short-loop handling, fragile on noisy data)
  and inductive miner is the recommended replacement.

- [ ] **Heuristic miner.** Frequency-based, noise-tolerant. The
  IM variants IMf (infrequent-noise filtering) and IMi
  (incremental) would extend this naturally; deferred to a
  follow-up when noisy real-world logs from a specific user
  motivate the work.

## Phase 13 — Better explanations *(later)*

Phase 8 distills XOR routing rules and AND-join quorum rules out of
trained weights. That's a start. Real decision-support deployments
need more.

- [x] **Counterfactual explanations.** `find_counterfactual(module,
  base_marking, *, flip_place, target_transition, ...)` binary-
  searches the input at one place to find the boundary at which
  the target transition's firing activation crosses 0.5. Supports
  both the marking channel (`flip_channel="marking"`) and the
  coloured-token value channel (`flip_channel="value"`, for CPN
  scenarios where the structural guard reads per-token values).
  Returns a `Counterfactual` bundling the target transition, the
  flipped input place and channel, original vs counterfactual
  input values, and original vs counterfactual activations.
  `prose_for_counterfactual(cf, *, input_label=None)` renders the
  result as a paragraph with directional language ("would need to
  be increased / decreased") and an optional input-label override
  for regulator-facing prose. Separate activation and interval
  tolerances let the search converge cleanly on both small-range
  (marking-channel [0, 1]) and large-range (value-channel [0,
  10000]) inputs. The "what would have flipped this decision"
  question — the actionable complement to anomaly scores — is now
  a one-call answer.
- [x] **Confidence intervals on extracted rules.** Bootstrap the
  training and report the distribution of crossover thresholds /
  AND-join quorum thresholds. `bootstrap_xor_rule(factory, traces,
  ...)` and `bootstrap_and_join_rule(factory, traces, ...)` resample
  the trace list with replacement N times (default 100), train a
  fresh module on each resample via ``train_on_traces``, extract
  the rule, and return an ``XORRuleCI`` / ``AndJoinRuleCI`` carrying
  the point estimate, the bootstrap distribution, a percentile-based
  confidence interval (default 95%), and a direction-agreement /
  quorum-agreement rate measuring how often the bootstrap rule
  matched the point estimate's structural form. Seeded RNG for
  reproducibility. The combination "*the crossover is 0.486 ±
  0.03 with 95% confidence over 100 resamples; direction agrees
  in 98% of resamples*" is the production-trust threshold any
  regulator would ask for.
- [x] **Sensitivity analysis.** `transition_sensitivity(module,
  base_marking, target_transition, *, base_values=None)` returns a
  `SensitivityReport` with per-input gradient magnitudes of the
  target's firing activation at the base point — both the marking
  channel and the coloured-token value channel. `input_importance(
  module, traces, *, attribute_to_marking, ...)` aggregates absolute
  gradients across a trace set and all scored transitions to rank
  the inputs the trained network leans on most. Both use torch
  autograd directly on the input tensors; gradients are local
  properties of the trained function (saturated regions produce
  smaller gradients), so the prose helper reports the base
  activation alongside the gradient ranking. `prose_for_sensitivity(
  report, *, top_n=3, input_labels=None)` renders a ranked list as
  a paragraph with directional language ("increasing this input
  raises / lowers the firing of …"). Together with the rule
  extractor and counterfactual analysis this completes the
  diagnostic toolkit: *which inputs matter → which threshold rules
  them → what change would flip the outcome*.
- [x] **Cross-variant comparison reports.** "These two variants
  agree on 87% of the input domain; here's the band where they
  diverge." `compare_variants(module_a, module_b, *, input_grid,
  input_value_grid=None, correspondence=None, tolerance=0.1,
  max_disagreement_samples=50)` runs both trained modules over a
  Cartesian-product grid of input points, pairs transitions by
  label (with an explicit `correspondence` override available),
  and reports the hard-agreement rate (same firing decision, both
  on the same side of 0.5), the soft-agreement rate (activations
  within `tolerance` of each other), per-transition agreement
  rates, and a bounded list of specific divergent grid points
  for inspection. Unmatched labels are recorded separately so
  callers can catch variants that differ structurally before
  reading the numbers. `prose_for_comparison_report(report, *,
  top_disagreement=3)` renders the result as a paragraph naming
  the worst-offender transitions; when there's no disagreement
  it says so explicitly. Pairs naturally with the cost-ranked
  refactoring scenario — bisimulation proves two variants are
  *behaviourally* equivalent, comparison reports show *where in
  the input domain* their soft routing actually overlaps.
- [x] **Prose explanations.** `prose_for_xor_rule(rule)` and
  `prose_for_and_join_rule(rule)` turn the structured rule objects
  into paragraph form. Both accept either the bare rule
  (point estimate only) or the corresponding CI variant — in the
  latter case the prose includes the bracket numbers and the
  direction-agreement / quorum-agreement percentage. An optional
  ``input_label`` substitutes a domain term for the raw place id
  (so a regulator-facing document can read *"application amount"*
  instead of *"p_application"*). Plain-text output, no
  dependencies, ready to drop into a report.

## Phase 14 — Making PETRA usable in production *(later)*

A passing test suite is not a deployable system. This phase covers
everything between "the library works on a developer's laptop" and
"a non-Python team can plug PETRA into their pipeline." No UI work —
that's the consuming application's call.

- [x] **PyPI packaging.** `pyproject.toml` with PEP 621 metadata
  (build backend `setuptools>=77`, distribution name `petra`,
  importable package `petri_net_nn`, dynamic version read from
  `petri_net_nn.__version__`, Python >=3.11, torch>=2.0, optional
  `[dev]` extra for pytest). `CHANGELOG.md` follows Keep a
  Changelog; the v0.1.0 entry captures the full feature surface
  shipped through Phases 1–13. PyPI metadata matches the user's
  existing `timeseries-formula-finder` package (same author,
  same `fleetingswallow.com` homepage). The first release is
  scaffolded but not pushed; the user does the one-time PyPI
  trusted-publisher setup and tags `v0.1.0` to fire the publish
  workflow.
- [x] **CI.** `.github/workflows/test.yml` runs the full pytest
  suite on every push to `main` and every pull request, matrix
  over Python 3.11 / 3.12. `.github/workflows/publish.yml`
  builds sdist + wheel and publishes to PyPI via *trusted
  publishing* (OIDC, no API token stored in the repo) when a
  `v*.*.*` git tag is pushed. The publish workflow file
  documents the one-time setup steps inline.
- [x] **Auto-built documentation site.** MkDocs Material with the
  `mkdocstrings` Python handler pulling API reference from each
  module's docstrings, the existing manuals (`BUSINESS_ANALYST_GUIDE.md`,
  `DEV_MANUAL.md`, `ROADMAP.md`) rendered as site pages, and
  `docs/index.md` as the landing page. Built and deployed by
  `.github/workflows/docs.yml` on every push to `main` and
  publishable manually via `workflow_dispatch`; uses
  `actions/deploy-pages` with OIDC (no API tokens). Published at
  <https://pcoz.github.io/formally-verified-learnable-process-intelligence/>.
  One-time GitHub setting: repo Settings → Pages → set source to
  "GitHub Actions" (documented inline in the workflow file).
  Strict-mode build catches broken cross-references; cross-links
  to files outside `docs/` (README, examples, source) use
  absolute GitHub URLs so they resolve cleanly in both the
  GitHub-rendered view and the MkDocs site.
- [x] **ONNX export.** `petri_net_nn.onnx_export.export_onnx(
  module, output_path, *, input_places, input_value_places=None,
  output_transitions=None, batch_size=1, opset_version=17,
  dynamic_batch=True)` wraps the trained `PetriNetModule`'s
  dict-input forward pass in a positional-args adapter and calls
  `torch.onnx.export`. The exported `.onnx` file runs unchanged
  in any ONNX runtime — onnxruntime in Python / C++ / Java /
  browser, plus the various accelerator stacks that consume
  ONNX. The export validates the requested input / output place
  and transition names against the net before tracing, marks
  the batch axis as dynamic by default, and returns a schema
  dict the caller can serialise as a JSON sidecar so
  consumers in other languages know which positional tensor
  corresponds to which place. Limitation carried forward:
  time-unrolled mode with non-uniform transition durations
  uses Python lists for the in-flight queue, which don't
  survive ONNX tracing — acyclic mode and uniform-duration
  time-unrolled mode export cleanly. The torch legacy exporter
  is used (`dynamo=False`) so the export path has no extra
  runtime dependency beyond torch itself; the optional
  `[onnx]` extra brings `onnxruntime` for the parity-test
  surface.
- [x] **Streaming evaluator.** `petri_net_nn.streaming` exposes
  `StreamingEvent`, `StreamingEvaluation`, and `StreamingEvaluator`.
  The evaluator maintains per-case state (event list + merged
  attribute dict, latest-event-wins on key conflicts) and scores
  on demand against the trained module. Two operating modes:
  *on-close* (the default; `on_event` buffers, `close_case`
  emits the final score and frees state) and *on-every-event*
  (every `on_event` returns a `StreamingEvaluation` against the
  partial trace, useful for live dashboards at higher per-event
  cost). Both push (`on_event`) and pull (`process_stream`)
  shapes are supported so the user can wire to whatever source
  they have — Kafka consumer, webhook handler, file-tail, batch
  backfill of a stored log. Scoring delegates to the existing
  offline `anomaly_score`, so streaming and batch share the same
  correctness story. Single-threaded by design; a multi-threaded
  deployment serialises through a single consumer.
- [x] **REST inference API.** `petri_net_nn.rest.build_app(module)`
  returns a FastAPI application that exposes the trained module
  over six JSON endpoints:
  - `GET /healthz` (liveness + version + module-stats),
  - `GET /schema` (places, transitions, labels, initial marking,
    flags for silent transitions / structural guards),
  - `POST /forward` (input marking + optional value channel →
    per-transition + per-place activations),
  - `POST /anomaly` (events + input marking → per-transition
    residuals + trace-level scalar),
  - `POST /counterfactual` (delegates to `find_counterfactual`,
    supports marking and value channels),
  - `POST /sensitivity` (delegates to `transition_sensitivity`).

  FastAPI auto-generates OpenAPI 3.x + Swagger UI at `/docs`.
  Pydantic models gate input validation and produce JSON
  schemas; the fastapi / pydantic imports are guarded at module
  load so `import petri_net_nn` works without the `[rest]`
  extra installed and `build_app` raises a clear ImportError
  if called without it. Optional `[rest]` extra brings fastapi
  and uvicorn; `[dev]` brings httpx for the FastAPI TestClient.
  Single-model-per-app by design — spin multiple FastAPI
  processes for multi-model serving. Auth / CORS / rate
  limiting are deliberately left to the caller's app layer.
- [x] **Workflow-engine integration toolkit** *(Python-side).* New
  `petri_net_nn.cli` module with three command-line entry points
  declared in `pyproject.toml`'s `[project.scripts]`:
  `petra-train` (TOML scenario → bundle), `petra-score`
  (bundle + traces → JSON per-trace scores), `petra-serve`
  (bundle → FastAPI REST app under uvicorn). The bundle format
  is two files: a `.pt` pickled module + a `.meta.json` sidecar
  carrying the scenario's input-marking and input-values specs
  so `petra-score` can map trace attributes onto place markings
  without needing the original `scenario.toml` at score time.

  Together with the REST API (Phase 14), the streaming evaluator
  (Phase 14), and the existing CSV/JSON/XES trace loaders, this
  is enough to wire PETRA into any workflow engine that emits
  structured audit events through *three documented integration
  patterns* (`docs/INTEGRATION_PATTERNS.md`):

  1. **REST webhook** — engine POSTs to `petra-serve`
     synchronously on each event-of-interest.
  2. **Audit-log tail** — engine exports its history table to
     CSV; `petra-score` runs on a cron schedule against it.
  3. **Streaming subscription** — engine publishes to Kafka /
     RabbitMQ / Redis Streams; PETRA's `StreamingEvaluator`
     consumes via a ~25-line consumer.

  The patterns work against the three named engines (Camunda,
  Activiti, Flowable) and any other system that can emit
  structured events to a database, queue, or HTTP webhook.

- [ ] **Engine-side JVM plugins.** Native execution-listener /
  command-interceptor / process-delegate bindings for Camunda,
  Activiti, and Flowable. Explicit follow-up — out of scope
  for the Python-only crate. The boundary between PETRA and
  an engine is currently JSON (for the three patterns above)
  or ONNX (for in-process JVM inference via `export_onnx` →
  ONNX Runtime for Java). Going further would mean a cross-
  language project (Maven build, JVM-side weight loading,
  per-engine extension shapes) that is properly its own
  repository.

- [ ] **Process-mining tool bridges** *(deeper than the existing
  PNML import).* ProM's PNML output is already consumed via
  `parse_pnml` (Phase 10); the missing piece is a single
  end-to-end recipe / wrapper for *discover → verify → train*
  that bundles ProM (or PM4Py / Celonis / Disco) with PETRA.
  Native structure discovery inside PETRA is Phase 12.

---

## Pending release: v0.2.0 to PyPI

The PyPI page for `petra-nn` is currently built from the
README that shipped with **v0.1.0** — it predates everything
done since:

* Phase 12 Inductive Miner (`discover_inductive`,
  `discover_and_train`) — *net-new public API*, not just docs.
* The README's *"Don't have a Petri net yet?"* section and
  the honest-framing calibration.
* The seven proposed worked-example scenarios listed below
  (whichever land before the release will ride along).

Because the additions are public-API-additive, the semver
bump is **minor** (v0.1.0 → **v0.2.0**), not patch. Steps:

- [ ] Bump `petri_net_nn.__version__` to `"0.2.0"`.
- [ ] Add a `v0.2.0` section at the top of `docs/CHANGELOG.md`.
- [ ] Commit, then `git tag v0.2.0 && git push origin v0.2.0`.
- [ ] The existing PyPI Trusted-Publisher workflow
  (`.github/workflows/publish.yml`) picks the tag up
  automatically and uploads the new wheel + sdist; the PyPI
  long-description refreshes on next page view.
- [ ] Sanity-check at <https://pypi.org/project/petra-nn/>
  that the new README is showing, including the discovery
  section and honest-framing block.

(There is no on-PyPI manual edit step — everything flows
from the tag.)

---

## Pending worked-example scenarios

The capabilities below all have full implementations and test
coverage, but no dedicated `examples/<scenario>/` directory
that walks through them end-to-end the way `credit_approval_coloured`
walks through the CPN-aware compiler and `mapk_pathway` walks
through SIF import. A new scenario for each (some can be
bundled) is the natural next layer of polish — every additional
worked example lowers the activation energy for a new user
landing on the repo and finding "the thing that shows my use
case."

Suggested groupings (one bullet = one new `examples/` scenario,
roughly):

- [x] **`discover_and_train_pipeline/`** — discovery via the
  Inductive Miner, then chain `discover_and_train` end-to-end
  on a synthetic four-trace loan-approval log with sequence,
  parallel, and exclusive-choice cuts. Demonstrates Phase 12.
  The adapter learned a new keyword in the process —
  `net.source = "discover"` invokes the basic Inductive Miner
  against the `[traces]` section, so the whole pipeline stays
  TOML-driven with no per-scenario Python boilerplate. Six
  tests pin the four canonical assertions: the mined net is
  sound by construction, every input trace replays on it,
  training through the full adapter path reduces the loss, and
  the `discover_and_train` one-call API reproduces the same
  result. README and BUSINESS_ANALYST_GUIDE cross-link to the
  new scenario from the discovery sections.

- [x] **`safe_refactoring/`** — Variant A (the unrefactored
  baseline loan-approval process) plus Variant B (Variant A
  with a silent τ audit-log transition inserted between triage
  and risk assessment). Strong bisimulation correctly rejects
  the pair as different; **weak bisimulation accepts them as
  equivalent** — the headline safe-refactoring claim. Cross-
  variant comparison sweeps the credit-score domain and pins
  hard-agreement on every grid point. Bootstrap CIs on each
  variant's distilled `credit_score → approve/decline` rule
  overlap and both land in the empirical decision band.
  Demonstrates Phase 11 weak bisim + Phase 13 cross-variant +
  bootstrap CIs. Six tests in
  `tests/scenarios/test_safe_refactoring.py`.

- [x] **`compliance_audit/`** — loan-approval process with an
  explicit audit-log step on the approval path, plus two
  regulatory invariants stated as CTL formulae:
  `AG (decided_approve → AF audit_logged)` ("every approved
  loan eventually fires the audit log") and
  `AG (enabled(t_decline) → has_token(p_credit_checked))`
  ("decline cannot fire before credit-check"). The scenario
  test pins `check_soundness`, `find_deadlocks`, both CTL
  invariants on the compliant net, and the
  CTL-fails-with-counterexample path on a deliberately-broken
  variant that bypasses the audit-log step — exactly the case
  auditors care about. Demonstrates Phase 11 soundness +
  deadlock localisation + CTL. Seven tests in
  `tests/scenarios/test_compliance_audit.py`.

- [x] **`regulator_ready_credit_approval/`** — coloured-token
  loan-approval net with the full Phase 13 diagnostic surface
  attached. Six tests pin: trained guard threshold lands in
  band; counterfactual binary-search on a declined £300
  application finds a flip point in the empirical decision
  band; sensitivity analysis confirms the application amount
  as the dominant input; prose helpers render counterfactual
  and sensitivity reports with `"application amount"`
  substituted for the raw place id; a custom bootstrap loop
  over the trace list produces a non-degenerate percentile CI
  on the learned guard threshold. The combination is the
  shape regulators (GDPR Article 22, SR 11-7, EU AI Act)
  expect of a trained decisioning model. Demonstrates Phase
  13's bootstrap CIs + counterfactuals + sensitivity + prose
  applied to a single business case.

- [x] **`realtime_monitoring/`** — simplified incident-
  management net trained on the happy-path lifecycle, then
  fed an interleaved live stream of three cases (two normal,
  one anomalous that skips the *Resolved* step) through
  `StreamingEvaluator`. Six tests pin the on-close
  emission-per-case-at-close pattern, the headline anomaly
  signal (anomalous case scores strictly higher than normal
  ones), the per-transition residual pinning to
  `t_resolved` (the skipped step — the actionable alert
  signal), state freeing on `close_case`, and the
  on-every-event live-dashboard mode plus the
  `process_stream` pull-shape consumer. Demonstrates Phase
  14 streaming evaluator.

- [x] **`onnx_deployment/`** — small loan-approval XOR-routing
  net trained from inline traces, exported via
  `export_onnx`, loaded into `onnxruntime`, parity-checked
  against the torch forward pass. Four tests pin: the
  exported `.onnx` file is written with a JSON-serialisable
  schema sidecar; torch and onnxruntime outputs match within
  1e-4 across a sweep of inputs; routing decisions are
  consistent across runtimes; the dynamic-batch axis works
  (a model exported with `batch_size=1` accepts a batch of
  50 at inference). Demonstrates Phase 14 ONNX export — the
  bridge to JVM / C++ / browser / mobile / accelerator
  deployments.

- [x] **`rest_serving/`** — small loan-approval XOR-routing
  model trained from inline traces, wrapped in a FastAPI app
  via `build_app(module)`, exercised end-to-end through
  FastAPI's `TestClient`. Eight tests pin every endpoint:
  `/healthz` reports module stats; `/schema` returns the
  net's structural inventory; `/forward` routes high
  risk_score to approve and low to decline; `/anomaly`
  scores a non-conforming trace higher than a conforming
  one; `/counterfactual` finds the flip-point for a declined
  application; `/sensitivity` reports a non-zero
  marking-channel gradient on the routing input; the
  auto-generated `/openapi.json` is reachable for
  client-side SDK generators. README documents the
  `petra-train` → `petra-serve` → `curl` production
  deployment shape. Demonstrates Phase 14 REST + CLI.

The patterns documented in `INTEGRATION_PATTERNS.md` overlap
with several of these (in particular the REST and streaming
ones); a scenario differs from the integration-patterns doc
by being an *executable, test-backed* artefact rather than a
code-snippet recipe. Both have value: the docs guide
integration design choices; the scenarios prove the wiring
actually works.

# What we are deliberately not building

Three things sit outside every phase, on purpose.

**No visualisation, no UI, no dashboards.** Petri-net diagrams,
activation heatmaps, anomaly overlays, trace animation, web apps,
desktop GUIs — none of this belongs in PETRA. PETRA produces numbers
and structure; presenting them is the consuming application's job.

**No continuous-time, continuous-state physics.** Fluid dynamics,
classical mechanics, analogue control. Petri nets are discrete; if a
problem has a sensible discretisation we'll catch it via Phase 9, but
trying to model the underlying physics is the wrong tool.

---

## Out-of-scope cross-cutting limits *(held)*

- Multi-token markings (per-place token counts > 1). All current nets
  are 1-bounded.
- Explicit time / clocks. BPMN timer events and durations aren't
  modelled.
- Probabilistic Petri nets — transition activations are not
  interpreted as firing probabilities (they're continuous activations
  in [0,1]).
- Engagement with formal-methods / neuro-symbolic AI research
  communities (§10 Step 5) — out of scope for code work.
