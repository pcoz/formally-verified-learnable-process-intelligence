# PETRA: A guide for process analysts

This guide explains what PETRA does, why each piece of it matters,
and how the pieces fit together — in language aimed at process
analysts, compliance officers, project managers, and anyone whose
job involves understanding or improving how an organisation
actually runs. There's no code in this guide, no Python, no
mathematics beyond what a workflow diagram already carries.

If you read this end to end, you will know:

- **What a Petri net is** and why PETRA uses them as its substrate.
- **How a BPMN diagram becomes a Petri net** (so PETRA can analyse it).
- **What each framework feature does in plain English** — coloured
  tokens, multi-token markings, inhibitor arcs, durations, firing
  rates, cross-pool composition.
- **What PETRA learns from your logs** and what it produces — readable
  rules, anomaly scores, equivalence proofs, cost rankings.
- **How the pieces compose end-to-end** into the unified analytical
  pipeline the [README](../README.md) walkthrough describes.

---

## 1. The one-minute version

PETRA takes two things from you:

1. A **structural model** of your process — usually a BPMN diagram
   you already have, but it can also come from a Petri-net file
   (PNML), a biology-pathway file (SIF), or be written by hand in
   a configuration file.
2. An **execution log** — *which steps actually ran, in what order,
   for each instance.* If you have a process-mining log already
   (XES, CSV, JSON), PETRA reads it directly.

PETRA then produces four things you can act on:

| Output | Example |
|---|---|
| **Readable decision rules** | *"If amount > £1,000 the application is routed to credit-review."* |
| **Anomaly scores pinned to specific steps** | *"In trace #4827 the credit-check step didn't fire when the data says it should have."* |
| **Proofs that two redesigns behave the same** | *"Process variant B is provably equivalent to variant A — no behaviour was lost or added."* |
| **Cost rankings between equivalent redesigns** | *"Variant B is 6× cheaper on the observed workload while doing provably the same thing."* |

The rest of this guide explains what each of these means and how
PETRA produces them.

---

## 2. The substrate: Petri nets

PETRA's entire job rests on representing your process as a **Petri
net**. A Petri net is a formal model invented by Carl Adam Petri
in 1962 for describing systems whose state changes at identifiable
moments — exactly the shape of a business process. There are only
three kinds of object:

| Element | What it is in a process | Visual convention |
|---|---|---|
| **Place** | A *state* something can be in (e.g. *"application submitted"*, *"awaiting review"*, *"approved"*). | A circle. |
| **Transition** | An *event* that moves work forward (e.g. *"review application"*, *"approve loan"*, *"send rejection"*). | A rectangle. |
| **Arc** | A *connection* — either *place → transition* (consumes work from this state) or *transition → place* (produces work into this state). | An arrow. |

Work in a Petri net is represented by **tokens**, which are small
dots that sit inside places. A token in a place means *"there is
one unit of work currently in this state."* When a transition
*fires*, it consumes one token from each of its input places and
produces one token in each of its output places. *Firing* is the
Petri-net word for *"this step just happened."*

### A worked example

A trivial loan-approval process — submit, review, decide:

```
   [Submitted] ──▶ ▢ review ──▶ [Reviewed] ──▶ ▢ approve ──▶ [Approved]
        ●                                                          
```

- *Places* (circles): **Submitted**, **Reviewed**, **Approved**.
- *Transitions* (rectangles): **review**, **approve**.
- *Initial token* (the ● in **Submitted**): one application waiting.

The token-game executes top-down:

1. The token sits in **Submitted**. The **review** transition is
   *enabled* — its single input place has a token.
2. **review** fires. The token in **Submitted** is consumed; a new
   token appears in **Reviewed**.
3. **approve** is now enabled. It fires; the token moves from
   **Reviewed** to **Approved**.
4. No transitions are enabled. The process has *terminated*.

The whole framework PETRA is built on is just elaborations of
this idea.

### Why Petri nets, rather than BPMN or flowcharts?

BPMN and flowcharts are *human-readable conveniences* on top of
the same underlying idea. Petri nets are the **formal substrate**
that lets you reason about a process precisely:

- *Mathematically defined* — no ambiguity about what each shape
  means.
- *Composable* — you can join two Petri nets at a shared place
  and get a valid larger Petri net, which is how PETRA handles
  multi-process collaborations.
- *Analysable* — there are well-understood algorithms for checking
  whether a Petri net deadlocks, terminates, or behaves like
  another. PETRA uses several of these.

You don't have to draw Petri nets by hand. PETRA reads BPMN
directly and translates internally — the next section shows the
correspondence.

---

## 3. From your BPMN diagram to a Petri net

PETRA's BPMN parser turns each BPMN element into the equivalent
Petri-net structure. You keep working in BPMN; the Petri-net
representation is built behind the scenes.

| BPMN element | What it represents | Petri-net translation |
|---|---|---|
| **Task** | A step of work (e.g. *"review application"*) | One transition, with one input place and one output place. |
| **Start event** | The entry point of the process | One source place (no input arc) holding an initial token. |
| **End event** | The termination point | One sink place (no output arc). |
| **Exclusive (XOR) gateway** | A decision — exactly one path is taken | One input place feeding multiple transitions, each leading to a different downstream branch. |
| **Parallel (AND) gateway** | A split where all branches run | One transition with multiple output places (split), or multiple input places feeding one transition (join). |
| **Sequence flow** | The arrow between elements | An arc through the corresponding place. |
| **Pool** | A participant in a collaboration (e.g. a department, a counterparty) | A self-contained Petri net for that participant. |
| **Message flow** | A message crossing pool boundaries | A *message place* connecting the sending transition to the receiving transition across pools. |

So a BPMN process you've drawn in Camunda Modeler or Bizagi gets
turned into a Petri net automatically. The names of the tasks,
gateways, and events carry through as labels on the corresponding
transitions and places, so anything PETRA tells you afterwards
refers back to your original BPMN vocabulary.

### Why this matters

Most ML systems applied to processes treat your BPMN diagram as
*reference documentation* — they learn from logs, ignore the
diagram, and report things in terms of opaque internal indices.
PETRA's structure of weights is **exactly** the structure of your
BPMN diagram. When PETRA later tells you *"the credit-check step
didn't fire"*, that's the BPMN task named *credit-check*, not a
guess.

---

## 4. The token game: how a Petri net "runs"

A Petri net's behaviour is the set of all token movements it
permits — the **token game**. The rules are:

1. A transition is **enabled** if every input place has at least
   one token.
2. An enabled transition may **fire** — when it does, it consumes
   one token from each input place and produces one token in each
   output place.
3. Multiple transitions may be enabled at once. Which fires first
   isn't determined by the Petri net alone — that's a property of
   the *execution log*, which records what actually happened on a
   given run.

The *initial marking* is the starting configuration — the set of
places that hold a token before anything has fired. A net plus an
initial marking defines a *reachable state space* — the collection
of all token configurations the net can reach. For business
processes this is usually finite; PETRA enumerates it when needed.

### Why the token game is useful

- It pins down what *"the process can do"* without ambiguity.
- It distinguishes the *structure* (what's possible) from the
  *behaviour* (what actually happened on a given run, recorded in
  the log).
- It lets PETRA compare two structures (does net A do everything
  net B does?) by comparing their reachable state spaces.

---

## 5. What PETRA adds to the picture

PETRA isn't just a Petri-net library. The classical Petri-net
world has been around for sixty years. PETRA adds **learning**:
you give it a Petri net **and** an execution log, and PETRA fits
a model of how the process *actually* behaves to the log, in a way
that stays interpretable at the granularity of your original
diagram.

The following sections explain the framework's modelling features
— the ways in which PETRA's Petri nets go beyond the bare
formalism so they can represent realistic processes.

---

## 6. Coloured tokens: when work carries data

A plain Petri-net token is just a marker — *"a unit of work is
here"*. But a real loan application carries data: the *amount*,
the *applicant ID*, the *risk score*. A real packaging line carries
the *bottle size*. A real signalling pathway carries the *signal
strength*.

**Coloured Petri nets** (sometimes called *CPN*) let each token
carry a value. PETRA supports a *scalar* form — each token carries
a single number — which is enough for most business decision rules.

### What you can do with coloured tokens

- **Route decisions on the value the token carries.** A transition
  can have a *guard* like *"only fire if the input token's amount is
  ≥ £1,000"*. The token's value travels structurally; the routing
  decision is declarative.
- **Have PETRA learn the threshold from data.** When the guard is
  declared in the simple form *(place, operator, value)* — say
  *(p_application, >=, 1000)* — PETRA's compiler builds a learnable
  threshold initialised at 1,000. Training on your logs refines the
  threshold to whatever your actual decision-makers were using.
  The [`credit_approval_coloured`](../examples/credit_approval_coloured/)
  scenario demonstrates this end-to-end: starting from 1,000, the
  threshold learns to a value in the observed decision band 900–1,500.
- **Express routing logic the simple form can't.** When the rule is
  *"approve if amount > credit-limit and credit-score > 600"* — two
  inputs, compound — you can supply a custom function that the
  compiler runs differentiably.

### Why this matters

Without coloured tokens, you'd have to flatten *amount* into a
*"high value"* / *"low value"* place activation in [0, 1] and
hope training rediscovers the threshold from log frequencies. With
coloured tokens, the actual amount travels with the token and the
threshold is *learned from data on the threshold's own scale.*

---

## 7. Multi-token markings: batches and queues

Plain Petri nets allow at most one token per place. Real processes
sometimes don't: a bottling line waits for **six** bottles to
accumulate before sealing a crate; a courier batches **N** packages
into one delivery run; an audit holds **all twelve** quarterly
reports before signing off.

PETRA supports **multi-token markings** through *arc weights*. An
arc weight of 6 means *"this firing consumes (or produces) six
tokens, not one."* The transition only becomes enabled when its
input place has accumulated enough tokens to satisfy every input
arc's weight.

The [`batch_packaging`](../examples/batch_packaging/) scenario
demonstrates this: a *bottle-to-crate* transition with input arc
weight 6 waits for six bottles in the *bottles-pending* place
before firing once and producing one *crate*.

### Why this matters

- **Batching is a first-class concept.** You don't have to model
  six bottles as six separate single-token transitions.
- **Bounded queues map naturally.** A buffer that holds up to N
  items is a place with N as its arc weight to the consuming
  transition.

---

## 8. Inhibitor arcs: mutual exclusion and negative preconditions

Sometimes a step shouldn't fire **unless something else is absent**.
A lock-acquire shouldn't fire unless the lock is currently free. A
loan can't be approved while it's already on hold. A reactor can't
start while a coolant alarm is active.

**Inhibitor arcs** express this. An inhibitor arc from place P to
transition T means *T can only fire when P is empty*. The
inhibitor place is **not consumed** by the firing — it's purely a
guard.

The [`resource_lock`](../examples/resource_lock/) scenario
demonstrates this with a two-client mutex: each client's
*lock-acquire* transition has an inhibitor arc from the
*lock-held* place. When client A holds the lock, client B's
acquire transition is structurally suppressed.

In the trained network, the inhibitor effect is implemented as a
multiplicative gate *(1 − activation(P))* — when the inhibitor
place is fully active, the gate multiplies the transition's
firing strength by zero.

### Why this matters

- **Mutex, semaphore, and other negative-precondition patterns**
  are first-class.
- **Compliance constraints** map naturally: *"a refund can't fire
  while a fraud investigation is open"* is an inhibitor arc from
  *investigation-open* to *refund*.

---

## 9. Transition durations: steps that take time

A plain Petri-net transition fires *instantly* — input tokens
disappear, output tokens appear, in the same step. Real processes
don't work that way. Paint takes three hours to cure; an underwriter
needs two days to review a complex case; a CI pipeline takes 45
minutes to run.

PETRA's **transition durations** let you say *"this transition,
once it fires, takes D time-steps to produce its output."* The
inputs are consumed at firing time, but the outputs only appear D
steps later. While the transition is *in progress*, the work is
invisible to other transitions — it can't be consumed because it
hasn't been produced yet.

The [`paint_shop`](../examples/paint_shop/) scenario demonstrates
this: the *cure* transition has duration 3. Parts that enter the
cure step at time *n* don't appear at the *cured* place until
time *n + 3*.

### Why this matters

- **Modelling cycle time.** You can express *"this step takes
  longer than that one"* directly.
- **Modelling staged processing.** A multi-day waiting period is a
  single transition with a duration, not a chain of dummy steps.

---

## 10. Firing rates: priorities and propensities

Two competing dispatchers, two competing approval routes, two
competing handlers. They're structurally identical (same shape of
arcs) but in practice one is *preferred* — high-priority dispatch
fires more eagerly for the same input.

PETRA's **firing rates** let you encode this prior. Each transition
has a rate multiplier; the higher the rate, the more eagerly the
transition fires on the same input. The rate is a *prior*: training
still refines it from data, so if your stated priority doesn't
match the observed dispatch pattern, the model learns the truth.

The [`priority_dispatch`](../examples/priority_dispatch/) scenario
demonstrates this with three handlers at rates 3.0 / 1.0 / 0.5 —
the rate-3 handler fires roughly three times as eagerly as the
rate-1 handler under identical conditions.

### Why this matters

- **You don't have to wait for training** to encode obvious
  domain knowledge ("urgent cases jump the queue").
- **The prior is testable** — if training pulls the rate down, that
  means the data disagrees with the stated priority, which is
  itself a finding.

---

## 11. Cross-pool composition: multi-process collaborations

Most BPMN diagrams have more than one pool — a customer interacting
with a bank, two services in a distributed protocol, three agents
in a coordination dance. The pools talk to each other via
*messages*.

PETRA composes pools through **shared message places**. Each pool
becomes its own internal Petri net (places and transitions all
prefixed by the pool name to avoid collisions). A message flow
*from sender's task to receiver's task* becomes a *message place*:
the sender's transition produces a token in the message place; the
receiver's transition consumes that token.

The [`distributed_consensus`](../examples/distributed_consensus/)
scenario demonstrates this with two-phase commit modelled as one
coordinator pool and two cohort pools, composed through
*prepare*, *vote*, and *commit* message places.

The [`multi_agent_coordination`](../examples/multi_agent_coordination/)
scenario goes further: three pools (auctioneer + two bidders)
exchanging six message types in a contract-net protocol.

### Why this matters

- **End-to-end visibility.** You can analyse *"does the bank's
  process compose correctly with the credit-bureau's process?"*
  rather than treating each in isolation.
- **Token conservation across boundaries.** PETRA tracks that
  messages produced by one pool are exactly the messages consumed
  by the other, so a mismatch (a message produced and never
  consumed) is structurally visible.

---

## 12. What PETRA learns from your logs

So far this guide has covered **the structure** — what you tell
PETRA about your process. Now the other half: **what PETRA does
with your execution logs.**

A log (XES, CSV, JSON, or inline in a configuration file) is a
collection of *traces*. Each trace is one execution of the
process — for a loan-approval log, one trace per loan
application. Each trace contains:

- **Attributes** — properties of the whole instance: *amount*,
  *applicant ID*, *channel*, *date*.
- **Events** — the steps that actually fired, in order, with any
  per-event attributes (*reviewer*, *outcome*, *duration*).

PETRA fits a **trained model** of the process to the log. The
trained model has exactly the same structure as the Petri net you
provided — one trainable weight per arc, one trainable threshold
per transition, and one learnable threshold for each *(place,
operator, value)* guard. Nothing else can be learned. The
architecture **is** the verified Petri net.

The training signal: each trace tells PETRA *which transitions
actually fired in that instance*. PETRA adjusts its weights so its
predicted activations match the observed firings, across all
traces in the log.

### What the trained model captures

- **Routing decisions.** When the XOR gateway in your BPMN
  diagram routes 70% to *credit-review* and 30% to *fast-track*,
  the weights along those arcs reflect that split.
- **Synchronisation rules.** When an AND-join only fires when all
  inputs are present, the threshold on the join transition
  reflects that.
- **Coloured thresholds.** When the *(amount, >=, 1000)* guard is
  the boundary that decision-makers use in practice, the learned
  threshold converges to that value.
- **Priority adjustments.** When the rate priors didn't quite
  match the observed dispatch, the weights compensate.

---

## 13. From learned weights to readable rules

A trained model is useful only if you can *read* what it learned.
PETRA includes a **rule extractor** that turns the learned weights
back into business vocabulary.

Two main rule shapes:

### XOR routing rules

For every XOR-shaped decision (one input place feeding multiple
competing transitions), PETRA extracts a *crossover threshold* —
the input value at which one branch becomes more active than the
other.

*Example:* in a credit-approval process with two routing
transitions, *route-to-quick-approval* and *route-to-credit-review*,
sharing an input value *risk-score*, PETRA might extract:

> *If risk-score < 0.486, route to quick-approval; otherwise route
> to credit-review.*

The 0.486 isn't a number you typed in. It was distilled from the
trained weights and matches the empirical decision boundary your
underwriters were applying — *in their own vocabulary.*

### AND-join rules

For every synchronisation (multiple input places feeding one
transition), PETRA extracts a *quorum* — how many inputs need to
be active for the join to fire.

*Example:* an audit transition that synchronises three
sub-reviews. PETRA might extract:

> *Audit fires when all three sub-reviews are complete.*

Or:

> *Audit fires when at least 2 of 3 sub-reviews are complete.*

The threshold is read from the learned weights. If the model
learned to fire on a majority rather than unanimity, the rule
extraction reflects that.

### Why this matters

The deepest critique of ML in business processes is the
**black-box problem**: a learned model spits out predictions but
nobody can read them. PETRA closes that loop. Every learned
parameter has a name from your domain, and the rule extractor
turns the parameters back into the prose form that an underwriter,
compliance officer, or process owner can read.

---

## 14. Spotting traces that don't fit

Once PETRA has trained on your logs, it can score *new* traces
against the learned model. A trace where every step fired exactly
as the model predicts gets a near-zero **anomaly score**. A trace
where a step that should have fired didn't — or a step that
shouldn't have fired did — gets a non-zero score, **pinned to the
specific transitions that disagreed**.

### What this gives you

| Output | Example |
|---|---|
| **Trace-level score** | *"This loan application has anomaly score 0.83 — high."* |
| **Per-transition residuals** | *"The credit-check step in this trace fired with confidence 0.05 against an expected 0.95 — likely skipped."* |
| **Rankings over a batch of traces** | *"Of the 1,200 applications processed last week, these 23 are the most anomalous; here's why each one diverged."* |

### Why this matters

Traditional anomaly-detection on process logs gives you
**trace-level scores** — *this trace is unusual* — without telling
you *why*. PETRA's residuals are at the **transition level** —
which step diverged, by how much, in either direction (fired when
shouldn't have, or didn't fire when should have). A compliance
officer can read a flagged trace, jump to the specific named
transition, and decide whether it's a process bug, a fraud signal,
or a benign edge case.

---

## 15. Asking "are these two variants the same?": strong bisimulation

A common question in process redesign: *"if we change the process
this way, does it still behave the same?"* Today, the answer is
usually *"run a shadow pilot for six months and see."* PETRA can
answer mechanically.

**Strong bisimulation** is a formal property of two Petri nets:
two nets are strongly bisimilar if they exhibit the same labelled
behaviour from their initial states — every transition one can
take is matched exactly by a labelled transition the other can
take, recursively, forever.

In practice this means: if PETRA says *"net A and net B are
strongly bisimilar"*, you have a **mathematical proof** that the
two variants are observably indistinguishable. They might be drawn
differently, have differently-named internal steps, or decompose
their gateways differently — but their observable behaviour is
identical.

This is the *core engine* of provably-safe refactoring. You can
propose a redesign, run bisimulation, and either:

- *Pass* — the redesign is provably equivalent; proceed.
- *Fail* — the redesign changes behaviour somehow; investigate which
  transitions diverge.

### Why this matters

Currently nobody proves process equivalence — they *test* it, by
running both variants and checking the outputs match. PETRA closes
that gap. The proof is a compile-time check, not a multi-month
shadow run.

---

## 16. Asking that question more leniently: weak bisimulation

Strong bisimulation is sometimes too strict. Consider a redesign
that adds an **internal logging step** — a transition that fires
just to record an audit entry, with no observable effect on the
process. Strong bisimulation says the two variants are *different*,
because the new variant has an extra transition the original
doesn't.

Most of the time, that's not what you want. The whole point of
the refactoring is that the logging step *shouldn't* count as a
behaviour change — it's an internal implementation detail.

**Weak bisimulation** solves this. You mark the logging
transition as *silent* (PETRA calls these *τ-transitions* after
the standard mathematical notation), and the equivalence check
collapses silent steps before the comparison.

### When weak bisimulation matters

| Refactoring pattern | Strong says | Weak says |
|---|---|---|
| Add an internal logging step | Different | Equivalent |
| Replace one no-op gate with a chain of two no-op gates | Different | Equivalent |
| Swap the order of two strictly internal handoffs | Different | Equivalent |
| Add an extra approval level (visible behaviour) | Different | Different (correctly) |
| Replace one visible step with a different one | Different | Different (correctly) |

The key feature: silent transitions are **only** silent to the
equivalence check. They still fire in the token game, still get
trained against the log, still get monitored for anomalies — the
silence is purely a property of the *equivalence comparison*.

### Why this matters

Weak bisimulation is what makes the cost-ranked refactoring story
actually work for real redesigns. In real life, almost every
variant differs from the reference by some internal detail nobody
cares about. Strong bisimulation would reject all of them; weak
bisimulation accepts the ones that are *behaviourally* equivalent.

---

## 17. Choosing between equivalent redesigns by cost

Bisimulation tells you *"these two redesigns do the same thing."*
The next question is *"which one costs less to run?"* PETRA
answers this by assigning per-step costs and computing the
**expected cost-to-completion** under each variant, against your
actual log distribution.

The recipe:

1. *Decide what cost means.* Wall-clock time, monetary cost,
   resource hours, energy, whatever you care about — assigned per
   transition.
2. *Train each variant on the same log.* This grounds the firing
   probabilities in your actual workload, not a stipulated model.
3. *Multiply firing probability by cost per transition, sum.* That
   gives expected cost-to-completion for an average instance.
4. *Rank.* The cheaper variant wins.

The [`cost_ranked_refactoring`](../examples/cost_ranked_refactoring/)
scenario demonstrates this on two variants of a loan-approval
process. Bisimulation proves they are equivalent; cost-ranking
shows that Variant B costs roughly one-sixth of Variant A on the
observed workload, while doing provably the same thing.

### Why this matters

This is the combination that nobody else has running. **Process
mining** tells you what happened. **Classical formal methods**
prove properties about a stipulated spec. **PETRA** does both at
once and adds *learned-from-actual-data cost ranking* on top.

For a business: this is the difference between *"the consultant
thinks Variant B is cheaper"* (judgement, expensive, slow) and
*"Variant B is mathematically equivalent to Variant A and 6×
cheaper on the workload we've actually observed"* (proof,
repeatable, audit-trail-friendly).

---

## 18. The ecosystem: where your data and models come from

PETRA's job is to **train, distil, score, and verify**. It is
deliberately **not** a structure-discovery tool, a modelling GUI,
or a workflow runtime. Other tools do those jobs well. PETRA
plugs into them through standard formats:

| Format | Where it comes from | What PETRA does with it |
|---|---|---|
| **BPMN** (`.bpmn` files) | Camunda Modeler, Bizagi, Signavio, any BPMN editor | Read it as the process structure. |
| **PNML** (Petri Net Markup Language) | CPN Tools, GreatSPN, TINA, ProM, Snoopy — any classical Petri-net tool | Read or write it as the process structure. |
| **SIF** (Simple Interaction Format) | Pathway Commons (Reactome, BioCyc, PID, etc.) | Read it as a biological pathway, mapping each entity to a place and each interaction to a transition. |
| **XES** (Extensible Event Stream) | Standard process-mining log format, used by Disco, ProM, Celonis exports | Read it as the execution log. |
| **CSV** | Flat-table process logs from any data warehouse or process-mining tool | Read it as the execution log; configurable case/activity columns. |
| **JSON** | Programmatic log dumps | Read it as the execution log. |

So you don't need to retool your existing pipeline to use PETRA.
Your BPMN files, your XES logs, your PNML exports — they all feed
in directly.

---

## 19. Putting it all together: the end-to-end pipeline

The pieces above compose into a single analytical pipeline. The
[README's bank-loan walkthrough](../README.md#using-the-whole-toolchain-together)
shows this concretely: two regional offices, two execution logs,
no documented "correct" process, and the chain of tools needed to
unify them.

| Stage | Tool | What it produces |
|---|---|---|
| **1. Discover** | ProM (process-mining tool, classical) | A Petri net inferred from each office's log. *You now have a structural model where before there was none.* |
| **2. Verify soundness** | CPN Tools | Proof that each discovered net is sound — terminates properly, doesn't deadlock, doesn't leak tokens. *The discovered models are well-formed Petri nets.* |
| **3. Throughput bounds** | GreatSPN | Closed-form maximum throughput for each office under stochastic firing assumptions. *Office A can handle 250/day, Office B 180/day.* |
| **4. Temporal invariants** | TINA | Model-checking proofs that regulatory invariants (*"every approved loan fires the audit-log transition"*) hold. *Office A passes; Office B violates the audit invariant on some paths.* |
| **5. Learn the dynamics** | PETRA | The four headline outputs: rules, anomalies, equivalence proofs, cost rankings. |

PETRA's contribution at stage 5:

1. **Train each net on its office's log** — the trained model
   captures how each office *actually* handles applications.
2. **Distil readable rules** — *Office A approves at amount >
   £5,000 with a strict credit-check gate; Office B at amount >
   £8,000 with a more lenient gate.*
3. **Run bisimulation between the two trained nets** — *not
   equivalent; the offices follow different rules.*
4. **Score held-out applications for anomalies** — *list the
   specific traces that violated the audit invariant in Office B.*
5. **Rank two proposed unified designs by cost** — *with
   bisimulation proving each is equivalent to a reference variant
   in behaviour, rank them by realised execution cost on the
   combined log distribution.*

The output is something you can act on: an evidence-backed
comparison, a verified equivalence claim (or proof that one
doesn't hold), a cost-ranked redesign, and a list of
compliance-flagged traces to investigate.

None of these outputs requires a multi-month consulting
engagement to produce. They are mechanical, repeatable, auditable
artefacts.

---

## Where to go next

- *I want to see a concrete scenario in my domain.* Browse
  [`examples/`](../examples/) — each scenario has its own README
  explaining what it demonstrates and why.
- *I want the technical API reference.* See [`DEV_MANUAL.md`](DEV_MANUAL.md).
- *I want the framing and the long-form roadmap.* See [`ROADMAP.md`](ROADMAP.md).
- *I want to read the source.* Start at
  [`petri_net_nn/petri_net.py`](../petri_net_nn/petri_net.py) — the
  data model — then
  [`petri_net_nn/compiler.py`](../petri_net_nn/compiler.py) for the
  compilation step, then
  [`petri_net_nn/bisimulation.py`](../petri_net_nn/bisimulation.py)
  for the equivalence checker.

---

## Glossary

A short reference for the terms used in this guide.

| Term | Meaning |
|---|---|
| **AND-join** | A transition that synchronises multiple input flows. All input places must have tokens for the transition to fire. |
| **Anomaly score** | A measure of how poorly a trace fits the trained model, pinned to specific named transitions. |
| **Arc** | A directed connection between a place and a transition (or vice versa) in a Petri net. |
| **Arc weight** | The number of tokens an arc consumes or produces per firing. Default 1; higher values express batching. |
| **Bisimulation** | A formal property that two systems exhibit identical observable behaviour from corresponding states. |
| **BPMN** | Business Process Model and Notation — the standard visual notation for business processes. |
| **Coloured Petri net (CPN)** | A Petri net where each token carries a value. PETRA supports a scalar form. |
| **Cost-ranked refactoring** | Choosing between provably-equivalent process variants by realised-execution cost. |
| **Firing** | The event of a transition consuming input tokens and producing output tokens. |
| **Firing rate** | A per-transition multiplier indicating how eagerly the transition fires. Higher rate = more eager. |
| **Guard** | A precondition on a transition's firing that reads the values carried by input tokens. |
| **Inhibitor arc** | An arc from a place to a transition expressing *"the transition can fire only if this place is empty."* |
| **LTS** (Labelled Transition System) | The graph of all reachable states of a Petri net, with edges labelled by the transitions that move between them. |
| **Marking** | A configuration of tokens across the places of a Petri net. |
| **Petri net** | A formal model of a discrete-event system, made of places, transitions, arcs, and tokens. |
| **Place** | A circle in a Petri net — represents a state or condition. |
| **PNML** | Petri Net Markup Language — the standard XML format for exchanging Petri nets between tools. |
| **Pool** | In BPMN, a participant in a collaboration. PETRA composes pools through shared message places. |
| **Reachability graph** | The graph of all markings the net can reach from its initial marking, with transitions as edges. |
| **SIF** | Simple Interaction Format — the tab-separated format Pathway Commons uses for biology pathways. |
| **Silent transition** (also τ) | A transition treated as invisible by the weak-bisimulation checker. Used for logging, no-op gates, internal handoffs. |
| **Strong bisimulation** | Two nets are strongly bisimilar if every transition each can take is matched exactly (label-for-label) by a transition the other can take. |
| **Token** | A dot in a place — represents one unit of work currently in that state. |
| **Trace** | One recorded execution of a process — a sequence of events for a single instance. |
| **Transition** | A rectangle in a Petri net — represents a step or event that moves work forward. |
| **Transition duration** | How many time-steps elapse between a transition firing and producing its output. |
| **Weak bisimulation** | Like strong bisimulation but tolerant of silent transitions — internal-only steps don't break the equivalence. |
| **XES** | Extensible Event Stream — the standard XML format for process-mining execution logs. |
| **XOR gateway** | An exclusive-choice decision: exactly one outgoing branch is taken. |
