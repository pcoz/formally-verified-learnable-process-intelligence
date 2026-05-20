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
  pipeline the [README](https://github.com/pcoz/formally-verified-learnable-process-intelligence#readme) walkthrough describes.

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
firing-propensity parameters that capture how the process *routes
through its structure*, in a way that stays interpretable at the
granularity of your original diagram.

### Three layers of guarantee

PETRA combines formal-methods machinery with learned dynamics. It
is worth being explicit about which guarantees come from where,
because the three layers carry different kinds of confidence:

| Layer | What it covers | What the guarantee is |
|---|---|---|
| **1 — the Petri-net substrate** | The structure itself: places, transitions, arcs, the reachable-marking graph. | **Mathematical proofs.** Bisimulation, Aalst soundness, deadlock localisation, CTL temporal-logic checking. These hold *regardless of any training* — they're properties of the net. |
| **2 — the compiled neural topology** | The trainable network's weight set. | **Preserved by construction.** Exactly one trainable weight per arc, one trainable threshold per transition, no parameters outside the flow relation. The layer-1 structural constraint carries through verbatim. |
| **3 — the learned parameters** | The values of those weights and thresholds after training. | **Empirical.** Rule extraction reads them; bootstrap CIs report their stability under data resampling; counterfactuals and sensitivity make the dependence on inputs auditable; cross-variant comparison reports empirical agreement. These are *evidence*, not *proof*. |

The distinction matters when you're talking to a regulator or a
model-risk committee. Saying *"the Petri net is sound"* is a
proof. Saying *"the trained threshold for the credit-check is
0.486 ± 0.02 with 98% direction-agreement across 100 bootstrap
resamples"* is strong empirical evidence — but it isn't a proof
in the layer-1 sense. PETRA gives you both kinds of confidence
in one tool, which is unusual; what's important is not to
conflate them.

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
  The [`credit_approval_coloured`](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/credit_approval_coloured)
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

The [`batch_packaging`](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/batch_packaging) scenario
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

The [`resource_lock`](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/resource_lock) scenario
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

The [`paint_shop`](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/paint_shop) scenario demonstrates
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

The [`priority_dispatch`](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/priority_dispatch) scenario
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

The [`distributed_consensus`](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/distributed_consensus)
scenario demonstrates this with two-phase commit modelled as one
coordinator pool and two cohort pools, composed through
*prepare*, *vote*, and *commit* message places.

The [`multi_agent_coordination`](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/multi_agent_coordination)
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

### What the training does (and doesn't) actually learn

This is worth being precise about, because the framing "PETRA
learns how the process behaves" can be read more broadly than
the training objective actually supports.

What PETRA's training **does** learn:

- **Per-transition firing propensities** — for each transition
  in the net, the probability it fires in an instance, conditional
  on the *input marking* derived from that instance's attributes.
- **Routing-decision rules** — when several transitions share an
  input place and the data conditions which one fires, the
  learned weights and thresholds recover that decision rule.
- **AND-join quorums** — when a synchronisation step waits for
  some-or-all of its inputs, the learned threshold captures
  the rule.
- **Coloured-Petri-net guard thresholds** — when a transition
  guard is declared structurally (`{place, op, value}`), the
  value is initialised at the modeller's declared boundary and
  refined from data.

What PETRA's training **does not** learn directly:

- **Step-by-step temporal dynamics.** The training objective is
  *which transitions fired in this instance* (a per-transition
  occurrence target), not *in what order they fired*. The
  underlying structure constrains valid orderings — only enabled
  transitions can fire under the token-game semantics — but the
  trained model isn't itself a sequence model.
- **Inter-step duration distributions.** Transition durations
  exist as a structural input (`add_transition(..., duration=N)`)
  that affects the time-unrolled compiler; they are not fitted
  from log timestamps.
- **Probabilities of trace prefixes or full event sequences.**
  PETRA doesn't compute "probability that this trace prefix
  continues with X next." That's a sequential-likelihood
  question for a different class of model.

This is enough to recover the most common decision and
synchronisation rules a process model carries — which is why
the rule-extraction and anomaly-detection outputs work — but it
is a more modest claim than "fully learns the process
dynamics."

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

### Confidence intervals on the extracted rule

A point-estimate rule (*"the threshold is 0.486"*) is fine for an
internal analyst's note, but a regulator-facing claim needs to
know **how stable the rule is** under reasonable variations in the
data. PETRA computes this via **bootstrap resampling**:

1. Resample the trace list *with replacement* N times (default
   100). Each resample is a plausible alternative dataset of the
   same size.
2. Train a fresh model on each resample. Extract the rule.
3. Report the **distribution** of the rule parameter (the
   crossover threshold for an XOR rule, the quorum threshold for
   an AND-join). The 2.5 / 97.5 percentiles give a 95% confidence
   interval.
4. Report the **direction agreement** — what fraction of the
   bootstrap samples produced a rule pointing the same way as the
   point estimate. If the routing direction flips on 30% of
   resamples, the rule is not stable enough to deploy.

A typical CI-augmented rule looks like:

> *When* application amount *is above £1,000 (95% confidence
> interval [£950, £1,070] over 100 bootstrap resamples), the
> trained model routes to* Credit Review *rather than* Fast
> Track. *The direction was consistent across* 98% *of the
> resamples.*

That's the form a regulator, an audit team, or a model-risk
committee can sign off on.

### Plain-English prose

Both the point estimate and the CI-augmented form can be
rendered as paragraph-form **prose** for inclusion in a report
or model-risk document. The prose helpers also let you
substitute a domain term (*"application amount"*) for the raw
internal place id (*"p_application"*) — important for
documents that go outside the engineering team.

### Counterfactual explanations

The CI tells you how stable the *rule* is. The next question a
customer-facing case worker or a regulator might ask is about a
*specific* decision: **"The model declined this loan. What would
the applicant have needed to change to be approved?"**

PETRA answers this with **counterfactual analysis**. Given a
trained model, a specific input scenario, and a target step
whose outcome you want to flip, the analysis binary-searches
one input attribute (e.g. *application amount*, *risk score*,
*credit limit*) to find the smallest change that would have
flipped the firing of that step.

For the credit-approval scenario, asking *"what amount would
have got this £100 application approved?"* returns something like:

> *To flip the outcome at* approve loan, *the input application
> amount would need to be increased from 100.000 to 1024.000. At
> that point the step would have started firing (activation
> changes from 0.05 to 0.50).*

The answer is *grounded in the trained model* — not a hand-set
threshold, not a stipulated rule. It tells the case worker the
specific number, in the relevant domain unit, that would have
flipped this specific instance.

### Why counterfactuals matter

Three audiences use this directly:

- **Customers and applicants.** *"Your application was declined
  because your amount of £100 is below the £1,024 boundary the
  model has learned. If your eligible amount were at or above
  that, the decision would have been to approve."* — auditable,
  actionable, individual-level.
- **Compliance officers.** Counterfactuals on flagged anomalies
  ("the credit-check step didn't fire — what input value would
  have triggered it?") localise the analysis to a single
  pivotable input rather than a vague heuristic.
- **Regulators.** A counterfactual is testable: change the input,
  re-run the model, observe the documented outcome. Together with
  CIs on the underlying rule, it's the form of explanation modern
  model-risk regimes (e.g. SR 11-7 in the US, PRA SS1/23 in the
  UK) require.

### Sensitivity: which input matters most?

Counterfactuals tell you what change to *one specific input*
would have flipped *one specific decision*. The natural prior
question is: **which input should I be looking at in the first
place?**

PETRA's **sensitivity analysis** computes the local rate of
change of a target step's firing activation with respect to each
input. Larger magnitude = the firing decision pivots more on
that input at this base point. Two scopes:

| Scope | Question it answers |
|---|---|
| **Local sensitivity at a single instance** | *For this specific loan application, which inputs is the model leaning on?* |
| **Aggregate input importance across a trace set** | *Across the whole population of cases we've trained on, which inputs is the model leaning on most overall?* |

A typical sensitivity output for a single instance reads:

> *At the base point, the firing of* approve loan *(activation
> 0.50) is most sensitive to:*
>
> *1.* application amount *(per-token value) — gradient +0.180;
> increasing it raises the firing.*
> *2.* applicant credit score *— gradient +0.045; increasing it
> raises the firing.*
> *3.* prior defaults *— gradient −0.012; increasing it lowers
> the firing.*

The numbers are *local gradients*: they tell you how much the
firing decision changes for a small change in each input. Inputs
with near-zero gradient are locally irrelevant — either the
model genuinely doesn't use them, or the case is in a saturated
regime far from the decision boundary.

Together with rules, CIs, and counterfactuals, this completes
the diagnostic toolkit:

| Question | Tool |
|---|---|
| *Which inputs does the model lean on?* | **Sensitivity** |
| *What's the rule it applies to that input?* | **Rule extraction** |
| *How stable is that rule?* | **Bootstrap CIs** |
| *What change to the input would flip this specific decision?* | **Counterfactual** |
| *Why is this trace flagged as unusual?* | **Anomaly score + prose explanation** |
| *Across the input domain, how often do these two variants agree?* | **Cross-variant comparison** |

### Cross-variant comparison: where do two variants actually agree?

Bisimulation proves two variants of a process are **behaviourally
equivalent** — every action the one can take, the other can match
exactly. That's the structural guarantee. The practical
follow-up question is more granular: **across the actual input
range a process sees in production, on what fraction of inputs do
the two variants make the same firing decisions, and where exactly
do they diverge?**

PETRA's cross-variant comparison samples the input space (a
Cartesian-product grid the modeller chooses), runs both trained
variants at each grid point, pairs corresponded transitions by
label (or by an explicit override when labels don't line up), and
reports:

- **Hard agreement rate** — what fraction of `(grid point,
  transition)` combinations the two variants reach the *same
  firing decision* (both above 0.5 or both below).
- **Soft agreement rate** — what fraction have activations
  *within tolerance* of each other (a stricter check that
  measures *how close* the two variants' confidences are, not
  just whether they reach the same decision).
- **Per-transition agreement** — which transitions agree the
  most reliably and which are the most prone to diverge. Useful
  for narrowing down where the variants behave differently.
- **Specific divergent samples** — the actual input points and
  the activations at those points. Lets a reviewer inspect *why*
  the variants disagreed in those bands.
- **Unmatched labels** — transitions present in one variant but
  not the other, surfaced as a flag rather than silently dropped.

A typical comparison report reads:

> *Variants compared across 121 input point(s); 6 transitions
> corresponded by label. Hard agreement (same firing decision):
> 87.3%. Soft agreement (activations within 0.10): 82.6%.
> Transitions most prone to disagreement (top 3): 'credit-review':
> agree on 71.4%; 'approve loan': agree on 78.5%; 'fast-track':
> agree on 94.2%.*

That kind of report sits naturally next to a bisimulation proof
and a cost-ranking analysis in a refactoring proposal: *"the
new variant is provably equivalent, agrees with the reference on
87% of the observed input domain, and costs 6× less to run".*

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

## 17. Is this process structurally sane? Soundness and deadlock checks

Before you train, before you compare variants, before you cost-rank
anything, there's a more basic question: *does this process model
even work?* PETRA's **soundness checker** answers it.

### What soundness means

A process model is **sound** (in van der Aalst's classical
formulation, the standard for workflow nets) when three properties
all hold:

| Property | Plain English | Failing case |
|---|---|---|
| **Option to complete** | From any reachable state, the process *can* reach the final state. | A state the process can land in but can never progress out of. *A one-way trip into a dead end.* |
| **Proper completion** | When the process reaches the final state, no work is still hanging around in other steps. | The final step fires but tokens are still parked in some other branch — *the audit signed off while work is still in progress*. |
| **No dead transitions** | Every step in the model is reachable — the model has no orphan transitions. | A step that's drawn in the diagram but nothing ever actually triggers it — *the modeller forgot a flow*. |

### What PETRA reports

`check_soundness(net)` returns a structured report:

- **Incomplete markings.** The specific states from which the
  final marking is unreachable — *option to complete* failures.
- **Lingering-token markings.** The specific states where the
  sink has tokens but other places still do too — *proper
  completion* failures.
- **Dead transitions.** A list of the transition names that are
  never enabled — *no dead transitions* failures.

If all three lists are empty, the net is sound and you can move
on to training, equivalence checking, and refactoring with
confidence the structural model is well-formed.

### Deadlock localisation

A close cousin: `find_deadlocks(net)` returns the **specific
states** where the process is stuck — states with no enabled
successors that aren't the intended final state. These are the
*root causes* of an option-to-complete failure: not "this net
can't complete cleanly" (the soundness summary) but "here's
exactly the token configuration where it gets stuck" (the actionable
detail).

### Why this matters

Currently, most BPMN tools catch only *syntactic* errors — *"this
arrow doesn't have a target", "this gateway has no condition"*.
Behavioural soundness — *does the process actually work as a
control-flow system?* — almost never gets checked, because there's
no widely-deployed tool that does it automatically.

PETRA bakes soundness into the same workflow that trains, scores,
and verifies. A scenario that ships with PETRA is required to pass
the soundness check — that's a bar everything in `examples/` clears.

---

## 18. Asking temporal questions about the process: CTL

Soundness checks tell you the model is *structurally* well-formed.
Bisimulation checks tell you two models are *equivalent*. But
there's a third kind of question process people often want to
ask: *does the process satisfy a specific temporal rule?*

Examples a compliance officer would recognise:

- *"Every approved loan eventually fires the audit-log step."*
- *"No decline can fire before the credit-check has fired."*
- *"From every state, the process can still reach completion."*
- *"There is some path along which the loan is approved within
  three steps."*

These are **temporal properties** — they're about how the process
unfolds over time, not just about a single instant. PETRA's
**CTL checker** answers them mechanically.

### What CTL is

CTL — *Computation Tree Logic* — extends ordinary logic
(*"AND"*, *"OR"*, *"NOT"*) with operators that quantify over the
tree of possible futures from each state:

| Operator | Reads as | Means |
|---|---|---|
| **AG φ** | "Always (on every path), φ" | φ holds at every reachable state, on every path. *The safety operator.* |
| **EF φ** | "Eventually (on some path), φ" | Some path reaches a state where φ holds. *Reachability.* |
| **AF φ** | "Eventually on every path, φ" | Every path eventually reaches a state where φ holds. *Liveness.* |
| **EG φ** | "Globally on some path, φ" | Some infinite path keeps φ true forever. *Existence of a stable behaviour.* |
| **EX φ** | "Some next state satisfies φ" | At least one immediate successor satisfies φ. |
| **AX φ** | "Every next state satisfies φ" | All immediate successors satisfy φ. |
| **A[φ U ψ]** | "On every path, φ until ψ" | Every path keeps φ true until ψ becomes true. *Ordering invariant.* |
| **E[φ U ψ]** | "On some path, φ until ψ" | Some path keeps φ true until ψ. |

### Worked examples for each headline question

**"Every approved loan eventually fires the audit-log step."**
This is the *response* property — a request is eventually matched
by a response. In CTL:

> *AG (approved → AF audit_log_fired)*

Read: *on every reachable state, if the loan is approved, then on
every future path some step eventually fires the audit log.*

**"No decline can fire before the credit-check has fired."** An
*ordering* property — one event must precede another:

> *A[¬decline_fired U credit_check_fired]*

Read: *on every path, decline stays unfired until the credit check
has fired.*

**"From every state, the process can still reach completion."**
A *no-deadlock* property — the process is never trapped:

> *AG EF process_complete*

Read: *at every reachable state, there is still some path to a
completion state.*

**"There is some path along which the loan is approved within
three steps."** A *bounded reachability* property:

> *EX EX EX approved*

Read: *there is a state three steps away that has the loan
approved.*

### What PETRA reports

`check_ctl(net, formula)` returns:

| Field | Meaning |
|---|---|
| `holds_at_initial` | *Yes/no answer for the property.* |
| `holds_at` | The full set of reachable states where the formula is satisfied — useful for understanding *where* the property fails. |
| `counterexample` | A specific reachable state that violates the formula, when one exists. *"Here's the state that breaks the rule"*, not just *"the rule is broken somewhere."* |

### Why this matters

The temporal-logic dimension is what classical formal-methods
tools (TINA, NuSMV, SPIN) have offered for decades. Bringing it
inside PETRA — alongside the trained model, the rule extractor,
the anomaly detector — means a compliance question and a process
question can be answered together, against the same structural
model, in the same tool. The bank-loan walkthrough's *Office B
violates the audit-log invariant* finding (today produced by
TINA) is now a one-line CTL query.

---

## 19. Choosing between equivalent redesigns by cost

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

The [`cost_ranked_refactoring`](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples/cost_ranked_refactoring)
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

## 20. The ecosystem: where your data and models come from

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

## 21. Putting it all together: the end-to-end pipeline

The pieces above compose into a single analytical pipeline. The
[README's bank-loan walkthrough](https://github.com/pcoz/formally-verified-learnable-process-intelligence#using-the-whole-toolchain-together)
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
  [`examples/`](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/examples) — each scenario has its own README
  explaining what it demonstrates and why.
- *I want the technical API reference.* See [`DEV_MANUAL.md`](DEV_MANUAL.md).
- *I want the framing and the long-form roadmap.* See [`ROADMAP.md`](ROADMAP.md).
- *I want to read the source.* Start at
  [`petri_net_nn/petri_net.py`](https://github.com/pcoz/formally-verified-learnable-process-intelligence/blob/main/petri_net_nn/petri_net.py) — the
  data model — then
  [`petri_net_nn/compiler.py`](https://github.com/pcoz/formally-verified-learnable-process-intelligence/blob/main/petri_net_nn/compiler.py) for the
  compilation step, then
  [`petri_net_nn/bisimulation.py`](https://github.com/pcoz/formally-verified-learnable-process-intelligence/blob/main/petri_net_nn/bisimulation.py)
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
| **Counterfactual** | An explanation that says what specific input change would have flipped a model's decision on a specific instance. *"If the amount had been £1,024 instead of £100, the loan would have been approved."* |
| **Cross-variant comparison** | Sampling the input space at a grid of points and tallying where two trained variants reach the same firing decision and where they diverge. Pairs naturally with bisimulation. |
| **CTL** (Computation Tree Logic) | A temporal logic for asking questions about how a process unfolds over time: *eventually*, *always*, *on every path*, *on some path*. |
| **Deadlock** | A reachable state with no enabled transitions and not the intended final state — the process is stuck. |
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
| **Sensitivity** | A measure of how much a model's decision changes when one input is nudged. Larger sensitivity = the model leans on that input more. |
| **Soundness** | Aalst's structural property of a workflow net: every reachable state can complete, completion leaves no lingering tokens, no transition is dead. |
| **Strong bisimulation** | Two nets are strongly bisimilar if every transition each can take is matched exactly (label-for-label) by a transition the other can take. |
| **Temporal property** | A claim about how a process unfolds over time, rather than about a single state — *eventually X*, *always Y*, *X until Y*. Checked using CTL. |
| **Token** | A dot in a place — represents one unit of work currently in that state. |
| **Trace** | One recorded execution of a process — a sequence of events for a single instance. |
| **Transition** | A rectangle in a Petri net — represents a step or event that moves work forward. |
| **Transition duration** | How many time-steps elapse between a transition firing and producing its output. |
| **Weak bisimulation** | Like strong bisimulation but tolerant of silent transitions — internal-only steps don't break the equivalence. |
| **XES** | Extensible Event Stream — the standard XML format for process-mining execution logs. |
| **XOR gateway** | An exclusive-choice decision: exactly one outgoing branch is taken. |
