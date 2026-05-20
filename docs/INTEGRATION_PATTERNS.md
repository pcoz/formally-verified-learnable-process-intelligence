# Integration patterns: PETRA + workflow engines

This guide is for **engineering teams who run a workflow engine**
(Camunda, Activiti, Flowable, or anything that emits structured
audit events) and want to wire PETRA into their stack — to score
production traces for anomalies, to surface routing rules, or to
provide a regulator-facing audit trail through the diagnostic
toolkit (counterfactuals, sensitivity, cross-variant comparison).

It is *not* a guide to writing a JVM-side Camunda plugin. See the
**What this is not** section below for an explicit boundary.

---

## What this enables

An engine team installs `petra-nn` and gets a working integration
without writing Python — only configuration and a small amount
of glue in whatever language the engine speaks.

The end-to-end story has four pieces, three of which the engine
team already has and one PETRA provides:

| Piece | Where it lives | What it does |
|---|---|---|
| **Audit / event source** | The engine | Camunda / Activiti / Flowable already emit structured events (BPMN task lifecycle, decision outcomes, process variables). Most teams already have this exported somewhere — a history database, a Kafka topic, an Elasticsearch index. |
| **Structural model** | The engine's BPMN deployment | PETRA's [BPMN parser](https://github.com/pcoz/formally-verified-learnable-process-intelligence/tree/main/petri_net_nn/bpmn.py) reads the same `.bpmn` file the engine deploys; no separate process model needs to be maintained. |
| **Trained PETRA model** | A `.pt` file produced by `petra-train` | One-time training step against a representative slice of the audit log. The resulting bundle (`.pt` + `.meta.json` sidecar) is the deployable unit. |
| **Production inference** | `petra-serve` or `petra-score` or `StreamingEvaluator` | Three different shapes for three different deployment models — synchronous HTTP per event, scheduled batch, or streaming subscription. |

Concretely, the workflow is:

1. Export a representative audit log to CSV (every named engine
   supports this through standard admin tools — the columns
   matter, not the source).
2. Author a small TOML scenario file pointing at the BPMN file
   and the CSV log, declaring which trace attributes feed which
   input places.
3. Run `petra-train scenario.toml -o model.pt` once. You now have
   a bundle.
4. Choose one of the three integration patterns (Section 3)
   based on how live you need the scoring to be.

---

## What this is *not*

This package is **the Python-side integration toolkit**. It does
*not* ship engine-side bindings — there's no Java / Groovy /
Kotlin code in this repository, no Camunda execution listener,
no Activiti command interceptor, no Flowable process delegate.

Why not:

- **JVM plugins are cross-language work.** A faithful Camunda
  plugin means a Maven project, a JVM-side serialisation story
  (PETRA's trained module is a torch tensor structure, not a
  format the JVM speaks natively), and a deployment shape that
  matches the engine team's existing patterns (OSGi? Spring
  Boot starter? plain JAR?). Those are choices an engine team
  is better-placed to make than this package.
- **The Python-side toolkit is enough for every integration
  shape worth supporting today.** All three integration patterns
  below work against an engine that emits structured events to
  *somewhere* — a database, a queue, an HTTP webhook. None of
  them need engine-side code.
- **The engine team's existing extension model is the right
  glue.** Camunda has External Tasks; Activiti and Flowable
  have execution listeners and history-event listeners. Those
  hooks call out to PETRA over HTTP (Pattern 1 below) or
  publish to a queue PETRA subscribes to (Pattern 3). That's
  the right separation of concerns — the engine knows what
  events fire when; PETRA knows how to score them.

If a team eventually wants a tighter binding (in-process
inference inside the JVM, no HTTP hop), the **ONNX export**
path (`petra-train` then `export_onnx`) is the recommended
route: load the resulting `.onnx` file into ONNX Runtime for
Java directly inside the engine process. PETRA emits ONNX
that runs unchanged on JVM-side ONNX Runtime.

---

## The three integration patterns

For all three patterns, you've already trained a model:

```
$ petra-train my_scenario.toml -o models/loan_approval.pt
training 'loan_approval' on 4827 trace(s) for 1500 step(s)...
saved model bundle: models/loan_approval.pt, loan_approval.pt.meta.json
  (final loss 0.0234)
```

What differs is *when* and *how* you call into PETRA from the
engine.

### Pattern 1: REST webhook — synchronous per-event scoring

**When to use:** the engine needs an answer back before the
business process continues (e.g. "is this trace looking
anomalous? if so, route to manual review").

**How it wires up:**

1. On the PETRA side, serve the bundle behind the FastAPI app:

   ```
   $ petra-serve models/loan_approval.pt --host 0.0.0.0 --port 8000
   serving 'loan_approval' on http://0.0.0.0:8000 (Swagger UI at /docs)
   ```

2. On the engine side, configure an extension point to call
   PETRA on each event-of-interest. The exact integration
   shape is engine-specific:

   | Engine | Extension hook |
   |---|---|
   | **Camunda 7 / 8** | *External Task* worker topic, OR a `JavaDelegate` execution listener that does an HTTP POST. |
   | **Activiti** | Execution listener on the relevant BPMN element; HTTP POST from a `Bean` task. |
   | **Flowable** | `FlowableEventListener` on `ACTIVITY_COMPLETED` events; HTTP POST. |
   | **Generic** | Any system that can issue an HTTP POST on each event of interest. |

3. The engine POSTs to PETRA's `/anomaly` endpoint with the
   trace-so-far:

   ```http
   POST /anomaly HTTP/1.1
   Host: petra.internal:8000
   Content-Type: application/json

   {
     "input_marking": {"p_application": 1.0},
     "input_values":  {"p_application": 5000.0},
     "events": [
       {"name": "Submit",       "attributes": {}},
       {"name": "CreditCheck",  "attributes": {}}
     ],
     "attributes": {"amount": "5000"}
   }
   ```

   PETRA responds with:

   ```json
   {
     "trace_score": 0.083,
     "per_transition_residuals": {
       "t_submit": 0.001,
       "t_credit_check": 0.020,
       "t_approve": 0.062
     }
   }
   ```

   The engine reads `trace_score`, decides whether to escalate.

**Tradeoffs.** Lowest-latency feedback, but the engine takes a
hard dependency on the PETRA service being up. Deploy PETRA
behind a load balancer if the engine's per-event volume is
high; PETRA is stateless per request, so horizontal scaling is
trivial.

### Pattern 2: Audit-log tail — scheduled batch scoring

**When to use:** the engine doesn't need real-time answers, but
you want a daily / hourly anomaly report dropped into a
monitoring dashboard or sent to a compliance officer's inbox.

**How it wires up:**

1. Configure the engine to expose its audit log as CSV
   (Camunda / Activiti / Flowable all support this through
   their REST admin APIs or via direct SQL against the
   history tables — `ACT_HI_PROCINST`, `ACT_HI_TASKINST`,
   `ACT_HI_VARINST`).

2. Schedule a cron job (or whatever your scheduler is) that
   runs:

   ```
   $ petra-score models/loan_approval.pt \
       --traces /var/exports/engine/history.csv \
       --format csv \
       --case-column proc_inst_id \
       --activity-column task_name \
       -o /var/reports/petra-$(date +%F).json
   scored 4827 trace(s) → /var/reports/petra-2026-05-20.json
   ```

   The output is a JSON document — one entry per trace, with
   `trace_score` and `per_transition_residuals` for each.

3. Feed the JSON into whatever monitoring / alerting / reporting
   surface your operation uses. The structure is intentionally
   `jq`-friendly:

   ```
   $ jq '.[] | select(.trace_score > 1.0) | .attributes.proc_inst_id' \
       /var/reports/petra-2026-05-20.json
   ```

**Tradeoffs.** Loose coupling — the engine doesn't even need to
know PETRA exists; it just needs to expose its history table.
Latency is whatever the cron interval is; storage cost is
trivial (the JSON output is one row per trace).

### Pattern 3: Streaming evaluator — subscribe to an event topic

**When to use:** the engine publishes events to a queue (Kafka,
RabbitMQ, Redis Streams) and you want live per-case scoring
without putting PETRA in the synchronous response path.

**How it wires up:**

1. The engine publishes events to a topic. Each event carries at
   minimum a `case_id`, an `activity` name, and a timestamp.

2. A small Python consumer reads from the topic and feeds the
   :class:`StreamingEvaluator`:

   ```python
   import torch
   from kafka import KafkaConsumer
   from petri_net_nn import StreamingEvaluator, StreamingEvent

   module = torch.load("models/loan_approval.pt", weights_only=False)
   evaluator = StreamingEvaluator(
       module,
       attribute_to_marking=lambda trace: {
           "p_application": float(trace.attributes.get("amount", 0)) / 10000.0
       },
       score_on_every_event=True,
   )
   consumer = KafkaConsumer("engine.loan.events", group_id="petra")
   for message in consumer:
       payload = json.loads(message.value)
       result = evaluator.on_event(
           StreamingEvent(
               case_id=payload["case_id"],
               name=payload["activity"],
               attributes=payload.get("attributes", {}),
           )
       )
       if result and result.trace_score > 1.0:
           publish_alert(result)
       if payload["activity"] in TERMINAL_ACTIVITIES:
           evaluator.close_case(payload["case_id"])
   ```

   That's a ~25-line consumer that gives you per-case
   live-updating anomaly scores against the trained model.

3. The consumer publishes alerts / dashboards from the
   evaluation outputs.

**Tradeoffs.** Decoupled from the engine entirely; no
synchronous dependency. Scoring is on the streaming consumer's
clock, not the engine's. The :class:`StreamingEvaluator` is
single-threaded by design; for high-throughput topics, shard
by `case_id` across multiple consumers.

---

## Choosing between the patterns

A rough decision table:

| Your situation | Recommended pattern |
|---|---|
| Engine needs to act on the score before continuing the process | **Pattern 1** (REST webhook). |
| You want a daily compliance report; engine doesn't need to know | **Pattern 2** (audit-log tail). |
| Engine already publishes to Kafka/RabbitMQ; you want live dashboards | **Pattern 3** (streaming). |
| Engine has no extension points but you can SQL-query its history | **Pattern 2** (audit-log tail). |
| You want in-process inference, no HTTP hop, JVM-side | Train with PETRA, **export to ONNX**, load into ONNX Runtime for Java directly inside the engine. |

---

## Producing the saved bundle

All three patterns load a `petra-train`-produced bundle. The
bundle is two files:

- `model.pt` — pickled :class:`PetriNetModule`. Pickle, so
  **don't load bundles from untrusted sources** — the same
  caveat as for any `torch.load` consumer.
- `model.pt.meta.json` — JSON sidecar with the scenario name,
  description, input-marking spec, and input-value spec used
  to train the model. Read by `petra-score` to map trace
  attributes onto place markings.

The pair must travel together. Keep them in the same directory.

A typical training command:

```
$ petra-train examples/cost_ranked_refactoring/scenario.toml \
    -o /var/petra/loan_v1.pt
training 'cost_ranked_refactoring' on 1000 trace(s) for 800 step(s)...
saved model bundle: /var/petra/loan_v1.pt, loan_v1.pt.meta.json
  (final loss 0.0421)
```

`petra-train` accepts `--steps`, `--lr`, and `--seed` to override
the values declared in the scenario's `[training]` section
without editing the TOML.

---

## Discovery — when you don't have a BPMN model yet

PETRA's first delivery requires the user to bring a Petri net.
If your engine has the audit log but no clean BPMN diagram
(many real deployments are in this situation), the recommended
path is:

1. Use a process-mining tool — **ProM** is the standard open
   one — to discover a Petri net from the audit log.
2. Export the discovered net as **PNML** (ProM's standard
   export format).
3. Point PETRA at the PNML file: `[net] source = "pnml_file"`
   in the scenario TOML.
4. Train normally with `petra-train`.

Native structure-discovery is the open
[Phase 12](https://github.com/pcoz/formally-verified-learnable-process-intelligence/blob/main/docs/ROADMAP.md)
in the PETRA roadmap (Alpha / Inductive / Heuristic miners
ported into PETRA so the discover step is a single library call).
Until that lands, ProM-then-PNML is the cleanest bridge.

---

## What runs where

A diagram of the layered story for an engine team's mental
model:

```
+-----------------------------------------------------------------+
|                       Workflow engine (JVM)                     |
|   Camunda / Activiti / Flowable / proprietary BPM platform      |
|                                                                 |
|   - executes BPMN processes                                     |
|   - writes audit log (history tables / events)                  |
|   - emits events through its existing extension model           |
+--------------------------------+--------------------------------+
                                 |
              one of three integration patterns
                                 |
                                 v
+-----------------------------------------------------------------+
|                          PETRA toolkit (Python)                 |
|                                                                 |
|   petra-train  →  loan_approval.pt + .meta.json                 |
|                                                                 |
|   Pattern 1: petra-serve  ──── /anomaly REST endpoint           |
|   Pattern 2: petra-score  ──── batch JSON output                |
|   Pattern 3: StreamingEvaluator  ──── live per-case scoring     |
|                                                                 |
|   For in-process JVM inference: export_onnx → ONNX Runtime      |
+-----------------------------------------------------------------+
```

The boundary between the two layers is **JSON** for Patterns 1 /
2 / 3 and **ONNX** for in-process JVM inference. Both are
standard interchange formats — no PETRA-specific binding required
on the engine side.
