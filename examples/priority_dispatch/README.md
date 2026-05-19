# Priority dispatch — handlers with rate-encoded priorities

Three handlers compete for incoming tickets. The modeller declares
*before training* that the express handler is three times as eager
to fire as the standard handler, and that the bulk handler is half
as eager. Training then refines those priors against observed
trace data. PETRA running on a workflow with **Phase 9 stochastic
firing rates**.

## What stochastic firing rates enable here

A trained network can in principle discover transition priorities
from data alone — but real workflows often have known priorities
that the modeller wants to *declare* and not have to learn from
scratch. Rate annotations let those priors travel with the
structural model:

```toml
[[net.transitions]]
id = "t_express"
rate = 3.0
```

In the compiler, the rate multiplies the pre-activation before the
sigmoid, so a high-rate transition fires more eagerly than its
siblings for the same inputs. Training still adjusts the learnable
weights and thresholds; the rate is a fixed prior on top.

The composition with other Phase 9 features is straightforward: a
transition can have a duration, a rate, and inhibitor guards all at
once. Each annotation contributes its own structural effect to the
compiled module's behaviour.

## When rates matter

Useful when you have a meaningful prior about transition propensity
that you don't want training to have to discover from limited data:

- Priority dispatch (express vs standard vs bulk handlers).
- Stochastic Petri-net modelling where rates correspond to
  exponential timing parameters.
- Service-level SLAs where one path is guaranteed faster than
  another by contract.
- Backwards-compatibility constraints: an existing process has
  measured-empirical firing frequencies; rates let you carry those
  into a new variant without re-training from scratch.

## Advantage over alternatives

- **vs. learnable thresholds alone**: training can discover
  priorities given enough data, but rates let modeller knowledge
  travel structurally. The trained model with rate priors converges
  faster and behaves predictably on out-of-distribution inputs.
- **vs. classical Stochastic Petri Nets**: classical SPN analysis
  produces continuous-time Markov chains; PETRA's rates plug
  cleanly into the trainable sigmoid relaxation, so you get the
  same prior with a learnable extension.
- **vs. weighting in monolithic ML**: monolithic models can have
  prior weights but they're buried in millions of parameters here
  the prior is a declarative annotation on a named transition.

## Files

- `scenario.toml` — three transitions with rate annotations 3.0,
  1.0, and 0.5 respectively
- `../../tests/scenarios/test_priority_dispatch.py` — test
  exercising the rate ordering before and after training
