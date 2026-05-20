# Credit approval with coloured tokens

A loan-application net where tokens carry the application *amount*
as their value, and the approve / decline transitions guard on
that value. PETRA running on a coloured-Petri-net pattern: routing
depends on the data carried by the token, not just on the token's
presence.

## What coloured Petri nets enable here

In the pre-Phase-9 substrate, tokens carried no information beyond
"a token exists at this place." Routing decisions had to encode the
application amount as a place *activation* in [0, 1] and rely on a
learned XOR threshold to map activation to approve / decline. That
works but flattens the amount onto a scalar channel and burns
training data discovering a threshold the loan officer already
knows.

With coloured tokens the amount travels with the token:

```toml
[[net.transitions]]
id = "t_approve"
guard = { place = "p_submitted", op = ">=", value = 1000.0 }

[[net.transitions]]
id = "t_decline"
guard = { place = "p_submitted", op = "<", value = 1000.0 }
```

The guard reads the value of the token that would be consumed
(here: the application amount in monetary units) and decides
whether the transition is allowed to fire. The decision rule is
declarative in the model rather than emergent from training.

## Advantage over alternatives

- **vs. encoding the amount as a place activation in [0, 1]**:
  encoding loses range and forces training to rediscover the
  threshold the modeller already knows. Coloured tokens carry the
  amount directly.
- **vs. classical CPN tools** (CPN Tools, CPN/ML): those handle the
  full ML-style colour sets but don't unify with neural training.
  PETRA's CPN-lite scopes to scalar token values and composes with
  the trainable substrate.
- **vs. business-rule engines**: rule engines apply a written rule
  to a payload but lack the structural Petri-net analysis that
  PETRA brings (bisimulation, soundness, anomaly residuals on
  named elements).

## CPN-aware compiler

This scenario also exercises the CPN-aware compiler: the trained
network reads the application amount as a per-token value channel
and routes on it through differentiable soft guards. Each
structural guard contributes a learnable `nn.Parameter` threshold
initialised at the TOML value (1000 here) and refined by training
against the observed routing in the trace data.

The scenario's training section supplies a mix of high- and
low-amount applications:

```toml
[training.input_marking]
p_submitted = { constant = 1.0 }

[training.input_values]
p_submitted = { attribute = "amount" }
```

After training, the learned thresholds sit in the empirical
decision band — between the largest observed decline (900) and the
smallest observed approve (1500). On held-out amounts, the
soft-guard routes correctly: amount 5000 fires `t_approve` and
suppresses `t_decline`; amount 300 does the opposite.

The token-game path (`fire_coloured` / `is_enabled_coloured`)
keeps using the original callable guard, so the structural and
trained views remain in sync: the declarative record is the
trainable face of the same rule the callable encodes.

## Files

- `scenario.toml` — net with declarative guards on the two
  routing transitions plus training traces driving the
  CPN-aware compiler
- `../../tests/scenarios/test_credit_approval_coloured.py` —
  adapter-driven tests for both the token-game path and the
  CPN-aware compiler path

## Running

```
python -m pytest tests/scenarios/test_credit_approval_coloured.py
```
