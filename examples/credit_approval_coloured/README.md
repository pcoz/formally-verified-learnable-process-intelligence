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

## Current scope

This first delivery of CPN in PETRA sits at the structural
token-game level. The compiler stays scalar — the trained network
still treats place activation as a single value in [0, 1] rather
than as a distribution over per-token values. CPN-aware compiler
integration (where the trained network reads token values and
routes on them) is a follow-up in the roadmap.

The scenario therefore exercises `fire_coloured` /
`is_enabled_coloured` directly rather than driving the neural
training loop. Future scenarios will use coloured tokens through
training once the compiler is taught about them.

## Files

- `scenario.toml` — net with declarative guards on the two
  routing transitions
- `../../tests/scenarios/test_credit_approval_coloured.py` —
  adapter-driven test exercising the coloured token-game

## Running

```
python -m pytest tests/scenarios/test_credit_approval_coloured.py
```
