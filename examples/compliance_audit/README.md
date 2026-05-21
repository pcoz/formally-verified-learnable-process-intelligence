# Compliance audit — regulatory invariants as CTL properties

A loan-approval process with an explicit audit-log step on the
approval path, and a compliance regime stated as **CTL
temporal-logic invariants** that the process must satisfy.
This scenario demonstrates how PETRA's structural verification
machinery — `check_soundness`, `find_deadlocks`, `check_ctl` —
turns a compliance regime into a set of mechanical checks that
run in milliseconds, before the process ever goes into
production.

## The compliance regime

Three classes of invariant, all expressed in CTL and verified
against the Petri net itself (no training, no log data):

1. **Audit-after-approve liveness.** *Every approved loan must
   eventually fire the audit-log step before closing.* In CTL:
   `AG (decided_approve → AF audit_logged)`. The reading: at
   every reachable state, *if* the loan is in the approved
   state, *then* on every future path the audit-log place
   eventually becomes marked.
2. **Decline-after-credit-check ordering.** *The decline
   transition cannot be enabled before the credit-check
   transition has fired.* In CTL:
   `AG (enabled(t_decline) → has_token(p_credit_checked))`.
   Rules out short-cutting the decision logic by declining
   before evaluating creditworthiness.
3. **Soundness and absence of deadlocks** (the Aalst conditions
   plus `find_deadlocks`). Every reachable state can reach the
   final marking; the final marking is the only one with the
   sink token; every transition is enabled in at least one
   reachable marking; no non-final state has no enabled
   successor.

## What the scenario test demonstrates

End-to-end against the compliant net loaded from
`scenario.toml`:

- `check_soundness` returns `is_sound=True`. The net carries
  no incomplete markings, no lingering-token markings, and no
  dead transitions.
- `find_deadlocks` returns `[]`. The only marking with no
  enabled successor is the intended final marking
  `{p_closed: 1}`.
- Both CTL invariants hold at the initial marking. The
  `CTLResult.counterexample` field is `None` for each
  formula, which is the cleaner "verified everywhere" signal.

The test also constructs a **deliberately broken variant** in
Python where the audit-log step can be skipped — the broken
variant has a direct *close-without-audit* transition from
`p_decided_approve` to `p_closed`, bypassing the mandatory
audit step. The audit-after-approve CTL invariant fails on the
broken variant, and `check_ctl` returns a counterexample
marking that witnesses the violation. This is the case
auditors care about most: the invariant catches the
non-compliant refactoring *before* deployment.

## Why this matters

Today's compliance verification on business processes is
typically a combination of:

- Tabular checklists ("does this process include an audit
  step?") reviewed manually;
- Periodic spot-checks of production traces;
- Hard-coded engine constraints that may or may not match the
  intended invariant.

None of those approaches give a *mechanical proof* that the
invariant holds across every reachable state of the process.
CTL model checking does. A compliance officer can author the
invariant once, in CTL, and the verification runs in
milliseconds on every refactoring — the same way unit tests
run on every commit in software engineering.

## Files

- `scenario.toml` — the compliant net structure. No traces or
  training section — the scenario is purely structural.
- `../../tests/scenarios/test_compliance_audit.py` — pins
  soundness, deadlock localisation, and the two CTL invariants
  on the compliant net, plus the CTL-failure-with-witness path
  on the broken variant.

## Running

```
python -m pytest tests/scenarios/test_compliance_audit.py
```
