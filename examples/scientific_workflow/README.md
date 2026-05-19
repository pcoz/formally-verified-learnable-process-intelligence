# Scientific workflow — PCR with quality gating

A polymerase chain reaction (PCR) protocol with a quality-driven
accept-vs-re-prep decision, modelled as a Petri net. PETRA running
on a laboratory protocol: the steps of the procedure are the
transitions, the quality readout drives the routing decision, and
procedural deviations (skipped steps, out-of-order operations) show
up as anomaly residuals pinned to the offending step.

## What this scenario shows

- A multi-step laboratory protocol fits the substrate as a sequential
  chain of `Place → Transition → Place` (the §5 Subnet 1 pattern)
  with a single XOR routing decision at the quality readout.
- The framework learns the amplification-quality → accept/re-prep
  rule from observed runs, just as it did for the biological
  signalling cascade and the manufacturing cell — same primitives,
  different domain.
- Protocol deviations (skipped denaturation step, double-measuring
  without re-annealing, accepting a low-quality sample) are
  flagged via §7.2 residuals pinned to the offending step's
  transition.

## Advantage over alternatives

- **vs. ELN / LIMS audit logs** (electronic lab notebooks /
  laboratory information management systems): audit logs record
  what happened but cannot tell you *which step deviated from the
  expected protocol structure*. The framework's structural prior
  provides that grounding.
- **vs. statistical quality control on PCR outputs**: SPC catches
  univariate drift in amplification yield but cannot catch
  procedural deviations (steps skipped, ordering wrong). The
  Petri-net substrate explicitly encodes procedure.
- **vs. workflow-management systems** (Snakemake, Nextflow, CWL):
  workflow managers verify a pipeline ran to completion but assume
  the *executed* steps match what was intended. Trace-level
  conformance via this framework catches the case where a step
  was executed but produced an unexpected pattern.
- **Cross-lab reproducibility**: Phase 2 bisimulation lets two
  labs' protocol variants be verified equivalent before comparing
  their results — closing a real reproducibility gap in molecular
  biology.

## Real-world source

- Mullis et al., "Specific Enzymatic Amplification of DNA In Vitro:
  The Polymerase Chain Reaction" (Cold Spring Harbor Symposia on
  Quantitative Biology, 1986)
- Saiki et al., "Primer-directed enzymatic amplification of DNA
  with a thermostable DNA polymerase" (Science, 1988)

PCR is run billions of times per year worldwide in clinical
diagnostics, forensics, and research. The structural pattern here
(denature → anneal → extend → measure → gate) is unchanged across
all variants (qPCR, RT-PCR, digital PCR).

## Files

- `scenario.toml`
- `../../tests/scenarios/test_scientific_workflow.py`

## Running

```
python -m pytest tests/scenarios/test_scientific_workflow.py
```
