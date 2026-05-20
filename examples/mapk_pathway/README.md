# MAPK1/MAPK3 signalling cascade (SIF)

A slice of the canonical **EGF → ERK signalling pathway** loaded
from a Simple Interaction Format (SIF) file — the format Pathway
Commons publishes Reactome-derived pathway data in. PETRA running
on real curated biology content with no hand-coding step.

## What this scenario shows

The Phase 10 closing piece: **SIF import**. The same
`source = "sif_file"` adapter accepts any Pathway Commons SIF
download — Reactome, BioCyc, PID, NCI Nature, Panther, HumanCyc,
KEGG. Once the file is in, the rest of PETRA — compilation,
forward pass, rule extraction, anomaly detection, bisimulation —
applies to it unchanged.

The cascade in `mapk_signalling.sif` covers the receptor → kinase
→ transcription-factor flow that essentially every cell-biology
textbook draws:

1. **Receptor binding.** EGF binds EGFR.
2. **Adapter recruitment.** EGFR recruits GRB2, which binds SOS1.
3. **Small GTPase.** SOS1 activates HRAS; HRAS activates RAF1.
4. **Kinase cascade.** RAF1 phosphorylates MAP2K1 and MAP2K2,
   which both phosphorylate MAPK1 and MAPK3 — the *ERK2* and
   *ERK1* MAP kinases.
5. **Substrate phosphorylation.** Activated MAPK1/3 phosphorylate
   ELK1 (a transcription factor) and RPS6KA1 (the p90 ribosomal
   S6 kinase).
6. **Transcription.** ELK1 controls expression of FOS.
7. **Negative regulation.** DUSP6, a dual-specificity phosphatase,
   acts on MAPK1 and MAPK3 (the SIF format expresses this as
   `controls-state-change-of` — without distinguishing activating
   from inhibitory effects, see [Limitations](#limitations) below).

Entity symbols follow HGNC; interaction types follow the Pathway
Commons v14 schema.

## Why this matters

Two angles:

- **Ecosystem citizenship.** Before SIF support, PETRA needed a
  hand-coded Petri net per biological scenario. After SIF support,
  any of the ~3,000 Reactome pathways (and the equivalent content
  in BioCyc, PID, etc.) is one download away.
- **Reactome as a Petri net.** Pathway databases store pathways
  as bigraph-like structures that map cleanly onto Petri nets —
  *places are molecule pools, transitions are reactions*. The
  fact that PETRA's substrate doesn't need to be specialised for
  biology is the whole point of the framing in `docs/ROADMAP.md`:
  one substrate, many domains.

## How to swap in a real Pathway Commons download

The bundled `mapk_signalling.sif` is a *curated slice* — real
HGNC symbols, real Pathway Commons interaction types, a
biologically faithful structure, but small enough to commit and
read. To run PETRA over a full Reactome pathway:

1. Visit [Pathway Commons](https://www.pathwaycommons.org/) or
   the [Pathway Commons archive](https://www.pathwaycommons.org/archives/PC2/).
2. Download the SIF for the pathway you want (or for the entire
   `Reactome` data source).
3. Replace `mapk_signalling.sif` in this directory with the
   downloaded file (or update the `path` field in `scenario.toml`).

No code changes needed — the adapter reads whatever SIF you point
it at.

## Limitations

This first SIF delivery preserves what the format carries and
loses what it doesn't:

- **Entity types are flattened.** SIF doesn't distinguish protein
  from gene from small molecule from complex; every entity becomes
  an opaque place. BioPAX carries this information through but
  is a much larger parsing job — see the [ROADMAP](../../docs/ROADMAP.md)
  Phase 10 carry-forward.
- **All interactions are directional.** Even nominally-symmetric
  ones like `in-complex-with`. The modeller can declare the
  opposite direction by adding a second triple.
- **Inhibitory effects.** SIF uses `controls-state-change-of` for
  both activating and inhibitory regulators. DUSP6 *dephosphorylates*
  MAPK1/3 (inhibitory) but the SIF triple looks identical to an
  activating one. Capturing the sign would need a typed-edge
  schema (e.g. SBGN's process-description glyphs) or a domain
  override.

## Files

- `mapk_signalling.sif` — 18 interactions, 13 entities, the
  canonical EGF → ERK cascade.
- `scenario.toml` — points the adapter at the SIF file.
- `../../tests/scenarios/test_mapk_pathway.py` — verifies the SIF
  loads cleanly, the resulting Petri net has the expected
  structure, and a forward pass propagates activation from EGF
  through to the MAPK substrates.

## Running

```
python -m pytest tests/scenarios/test_mapk_pathway.py
```
