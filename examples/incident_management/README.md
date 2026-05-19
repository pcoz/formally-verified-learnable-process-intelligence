# Incident management — real BPI Challenge 2013 data

PETRA running on real public data: the **BPI Challenge 2013
incidents** dataset from Volvo IT, comprising 7,554 actual
incident tickets recorded inside their ITIL service-management
system. The compressed XES file (1.3 MB) is committed to the
repo; the adapter reads gzipped XES transparently.

## What this scenario demonstrates

- **Real-world XES ingestion**: the file is the unmodified public
  release, parsed via `parse_xes` with on-the-fly gzip
  decompression.
- **Adapter handling of event-level routing attributes**: the
  BPI 2013 log records `impact` (Low / Medium / High / Major) at
  the event level, never on the trace itself. The new
  `promote_event_attrs` adapter option lifts it to trace level so
  the training pipeline can read it.
- **Structural prior over messy real data**: the canonical net
  here models the happy-path ITIL incident lifecycle
  (In Progress → Resolved → Closed). The real log has substantial
  back-and-forth between In-Progress and Awaiting-Assignment that
  the simplified net doesn't try to fit — PETRA's training learns
  the relative frequency of each modelled transition firing
  given the real trace distribution.
- **Anomaly detection on real traces**: deviations from the
  canonical happy path (e.g. an incident that closes without
  reaching Resolved) produce non-zero residuals on the
  unmatched transition.

## Source

- **Dataset:** BPI Challenge 2013 — incidents
- **Publisher:** Steeman, W. (2014), Ghent University
- **Provider:** 4TU.ResearchData
  ([landing page](https://data.4tu.nl/articles/dataset/_/12693914))
- **DOI:** 10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee
- **License:** CC BY 4.0 (4TU.ResearchData default for BPI data)
- **Original organisation:** Volvo IT Belgium, incident
  management workflow
- **Citation:**
  Steeman, W. (2014). BPI Challenge 2013, incidents.
  4TU.ResearchData. https://doi.org/10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee

## Scope

The scenario trains on the **full 7,554-trace public release** —
the entire BPI Challenge 2013 incidents corpus, unedited. The
adapter reads the 1.3 MB gzipped XES file directly.

The Petri net here is a deliberate simplification of the real
process. Process discovery from logs (Phase 12 of the roadmap)
would derive a more faithful structure automatically; this
scenario uses a hand-crafted canonical happy-path net to
demonstrate the *real-data-ingestion* path rather than the
discovery path.

## Files

- `data/incidents.xes.gz` — the committed real public dataset
- `scenario.toml` — net, training config, BPI 2013 source pointer
- `../../tests/scenarios/test_incident_management.py` — end-to-end test

## Running

```
python -m pytest tests/scenarios/test_incident_management.py
```
