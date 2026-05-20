# API Reference

PETRA's public API. Each page below is auto-generated from the
docstrings of the corresponding module — when you change a
docstring in the source code, the next push to `main` rebuilds
this site and the changes appear here automatically.

| Module | What it covers |
|---|---|
| [PetriNet](petri_net.md) | Core data model — places, transitions, arcs, markings; token-game semantics for both plain and coloured Petri nets. |
| [PetriNetModule](compiler.md) | The differentiable compiler that turns a `PetriNet` into a `torch.nn.Module`. |
| [Adapter](adapter.md) | TOML scenario loader (`load_scenario` + `ScenarioContext`). |
| [BPMN](bpmn.md) | BPMN 2.0 → `PetriNet` parser. |
| [PNML](pnml.md) | PNML 2009 P/T-net import / export — the standard Petri-net interchange format. |
| [SIF](sif.md) | Pathway Commons / Reactome biology pathway import. |
| [XES](xes.md) | IEEE XES execution log loader (plain + gzipped). |
| [Traces](traces.md) | Training (`train_on_traces`), anomaly scoring, sharpness annealing, AUC, expected-cost. |
| [Anomalies](anomalies.md) | Corruption generators (`drop_event`, `insert_event`, `swap_event_labels`, `shuffle_events`) plus the non-structural `FrequencyBaseline`. |
| [Bisimulation](bisimulation.md) | Strong (`are_bisimilar`) and weak (`are_weakly_bisimilar`) bisimulation, plus the reachability-graph foundation they share. |
| [Soundness](soundness.md) | Aalst soundness verification (`check_soundness`) and deadlock localisation (`find_deadlocks`). |
| [CTL](ctl.md) | Computation Tree Logic model checking — six-primitive AST plus the derived `AG` / `AF` / `EF` / `AX` / `AU` constructors. |
| [Interpretability](interpretability.md) | Rule extraction, bootstrap CIs, counterfactuals, sensitivity analysis, cross-variant comparison reports, prose explainers. |
| [Subnets](subnets.md) | Hand-built reference subnets — the five canonical workflow-net building blocks. |
| [ONNX export](onnx_export.md) | Export a trained `PetriNetModule` to ONNX for deployment to any ONNX runtime (C++, Java, browser, mobile, edge). |
