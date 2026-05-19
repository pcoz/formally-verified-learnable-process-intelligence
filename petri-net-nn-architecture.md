# Petri Net Neural Network Architecture for BPMN Process Reasoning

**Document type:** Research architecture proposal  
**Status:** Speculative — theoretical foundations are real, implementation does not exist  
**Audience:** Research team and external collaborators  
**Date:** May 2026

---

## Honest Framing

This document describes a research direction, not a buildable system. The theoretical connections between Petri nets, BPMN process representations, and neural network architectures are real and documented in existing literature. The specific combination proposed here — using verified BPMN workflow nets as the structural substrate for a neural network — does not exist and would require original research to build.

The architecture described here is a research target, not an engineering specification. It is written to be precise enough to evaluate whether the direction is worth pursuing, not to imply the work is done.

---

## 1. The Core Idea

A BPMN process, when expressed as a sound workflow net (Aalst 1997), produces a formally verified process graph. That graph is a Petri net. A Petri net is a mathematical object with well-understood computational properties. The proposal is to use that Petri net directly as the architecture of a neural network — so that the network's structure is the process, not a learned approximation of it.

The network does not learn about the process. The network is the process. The learning happens within the structural constraints the Petri net imposes.

---

## 2. Petri Net Fundamentals

A Petri net N = (P, T, F, M₀) where:

- **P** is a finite set of places
- **T** is a finite set of transitions  
- **F ⊆ (P×T) ∪ (T×P)** is the flow relation
- **M₀: P → ℕ** is the initial marking

A transition t ∈ T is **enabled** under marking M if every input place of t holds at least one token. Firing t removes one token from each input place and adds one token to each output place.

This is the computational substrate. It is discrete, formal, and well-understood.

---

## 3. Mapping BPMN to Petri Nets

BPMN maps to Petri nets via a well-defined structural translation documented in Aalst's workflow net literature:

| BPMN construct | Petri net element |
|---|---|
| Activity / task | Place (pre) + Transition + Place (post) |
| Process state | Place |
| Token of control | Token in corresponding place |
| Guard condition | Transition pre-condition |
| Parallel gateway (AND-split) | Transition with multiple output places |
| Exclusive gateway (XOR-split) | Multiple transitions with shared input place |
| Synchronisation (AND-join) | Transition with multiple input places |
| Event-based compensation | Additional transition subnet |
| Message flow between pools | Transition spanning subnet boundary |

Sound workflow net verification (Aalst) establishes:

- **Reachability** → every place is reachable from M₀
- **No deadlock** → every reachable marking has at least one enabled transition
- **Proper completion** → the final place is reachable and uniquely marked at completion
- **Bisimulation equivalence** → two subnets are bisimulation equivalent in the Petri net sense

A verified BPMN process expressed as a sound workflow net has proven reachability, liveness, and boundedness properties — the formal substrate this architecture requires.

---

## 4. The Neural Network Mapping

### 4.1 Structural correspondence

Each element of the Petri net corresponds to a neural network element:

| Petri net element | Neural network element |
|---|---|
| Place p | Neuron with activation a(p) ∈ [0,1] |
| Token in place p | High activation: a(p) → 1 |
| Empty place p | Low activation: a(p) → 0 |
| Transition t | Weighted aggregation function |
| Flow arc (p,t) | Input weight w(p,t) |
| Flow arc (t,p) | Output weight w(t,p) |
| Firing rule | Activation threshold with structural constraint |
| Initial marking M₀ | Input layer activation pattern |
| Final marking | Output layer activation pattern |

### 4.2 The continuous relaxation

Petri nets are discrete. Neural networks are continuous. The bridge is a **continuous relaxation** of the token firing rule:

Instead of: transition t fires if all input places hold tokens (discrete, binary)

Use: transition t activates with strength proportional to the product of input place activations (continuous, differentiable)

```
activation(t) = σ(Σ w(p,t) · a(p) - θ(t))
```

where σ is a sigmoid, w(p,t) are learned weights, and θ(t) is a learned threshold.

The output activation of each downstream place:

```
a(p) = Σ_{t: (t,p) ∈ F} activation(t) · w(t,p)
```

This is differentiable end-to-end. Standard backpropagation applies.

### 4.3 The structural constraint — what makes this novel

In a standard neural network, any neuron can connect to any other. The architecture is arbitrary.

In the Petri net neural network, connections are determined by the verified BPMN process. A neuron corresponding to place p can only receive input from transitions that have p as an output place in the Petri net. It can only send output to transitions that have p as an input place.

This is not a soft constraint learned during training. It is a hard structural constraint imposed by the compiled process graph. Weights outside this structure are zero by construction and cannot be learned away from zero.

The consequence: the network cannot represent computations that the underlying process cannot reach. The workflow net's soundness guarantees propagate into the network's representational capacity. The network is formally bounded by the process semantics.

### 4.4 What is learned

Given the fixed structural constraints, what does training learn?

- **Transition weights w(p,t):** how strongly each input place contributes to transition activation
- **Thresholds θ(t):** how much cumulative input is required to activate each transition
- **Output weights w(t,p):** how strongly each transition contributes to downstream place activation

These learned parameters determine, within the structurally fixed topology, how the network responds to specific attribute patterns in process execution data. The structure is the process. The learned weights are the process's characteristic behaviour under the observed data distribution.

---

## 5. Five Petri Net Subnets — The Elemental Building Blocks

BPMN processes compose from a small set of elemental Petri net patterns. These are the building blocks from which all BPMN process graphs are assembled.

### Subnet 1: Sequential execution

```
[P_before] --T_step--> [P_after]
```

Neural interpretation: a single-input single-output neuron. Activation flows from before to after when the step fires. The weight w(P_before, T_step) is learned — how strongly the prior state activates the transition.

Use in BPMN: every sequential task in a process flow. The most common element.

### Subnet 2: Exclusive choice (XOR-gateway)

```
              --T_route_A--> [P_path_A]
[P_decision] <
              --T_route_B--> [P_path_B]
```

Neural interpretation: a one-to-many neuron with competing transitions. The transition with the highest activation wins. In the continuous relaxation, both paths receive partial activation proportional to their transition weights — a soft routing rather than hard selection.

Training learns which input attributes route to which path. The structural constraint ensures only the declared paths are possible.

Use in BPMN: every XOR gateway. In a business rules context — route to ReviewProcess or ApprovalProcess based on transaction attributes.

### Subnet 3: Parallel split (AND-gateway)

```
              --T_spawn--> [P_branch_A]
[P_ready]    <
              --T_spawn--> [P_branch_B]
```

In a true AND-split, a single transition produces tokens in multiple places simultaneously. Neural interpretation: one transition with multiple output connections, all activated simultaneously.

Training learns the threshold at which the parallel split fires — how much cumulative evidence is required before both branches are activated.

Use in BPMN: every parallel gateway. One event triggers simultaneous parallel subprocess instances.

### Subnet 4: Synchronisation (AND-join)

```
[P_branch_A] --\
                T_merge --> [P_unified]
[P_branch_B] --/
```

Both input places must hold tokens before the transition fires. Neural interpretation: a multi-input neuron where all inputs must be sufficiently activated before the output fires. The threshold θ(T_merge) is set high — requiring cumulative input from all branches.

This is the hardest subnet to relax continuously. A partial firing — where one branch is complete and the other is not — should not propagate. Training must learn a near-step threshold. In practice this requires careful initialisation.

Use in BPMN: every AND-join. Both the risk assessment and the document verification must complete before approval proceeds.

### Subnet 5: Saga compensation

```
[P_active] --T_succeed--> [P_complete]
[P_active] --T_fail-----> [P_compensating]
[P_compensating] --T_compensate--> [P_initial]
```

The compensation path returns the process to a prior state when the main path fails. Neural interpretation: two competing transitions from the active place — success and failure — with the failure path leading to a compensation subnet that reactivates the initial place.

Training learns the failure threshold — what attribute patterns predict failure strongly enough to activate the compensation path rather than the success path.

Use in BPMN: every compensation event. The main path fails, compensation fires, the process returns to a recoverable state.

---

## 6. Composition

BPMN processes are composed from these five subnets. The Petri net neural network for a complete BPMN process is the composition of its elemental subnets, preserving all structural constraints from the compiler.

A simple approval process composes:

- Subnet 1 (sequential): submitted → triaged
- Subnet 2 (XOR): triage routes to standard or expedited process
- Subnet 3 (AND-split): expedited process spawns review and notification simultaneously
- Subnet 4 (AND-join): both must complete before decision is issued
- Subnet 5 (compensation): review fails, compensation fires, process returns to submission

The composed network has exactly the topology of the verified BPMN process. No additional connections. No missing connections. The architecture is the process.

---

## 7. What This Enables

### 7.1 Process execution prediction

Given a partial process execution — some places are activated, others are not — the network predicts which transitions will fire next and which places will become active. This is forward inference over the process structure.

Training data: historical BPMN process execution logs. Each trace is a sequence of place activations. The network learns to predict the next activation given the current state.

### 7.2 Anomaly detection — the coloring book method grounded formally

A process instance that produces an unusual activation pattern — unexpected transition weights, unexpected routing at XOR gateways, unexpected failure rates at saga transitions — is anomalous relative to the learned distribution over the fixed process structure.

This is the financial crime detection idea grounded in the Petri net substrate. The fingerprint is the learned weight distribution over the process graph. Anomalies are deviations from that distribution. The structure ensures deviations are interpretable — which subnet, which transition, which place shows the anomaly.

### 7.3 Bisimulation in the neural substrate

Two subnets that are bisimulation equivalent in the Petri net sense will learn identical weight distributions given the same training data. The bisimulation checker identifies these structurally. The neural network confirms them empirically. The combination — structural equivalence verified by the compiler, behavioural equivalence confirmed by identical learned weights — is a stronger finding than either alone.

---

## 8. Open Research Problems

These are not engineering problems with known solutions. They are research problems that must be solved for this architecture to work in practice.

**The discrete-continuous interface.** The continuous relaxation of token firing is mathematically convenient but may not faithfully represent Petri net semantics. Specifically, AND-join synchronisation requires near-step activation functions that are difficult to train with standard gradient methods. Alternative approaches — straight-through estimators, Gumbel-softmax, or discrete diffusion — need evaluation against the specific BPMN process patterns being modelled.

**Training data requirements.** The network learns from process execution traces. How many traces are needed for stable weight estimates at each subnet type? Sequential and XOR subnets likely need fewer. AND-joins and saga compensations — which fire less frequently — may need many more traces to learn reliable weights. Empirical characterisation is needed.

**Cross-pool composition.** Message flows in BPMN span process pool boundaries. The Petri net subnet for a cross-pool message has a transition that produces tokens in a place defined in a different pool's net. Composing multiple pool nets correctly — handling shared event spaces, avoiding token conservation violations at boundaries — requires formal treatment. Aalst's inter-organisational workflow nets provide a starting point but compositional neural training across boundaries is an open problem.

**Interpretability of learned weights.** The structural interpretability of the architecture — which subnet shows the anomaly — is a genuine advantage over black-box ML. But the learned weights within each subnet are not directly interpretable. A weight w(P_active, T_dispatch) tells you how strongly the active state drives dispatch, but not why. The next research question is whether the learned weights can be distilled back into readable decision rules — closing the loop from neural learning to interpretable business logic.

---

## 9. Relationship to Existing Work

This architecture draws on and extends several existing research threads:

**Workflow nets** — Aalst's foundational work on sound workflow nets as a Petri net subclass. Sound workflow net verification is the formal foundation for the reachability and liveness properties this architecture depends on.

**Graph neural networks for process mining** — existing work applying GNNs to process graphs for conformance checking and anomaly detection (Tax et al., Bukhsh et al.). The Petri net neural network is structurally more constrained than a general GNN — the architecture is the verified workflow net, not learned from process data.

**Neuro-symbolic AI** — Scallop, DeepProbLog, and related systems that combine neural and symbolic reasoning. The Petri net neural network is a specific instance of this paradigm where the symbolic substrate is a compiler-verified workflow net.

**Spiking neural networks** — networks that model discrete spike propagation rather than continuous activations. The token firing model is analogous to spike propagation. SNN training methods — STDP, surrogate gradient methods — may be applicable to the discrete token firing case.

The novel contribution, if the research programme succeeds, is the specific combination: a verified BPMN workflow net used as a fixed neural network architecture, where the workflow net's soundness properties propagate into the network's representational constraints, and where learned weights are interpretable at the granularity of named BPMN elements.

---

## 10. Next Steps

For this to move from research direction to research programme:

**Step 1:** Implement the Petri net extraction from a standard BPMN process description. The mapping from BPMN to workflow nets is well-defined in the literature. The output is a formal Petri net object that serves as input to the neural network architecture generator. Tools such as ProM provide reference implementations of this extraction.

**Step 2:** Implement the continuous relaxation for the five elemental subnets. Build a differentiable version of each subnet and verify that backpropagation produces sensible weight updates on synthetic data.

**Step 3:** Train on process execution logs from a BPMN-instrumented system. Process-aware information systems that log BPMN execution events provide the necessary training data. The IEEE XES standard defines a portable format for such logs. Begin with sequential and XOR subnets — the most common and tractable cases.

**Step 4:** Evaluate anomaly detection on known anomalous traces. Deliberately introduce process instances with known structural anomalies and measure whether the learned weight distribution flags them.

**Step 5:** Engage the formal methods and neuro-symbolic AI research communities. This is a publishable research direction. External collaboration will accelerate progress on the open problems in section 8.

---

*This document is a research architecture proposal. The theoretical foundations are sound. The implementation does not exist. The open problems in section 8 are genuine research problems, not engineering tasks. Anyone reading this document as a description of a working system is misreading it.*
