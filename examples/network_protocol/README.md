# Network protocol — TCP 3-way handshake + attack detection

The TCP 3-way handshake modelled as a Petri net, used to demonstrate
attack-pattern anomaly detection on protocol traces. Validates the
framing claims that the substrate covers network protocol state
machines and that anomaly detection generalises to attack patterns.

## What this scenario shows

- A canonical network protocol state machine compiles into a
  Petri net via the same primitives that handle BPMN.
- Cross-pool composition (client + server, three shared message
  places: SYN, SYN-ACK, ACK) is the right structural fit — the
  server cannot fire `recv_syn` until the client has produced the
  SYN message; the client cannot complete until the server's
  SYN-ACK is in flight.
- After training on normal handshake traces, three classes of
  attack pattern are detected via §7.2 anomaly scores:
  - **SYN flood** — many SYN events without completion; the trace
    is missing `recv ACK, established`.
  - **Half-open connection** — handshake stops at SYN-RECEIVED;
    missing `recv SYN-ACK, send ACK` and downstream events.
  - **Out-of-order ACK** — ACK arrives before SYN-ACK; events in
    the wrong sequence.

## Advantage over alternatives

- **vs. signature-based IDS** (e.g. Snort rules): signatures match
  byte patterns and miss novel attacks. The structural prior here
  flags any deviation from the protocol's *expected firing
  distribution*, including patterns the rule-writer didn't
  anticipate.
- **vs. flow-based anomaly detection** (NetFlow, Bro/Zeek): flow
  records aggregate packets but lose the per-session state-machine
  structure. The substrate keeps state explicitly so half-open
  attacks (which look statistically similar to legitimate slow
  connections) show up as structural deviations.
- **vs. ML-based intrusion detection** (e.g. autoencoders on
  packet features): ML detectors can flag anomalies but cannot
  explain *which protocol state transition* failed. Here the
  residual pins to specific transitions ("`recv ACK, established`
  expected to fire, did not"), giving an analyst a directly
  actionable signal.
- **vs. TLA+ / model checking** of the protocol: model checking
  proves the spec is correct under all schedules. This framework
  monitors that the deployed protocol is *being executed* as
  specified — complementary.

## Real-world source

- RFC 793 — Transmission Control Protocol, Postel 1981
- RFC 9293 — Transmission Control Protocol (revised 2022)
- CERT advisory CA-1996-21 (TCP SYN flooding)

The 3-way handshake state machine is unchanged across both RFCs;
the place/transition graph here corresponds directly to the state
diagram in §3.2 of RFC 9293.

## Files

- `scenario.toml` — full specification
- `../../tests/scenarios/test_network_protocol.py` — adapter-driven
  test covering load, training, and detection of all three attack
  patterns

## Running

```
python -m pytest tests/scenarios/test_network_protocol.py
```
