"""Simple Interaction Format (SIF) import.

SIF is the de facto interchange format used by **Pathway Commons** —
the public hub that aggregates Reactome, BioCyc, PID, NCI Nature,
Panther, HumanCyc, KEGG and others, and the canonical source of
real curated biological-pathway content. Adding SIF support here
gives PETRA a direct on-ramp to that body of data.

Each SIF line is a tab-separated triple::

    ENTITY_A    interaction_type    ENTITY_B

The parser maps every unique entity (gene symbol, small molecule,
complex name) to a **place** and every interaction triple to a
**transition** with one input arc from ``ENTITY_A`` and one output
arc to ``ENTITY_B``. Duplicate triples are deduplicated. The flow
is directional even for nominally-symmetric interaction types
(``in-complex-with``, ``interacts-with``, ``neighbor-of``) — when
a modeller wants symmetric handling they can add a second triple
in the opposite direction.

Standard Pathway Commons interaction types — the parser is not
opinionated about which strings are valid, but for reference these
are the ones the PC v14 schema documents:

  * ``controls-state-change-of``
  * ``controls-phosphorylation-of``
  * ``controls-expression-of``
  * ``controls-transport-of``
  * ``catalysis-precedes``
  * ``in-complex-with``
  * ``interacts-with``
  * ``neighbor-of``
  * ``chemical-affects``
  * ``consumption-controlled-by``
  * ``controls-production-of``
  * ``controls-transport-of-chemical``
  * ``reacts-with``
  * ``used-to-produce``

Comment lines starting with ``#`` and blank lines are skipped.
Lines with extra columns beyond the standard three (e.g.
EXTENDED_BINARY_SIF, which adds mediator IDs and data-source
columns) are accepted — only the first three columns are read.

What this importer deliberately does *not* do (yet):

  * Inflect interaction direction by type — every triple becomes a
    directed transition src → dst, even for symmetric interactions
    like ``in-complex-with``. The Petri-net flow is directional by
    construction.
  * Type entities (protein, gene, small molecule, complex). PETRA
    treats every place as a single colourless slot. BioPAX support
    would carry that information through; SIF lost it before
    PETRA ever sees the file.
"""
from __future__ import annotations

from pathlib import Path

from petri_net_nn.petri_net import PetriNet


def parse_sif(path: str | Path) -> PetriNet:
    """Parse a SIF file into a ``PetriNet``.

    Each unique entity becomes a place; each unique interaction
    triple becomes a transition with one input arc (entity_a →
    transition) and one output arc (transition → entity_b).
    Duplicate triples are silently deduplicated, so re-importing
    the same file is idempotent.

    The transition id is ``<src>__<interaction>__<dst>``; the
    label is the natural-language form ``"src interaction dst"``
    so distilled rules and anomaly explanations refer to the
    triple in its original SIF vocabulary.

    Raises ``ValueError`` if any non-comment, non-blank line cannot
    be parsed as a 3-or-more-column tab-separated row with all
    three core fields non-empty.
    """
    path = Path(path)
    net = PetriNet()
    seen: set[tuple[str, str, str]] = set()

    with path.open(encoding="utf-8") as fh:
        for line_num, raw in enumerate(fh, start=1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            # SIF is tab-separated. Extended SIF variants append more
            # columns after the core triple; only the first three are
            # used here.
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(
                    f"{path}:{line_num}: expected at least 3 tab-separated "
                    f"fields (entity_a, interaction_type, entity_b), got "
                    f"{len(parts)}: {line!r}"
                )
            src = parts[0].strip()
            interaction = parts[1].strip()
            dst = parts[2].strip()
            if not src or not interaction or not dst:
                raise ValueError(
                    f"{path}:{line_num}: one of entity_a / interaction_type "
                    f"/ entity_b is empty: {line!r}"
                )

            triple = (src, interaction, dst)
            if triple in seen:
                continue
            seen.add(triple)

            for entity in (src, dst):
                if entity not in net.places:
                    net.add_place(entity, label=entity)

            t_id = f"{src}__{interaction}__{dst}"
            t_label = f"{src} {interaction} {dst}"
            net.add_transition(t_id, label=t_label)
            net.add_arc(src, t_id)
            net.add_arc(t_id, dst)

    return net
