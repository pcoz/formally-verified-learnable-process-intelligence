"""Tests for the SIF (Simple Interaction Format) importer.

SIF is what Pathway Commons publishes biology pathway data in.
Tab-separated triples — ``entity_a  interaction_type  entity_b`` —
plus comments and blank lines. The parser should:

  * map each unique entity to a place;
  * map each unique triple to a transition with one input arc and
    one output arc;
  * dedupe duplicate triples idempotently;
  * skip ``#`` comments and blank lines;
  * accept extended-SIF rows (more columns) by reading only the
    first three;
  * reject malformed rows with a useful error message.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from petri_net_nn import parse_sif


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "pathway.sif"
    p.write_text(content, encoding="utf-8")
    return p


def test_single_triple_produces_two_places_one_transition_two_arcs(tmp_path):
    path = _write(tmp_path, "RAF1\tcontrols-phosphorylation-of\tMAP2K1\n")
    net = parse_sif(path)
    assert net.places == {"RAF1", "MAP2K1"}
    assert net.transitions == {"RAF1__controls-phosphorylation-of__MAP2K1"}
    assert ("RAF1", "RAF1__controls-phosphorylation-of__MAP2K1") in net.flow
    assert ("RAF1__controls-phosphorylation-of__MAP2K1", "MAP2K1") in net.flow
    # The label preserves the original triple for downstream rule
    # extraction and anomaly explanations.
    assert (
        net.transition_labels["RAF1__controls-phosphorylation-of__MAP2K1"]
        == "RAF1 controls-phosphorylation-of MAP2K1"
    )


def test_shared_entity_becomes_a_single_place(tmp_path):
    """MAP2K1 appears in two triples — it should be one place."""
    path = _write(
        tmp_path,
        "RAF1\tcontrols-phosphorylation-of\tMAP2K1\n"
        "MAP2K1\tcontrols-phosphorylation-of\tMAPK1\n",
    )
    net = parse_sif(path)
    assert net.places == {"RAF1", "MAP2K1", "MAPK1"}
    assert len(net.transitions) == 2


def test_duplicate_triples_are_deduplicated(tmp_path):
    """The same triple repeated produces one transition, not two."""
    path = _write(
        tmp_path,
        "RAF1\tcontrols-phosphorylation-of\tMAP2K1\n"
        "RAF1\tcontrols-phosphorylation-of\tMAP2K1\n",
    )
    net = parse_sif(path)
    assert len(net.transitions) == 1


def test_comment_lines_and_blank_lines_are_skipped(tmp_path):
    path = _write(
        tmp_path,
        "# This is a header comment\n"
        "\n"
        "RAF1\tcontrols-phosphorylation-of\tMAP2K1\n"
        "  \n"
        "# Another comment between rows\n"
        "MAP2K1\tcontrols-phosphorylation-of\tMAPK1\n",
    )
    net = parse_sif(path)
    assert net.places == {"RAF1", "MAP2K1", "MAPK1"}
    assert len(net.transitions) == 2


def test_extended_sif_rows_are_read_using_first_three_columns(tmp_path):
    """EXTENDED_BINARY_SIF appends mediator/source/pathway columns
    after the core triple. The parser should accept them and
    ignore everything beyond column three."""
    path = _write(
        tmp_path,
        "RAF1\tcontrols-phosphorylation-of\tMAP2K1\tmediator_id_1\treactome\tMAPK signaling\n",
    )
    net = parse_sif(path)
    assert net.places == {"RAF1", "MAP2K1"}
    assert net.transitions == {"RAF1__controls-phosphorylation-of__MAP2K1"}


def test_malformed_row_raises_value_error_with_line_number(tmp_path):
    """A row with fewer than 3 tab-separated fields is a hard error
    — silently dropping malformed data would hide source-file
    bugs."""
    path = _write(
        tmp_path,
        "RAF1\tcontrols-phosphorylation-of\tMAP2K1\n"
        "this row is malformed\n"
        "MAP2K1\tcontrols-phosphorylation-of\tMAPK1\n",
    )
    with pytest.raises(ValueError) as exc:
        parse_sif(path)
    assert ":2:" in str(exc.value)  # line number


def test_empty_field_raises_value_error(tmp_path):
    """A row with three columns but one of them empty is a hard
    error — the SIF is structurally broken at that line."""
    path = _write(
        tmp_path,
        "RAF1\t\tMAP2K1\n",
    )
    with pytest.raises(ValueError) as exc:
        parse_sif(path)
    assert "empty" in str(exc.value).lower()


def test_self_loop_is_accepted(tmp_path):
    """`EGFR controls-state-change-of EGFR` is a self-loop
    (autophosphorylation in the biology). The parser produces a
    transition with EGFR as both input and output — a valid
    Petri-net structure that the rest of PETRA handles."""
    path = _write(tmp_path, "EGFR\tcontrols-state-change-of\tEGFR\n")
    net = parse_sif(path)
    assert net.places == {"EGFR"}
    assert net.transitions == {"EGFR__controls-state-change-of__EGFR"}
    assert net.validate() == []


def test_parsed_net_is_well_formed(tmp_path):
    """validate() returns no issues on a multi-triple parse — the
    parser produces a structurally sound net out of the box."""
    path = _write(
        tmp_path,
        "RAF1\tcontrols-phosphorylation-of\tMAP2K1\n"
        "MAP2K1\tcontrols-phosphorylation-of\tMAPK1\n"
        "MAPK1\tcontrols-phosphorylation-of\tELK1\n",
    )
    net = parse_sif(path)
    assert net.validate() == []


def test_carriage_return_line_endings_are_stripped(tmp_path):
    """Files originating on Windows often have ``\\r\\n`` line endings;
    parser must strip the carriage return or the last field on each
    row would carry an embedded ``\\r``."""
    path = _write(
        tmp_path,
        "RAF1\tcontrols-phosphorylation-of\tMAP2K1\r\n"
        "MAP2K1\tcontrols-phosphorylation-of\tMAPK1\r\n",
    )
    net = parse_sif(path)
    assert "MAPK1" in net.places
    assert "MAPK1\r" not in net.places
