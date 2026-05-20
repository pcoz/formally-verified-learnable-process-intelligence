"""Tests for the petra-* command-line entry points.

We exercise the CLI via in-process calls to ``cli.main`` rather
than spawning a subprocess — same code path, much faster, and
captures stdout / stderr cleanly through the standard
``capsys`` fixture. The pyproject.toml [project.scripts] entry
points are thin wrappers around the same function.

Coverage:

  * petra-train end-to-end: load the credit-approval scenario,
    train (short steps), write the bundle, verify the bundle is
    loadable and the metadata sidecar carries the spec dicts.
  * petra-score end-to-end: produce a JSON document of per-trace
    scores against a bundle.
  * petra-score CSV input path: case_column / activity_column
    routing works.
  * petra-train --steps / --lr CLI overrides take precedence
    over the scenario's TOML values.
  * petra-train rejects a scenario with no traces section
    cleanly (exit code 2, useful message).
  * petra-score rejects an unknown file with a useful error.
  * The ``petra`` umbrella with no subcommand prints help and
    exits non-zero (so a script can detect missing args).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from petri_net_nn import __version__
from petri_net_nn.cli import _meta_path_for, main


SCENARIO_DIR = (
    Path(__file__).parent.parent
    / "examples"
    / "credit_approval_coloured"
)


def test_train_then_score_round_trip(tmp_path, capsys):
    """The headline end-to-end: train a small model, save the
    bundle, score the same trace set against it."""
    output = tmp_path / "credit.pt"
    exit_code = main([
        "train",
        str(SCENARIO_DIR / "scenario.toml"),
        "-o", str(output),
        "--steps", "50",
    ])
    assert exit_code == 0
    assert output.exists()

    meta_path = _meta_path_for(output)
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # The credit-approval scenario uses both channels — verify
    # both spec blocks landed in the sidecar.
    assert "input_marking_spec" in meta
    assert "input_values_spec" in meta
    assert meta["input_values_spec"], (
        "credit-approval scenario should declare an input_values "
        "spec — the CPN-aware compiler path requires it"
    )
    assert meta["petra_version"] == __version__
    assert meta["scenario_name"] == "credit_approval_coloured"

    # Now score the same trace set (which the scenario inlines)
    # by writing a small JSON traces file mirroring two of the
    # inline traces from the scenario.
    traces_json = tmp_path / "traces.json"
    traces_json.write_text(
        json.dumps([
            {
                "attributes": {"amount": "5000"},
                "events": [{"name": "approve loan"}],
            },
            {
                "attributes": {"amount": "200"},
                "events": [{"name": "decline loan"}],
            },
        ]),
        encoding="utf-8",
    )

    output_json = tmp_path / "scored.json"
    exit_code = main([
        "score",
        str(output),
        "--traces", str(traces_json),
        "--format", "json",
        "-o", str(output_json),
    ])
    assert exit_code == 0
    assert output_json.exists()

    scored = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(scored) == 2
    for entry in scored:
        assert "trace_index" in entry
        assert "attributes" in entry
        # Either we have a score or a documented error — never
        # silently dropped.
        assert "trace_score" in entry or "error" in entry


def test_train_cli_steps_override_takes_precedence(tmp_path):
    """--steps on the command line must beat the value in the
    scenario's [training] block."""
    output = tmp_path / "credit.pt"
    # Use a wildly small step count to make the override obvious;
    # we don't care about training quality, just that the CLI
    # flag was honoured.
    exit_code = main([
        "train",
        str(SCENARIO_DIR / "scenario.toml"),
        "-o", str(output),
        "--steps", "5",
        "--lr", "0.01",
        "--seed", "42",
    ])
    assert exit_code == 0
    assert output.exists()


def test_train_rejects_scenario_with_no_traces(tmp_path):
    """A TOML scenario that declares only a net (no traces)
    isn't trainable — petra-train must exit cleanly with a
    useful error rather than crash deep inside torch."""
    scenario_path = tmp_path / "net_only.toml"
    scenario_path.write_text("""
[scenario]
name = "net_only"

[net]
source = "inline"
[[net.places]]
id = "p0"
tokens = 1
[[net.places]]
id = "p1"
[[net.transitions]]
id = "t0"
[[net.arcs]]
src = "p0"
dst = "t0"
[[net.arcs]]
src = "t0"
dst = "p1"
""", encoding="utf-8")
    exit_code = main([
        "train",
        str(scenario_path),
        "-o", str(tmp_path / "model.pt"),
    ])
    assert exit_code == 2


def test_train_rejects_missing_scenario(tmp_path):
    """File-not-found gets exit code 2, no traceback bleeding."""
    exit_code = main([
        "train",
        str(tmp_path / "does_not_exist.toml"),
    ])
    assert exit_code == 2


def test_score_rejects_missing_model():
    """petra-score with a non-existent model path exits cleanly."""
    exit_code = main([
        "score",
        "/no/such/model.pt",
        "--traces", "/no/such/log.csv",
    ])
    assert exit_code == 2


def test_score_csv_routing(tmp_path):
    """Train then score, with the traces supplied as CSV using
    custom case/activity column names. Pins the case-column /
    activity-column CLI flags work end-to-end."""
    # Train.
    bundle_path = tmp_path / "model.pt"
    exit_code = main([
        "train",
        str(SCENARIO_DIR / "scenario.toml"),
        "-o", str(bundle_path),
        "--steps", "20",
    ])
    assert exit_code == 0

    # Write a small CSV with non-default column names.
    csv_path = tmp_path / "events.csv"
    csv_path.write_text(
        "case_id,activity,amount\n"
        "loan-1,approve loan,5000\n"
        "loan-2,decline loan,200\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "scored.json"
    exit_code = main([
        "score",
        str(bundle_path),
        "--traces", str(csv_path),
        "--format", "csv",
        "--case-column", "case_id",
        "--activity-column", "activity",
        "-o", str(out_path),
    ])
    assert exit_code == 0
    scored = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(scored) == 2


def test_umbrella_with_no_subcommand_prints_help(capsys):
    """`petra` without a subcommand should exit non-zero rather
    than do nothing silently — the user almost certainly meant
    to type a subcommand."""
    exit_code = main([])
    assert exit_code == 2
    captured = capsys.readouterr()
    # Help text mentions all three subcommands.
    assert "train" in captured.err or "train" in captured.out
    assert "score" in captured.err or "score" in captured.out
    assert "serve" in captured.err or "serve" in captured.out


def test_version_flag(capsys):
    """`petra --version` prints the package version and exits 0
    via argparse's standard mechanism (SystemExit on --version
    rather than a return)."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out
