"""Regression: run_florian_core.py command provenance must be complete and executable.

Nora's review of 2c7cd6b found the recorded command omitted the required
``--gate-commit`` and the run's ``--g0``, so it could not be re-executed.
"""

import json
import shlex
from pathlib import Path

import pytest

from run_florian_core import argv_from_args, build_parser, command_from_args

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "argv",
    [
        ["--gate-commit", "ee442bc2c0e4e776dbda9c96754e3f00f9c97e4e", "--g0", "PASS: x; y (4 cells, 2.7e-15 mV)"],
        ["--seeds", "3", "--epochs", "4", "--workers", "7", "--gate-commit", "SMOKE", "--g0", "it's \"quoted\" $HOME"],
        ["--gate-commit", "abc123"],
    ],
)
def test_command_round_trips_every_parsed_argument(argv):
    ap = build_parser()
    args = ap.parse_args(argv)
    cmd = command_from_args(args, "/opt/py bin/python3.11")
    tokens = shlex.split(cmd)
    assert tokens[:2] == ["/opt/py bin/python3.11", "run_florian_core.py"]
    reparsed = ap.parse_args(tokens[2:])
    assert vars(reparsed) == vars(args)
    # Every option the parser knows about is present explicitly, including defaults.
    for action in ap._actions:
        if action.option_strings and action.dest != "help":
            assert action.option_strings[0] in tokens


def test_alternate_gate_and_g0_change_the_recorded_command():
    ap = build_parser()
    a = command_from_args(ap.parse_args(["--gate-commit", "AAA", "--g0", "first"]))
    b = command_from_args(ap.parse_args(["--gate-commit", "BBB", "--g0", "second"]))
    assert "AAA" in a and "first" in a and "BBB" in b and "second" in b and a != b
    assert argv_from_args(ap.parse_args(["--gate-commit", "AAA"]))[-4:] == ["--gate-commit", "AAA", "--g0", "not recorded"]


def test_committed_summary_command_is_executable_and_matches_argv():
    summary = json.loads((ROOT / "results_florian_core" / "summary.json").read_text())
    ap = build_parser()
    argv = summary["argv"]
    assert argv[1] == "run_florian_core.py"
    args = ap.parse_args(argv[2:])  # would SystemExit(2) if --gate-commit were missing
    assert args.gate_commit == summary["gate_commit"]
    assert args.g0 == summary["G0"]
    assert shlex.split(summary["command"]) == argv
