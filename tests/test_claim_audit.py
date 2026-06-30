from __future__ import annotations

import json
from pathlib import Path

from morphtamp_x_v2.claim_audit import build_claim_audit, write_claim_audit_markdown
from morphtamp_x_v2.cli import parser


ROOT = Path(__file__).resolve().parents[1]


def test_claim_audit_maps_final_summary_to_supported_research_claims():
    summary = json.loads(
        (ROOT / "evidence" / "heldout_panda_final_summary.json").read_text(
            encoding="utf-8"
        )
    )

    audit = build_claim_audit(summary)

    assert audit["schema_version"] == 1
    assert audit["overall"]["supported_claims"] >= 6
    assert audit["overall"]["unsupported_claims"] >= 1
    by_id = {claim["claim_id"]: claim for claim in audit["claims"]}

    assert by_id["heldout_success"]["supported"] is True
    assert by_id["singularity_margin"]["supported"] is True
    assert by_id["condition_number"]["supported"] is True
    assert by_id["hardware_transfer"]["supported"] is False
    assert "simulation" in by_id["hardware_transfer"]["caveat"].lower()


def test_claim_audit_markdown_is_report_ready():
    summary = {
        "optimization": {
            "total_candidates": 10,
            "feasible_candidates": 3,
            "best_design": "opt_demo",
            "best_morphology_cost": 0.75,
            "best_minimum_reach_margin": 0.05,
            "minimum_reach_margin_constraint": 0.03,
            "best_minimum_sigma": 0.2,
            "constraints": {"minimum_sigma": 0.08, "maximum_condition_number": 30},
            "best_maximum_condition_number": 4.0,
            "best_success_rate": 1.0,
        },
        "robustness": {
            "successful_runs": 12,
            "total_runs": 12,
            "success_rate": 1.0,
            "min_sigma": 0.19,
            "max_condition_number": 5.0,
            "objects": ["cube", "sphere"],
            "tasks": ["under_bridge"],
        },
        "failure_analysis": {"failed_runs": 0, "reason_codes": {}},
    }
    audit = build_claim_audit(summary)
    markdown = write_claim_audit_markdown(audit)

    assert "Claim Audit" in markdown
    assert "heldout_success" in markdown
    assert "hardware_transfer" in markdown
    assert "No" in markdown


def test_cli_accepts_claim_audit_command(tmp_path):
    args = parser().parse_args(
        [
            "claim-audit",
            "--input",
            str(tmp_path / "summary.json"),
            "--output-json",
            str(tmp_path / "audit.json"),
            "--output-md",
            str(tmp_path / "audit.md"),
        ]
    )

    assert args.command == "claim-audit"
    assert args.input.name == "summary.json"
