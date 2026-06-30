from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClaimRule:
    claim_id: str
    claim: str
    evidence_path: str
    relation: str
    threshold: float | int | str | bool | None
    caveat: str


def _get(payload: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _evaluate(observed: Any, relation: str, threshold: Any) -> bool:
    if relation == "exists":
        return observed is not None and observed != ""
    if relation == "equals":
        return observed == threshold
    if relation == "truthy":
        return bool(observed)
    observed_float = _as_float(observed)
    threshold_float = _as_float(threshold)
    if observed_float is None or threshold_float is None:
        return False
    if relation == ">=":
        return observed_float >= threshold_float
    if relation == "<=":
        return observed_float <= threshold_float
    if relation == ">":
        return observed_float > threshold_float
    if relation == "<":
        return observed_float < threshold_float
    raise ValueError(f"unsupported claim relation {relation!r}")


def _rules(summary: dict[str, Any]) -> tuple[ClaimRule, ...]:
    min_reach_margin = _get(summary, "optimization.minimum_reach_margin_constraint", 0.0)
    min_sigma = _get(summary, "optimization.constraints.minimum_sigma", 0.0)
    max_condition = _get(summary, "optimization.constraints.maximum_condition_number", 1e9)
    total_robustness = _get(summary, "robustness.total_runs", 0)
    return (
        ClaimRule(
            "optimization_search",
            "The morphology optimizer evaluated a non-trivial candidate set.",
            "optimization.total_candidates",
            ">=",
            1,
            "This checks search execution, not global optimality.",
        ),
        ClaimRule(
            "feasible_design_found",
            "The search found at least one feasible morphology design.",
            "optimization.feasible_candidates",
            ">",
            0,
            "Feasibility is defined by the implemented simulation constraints.",
        ),
        ClaimRule(
            "best_design_selected",
            "A best morphology design is explicitly selected.",
            "optimization.best_design",
            "exists",
            None,
            "Selection depends on the configured cost and constraint model.",
        ),
        ClaimRule(
            "heldout_success",
            "The selected design preserves full held-out task success.",
            "optimization.best_success_rate",
            ">=",
            1.0,
            "Held-out success is bounded to the encoded object/task protocol.",
        ),
        ClaimRule(
            "reach_margin",
            "The selected design satisfies the required robust reach margin.",
            "optimization.best_minimum_reach_margin",
            ">=",
            min_reach_margin,
            "Reach margin is an equivalent morphology proxy, not hardware clearance.",
        ),
        ClaimRule(
            "singularity_margin",
            "The selected design satisfies the minimum Jacobian singular-value threshold.",
            "optimization.best_minimum_sigma",
            ">=",
            min_sigma,
            "The metric is computed on Panda replay states in simulation.",
        ),
        ClaimRule(
            "condition_number",
            "The selected design satisfies the maximum Jacobian condition-number threshold.",
            "optimization.best_maximum_condition_number",
            "<=",
            max_condition,
            "Condition number is a kinematic quality metric, not a force guarantee.",
        ),
        ClaimRule(
            "robustness_runs",
            "A non-empty robustness benchmark was executed.",
            "robustness.total_runs",
            ">=",
            max(1, int(total_robustness or 0)),
            "The count should be interpreted with the listed perturbation settings.",
        ),
        ClaimRule(
            "robustness_success",
            "The final robustness benchmark has no failed runs.",
            "robustness.success_rate",
            ">=",
            1.0,
            "This does not imply unbounded robustness outside the perturbation radius.",
        ),
        ClaimRule(
            "failure_free_summary",
            "Failure analysis reports zero failed robustness entries.",
            "failure_analysis.failed_runs",
            "equals",
            0,
            "Only failures recorded by the current evaluator are counted.",
        ),
        ClaimRule(
            "hardware_transfer",
            "The results prove real-world hardware or manufacturable Panda redesign performance.",
            "unsupported.hardware_transfer",
            "truthy",
            True,
            "No hardware, force-closure, tactile, actuator-torque, or sim-to-real evidence is present; this remains a simulation claim.",
        ),
    )


def _claim_payload(rule: ClaimRule, summary: dict[str, Any]) -> dict[str, Any]:
    observed = _get(summary, rule.evidence_path)
    supported = _evaluate(observed, rule.relation, rule.threshold)
    return {
        "claim_id": rule.claim_id,
        "claim": rule.claim,
        "evidence_path": rule.evidence_path,
        "observed": observed,
        "relation": rule.relation,
        "threshold": rule.threshold,
        "supported": supported,
        "caveat": rule.caveat,
    }


def build_claim_audit(summary: dict[str, Any]) -> dict[str, Any]:
    claims = [_claim_payload(rule, summary) for rule in _rules(summary)]
    supported = sum(1 for claim in claims if claim["supported"])
    unsupported = len(claims) - supported
    return {
        "schema_version": 1,
        "overall": {
            "claims": len(claims),
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "research_claim_safe": unsupported >= 1 and supported >= 6,
        },
        "claims": claims,
    }


def write_claim_audit_markdown(audit: dict[str, Any]) -> str:
    rows = [
        "# Claim Audit",
        "",
        "This table maps report-level claims to concrete result fields. "
        "It is intentionally conservative: simulation-supported claims are "
        "separated from hardware or sim-to-real claims.",
        "",
        "| Claim ID | Supported | Evidence path | Observed | Threshold | Caveat |",
        "|---|---:|---|---:|---:|---|",
    ]
    for claim in audit["claims"]:
        supported = "Yes" if claim["supported"] else "No"
        observed = claim["observed"]
        threshold = claim["threshold"]
        rows.append(
            f"| `{claim['claim_id']}` | {supported} | `{claim['evidence_path']}` | "
            f"{json.dumps(observed, ensure_ascii=False)} | "
            f"{json.dumps(threshold, ensure_ascii=False)} | {claim['caveat']} |"
        )
    rows.extend(
        [
            "",
            "## Safe wording",
            "",
            "The current evidence supports a bounded MuJoCo/Panda simulation claim "
            "about task-aware equivalent morphology optimization. It does not support "
            "claims of hardware validation, force-closure grasping, or physically "
            "manufacturable Panda redesign.",
            "",
        ]
    )
    return "\n".join(rows)


def load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_claim_audit_files(
    summary: dict[str, Any],
    *,
    output_json: str | Path,
    output_md: str | Path | None = None,
) -> dict[str, Any]:
    audit = build_claim_audit(summary)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if output_md is not None:
        output_md = Path(output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(write_claim_audit_markdown(audit), encoding="utf-8")
    return audit
