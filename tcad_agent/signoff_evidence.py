from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    name: str
    status: str
    required: bool = True
    detail: dict[str, Any] = Field(default_factory=dict)


class SignoffEvidencePack(BaseModel):
    verdict: str
    label_zh: str
    score: float
    required_items: list[EvidenceItem] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    next_actions: list[dict[str, Any]] = Field(default_factory=list)


def check_codes(checks: list[Any], severity: str | None = None) -> list[str]:
    codes: list[str] = []
    for check in checks:
        check_severity = getattr(getattr(check, "severity", None), "value", None) or getattr(check, "severity", None)
        if severity and check_severity != severity:
            continue
        code = getattr(check, "code", None)
        if code:
            codes.append(str(code))
    return codes


def has_artifacts(state: dict[str, Any]) -> bool:
    summary = state.get("final_summary") or {}
    artifacts = summary.get("artifacts") or {}
    return bool(artifacts.get("plot") or artifacts.get("csv") or artifacts.get("summary"))


def has_convergence(state: dict[str, Any], matrix: dict[str, Any], metrics: dict[str, Any]) -> bool:
    return (
        matrix.get("convergence_evidence") == "present"
        or state.get("tool_name") in {"mesh_convergence", "tool_convergence"}
        or "relative_delta" in metrics
    )


def merged_metrics(state: dict[str, Any]) -> dict[str, Any]:
    summary = state.get("final_summary") or {}
    metrics = dict(summary.get("metrics") or {})
    quality_metrics = ((state.get("quality_report") or {}).get("metrics") or {})
    metrics.update(quality_metrics)
    return metrics


def evidence_status(condition: bool) -> str:
    return "present" if condition else "missing"


def build_signoff_evidence_pack(
    state: dict[str, Any],
    checks: list[Any],
    *,
    evidence_matrix: dict[str, Any],
    credibility: dict[str, Any] | None = None,
) -> SignoffEvidencePack:
    quality = state.get("quality_report") or {}
    metrics = merged_metrics(state)
    deck = state.get("tcad_deck_spec") or (state.get("request") or {}).get("tcad_deck_spec") or {}
    signoff = deck.get("signoff_requirements") if isinstance(deck, dict) else {}
    if not isinstance(signoff, dict):
        signoff = {}
    strict = isinstance(signoff, dict) and signoff.get("required_level") == "engineering_signoff"
    warning_codes = check_codes(checks, "warning")
    error_codes = check_codes(checks, "error")
    capability = evidence_matrix.get("capability_boundary")

    required_names = [
        "quality_report",
        "curve_artifacts",
        "structured_tcad_spec",
        "physical_benchmark",
    ]
    if strict or signoff.get("require_convergence_evidence"):
        required_names.append("convergence_evidence")
    if strict or signoff.get("measured_curve_path") or signoff.get("golden_metrics"):
        required_names.append("golden_or_measured_comparison")

    items = [
        EvidenceItem(
            name="quality_report",
            status="passed" if quality.get("status") == "passed" else "failed" if quality.get("status") == "failed" else evidence_status(bool(quality)),
            detail={"quality_status": quality.get("status")},
        ),
        EvidenceItem(
            name="curve_artifacts",
            status=evidence_status(has_artifacts(state) or evidence_matrix.get("curve_artifacts") == "present"),
            detail={"matrix": evidence_matrix.get("curve_artifacts")},
        ),
        EvidenceItem(
            name="structured_tcad_spec",
            status=evidence_status(bool(deck) or evidence_matrix.get("deck_spec") == "present"),
            detail={"deck_spec": evidence_matrix.get("deck_spec")},
        ),
        EvidenceItem(
            name="physical_benchmark",
            status=evidence_status(evidence_matrix.get("physical_benchmark") == "present"),
            detail={"warning_codes": warning_codes[:8], "error_codes": error_codes[:8]},
        ),
        EvidenceItem(
            name="convergence_evidence",
            status=evidence_status(has_convergence(state, evidence_matrix, metrics)),
            required="convergence_evidence" in required_names,
            detail={"matrix": evidence_matrix.get("convergence_evidence")},
        ),
        EvidenceItem(
            name="golden_or_measured_comparison",
            status=evidence_status(evidence_matrix.get("golden_or_measured_comparison") == "present"),
            required="golden_or_measured_comparison" in required_names,
            detail={"matrix": evidence_matrix.get("golden_or_measured_comparison")},
        ),
    ]

    required_items = [item for item in items if item.required or item.name in required_names]
    missing = [item.name for item in required_items if item.status in {"missing", "failed"}]
    blocking: list[str] = []
    risk_notes: list[str] = []
    if error_codes:
        blocking.extend(error_codes)
    if capability == "planned_runner_missing":
        blocking.append("planned industrial runner missing")
    if capability == "compact_baseline":
        risk_notes.append("compact baseline is not signoff evidence")
    if any(code.startswith("physics_1d_") for code in warning_codes):
        risk_notes.append("physics_1d evidence needs mesh convergence and measured/golden correlation before strong signoff")
    if warning_codes:
        risk_notes.extend(warning_codes[:8])

    if blocking:
        verdict = "blocked"
        label = "不可签核"
    elif missing:
        verdict = "conditional"
        label = "证据不完整"
    elif risk_notes:
        verdict = "conditional"
        label = "有条件可用"
    else:
        verdict = "ready"
        label = "签核证据齐套"

    score = 1.0
    score -= 0.22 * len(blocking)
    score -= 0.12 * len(missing)
    score -= 0.04 * min(len(risk_notes), 5)
    if credibility and credibility.get("score") is not None:
        score = min(score, float(credibility["score"]))
    score = round(max(score, 0.0), 3)

    next_actions: list[dict[str, Any]] = []
    if "convergence_evidence" in missing:
        next_actions.append({"action": "run_tool_convergence", "reason": "补 mesh/model/bias 收敛证据"})
    if "golden_or_measured_comparison" in missing:
        next_actions.append({"action": "add_golden_or_measured_comparison", "reason": "补 golden/实测曲线或关键指标对比"})
    if "structured_tcad_spec" in missing:
        next_actions.append({"action": "attach_tcad_deck_spec", "reason": "补结构化 TCAD deck/spec"})
    if capability == "compact_baseline":
        next_actions.append({"action": "promote_to_higher_fidelity_runner", "reason": "compact baseline 不能用于最终签核"})
    if capability == "planned_runner_missing":
        next_actions.append({"action": "implement_runner_quality_benchmark", "reason": "planned 工业模板必须先实现真实 runner"})
    if "physics_1d_mesh_convergence_missing" in warning_codes:
        next_actions.append({"action": "run_mesh_or_model_convergence", "reason": "physics_1d 结果需要 mesh/model/bias 收敛证据"})
    if "physics_1d_reference_correlation_missing" in warning_codes:
        next_actions.append({"action": "run_golden_or_measured_correlation", "reason": "physics_1d 结果需要和实测/golden 曲线建立相关性"})
    if error_codes or warning_codes:
        next_actions.append({"action": "run_repair_executor", "reason": "用 benchmark/quality issue 生成修复计划"})

    return SignoffEvidencePack(
        verdict=verdict,
        label_zh=label,
        score=score,
        required_items=required_items,
        missing_evidence=missing,
        blocking_reasons=blocking,
        risk_notes=risk_notes,
        next_actions=next_actions,
    )
