from __future__ import annotations

import argparse
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from tcad_agent.deck_writer import write_deck_artifacts
from tcad_agent.metrics import (
    extract_diode_reverse_metrics,
    extract_pn_iv_metrics,
    load_iv_points,
)
from tcad_agent.physical_quality import check_diode_breakdown_physics
from tcad_agent.tools.pn_junction_iv import PNJunctionIVRequest, run_pn_junction_iv_sweep
from tcad_agent.tools.result_judge import IssueSeverity, QualityIssue, QualityStatus, add_issue


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DiodeBreakdownStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DiodeBreakdownRequest(BaseModel):
    start: float = Field(default=0.0, description="Start bias in volts.")
    stop: float = Field(default=-5.0, description="Reverse stop bias in volts.")
    step: float = Field(default=0.5, gt=0.0, description="Initial bias step magnitude in volts.")
    min_step: float = Field(default=0.0625, gt=0.0)
    max_attempts: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=300.0, gt=0.0)
    breakdown_current_a: float = Field(default=1e-6, gt=0.0)
    leakage_voltage_v: float = Field(default=-1.0)
    require_breakdown: bool = False
    quality_min_points: int = Field(default=3, ge=1)
    quality_max_abs_current_a: float = Field(default=1.0, gt=0.0)
    quality_max_leakage_abs_current_a: float = Field(default=1e-3, gt=0.0)
    quality_max_convergence_failures: int = Field(default=0, ge=0)
    length_um: float = Field(default=0.1, gt=0.0)
    junction_um: float = Field(default=0.05, gt=0.0)
    p_doping_cm3: float = Field(default=1.0e18, gt=0.0)
    n_doping_cm3: float = Field(default=1.0e18, gt=0.0)
    temperature_k: float = Field(default=300.0, gt=0.0)
    electron_lifetime_s: float = Field(default=1.0e-8, gt=0.0)
    hole_lifetime_s: float = Field(default=1.0e-8, gt=0.0)
    contact_spacing_um: float = Field(default=0.001, gt=0.0)
    junction_spacing_um: float = Field(default=1.0e-5, gt=0.0)
    run_id: str | None = None
    run_root: Path = PROJECT_ROOT / "runs" / "agent_tools"
    resume: bool = False
    tcad_deck_spec: dict[str, Any] | None = None
    tcad_deck_mutations: list[dict[str, Any]] = Field(default_factory=list)
    deck_patch_history: list[dict[str, Any]] = Field(default_factory=list)
    source_deck_path: str | None = None
    repair_source_state_path: str | None = None
    repair_baseline_state_path: str | None = None

    @model_validator(mode="after")
    def validate_request(self) -> "DiodeBreakdownRequest":
        if self.stop == self.start:
            raise ValueError("stop must differ from start")
        if min(self.start, self.stop) >= 0:
            raise ValueError("diode breakdown/leakage sweep requires at least one negative reverse-bias point")
        if self.leakage_voltage_v > 0:
            raise ValueError("leakage_voltage_v must be zero or negative")
        if self.min_step > self.step:
            raise ValueError("min_step must be less than or equal to step")
        if self.junction_um >= self.length_um:
            raise ValueError("junction_um must be less than length_um")
        return self


class DiodeBreakdownState(BaseModel):
    tool_name: str = "diode_breakdown_leakage_sweep"
    status: DiodeBreakdownStatus
    run_id: str
    run_dir: str
    request: dict[str, Any]
    created_at: str
    updated_at: str
    pn_state_path: str | None = None
    final_summary: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None
    next_action: str | None = None
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    tcad_deck_spec: dict[str, Any] | None = None
    tcad_deck_mutations: list[dict[str, Any]] = Field(default_factory=list)


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_state(state: DiodeBreakdownState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")


def load_state(path: Path) -> DiodeBreakdownState:
    return DiodeBreakdownState.model_validate_json(path.read_text(encoding="utf-8"))


def create_initial_state(request: DiodeBreakdownRequest, run_id: str, run_dir: Path) -> DiodeBreakdownState:
    now = utc_timestamp()
    return DiodeBreakdownState(
        status=DiodeBreakdownStatus.RUNNING,
        run_id=run_id,
        run_dir=str(run_dir),
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        tcad_deck_spec=request.tcad_deck_spec,
        tcad_deck_mutations=request.tcad_deck_mutations,
        next_action="launch reverse-bias PN junction sweep",
        checkpoint={
            "current_step_v": request.step,
            "breakdown_current_a": request.breakdown_current_a,
            "leakage_voltage_v": request.leakage_voltage_v,
        },
    )


def prepare_state(request: DiodeBreakdownRequest) -> tuple[DiodeBreakdownState, Path]:
    run_id = request.run_id or default_run_id()
    run_dir = request.run_root / "diode_breakdown" / run_id
    state_path = run_dir / "state.json"
    if request.resume:
        if not state_path.exists():
            raise FileNotFoundError(f"Cannot resume; state file does not exist: {state_path}")
        return load_state(state_path), state_path
    run_dir.mkdir(parents=True, exist_ok=False)
    state = create_initial_state(request, run_id, run_dir)
    write_state(state, state_path)
    return state, state_path


def pn_request_from_diode_request(request: DiodeBreakdownRequest, state: DiodeBreakdownState) -> PNJunctionIVRequest:
    run_dir = Path(state.run_dir)
    return PNJunctionIVRequest(
        start=request.start,
        stop=request.stop,
        step=request.step,
        min_step=request.min_step,
        max_attempts=request.max_attempts,
        timeout_seconds=request.timeout_seconds,
        quality_min_points=request.quality_min_points,
        quality_max_abs_current_a=request.quality_max_abs_current_a,
        quality_max_convergence_failures=request.quality_max_convergence_failures,
        length_um=request.length_um,
        junction_um=request.junction_um,
        p_doping_cm3=request.p_doping_cm3,
        n_doping_cm3=request.n_doping_cm3,
        temperature_k=request.temperature_k,
        electron_lifetime_s=request.electron_lifetime_s,
        hole_lifetime_s=request.hole_lifetime_s,
        contact_spacing_um=request.contact_spacing_um,
        junction_spacing_um=request.junction_spacing_um,
        run_id=f"{state.run_id}_pn_reverse",
        run_root=run_dir / "inner_agent_tools",
        resume=False,
    )


def count_convergence_failures(pn_state: dict[str, Any]) -> int:
    return sum(1 for attempt in pn_state.get("attempts") or [] if attempt.get("failure_class") == "convergence")


def choose_recommended_action(status: QualityStatus, issues: list[QualityIssue], metrics: dict[str, Any]) -> str:
    codes = {issue.code for issue in issues}
    if status == QualityStatus.FAILED:
        return "rerun the reverse sweep after fixing failed artifacts or sweep settings"
    if "too_many_convergence_failures" in codes:
        return "rerun with a smaller initial reverse-bias step or relaxed solver settings"
    if "reverse_current_not_monotonic" in codes:
        return "review the reverse IV curve and rerun with a smaller step around the suspicious segment"
    if "leakage_exceeds_policy" in codes:
        return "treat leakage as physically suspicious; verify geometry, doping, lifetime, and boundary conditions"
    if not metrics.get("breakdown_detected"):
        return "accept leakage metrics; extend reverse stop voltage if a breakdown voltage is required"
    return "accept extracted leakage and breakdown metrics"


def judge_diode_breakdown(
    summary: dict[str, Any],
    pn_state: dict[str, Any],
    request: DiodeBreakdownRequest,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    issues: list[QualityIssue] = []
    if summary.get("status") != "completed":
        add_issue(
            issues,
            "summary_not_completed",
            IssueSeverity.ERROR,
            "Underlying PN runner summary status is not completed.",
            {"status": summary.get("status")},
        )
    if metrics.get("points", 0) < request.quality_min_points:
        add_issue(
            issues,
            "too_few_points",
            IssueSeverity.WARNING,
            "Reverse sweep has too few points for leakage/breakdown extraction.",
            {"points": metrics.get("points"), "min_points": request.quality_min_points},
        )
    if metrics.get("reverse_points", 0) < 1:
        add_issue(
            issues,
            "no_reverse_bias_points",
            IssueSeverity.ERROR,
            "Sweep did not include any negative reverse-bias points.",
            {"voltage_range_v": metrics.get("voltage_range_v")},
        )
    leakage_abs = metrics.get("leakage_abs_current_at_target_a")
    if leakage_abs is not None and float(leakage_abs) > request.quality_max_leakage_abs_current_a:
        add_issue(
            issues,
            "leakage_exceeds_policy",
            IssueSeverity.WARNING,
            "Reverse leakage at the target voltage exceeds the configured threshold.",
            {
                "leakage_abs_current_at_target_a": leakage_abs,
                "threshold_a": request.quality_max_leakage_abs_current_a,
                "leakage_voltage_used_v": metrics.get("leakage_voltage_used_v"),
            },
        )
    max_reverse = metrics.get("max_reverse_abs_current_a")
    if max_reverse is not None and float(max_reverse) > request.quality_max_abs_current_a:
        add_issue(
            issues,
            "current_exceeds_policy",
            IssueSeverity.WARNING,
            "Reverse current exceeds the configured plausibility threshold.",
            {"max_reverse_abs_current_a": max_reverse, "threshold_a": request.quality_max_abs_current_a},
        )
    if int(metrics.get("reverse_current_shape_violations") or 0) > 0:
        add_issue(
            issues,
            "reverse_current_not_monotonic",
            IssueSeverity.WARNING,
            "Absolute reverse current decreases as reverse-bias magnitude increases.",
            {"violations": metrics.get("reverse_current_shape_violations")},
        )
    if request.require_breakdown and not metrics.get("breakdown_detected"):
        add_issue(
            issues,
            "breakdown_not_reached",
            IssueSeverity.WARNING,
            "Breakdown threshold was not reached inside the simulated reverse-bias range.",
            {
                "threshold_a": request.breakdown_current_a,
                "min_reverse_voltage_v": metrics.get("min_reverse_voltage_v"),
            },
        )
    convergence_failures = count_convergence_failures(pn_state)
    if convergence_failures > request.quality_max_convergence_failures:
        add_issue(
            issues,
            "too_many_convergence_failures",
            IssueSeverity.WARNING,
            "Underlying PN sweep completed only after convergence failures.",
            {
                "convergence_failures": convergence_failures,
                "max_allowed": request.quality_max_convergence_failures,
            },
        )
    physical_params = dict(summary.get("parameters") or {})
    physical_params.update(request.model_dump(mode="json"))
    for physical_issue in check_diode_breakdown_physics(metrics, physical_params):
        add_issue(
            issues,
            str(physical_issue.get("code")),
            IssueSeverity.ERROR if physical_issue.get("severity") == "error" else IssueSeverity.WARNING,
            str(physical_issue.get("message") or physical_issue.get("code")),
            dict(physical_issue.get("evidence") or {}),
        )
    status = QualityStatus.FAILED if any(issue.severity == IssueSeverity.ERROR for issue in issues) else (
        QualityStatus.SUSPICIOUS if issues else QualityStatus.PASSED
    )
    return {
        "status": status,
        "issues": [issue.model_dump(mode="json") for issue in issues],
        "metrics": metrics,
        "recommended_next_action": choose_recommended_action(status, issues, metrics),
    }


def augment_summary(
    summary: dict[str, Any],
    request: DiodeBreakdownRequest,
) -> dict[str, Any]:
    artifacts = summary.get("artifacts") or {}
    csv_path = artifacts.get("csv")
    if not csv_path:
        raise FileNotFoundError("Underlying PN summary does not include IV CSV artifact.")
    points = load_iv_points(Path(csv_path))
    temperature_k = float((summary.get("parameters") or {}).get("temperature_k") or request.temperature_k)
    metrics = extract_pn_iv_metrics(
        points,
        temperature_k=temperature_k,
        breakdown_current_a=request.breakdown_current_a,
    )
    metrics.update(
        extract_diode_reverse_metrics(
            points,
            leakage_voltage_v=request.leakage_voltage_v,
            breakdown_current_a=request.breakdown_current_a,
        )
    )
    augmented = dict(summary)
    augmented["task"] = "diode_breakdown_leakage_sweep"
    augmented["source_task"] = summary.get("task")
    augmented["breakdown_current_threshold_a"] = request.breakdown_current_a
    augmented["leakage_voltage_target_v"] = request.leakage_voltage_v
    augmented["diode_reverse_metrics"] = metrics
    augmented["extracted_metrics"] = metrics
    return augmented


def run_diode_breakdown_sweep(request: DiodeBreakdownRequest) -> dict[str, Any]:
    state, state_path = prepare_state(request)
    if state.status == DiodeBreakdownStatus.COMPLETED:
        return state.model_dump(mode="json")

    pn_request = pn_request_from_diode_request(request, state)
    pn_state = run_pn_junction_iv_sweep(pn_request)
    pn_state_path = Path(pn_request.run_root) / "pn_junction_iv" / str(pn_request.run_id) / "state.json"
    state.pn_state_path = str(pn_state_path)
    state.checkpoint["pn_state_path"] = str(pn_state_path)
    write_state(state, state_path)

    if pn_state.get("status") != "completed" or not pn_state.get("final_summary"):
        state.status = DiodeBreakdownStatus.FAILED
        state.next_action = "inspect underlying PN reverse sweep failure"
        state.checkpoint["last_failure_reason"] = (pn_state.get("checkpoint") or {}).get("last_failure_reason")
        write_state(state, state_path)
        return state.model_dump(mode="json")

    summary = augment_summary(pn_state["final_summary"], request)
    quality_report = judge_diode_breakdown(
        summary,
        pn_state,
        request,
        summary["diode_reverse_metrics"],
    )
    deck_artifacts = write_deck_artifacts(
        Path(state.run_dir),
        tool_name="diode_breakdown_leakage_sweep",
        request=request.model_dump(mode="json"),
        deck_spec=request.tcad_deck_spec,
        mutations=request.tcad_deck_mutations,
        source_goal_text=(request.tcad_deck_spec or {}).get("source_goal_text") if request.tcad_deck_spec else None,
    )
    summary.setdefault("artifacts", {}).update(deck_artifacts)
    summary["tcad_deck_spec"] = request.tcad_deck_spec
    summary["tcad_deck_mutations"] = request.tcad_deck_mutations
    state.status = DiodeBreakdownStatus.COMPLETED
    state.final_summary = summary
    state.quality_report = quality_report
    state.next_action = quality_report["recommended_next_action"]
    state.checkpoint = {
        "pn_state_path": str(pn_state_path),
        "quality_status": quality_report["status"],
        "breakdown_detected": quality_report["metrics"].get("breakdown_detected"),
        "breakdown_voltage_at_threshold_v": quality_report["metrics"].get("breakdown_voltage_at_threshold_v"),
        "leakage_abs_current_at_target_a": quality_report["metrics"].get("leakage_abs_current_at_target_a"),
    }
    write_state(state, state_path)
    return state.model_dump(mode="json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-callable diode reverse leakage / breakdown sweep tool.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stop", type=float, default=-5.0)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--min-step", type=float, default=0.0625)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--breakdown-current-a", type=float, default=1e-6)
    parser.add_argument("--leakage-voltage-v", type=float, default=-1.0)
    parser.add_argument("--require-breakdown", action="store_true")
    parser.add_argument("--quality-min-points", type=int, default=3)
    parser.add_argument("--quality-max-abs-current-a", type=float, default=1.0)
    parser.add_argument("--quality-max-leakage-abs-current-a", type=float, default=1e-3)
    parser.add_argument("--quality-max-convergence-failures", type=int, default=0)
    parser.add_argument("--length-um", type=float, default=0.1)
    parser.add_argument("--junction-um", type=float, default=0.05)
    parser.add_argument("--p-doping-cm3", type=float, default=1.0e18)
    parser.add_argument("--n-doping-cm3", type=float, default=1.0e18)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--electron-lifetime-s", type=float, default=1.0e-8)
    parser.add_argument("--hole-lifetime-s", type=float, default=1.0e-8)
    parser.add_argument("--contact-spacing-um", type=float, default=0.001)
    parser.add_argument("--junction-spacing-um", type=float, default=1.0e-5)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs" / "agent_tools")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> DiodeBreakdownRequest:
    return DiodeBreakdownRequest(
        start=args.start,
        stop=args.stop,
        step=args.step,
        min_step=args.min_step,
        max_attempts=args.max_attempts,
        timeout_seconds=args.timeout_seconds,
        breakdown_current_a=args.breakdown_current_a,
        leakage_voltage_v=args.leakage_voltage_v,
        require_breakdown=args.require_breakdown,
        quality_min_points=args.quality_min_points,
        quality_max_abs_current_a=args.quality_max_abs_current_a,
        quality_max_leakage_abs_current_a=args.quality_max_leakage_abs_current_a,
        quality_max_convergence_failures=args.quality_max_convergence_failures,
        length_um=args.length_um,
        junction_um=args.junction_um,
        p_doping_cm3=args.p_doping_cm3,
        n_doping_cm3=args.n_doping_cm3,
        temperature_k=args.temperature_k,
        electron_lifetime_s=args.electron_lifetime_s,
        hole_lifetime_s=args.hole_lifetime_s,
        contact_spacing_um=args.contact_spacing_um,
        junction_spacing_um=args.junction_spacing_um,
        run_id=args.run_id,
        run_root=args.run_root,
        resume=args.resume,
    )


def main() -> None:
    try:
        result = run_diode_breakdown_sweep(request_from_args(parse_args()))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit(0 if result["status"] == DiodeBreakdownStatus.COMPLETED else 1)
    except (ValidationError, ValueError, FileExistsError, FileNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "tool_name": "diode_breakdown_leakage_sweep",
                    "status": DiodeBreakdownStatus.FAILED,
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
