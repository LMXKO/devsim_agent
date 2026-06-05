from __future__ import annotations

import argparse
import json
import math
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.metrics import IVPoint, extract_pn_iv_metrics, load_iv_points
from tcad_agent.physical_quality import check_parameter_sanity


class QualityStatus(str, Enum):
    PASSED = "passed"
    SUSPICIOUS = "suspicious"
    FAILED = "failed"


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class QualityIssue(BaseModel):
    code: str
    severity: IssueSeverity
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class QualityPolicy(BaseModel):
    min_points: int = 3
    max_abs_current_a: float = 1.0
    monotonic_tolerance_a: float = 1e-18
    max_convergence_failures: int = 0
    min_ideality_factor: float = 0.5
    max_ideality_factor: float = 5.0
    min_rectification_ratio: float = 10.0
    max_voltage_span_v: float = 200.0
    min_temperature_k: float = 150.0
    max_temperature_k: float = 500.0


class QualityReport(BaseModel):
    status: QualityStatus
    issues: list[QualityIssue]
    metrics: dict[str, Any]
    recommended_next_action: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_iv_csv(path: Path) -> list[IVPoint]:
    return load_iv_points(path)


def add_issue(
    issues: list[QualityIssue],
    code: str,
    severity: IssueSeverity,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    issues.append(
        QualityIssue(
            code=code,
            severity=severity,
            message=message,
            evidence=evidence or {},
        )
    )


def is_finite(value: float) -> bool:
    return math.isfinite(value)


def count_convergence_failures(attempts: list[dict[str, Any]]) -> int:
    return sum(1 for attempt in attempts if attempt.get("failure_class") == "convergence")


def check_required_artifacts(summary: dict[str, Any], issues: list[QualityIssue]) -> None:
    artifacts = summary.get("artifacts") or {}
    required = {
        "csv": IssueSeverity.ERROR,
        "log": IssueSeverity.WARNING,
        "plot": IssueSeverity.WARNING,
        "tecplot": IssueSeverity.WARNING,
    }
    for name, severity in required.items():
        value = artifacts.get(name)
        if not value:
            add_issue(
                issues,
                f"missing_artifact_{name}",
                severity,
                f"Summary does not include artifact path for {name}.",
            )
            continue
        if not Path(value).exists():
            add_issue(
                issues,
                f"missing_artifact_file_{name}",
                severity,
                f"Artifact file for {name} does not exist.",
                {"path": value},
            )


def check_iv_points(points: list[IVPoint], policy: QualityPolicy, issues: list[QualityIssue]) -> None:
    if not points:
        add_issue(
            issues,
            "empty_iv_curve",
            IssueSeverity.ERROR,
            "IV sweep has no data points.",
        )
        return

    if len(points) < policy.min_points:
        add_issue(
            issues,
            "too_few_points",
            IssueSeverity.WARNING,
            "IV sweep has too few points for a reliable trend check.",
            {"points": len(points), "min_points": policy.min_points},
        )

    for index, point in enumerate(points):
        values = point.model_dump()
        nonfinite = {key: value for key, value in values.items() if not is_finite(value)}
        if nonfinite:
            add_issue(
                issues,
                "nonfinite_value",
                IssueSeverity.ERROR,
                "IV sweep contains NaN or infinite values.",
                {"row": index, "values": nonfinite},
            )

    voltages = [point.voltage_v for point in points]
    increasing = all(right >= left for left, right in zip(voltages[:-1], voltages[1:]))
    decreasing = all(right <= left for left, right in zip(voltages[:-1], voltages[1:]))
    if not (increasing or decreasing):
        add_issue(
            issues,
            "voltage_not_monotonic",
            IssueSeverity.ERROR,
            "Voltage sweep is not monotonic.",
            {"voltages": voltages},
        )

    abs_currents = [abs(point.total_current_a) for point in points]
    max_abs_current = max(abs_currents)
    if max_abs_current > policy.max_abs_current_a:
        add_issue(
            issues,
            "current_exceeds_policy",
            IssueSeverity.WARNING,
            "Total current exceeds the configured plausibility threshold.",
            {
                "max_abs_current_a": max_abs_current,
                "threshold_a": policy.max_abs_current_a,
            },
        )

    check_current_shape(points, policy, issues)


def check_current_shape(points: list[IVPoint], policy: QualityPolicy, issues: list[QualityIssue]) -> None:
    forward_points = sorted(
        [point for point in points if point.voltage_v >= 0],
        key=lambda point: point.voltage_v,
    )
    forward_abs = [abs(point.total_current_a) for point in forward_points]
    for index, (left, right) in enumerate(zip(forward_abs, forward_abs[1:]), start=1):
        if right + policy.monotonic_tolerance_a < left:
            add_issue(
                issues,
                "current_not_monotonic",
                IssueSeverity.WARNING,
                "Absolute total current decreases during forward voltage sweep.",
                {"row": index, "previous_abs_current_a": left, "current_abs_current_a": right},
            )

    reverse_points = sorted(
        [point for point in points if point.voltage_v <= 0],
        key=lambda point: abs(point.voltage_v),
    )
    reverse_abs = [abs(point.total_current_a) for point in reverse_points]
    for index, (left, right) in enumerate(zip(reverse_abs, reverse_abs[1:]), start=1):
        if right + policy.monotonic_tolerance_a < left:
            add_issue(
                issues,
                "reverse_current_not_monotonic",
                IssueSeverity.WARNING,
                "Absolute total current decreases as reverse-bias magnitude increases.",
                {"row": index, "previous_abs_current_a": left, "current_abs_current_a": right},
            )


def summary_temperature_k(summary: dict[str, Any]) -> float:
    params = summary.get("parameters") or {}
    try:
        return float(params.get("temperature_k") or 300.0)
    except (TypeError, ValueError):
        return 300.0


def summarize_metrics(
    points: list[IVPoint],
    attempts: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    if not points:
        return {
            "points": 0,
            "convergence_failures": count_convergence_failures(attempts),
        }
    metrics = extract_pn_iv_metrics(points, temperature_k=summary_temperature_k(summary))
    metrics["convergence_failures"] = count_convergence_failures(attempts)
    return metrics


def check_physical_metrics(
    metrics: dict[str, Any],
    summary: dict[str, Any],
    policy: QualityPolicy,
    issues: list[QualityIssue],
) -> None:
    voltage_range = metrics.get("voltage_range_v") or []
    if len(voltage_range) == 2 and abs(float(voltage_range[1]) - float(voltage_range[0])) > policy.max_voltage_span_v:
        add_issue(
            issues,
            "voltage_span_unusually_large",
            IssueSeverity.WARNING,
            "Voltage span is unusually large for a PN IV run; check voltage units.",
            {"voltage_range_v": voltage_range, "max_voltage_span_v": policy.max_voltage_span_v},
        )

    temperature_k = summary_temperature_k(summary)
    if not policy.min_temperature_k <= temperature_k <= policy.max_temperature_k:
        add_issue(
            issues,
            "temperature_out_of_expected_range",
            IssueSeverity.WARNING,
            "Device temperature is outside the expected sanity range.",
            {
                "temperature_k": temperature_k,
                "expected_range_k": [policy.min_temperature_k, policy.max_temperature_k],
            },
        )

    ideality = metrics.get("ideality_factor_estimate")
    if ideality is not None and not policy.min_ideality_factor <= float(ideality) <= policy.max_ideality_factor:
        add_issue(
            issues,
            "ideality_factor_out_of_range",
            IssueSeverity.WARNING,
            "Estimated diode ideality factor is outside the configured plausibility range.",
            {
                "ideality_factor_estimate": ideality,
                "expected_range": [policy.min_ideality_factor, policy.max_ideality_factor],
            },
        )

    rectification = metrics.get("rectification_ratio_final_to_leakage")
    if (
        rectification is not None
        and int(metrics.get("forward_points") or 0) > 0
        and float(rectification) < policy.min_rectification_ratio
    ):
        add_issue(
            issues,
            "low_rectification_ratio",
            IssueSeverity.WARNING,
            "Forward current is not much larger than leakage/current near zero bias.",
            {
                "rectification_ratio_final_to_leakage": rectification,
                "min_rectification_ratio": policy.min_rectification_ratio,
            },
        )

    for physical_issue in check_parameter_sanity(summary.get("parameters") or {}, check_temperature=False):
        add_issue(
            issues,
            str(physical_issue["code"]),
            IssueSeverity(str(physical_issue["severity"])),
            str(physical_issue["message"]),
            physical_issue.get("evidence") or {},
        )


def choose_status(issues: list[QualityIssue]) -> QualityStatus:
    if any(issue.severity == IssueSeverity.ERROR for issue in issues):
        return QualityStatus.FAILED
    if issues:
        return QualityStatus.SUSPICIOUS
    return QualityStatus.PASSED


def choose_recommended_action(status: QualityStatus, issues: list[QualityIssue]) -> str:
    codes = {issue.code for issue in issues}
    if status == QualityStatus.PASSED:
        return "accept result artifacts and proceed to the next TCAD task"
    if "empty_iv_curve" in codes or "nonfinite_value" in codes or "voltage_not_monotonic" in codes:
        return "rerun the simulation with corrected sweep settings before using the result"
    if "current_exceeds_policy" in codes:
        return "treat the run as numerically completed but physically suspicious; narrow the voltage range or tighten the device model before accepting"
    if "too_many_convergence_failures" in codes:
        return "review the solver log and rerun with a smaller initial bias step"
    if status == QualityStatus.SUSPICIOUS:
        return "review warnings before accepting the result"
    return "stop and inspect failed artifacts"


def judge_pn_junction_iv(
    summary: dict[str, Any],
    attempts: list[dict[str, Any]] | None = None,
    policy: QualityPolicy | None = None,
) -> QualityReport:
    attempts = attempts or []
    policy = policy or QualityPolicy()
    issues: list[QualityIssue] = []

    if summary.get("status") != "completed":
        add_issue(
            issues,
            "summary_not_completed",
            IssueSeverity.ERROR,
            "Runner summary status is not completed.",
            {"status": summary.get("status")},
        )

    check_required_artifacts(summary, issues)
    csv_path_value = (summary.get("artifacts") or {}).get("csv")
    points: list[IVPoint] = []
    if csv_path_value and Path(csv_path_value).exists():
        try:
            points = load_iv_csv(Path(csv_path_value))
        except (KeyError, ValueError) as exc:
            add_issue(
                issues,
                "csv_parse_error",
                IssueSeverity.ERROR,
                "Could not parse IV CSV.",
                {"path": csv_path_value, "error": str(exc)},
            )
    else:
        add_issue(
            issues,
            "csv_missing",
            IssueSeverity.ERROR,
            "IV CSV is required for result judging.",
            {"path": csv_path_value},
        )

    check_iv_points(points, policy, issues)

    convergence_failures = count_convergence_failures(attempts)
    if convergence_failures > policy.max_convergence_failures:
        add_issue(
            issues,
            "too_many_convergence_failures",
            IssueSeverity.WARNING,
            "Run completed only after convergence failures.",
            {
                "convergence_failures": convergence_failures,
                "max_allowed": policy.max_convergence_failures,
            },
        )

    metrics = summarize_metrics(points, attempts, summary)
    check_physical_metrics(metrics, summary, policy, issues)
    status = choose_status(issues)
    return QualityReport(
        status=status,
        issues=issues,
        metrics=metrics,
        recommended_next_action=choose_recommended_action(status, issues),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge PN junction IV result quality.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--max-abs-current-a", type=float, default=1.0)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--max-convergence-failures", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_json(args.summary)
    attempts = []
    if args.state:
        attempts = load_json(args.state).get("attempts", [])
    report = judge_pn_junction_iv(
        summary,
        attempts=attempts,
        policy=QualityPolicy(
            min_points=args.min_points,
            max_abs_current_a=args.max_abs_current_a,
            max_convergence_failures=args.max_convergence_failures,
        ),
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if report.status != QualityStatus.FAILED else 1)


if __name__ == "__main__":
    main()
