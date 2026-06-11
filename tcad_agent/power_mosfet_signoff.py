from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.golden_curve import GoldenCurveComparisonRequest, run_golden_curve_comparison
from tcad_agent.physical_benchmark import run_physical_benchmark
from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.tool_convergence import ToolConvergenceRequest, run_tool_convergence
from tcad_agent.tools.extended_device_sweep import ExtendedDeviceRequest, ExtendedDeviceType, run_extended_device_sweep


class PowerMOSFETSignoffRequest(BaseModel):
    run_id: str | None = None
    run_root: Path = PROJECT_ROOT / "runs" / "power_mosfet_signoff"
    execute: bool = True
    baseline_request: dict[str, Any] = Field(default_factory=dict)
    run_convergence: bool = True
    convergence_values: list[float] = Field(default_factory=lambda: [0.02, 0.01, 0.005])
    convergence_relative_tolerance: float = 0.2
    reference_curve_path: Path | None = None
    timeout_seconds: float = 300.0


class PowerMOSFETSignoffState(BaseModel):
    tool_name: str = "power_mosfet_signoff"
    schema_version: str = "actsoft.tcad.power_mosfet_signoff.v1"
    status: str
    run_id: str
    run_dir: str
    created_at: str
    updated_at: str
    request: dict[str, Any]
    baseline_state_path: str | None = None
    benchmark_path: str | None = None
    convergence_state_path: str | None = None
    golden_comparison_state_path: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    final_summary: dict[str, Any] = Field(default_factory=dict)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    signoff_gate: dict[str, Any] = Field(default_factory=dict)
    next_action: str | None = None
    failure_reason: str | None = None
    state_path: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_run_id() -> str:
    return f"power_mos_signoff_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def default_baseline_request(request: PowerMOSFETSignoffRequest, run_dir: Path) -> dict[str, Any]:
    payload = {
        "device_type": ExtendedDeviceType.POWER_MOSFET_BV_RON.value,
        "fidelity": "devsim_2d_field_plate",
        "evidence_level": "tcad_executable",
        "start": 0.0,
        "stop": -60.0,
        "step": 10.0,
        "quality_min_points": 3,
        "timeout_seconds": request.timeout_seconds,
        "run_id": f"{request.run_id or 'power_mos_signoff'}_baseline_2d",
        "run_root": str(run_dir / "baseline"),
    }
    payload.update(request.baseline_request)
    payload["device_type"] = ExtendedDeviceType.POWER_MOSFET_BV_RON.value
    payload["fidelity"] = "devsim_2d_field_plate"
    payload["evidence_level"] = "tcad_executable"
    return payload


def state_path_for_extended_result(result: Any) -> str:
    run_dir = Path(str(result.run_dir if hasattr(result, "run_dir") else result.get("run_dir")))
    return str((run_dir / "state.json").resolve())


def build_signoff_gate(
    *,
    baseline_status: str | None,
    benchmark: dict[str, Any] | None,
    convergence: dict[str, Any] | None,
    golden: dict[str, Any] | None,
) -> dict[str, Any]:
    benchmark_summary = (benchmark or {}).get("summary") or {}
    benchmark_pack = benchmark_summary.get("signoff_evidence_pack") or {}
    convergence_quality = (convergence or {}).get("quality_report") or {}
    golden_quality = (golden or {}).get("quality_report") or {}
    missing: list[str] = []
    blocking: list[str] = []
    if baseline_status != "completed":
        blocking.append("power_mosfet_2d_baseline_not_completed")
    if not benchmark:
        missing.append("physical_benchmark")
    elif benchmark.get("status") == "failed":
        blocking.extend(benchmark_summary.get("blocking_codes") or ["physical_benchmark_failed"])
    if not convergence:
        missing.append("mesh_model_convergence")
    elif convergence_quality.get("status") in {"failed"}:
        blocking.append("mesh_model_convergence_failed")
    elif convergence_quality.get("status") in {"suspicious", "planned"}:
        missing.append("mesh_model_convergence_clean_pass")
    if not golden:
        missing.append("golden_or_measured_correlation")
    elif golden_quality.get("status") == "failed":
        blocking.append("golden_or_measured_correlation_failed")
    elif golden_quality.get("status") in {"suspicious", "planned"}:
        missing.append("golden_or_measured_correlation_clean_pass")
    if benchmark_pack.get("blocking_reasons"):
        blocking.extend(str(item) for item in benchmark_pack.get("blocking_reasons") or [])
    verdict = "blocked" if blocking else "conditional" if missing else "ready"
    next_actions = [
        item
        for item in [
            {"action": "run_tool_convergence", "reason": "补 Power MOSFET 2D mesh/model convergence"}
            if "mesh_model_convergence" in missing or "mesh_model_convergence_clean_pass" in missing
            else None,
            {"action": "add_golden_or_measured_correlation", "reason": "补实测/golden 曲线相关性"}
            if any("golden_or_measured" in item for item in missing)
            else None,
            {"action": "inspect_blocking_reason", "reason": ", ".join(blocking[:4])}
            if blocking
            else None,
        ]
        if item is not None
    ]
    return {
        "schema_version": "actsoft.tcad.power_mosfet_signoff_gate.v1",
        "verdict": verdict,
        "baseline_status": baseline_status,
        "missing_evidence": list(dict.fromkeys(missing)),
        "blocking_reasons": list(dict.fromkeys(blocking)),
        "benchmark_signoff_pack": benchmark_pack,
        "next_actions": next_actions,
    }


def run_power_mosfet_signoff(request: PowerMOSFETSignoffRequest) -> PowerMOSFETSignoffState:
    run_id = request.run_id or default_run_id()
    run_dir = (request.run_root / run_id).resolve()
    state_path = run_dir / "state.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    now = utc_timestamp()
    artifacts: dict[str, str] = {}
    baseline_state_path: str | None = None
    benchmark_payload: dict[str, Any] | None = None
    convergence_payload: dict[str, Any] | None = None
    golden_payload: dict[str, Any] | None = None
    failure_reason: str | None = None
    try:
        baseline_request = default_baseline_request(request.model_copy(update={"run_id": run_id}), run_dir)
        if request.execute:
            baseline = run_extended_device_sweep(ExtendedDeviceRequest.model_validate(baseline_request))
            baseline_state_path = state_path_for_extended_result(baseline)
            artifacts["baseline_state"] = baseline_state_path
            baseline_summary = baseline.final_summary or {}
            artifacts.update({f"baseline_{key}": value for key, value in (baseline_summary.get("artifacts") or {}).items()})
            benchmark = run_physical_benchmark(Path(baseline_state_path), output_path=run_dir / "physical_benchmark.json")
            benchmark_payload = benchmark.model_dump(mode="json")
            artifacts["physical_benchmark"] = benchmark.benchmark_path or str(run_dir / "physical_benchmark.json")
            if request.run_convergence:
                convergence = run_tool_convergence(
                    ToolConvergenceRequest(
                        convergence_id=f"{run_id}_mesh_model_convergence",
                        tool_name="extended_device_sweep",
                        base_request=baseline_request,
                        axis_path="power_mos_junction_mesh_spacing_um",
                        values=request.convergence_values,
                        metric_path="quality_report.metrics.max_electric_field_v_per_cm",
                        relative_tolerance=request.convergence_relative_tolerance,
                        execute=True,
                        overwrite=True,
                        convergence_root=run_dir / "tool_convergence",
                    )
                )
                convergence_payload = convergence.model_dump(mode="json")
                artifacts["tool_convergence"] = str(Path(convergence.convergence_dir) / "state.json")
            if request.reference_curve_path:
                golden = run_golden_curve_comparison(
                    GoldenCurveComparisonRequest(
                        comparison_id=f"{run_id}_golden",
                        source_state_path=Path(baseline_state_path),
                        reference_curve_path=request.reference_curve_path,
                        run_root=run_dir / "golden_curve",
                    )
                )
                golden_payload = golden.model_dump(mode="json")
                artifacts["golden_curve_comparison"] = str(Path(golden.comparison_dir) / "state.json")
        else:
            artifacts["planned_baseline_request"] = str(run_dir / "planned_baseline_request.json")
            write_json(run_dir / "planned_baseline_request.json", baseline_request)
    except Exception as exc:
        failure_reason = str(exc)

    signoff_gate = build_signoff_gate(
        baseline_status=(benchmark_payload or {}).get("source_tool_name") and "completed" if baseline_state_path else None,
        benchmark=benchmark_payload,
        convergence=convergence_payload,
        golden=golden_payload,
    )
    signoff_gate_path = run_dir / "signoff_gate.json"
    artifacts["signoff_gate"] = str(signoff_gate_path)
    status = "planned" if not request.execute else "failed" if failure_reason or signoff_gate["blocking_reasons"] else "completed"
    quality_status = "planned" if status == "planned" else "failed" if status == "failed" else "passed" if signoff_gate["verdict"] == "ready" else "suspicious"
    state = PowerMOSFETSignoffState(
        status=status,
        run_id=run_id,
        run_dir=str(run_dir),
        created_at=now,
        updated_at=utc_timestamp(),
        request=request.model_dump(mode="json"),
        baseline_state_path=baseline_state_path,
        benchmark_path=artifacts.get("physical_benchmark"),
        convergence_state_path=artifacts.get("tool_convergence"),
        golden_comparison_state_path=artifacts.get("golden_curve_comparison"),
        artifacts=artifacts,
        final_summary={
            "artifacts": artifacts,
            "metrics": {
                "baseline_completed": bool(baseline_state_path),
                "physical_benchmark_present": bool(benchmark_payload),
                "convergence_present": bool(convergence_payload),
                "golden_or_measured_present": bool(golden_payload),
                "signoff_verdict": signoff_gate["verdict"],
            },
        },
        quality_report={
            "status": quality_status,
            "issues": [
                {"code": "power_mosfet_signoff_missing_evidence", "severity": "warning", "missing": signoff_gate["missing_evidence"]}
            ]
            if signoff_gate["missing_evidence"]
            else [],
            "metrics": {"signoff_verdict": signoff_gate["verdict"]},
            "recommended_next_action": "run missing signoff evidence steps" if signoff_gate["verdict"] != "ready" else "ready for bounded engineering conclusion",
        },
        signoff_gate=signoff_gate,
        next_action="close missing signoff evidence" if signoff_gate["verdict"] != "ready" else "generate engineering signoff conclusion",
        failure_reason=failure_reason,
        state_path=str(state_path),
    )
    write_json(signoff_gate_path, signoff_gate)
    write_json(state_path, state.model_dump(mode="json"))
    return state
