from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.reporting import final_artifacts, final_metrics
from tcad_agent.sentaurus_lineage import SentaurusLineageArchiveRequest, build_sentaurus_lineage_archive
from tcad_agent.sentaurus_mutation_effect import SentaurusMutationEffectRequest, analyze_sentaurus_mutation_effect
from tcad_agent.task_spec import PROJECT_ROOT


class SentaurusReplayCheck(BaseModel):
    code: str
    status: str
    message: str
    observed: dict[str, Any] = Field(default_factory=dict)


class SentaurusReplayRequest(BaseModel):
    source_state_path: Path | None = None
    baseline_state_path: Path | None = None
    mutation_state_path: Path | None = None
    candidate: dict[str, Any] = Field(default_factory=dict)
    goal_text: str = ""
    output_dir: Path = PROJECT_ROOT / "runs" / "sentaurus_replay"
    output_path: Path | None = None


class SentaurusReplayResult(BaseModel):
    tool_name: str = "sentaurus_replay"
    schema_version: str = "actsoft.tcad.sentaurus_replay.v1"
    status: str
    source_state_path: str | None = None
    baseline_state_path: str | None = None
    mutation_state_path: str | None = None
    checks: list[SentaurusReplayCheck] = Field(default_factory=list)
    mutation_effect: dict[str, Any] | None = None
    lineage_archive: dict[str, Any] | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    final_summary: dict[str, Any] = Field(default_factory=dict)
    output_path: str | None = None
    failure_reason: str | None = None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def check(code: str, status: str, message: str, observed: dict[str, Any] | None = None) -> SentaurusReplayCheck:
    return SentaurusReplayCheck(code=code, status=status, message=message, observed=observed or {})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def validate_sentaurus_state(path: Path) -> tuple[dict[str, Any] | None, list[SentaurusReplayCheck]]:
    checks: list[SentaurusReplayCheck] = []
    if not path.exists():
        return None, [check("sentaurus_replay_state_missing", "failed", "Replay state path does not exist.", {"state_path": str(path)})]
    try:
        state = read_json(path)
    except Exception as exc:
        return None, [check("sentaurus_replay_state_unreadable", "failed", "Replay state is not readable JSON.", {"state_path": str(path), "error": str(exc)})]
    if state.get("tool_name") == "sentaurus_run":
        checks.append(check("sentaurus_replay_state_is_sentaurus", "passed", "State is a Sentaurus adapter state.", {"state_path": str(path)}))
    else:
        checks.append(check("sentaurus_replay_state_not_sentaurus", "failed", "State is not a Sentaurus adapter state.", {"tool_name": state.get("tool_name")}))
    metrics = final_metrics(state)
    artifacts = final_artifacts(state)
    curve_path = metrics.get("curve_path") or artifacts.get("sentaurus_curve_csv")
    if curve_path and Path(str(curve_path)).exists():
        checks.append(check("sentaurus_replay_curve_available", "passed", "Replay state has an inspectable CSV curve.", {"curve_path": str(curve_path)}))
    else:
        checks.append(check("sentaurus_replay_curve_missing", "warning", "Replay state has no readable CSV curve; mutation effect may be metric-only.", {"curve_path": str(curve_path) if curve_path else None}))
    if metrics.get("solver_backend") == "sentaurus":
        checks.append(check("sentaurus_replay_backend_sentaurus", "passed", "State records Sentaurus solver backend."))
    else:
        checks.append(check("sentaurus_replay_backend_unknown", "warning", "State does not record Sentaurus solver backend.", {"solver_backend": metrics.get("solver_backend")}))
    return state, checks


def status_from_checks(checks: list[SentaurusReplayCheck]) -> str:
    return "failed" if any(item.status == "failed" for item in checks) else "completed"


def run_sentaurus_replay(request: SentaurusReplayRequest) -> SentaurusReplayResult:
    output_dir = request.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[SentaurusReplayCheck] = []
    artifacts: dict[str, str] = {}
    source_path = request.source_state_path or request.mutation_state_path or request.baseline_state_path
    source_state = None
    if source_path:
        source_state, source_checks = validate_sentaurus_state(source_path.expanduser())
        checks.extend(source_checks)
    else:
        checks.append(check("sentaurus_replay_source_missing", "failed", "Provide source_state_path or baseline/mutation state paths."))

    mutation_effect = None
    if request.baseline_state_path and request.mutation_state_path:
        _, baseline_checks = validate_sentaurus_state(request.baseline_state_path.expanduser())
        _, mutation_checks = validate_sentaurus_state(request.mutation_state_path.expanduser())
        checks.extend(baseline_checks)
        checks.extend(mutation_checks)
        if not any(item.status == "failed" for item in baseline_checks + mutation_checks):
            effect_path = output_dir / "sentaurus_replay_mutation_effect.json"
            overlay_path = output_dir / "sentaurus_replay_overlay.svg"
            effect = analyze_sentaurus_mutation_effect(
                SentaurusMutationEffectRequest(
                    baseline_state_path=request.baseline_state_path.expanduser(),
                    mutation_state_path=request.mutation_state_path.expanduser(),
                    candidate=request.candidate,
                    goal_text=request.goal_text,
                    output_path=effect_path,
                    overlay_output_path=overlay_path,
                )
            )
            mutation_effect = effect.model_dump(mode="json")
            artifacts["mutation_effect"] = str(effect_path)
            if effect.overlay_svg_path:
                artifacts["overlay_svg"] = effect.overlay_svg_path
            checks.append(check("sentaurus_replay_mutation_effect_completed", "passed", "Baseline/mutation replay produced mutation-effect analysis.", {"decision": effect.decision}))

    lineage_archive = None
    lineage_source = request.mutation_state_path or request.source_state_path
    if lineage_source and lineage_source.expanduser().exists():
        lineage_path = output_dir / "sentaurus_replay_lineage_archive.json"
        archive = build_sentaurus_lineage_archive(
            SentaurusLineageArchiveRequest(source_state_path=lineage_source.expanduser(), output_path=lineage_path)
        )
        lineage_archive = archive.model_dump(mode="json")
        artifacts["lineage_archive"] = str(lineage_path)
        checks.append(check("sentaurus_replay_lineage_completed", "passed" if archive.status == "completed" else "failed", "Replay built lineage archive.", {"entries": len(archive.entries), "status": archive.status}))

    result = SentaurusReplayResult(
        status=status_from_checks(checks),
        source_state_path=str(source_path.expanduser()) if source_path else None,
        baseline_state_path=str(request.baseline_state_path.expanduser()) if request.baseline_state_path else None,
        mutation_state_path=str(request.mutation_state_path.expanduser()) if request.mutation_state_path else None,
        checks=checks,
        mutation_effect=mutation_effect,
        lineage_archive=lineage_archive,
        artifacts=artifacts,
        final_summary={
            "metrics": {
                "sentaurus_replay_only": True,
                "tcad_solver_invoked": False,
                "source_had_solver_invoked": bool(final_metrics(source_state or {}).get("tcad_solver_invoked")),
            },
            "data_provenance": {
                "does_not_run_sentaurus": True,
                "consumes_existing_logs_curves_states_only": True,
            },
        },
    )
    if result.status == "failed":
        result.failure_reason = next((item.code for item in checks if item.status == "failed"), "sentaurus_replay_failed")
    output_path = request.output_path.expanduser().resolve() if request.output_path else output_dir / "sentaurus_replay_state.json"
    result.output_path = str(output_path)
    write_json(output_path, result.model_dump(mode="json"))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay existing Sentaurus adapter states/logs/curves without running Sentaurus.")
    parser.add_argument("--state", "--source-state-path", dest="source_state_path", type=Path, default=None)
    parser.add_argument("--baseline", "--baseline-state-path", dest="baseline_state_path", type=Path, default=None)
    parser.add_argument("--mutation", "--mutation-state-path", dest="mutation_state_path", type=Path, default=None)
    parser.add_argument("--goal", "--goal-text", dest="goal_text", default="")
    parser.add_argument("--candidate-json", default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "sentaurus_replay")
    parser.add_argument("--output", "--output-path", dest="output_path", type=Path, default=None)
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> SentaurusReplayRequest:
    candidate = json.loads(args.candidate_json) if args.candidate_json else {}
    if not isinstance(candidate, dict):
        raise ValueError("--candidate-json must decode to a JSON object")
    return SentaurusReplayRequest(
        source_state_path=args.source_state_path,
        baseline_state_path=args.baseline_state_path,
        mutation_state_path=args.mutation_state_path,
        candidate=candidate,
        goal_text=args.goal_text,
        output_dir=args.output_dir,
        output_path=args.output_path,
    )


def main() -> None:
    result = run_sentaurus_replay(request_from_args(parse_args()))
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
