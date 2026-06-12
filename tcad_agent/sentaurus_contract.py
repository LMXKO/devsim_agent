from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.sentaurus import SentaurusRunRequest, SentaurusRuntimeProfile, run_sentaurus
from tcad_agent.sentaurus_deck import apply_sentaurus_semantic_patch_text, parse_sentaurus_deck_file
from tcad_agent.task_spec import PROJECT_ROOT


class SentaurusContractCheck(BaseModel):
    code: str
    status: str
    message: str
    observed: dict[str, Any] = Field(default_factory=dict)


class SentaurusContractCase(BaseModel):
    case_id: str
    description: str = ""
    deck_files: list[str] = Field(default_factory=lambda: ["device.cmd"])
    public_reference_urls: list[str] = Field(default_factory=list)
    expected_sections: list[list[str] | str] = Field(default_factory=list)
    expected_variables: list[str] = Field(default_factory=list)
    expected_assignments: list[dict[str, Any]] = Field(default_factory=list)
    semantic_patch_smoke: list[dict[str, Any]] = Field(default_factory=list)
    required_curve_columns: list[str] = Field(default_factory=list)
    fake_backend: dict[str, Any] = Field(default_factory=dict)


class SentaurusContractResult(BaseModel):
    tool_name: str = "sentaurus_contract"
    status: str
    case_id: str
    project_path: str
    created_at: str
    checks: list[SentaurusContractCheck] = Field(default_factory=list)
    parsed_decks: list[dict[str, Any]] = Field(default_factory=list)
    sentaurus_state_path: str | None = None
    report_path: str | None = None
    final_summary: dict[str, Any] = Field(default_factory=dict)


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def default_fixture_root() -> Path:
    return PROJECT_ROOT / "tcad_agent" / "examples" / "sentaurus_fixtures"


def manifest_path_for_project(project_path: Path) -> Path:
    direct = project_path / "actsoft_sentaurus_contract.json"
    if direct.exists():
        return direct
    raise FileNotFoundError(f"Sentaurus contract manifest not found under: {project_path}")


def load_contract_case(project_path: Path) -> SentaurusContractCase:
    return SentaurusContractCase.model_validate_json(manifest_path_for_project(project_path).read_text(encoding="utf-8"))


def discover_contract_projects(root: Path | None = None) -> list[Path]:
    actual_root = root or default_fixture_root()
    if not actual_root.exists():
        return []
    return sorted(path.parent for path in actual_root.rglob("actsoft_sentaurus_contract.json"))


def normalize_path(value: list[str] | str) -> list[str]:
    if isinstance(value, str):
        return [part for part in value.replace("/", ".").split(".") if part]
    return [str(part) for part in value if str(part)]


def has_section_path(paths: list[list[str]], expected: list[str]) -> bool:
    expected_lower = [part.lower() for part in expected]
    for path in paths:
        lowered = [part.lower() for part in path]
        if lowered == expected_lower:
            return True
    return False


def value_matches(actual: str | None, expected: Any) -> bool:
    if actual is None:
        return False
    normalized = actual.strip().strip("\"'")
    return normalized == str(expected)


def pass_check(code: str, message: str, observed: dict[str, Any] | None = None) -> SentaurusContractCheck:
    return SentaurusContractCheck(code=code, status="passed", message=message, observed=observed or {})


def fail_check(code: str, message: str, observed: dict[str, Any] | None = None) -> SentaurusContractCheck:
    return SentaurusContractCheck(code=code, status="failed", message=message, observed=observed or {})


def validate_deck_ir(project_path: Path, case: SentaurusContractCase) -> tuple[list[SentaurusContractCheck], list[dict[str, Any]]]:
    checks: list[SentaurusContractCheck] = []
    parsed: list[dict[str, Any]] = []
    for raw_deck in case.deck_files:
        deck_path = project_path / raw_deck
        if not deck_path.exists():
            checks.append(fail_check("sentaurus_contract_deck_missing", "Deck file listed in contract is missing.", {"deck_file": raw_deck}))
            continue
        ir = parse_sentaurus_deck_file(deck_path)
        parsed.append(
            {
                "deck_file": raw_deck,
                "sections": [section.model_dump(mode="json") for section in ir.sections],
                "variables": [variable.model_dump(mode="json") for variable in ir.set_variables],
                "assignments": [assignment.model_dump(mode="json") for assignment in ir.assignments],
                "warnings": ir.warnings,
            }
        )
        section_paths = [section.path for section in ir.sections]
        variables = {variable.key.lower(): variable.value for variable in ir.set_variables}
        assignment_records = [
            {
                "section_path": assignment.section_path,
                "key": assignment.key,
                "value": assignment.value,
                "line": assignment.line,
            }
            for assignment in ir.assignments
        ]
        for expected in case.expected_sections:
            normalized = normalize_path(expected)
            if has_section_path(section_paths, normalized):
                checks.append(pass_check("sentaurus_contract_section_present", "Expected Sentaurus section path is present.", {"deck_file": raw_deck, "section_path": normalized}))
            else:
                checks.append(fail_check("sentaurus_contract_section_missing", "Expected Sentaurus section path is missing.", {"deck_file": raw_deck, "section_path": normalized, "observed_paths": section_paths}))
        for variable in case.expected_variables:
            if variable.lower() in variables:
                checks.append(pass_check("sentaurus_contract_variable_present", "Expected set/#define variable is present.", {"deck_file": raw_deck, "variable": variable}))
            else:
                checks.append(fail_check("sentaurus_contract_variable_missing", "Expected set/#define variable is missing.", {"deck_file": raw_deck, "variable": variable}))
        for expected_assignment in case.expected_assignments:
            expected_path = normalize_path(expected_assignment.get("section_path") or [])
            expected_key = str(expected_assignment.get("key") or "")
            expected_value = expected_assignment.get("value")
            matched = False
            for assignment in assignment_records:
                if expected_path and not has_section_path([assignment["section_path"]], expected_path):
                    continue
                if assignment["key"].lower() != expected_key.lower():
                    continue
                if "value" in expected_assignment and not value_matches(str(assignment["value"]), expected_value):
                    continue
                matched = True
                break
            if matched:
                checks.append(pass_check("sentaurus_contract_assignment_present", "Expected assignment is present.", {"deck_file": raw_deck, "assignment": expected_assignment}))
            else:
                checks.append(fail_check("sentaurus_contract_assignment_missing", "Expected assignment is missing.", {"deck_file": raw_deck, "assignment": expected_assignment}))
        if ir.warnings:
            checks.append(fail_check("sentaurus_contract_deck_ir_warning", "Deck IR parser produced warnings.", {"deck_file": raw_deck, "warnings": ir.warnings}))
    return checks, parsed


def validate_semantic_patch_smoke(project_path: Path, case: SentaurusContractCase) -> list[SentaurusContractCheck]:
    checks: list[SentaurusContractCheck] = []
    for patch in case.semantic_patch_smoke:
        deck_file = str(patch.get("file") or "")
        deck_path = project_path / deck_file
        if not deck_path.exists():
            checks.append(fail_check("sentaurus_contract_patch_deck_missing", "Semantic patch target deck is missing.", {"patch": patch}))
            continue
        updated, record, _ = apply_sentaurus_semantic_patch_text(
            deck_path.read_text(encoding="utf-8", errors="replace"),
            patch,
            source_path=deck_file,
        )
        expected_contains = patch.get("expected_contains")
        if record.get("verified") and (not expected_contains or str(expected_contains) in updated):
            checks.append(pass_check("sentaurus_contract_semantic_patch_verified", "Semantic patch verified against fixture deck.", {"patch": patch, "record": record}))
        else:
            checks.append(fail_check("sentaurus_contract_semantic_patch_failed", "Semantic patch did not verify against fixture deck.", {"patch": patch, "record": record}))
    return checks


def validate_curve_contract(state: dict[str, Any], case: SentaurusContractCase) -> list[SentaurusContractCheck]:
    checks: list[SentaurusContractCheck] = []
    metrics = ((state.get("quality_report") or {}).get("metrics") or {})
    curve_path = metrics.get("curve_path")
    x_key = metrics.get("curve_x_key")
    y_key = metrics.get("curve_y_key")
    observed_columns = [key for key in [x_key, y_key, metrics.get("curve_field_key")] if key]
    missing = [column for column in case.required_curve_columns if column not in observed_columns]
    if not case.required_curve_columns:
        return checks
    if curve_path and not missing:
        checks.append(pass_check("sentaurus_contract_curve_columns_present", "Fake backend produced required interface CSV columns.", {"curve_path": curve_path, "columns": observed_columns}))
    else:
        checks.append(fail_check("sentaurus_contract_curve_columns_missing", "Required interface CSV columns were not extracted.", {"curve_path": curve_path, "columns": observed_columns, "missing": missing}))
    return checks


def read_fake_backend_manifest(project_dir: Path) -> dict[str, Any]:
    path = project_dir / "actsoft_sentaurus_contract.json"
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def fake_backend_rows(manifest: dict[str, Any]) -> list[dict[str, str]]:
    backend = manifest.get("fake_backend") if isinstance(manifest.get("fake_backend"), dict) else {}
    rows = backend.get("curve_rows") if isinstance(backend.get("curve_rows"), list) else None
    if rows:
        return [{str(key): str(value) for key, value in row.items()} for row in rows if isinstance(row, dict)]
    return [
        {"voltage_v": "0", "current_a": "1e-12", "electric_field_v_per_cm": "1e4"},
        {"voltage_v": "-10", "current_a": "1e-9", "electric_field_v_per_cm": "2e5"},
        {"voltage_v": "-20", "current_a": "1e-6", "electric_field_v_per_cm": "8e5"},
    ]


def write_fake_backend_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_fake_backend_cli(project_dir: Path | None = None) -> dict[str, Any]:
    actual_project = project_dir or Path.cwd()
    manifest = read_fake_backend_manifest(actual_project)
    backend = manifest.get("fake_backend") if isinstance(manifest.get("fake_backend"), dict) else {}
    log_name = str(backend.get("log_name") or "contract_des.log")
    csv_name = str(backend.get("csv_name") or "sentaurus_contract_extract.csv")
    plt_name = str(backend.get("plt_name") or "contract_des.plt")
    (actual_project / log_name).write_text(
        "ACTSOFT SENTaurus CONTRACT FAKE BACKEND\n"
        "interface_contract_only=true\n"
        "This output validates agent IO only and is not a Sentaurus physics result.\n"
        "Sentaurus Device finished\n",
        encoding="utf-8",
    )
    (actual_project / plt_name).write_text(
        "ACTSOFT interface placeholder for artifact collection only.\n",
        encoding="utf-8",
    )
    write_fake_backend_csv(actual_project / csv_name, fake_backend_rows(manifest))
    return {"status": "completed", "interface_contract_only": True, "csv": csv_name}


def run_fake_sentaurus_contract(project_path: Path, case: SentaurusContractCase, *, output_root: Path) -> tuple[dict[str, Any], list[SentaurusContractCheck]]:
    output_root.mkdir(parents=True, exist_ok=True)
    profile = SentaurusRuntimeProfile(
        profile_id="sentaurus_contract_fake_backend",
        commands={"sdevice": sys.executable},
        allowed_project_roots=[project_path.parent],
        run_root=output_root,
        env={"PYTHONPATH": str(PROJECT_ROOT)},
        default_flow=["sdevice"],
        curve_globs=["*.csv", "*_extract.csv", "*_iv.csv"],
    )
    request = SentaurusRunRequest(
        goal_text=f"Run Sentaurus contract fixture {case.case_id}",
        project_path=project_path,
        profile=profile,
        run_id=f"{case.case_id}_contract",
        flow=["sdevice"],
        command_args={"sdevice": ["-m", "tcad_agent.sentaurus_contract", "--fake-backend"]},
        deck_files=case.deck_files,
        timeout_seconds=30,
    )
    state = run_sentaurus(request).model_dump(mode="json")
    checks = validate_curve_contract(state, case)
    if state.get("status") == "completed":
        checks.append(pass_check("sentaurus_contract_fake_backend_completed", "Fake backend completed the Sentaurus runner interface contract.", {"state_path": state.get("state_path")}))
    else:
        checks.append(fail_check("sentaurus_contract_fake_backend_failed", "Fake backend did not complete the runner contract.", {"state": state}))
    return state, checks


def status_from_checks(checks: list[SentaurusContractCheck]) -> str:
    return "failed" if any(check.status == "failed" for check in checks) else "passed"


def validate_sentaurus_contract(
    project_path: Path,
    *,
    run_fake_e2e: bool = False,
    output_root: Path | None = None,
    report_path: Path | None = None,
) -> SentaurusContractResult:
    actual_project = project_path.expanduser().resolve()
    case = load_contract_case(actual_project)
    checks, parsed = validate_deck_ir(actual_project, case)
    checks.extend(validate_semantic_patch_smoke(actual_project, case))
    sentaurus_state_path = None
    if run_fake_e2e:
        state, fake_checks = run_fake_sentaurus_contract(
            actual_project,
            case,
            output_root=(output_root or PROJECT_ROOT / "runs" / "sentaurus_contract"),
        )
        checks.extend(fake_checks)
        sentaurus_state_path = state.get("state_path")
    result = SentaurusContractResult(
        status=status_from_checks(checks),
        case_id=case.case_id,
        project_path=str(actual_project),
        created_at=utc_timestamp(),
        checks=checks,
        parsed_decks=parsed,
        sentaurus_state_path=sentaurus_state_path,
        final_summary={
            "checks_total": len(checks),
            "checks_failed": sum(1 for check in checks if check.status == "failed"),
            "run_fake_e2e": run_fake_e2e,
            "public_reference_urls": case.public_reference_urls,
            "scope": "offline Sentaurus agent contract validation; no real Sentaurus physics is simulated",
        },
    )
    if report_path:
        write_json(report_path, result.model_dump(mode="json"))
        result.report_path = str(report_path)
        write_json(report_path, result.model_dump(mode="json"))
    return result


def validate_fixture_corpus(
    root: Path | None = None,
    *,
    run_fake_e2e: bool = False,
    output_root: Path | None = None,
) -> list[SentaurusContractResult]:
    results: list[SentaurusContractResult] = []
    for project in discover_contract_projects(root):
        case_output = (output_root or PROJECT_ROOT / "runs" / "sentaurus_contract") / project.name
        results.append(validate_sentaurus_contract(project, run_fake_e2e=run_fake_e2e, output_root=case_output))
    return results


def main() -> None:
    if "--fake-backend" not in sys.argv[1:]:
        print(json.dumps({"status": "failed", "failure_reason": "expected --fake-backend"}, ensure_ascii=False))
        raise SystemExit(2)
    print(json.dumps(run_fake_backend_cli(), ensure_ascii=False))


if __name__ == "__main__":
    main()
