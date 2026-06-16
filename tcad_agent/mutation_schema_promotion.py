from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.mutation_vocabulary import MutationVocabularyEntry, mutation_class_ids
from tcad_agent.task_spec import PROJECT_ROOT


class MutationSchemaPromotionCheck(BaseModel):
    code: str
    status: str
    message: str
    observed: dict[str, Any] = Field(default_factory=dict)


class MutationSchemaPromotionRequest(BaseModel):
    schema_extension_path: Path
    candidate_id: str | None = None
    vocabulary_path: Path = PROJECT_ROOT / "tcad_agent" / "mutation_vocabulary.py"
    output_dir: Path = PROJECT_ROOT / "runs" / "mutation_schema_promotions"
    output_path: Path | None = None
    apply: bool = False
    confirmed: bool = False


class MutationSchemaPromotionResult(BaseModel):
    tool_name: str = "mutation_schema_promotion"
    schema_version: str = "actsoft.tcad.mutation_schema_promotion.v1"
    status: str
    schema_extension_path: str
    selected_class_id: str | None = None
    checks: list[MutationSchemaPromotionCheck] = Field(default_factory=list)
    mutation_vocabulary_patch: str | None = None
    generated_test_source: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    applied: bool = False
    output_path: str | None = None
    failure_reason: str | None = None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def passed(code: str, message: str, observed: dict[str, Any] | None = None) -> MutationSchemaPromotionCheck:
    return MutationSchemaPromotionCheck(code=code, status="passed", message=message, observed=observed or {})


def failed(code: str, message: str, observed: dict[str, Any] | None = None) -> MutationSchemaPromotionCheck:
    return MutationSchemaPromotionCheck(code=code, status="failed", message=message, observed=observed or {})


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def candidate_from_extension(extension: dict[str, Any], candidate_id: str | None) -> dict[str, Any] | None:
    selected = extension.get("selected_candidate")
    candidates = extension.get("candidates")
    if candidate_id and isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, dict) and str(item.get("class_id") or item.get("candidate_id") or "") == candidate_id:
                return item
    if isinstance(selected, dict):
        return selected
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, dict) and item.get("ready_for_review"):
                return item
    return None


def list_verified(records: Any) -> list[dict[str, Any]]:
    return [record for record in records or [] if isinstance(record, dict) and record.get("verified")]


def validate_candidate(extension: dict[str, Any], candidate: dict[str, Any] | None) -> tuple[list[MutationSchemaPromotionCheck], MutationVocabularyEntry | None]:
    checks: list[MutationSchemaPromotionCheck] = []
    if candidate is None:
        checks.append(failed("missing_schema_candidate", "No selected or ready candidate was found in the schema extension package."))
        return checks, None
    gate = extension.get("public_evidence_dossier", {}).get("evidence_gate") if isinstance(extension.get("public_evidence_dossier"), dict) else {}
    if isinstance(gate, dict) and gate.get("passed"):
        checks.append(passed("public_evidence_gate_passed", "Public evidence gate passed.", {"source_count": gate.get("source_count")}))
    else:
        checks.append(failed("public_evidence_gate_missing", "Public evidence gate did not pass."))
    if candidate.get("ready_for_review"):
        checks.append(passed("candidate_ready_for_review", "Candidate is marked ready_for_review."))
    else:
        checks.append(failed("candidate_not_ready_for_review", "Candidate is not marked ready_for_review."))
    validation_records = list_verified(candidate.get("validation_records"))
    fixture_records = list_verified(candidate.get("fixture_validation_records"))
    if validation_records:
        checks.append(passed("deck_patch_validation_verified", "At least one local deck semantic patch validation record is verified.", {"verified_count": len(validation_records)}))
    else:
        checks.append(failed("deck_patch_validation_missing", "No verified local deck semantic patch validation record was found."))
    if fixture_records:
        checks.append(passed("fixture_validation_verified", "Fixture deck semantic patch validation is verified.", {"verified_count": len(fixture_records)}))
    else:
        checks.append(failed("fixture_validation_missing", "No verified fixture semantic patch validation record was found."))
    schema = candidate.get("schema_patch") if isinstance(candidate.get("schema_patch"), dict) else {}
    try:
        entry = MutationVocabularyEntry.model_validate(schema)
        checks.append(passed("schema_patch_validates", "schema_patch validates as MutationVocabularyEntry.", {"class_id": entry.class_id}))
    except Exception as exc:
        checks.append(failed("schema_patch_invalid", "schema_patch does not validate as MutationVocabularyEntry.", {"error": str(exc)}))
        return checks, None
    if entry.class_id in set(mutation_class_ids()):
        checks.append(failed("class_id_already_exists", "Mutation vocabulary class_id already exists.", {"class_id": entry.class_id}))
    else:
        checks.append(passed("class_id_is_new", "Mutation vocabulary class_id is new.", {"class_id": entry.class_id}))
    if entry.semantic_patch_operations:
        checks.append(passed("semantic_operations_declared", "Semantic patch operations are declared.", {"operations": entry.semantic_patch_operations}))
    else:
        checks.append(failed("semantic_operations_missing", "No semantic patch operations were declared."))
    return checks, entry


def py_literal(value: Any, indent: int = 8) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        if not value:
            return "[]"
        one_line = repr(value)
        if len(one_line) <= 120 and "\n" not in one_line:
            return one_line
        inner_indent = " " * (indent + 4)
        closing_indent = " " * indent
        items = ",\n".join(f"{inner_indent}{py_literal(item, indent + 4)}" for item in value)
        return "[\n" + items + ",\n" + closing_indent + "]"
    if isinstance(value, dict):
        return repr(value)
    return repr(value)


def render_entry(entry: MutationVocabularyEntry) -> str:
    data = entry.model_dump(mode="json")
    lines = ["    MutationVocabularyEntry("]
    field_order = [
        "class_id",
        "display_name",
        "target_kind",
        "default_risk_level",
        "requires_user_confirmation",
        "variable_name_tokens",
        "goal_tags",
        "primary_metrics",
        "tradeoff_metrics",
        "semantic_patch_operations",
        "expected_curve_evidence",
        "stop_conditions",
        "public_source_ids",
        "notes",
    ]
    for key in field_order:
        value = data.get(key)
        if key == "requires_user_confirmation" and value is False:
            continue
        if value is None or value == [] or value == {}:
            continue
        lines.append(f"        {key}={py_literal(value, 8)},")
    lines.append("    ),")
    return "\n".join(lines)


def insert_entry_source(source: str, entry_source: str) -> str:
    marker = "\n)\n\n\ndef list_mutation_vocabulary"
    index = source.find(marker)
    if index < 0:
        raise ValueError("Could not find MUTATION_VOCABULARY tuple closing marker.")
    return source[:index] + "\n" + entry_source + source[index:]


def unified_diff(before: str, after: str, path: Path) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
            lineterm="",
        )
    )


def test_module_name(class_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", class_id).strip("_").lower()
    return f"test_mutation_vocabulary_promoted_{safe}.py"


def render_test_source(entry: MutationVocabularyEntry) -> str:
    variable = "_".join(entry.variable_name_tokens[0]) if entry.variable_name_tokens else entry.class_id.upper()
    return f'''from __future__ import annotations

import unittest

from tcad_agent.mutation_vocabulary import classify_mutation_variable, mutation_entry


class PromotedMutationVocabularyTest(unittest.TestCase):
    def test_{entry.class_id}_entry_is_classifiable(self) -> None:
        entry = mutation_entry("{entry.class_id}")

        self.assertIsNotNone(entry)
        self.assertEqual(entry.display_name, {entry.display_name!r})
        self.assertIn("{entry.class_id}", classify_mutation_variable({variable!r}))
        self.assertTrue(entry.semantic_patch_operations)
        self.assertTrue(entry.public_source_ids)


if __name__ == "__main__":
    unittest.main()
'''


def status_from_checks(checks: list[MutationSchemaPromotionCheck], *, apply: bool, confirmed: bool) -> tuple[str, str | None]:
    failed_checks = [check for check in checks if check.status == "failed"]
    if failed_checks:
        return "blocked", failed_checks[0].code
    if apply and not confirmed:
        return "blocked_needs_confirmation", "apply_requires_confirmed_true"
    if apply and confirmed:
        return "applied", None
    return "ready_for_confirmation", None


def run_mutation_schema_promotion(request: MutationSchemaPromotionRequest) -> MutationSchemaPromotionResult:
    output_dir = request.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    extension_path = request.schema_extension_path.expanduser().resolve()
    artifacts: dict[str, str] = {}
    try:
        extension = load_json(extension_path)
        candidate = candidate_from_extension(extension, request.candidate_id)
        checks, entry = validate_candidate(extension, candidate)
        patch_text = None
        test_source = None
        if entry is not None:
            vocabulary_path = request.vocabulary_path.expanduser().resolve()
            before = vocabulary_path.read_text(encoding="utf-8") if vocabulary_path.exists() else ""
            after = insert_entry_source(before, render_entry(entry))
            patch_text = unified_diff(before, after, vocabulary_path)
            patch_path = output_dir / "mutation_vocabulary_patch.diff"
            write_text(patch_path, patch_text)
            artifacts["mutation_vocabulary_patch"] = str(patch_path)
            test_source = render_test_source(entry)
            test_path = output_dir / test_module_name(entry.class_id)
            write_text(test_path, test_source)
            artifacts["generated_test_source"] = str(test_path)
        status, failure_reason = status_from_checks(checks, apply=request.apply, confirmed=request.confirmed)
        applied = False
        if status == "applied" and entry is not None and patch_text is not None:
            vocabulary_path = request.vocabulary_path.expanduser().resolve()
            before = vocabulary_path.read_text(encoding="utf-8")
            after = insert_entry_source(before, render_entry(entry))
            vocabulary_path.write_text(after, encoding="utf-8")
            applied = True
        result = MutationSchemaPromotionResult(
            status=status,
            schema_extension_path=str(extension_path),
            selected_class_id=entry.class_id if entry else None,
            checks=checks,
            mutation_vocabulary_patch=patch_text,
            generated_test_source=test_source,
            artifacts=artifacts,
            applied=applied,
            failure_reason=failure_reason,
        )
    except Exception as exc:
        result = MutationSchemaPromotionResult(
            status="failed",
            schema_extension_path=str(extension_path),
            failure_reason=str(exc),
        )
    output_path = request.output_path.expanduser().resolve() if request.output_path else output_dir / "mutation_schema_promotion.json"
    result.output_path = str(output_path)
    result.artifacts["mutation_schema_promotion"] = str(output_path)
    write_json(output_path, result.model_dump(mode="json"))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate a mutation schema extension package before static vocabulary promotion.")
    parser.add_argument("--schema-extension", "--schema-extension-path", dest="schema_extension_path", type=Path, required=True)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--vocabulary-path", type=Path, default=PROJECT_ROOT / "tcad_agent" / "mutation_vocabulary.py")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "mutation_schema_promotions")
    parser.add_argument("--output", "--output-path", dest="output_path", type=Path, default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirmed", action="store_true")
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> MutationSchemaPromotionRequest:
    return MutationSchemaPromotionRequest(
        schema_extension_path=args.schema_extension_path,
        candidate_id=args.candidate_id,
        vocabulary_path=args.vocabulary_path,
        output_dir=args.output_dir,
        output_path=args.output_path,
        apply=args.apply,
        confirmed=args.confirmed,
    )


def main() -> None:
    result = run_mutation_schema_promotion(request_from_args(parse_args()))
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status in {"ready_for_confirmation", "blocked_needs_confirmation", "applied"} else 1)


if __name__ == "__main__":
    main()
