from __future__ import annotations

import ast
import difflib
import json
import math
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


SECTION_KEYWORDS = {
    "geometry": [
        "geometry",
        "region",
        "contact",
        "electrode",
        "field_plate",
        "field plate",
        "guard_ring",
        "guard ring",
        "junction",
        "oxide",
        "trench",
        "radius",
        "length",
        "width",
        "depth",
    ],
    "mesh": ["mesh", "grid", "node", "spacing", "refine", "element"],
    "doping": ["doping", "implant", "dose", "profile", "concentration", "acceptor", "donor"],
    "model": [
        "model",
        "equation",
        "mobility",
        "recombination",
        "srh",
        "lifetime",
        "avalanche",
        "impact",
        "trap",
        "parameter",
    ],
    "bias": ["bias", "sweep", "solve", "voltage", "ramp", "gate", "drain", "anode", "cathode", "terminal"],
}


class DeckAssignment(BaseModel):
    name: str
    value: Any = None
    section: str
    line: int


class DeckCall(BaseModel):
    function: str
    section: str
    line: int
    keywords: dict[str, Any] = Field(default_factory=dict)


class DeckIRSection(BaseModel):
    name: str
    start_line: int
    end_line: int
    symbols: list[str] = Field(default_factory=list)
    calls: list[str] = Field(default_factory=list)
    text_preview: str = ""


class DeckSourceIR(BaseModel):
    schema_version: str = "actsoft.tcad.deck_source_ir.v1"
    source_path: str | None = None
    sections: list[DeckIRSection] = Field(default_factory=list)
    assignments: list[DeckAssignment] = Field(default_factory=list)
    calls: list[DeckCall] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)


class DeckPatchResult(BaseModel):
    schema_version: str = "actsoft.tcad.deck_semantic_patch.v1"
    source_path: str | None = None
    patched_source_path: str | None = None
    diff_path: str | None = None
    ir_path: str | None = None
    applied_patches: list[dict[str, Any]] = Field(default_factory=list)
    unapplied_patches: list[dict[str, Any]] = Field(default_factory=list)
    unified_diff: str = ""
    patched_source: str | None = None
    ir: DeckSourceIR | None = None


def normalized_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return re.sub(r"_+", "_", cleaned)


def literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def finite_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr, *target_names(node.value)]
    if isinstance(node, ast.Tuple):
        output: list[str] = []
        for item in node.elts:
            output.extend(target_names(item))
        return output
    if isinstance(node, ast.Subscript):
        names = target_names(node.value)
        key = literal_value(node.slice)
        if isinstance(key, str):
            names.append(key)
        return names
    return []


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def classify_section(text: str, symbols: list[str] | None = None, preferred: str | None = None) -> str:
    combined = normalized_name(" ".join([text, *(symbols or [])]).replace("_", " "))
    if preferred and any(normalized_name(keyword) in combined for keyword in SECTION_KEYWORDS.get(preferred, [])):
        return preferred
    scores: dict[str, int] = {}
    for section, keywords in SECTION_KEYWORDS.items():
        scores[section] = sum(1 for keyword in keywords if normalized_name(keyword) in combined)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def source_segment(lines: list[str], start_line: int, end_line: int) -> str:
    return "\n".join(lines[start_line - 1 : end_line])


def parse_devsim_deck_source(source: str, *, source_path: str | None = None) -> DeckSourceIR:
    lines = source.splitlines()
    warnings: list[str] = []
    sections: list[DeckIRSection] = []
    assignments: list[DeckAssignment] = []
    calls: list[DeckCall] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return DeckSourceIR(
            source_path=source_path,
            parse_warnings=[f"syntax_error:{exc.lineno}:{exc.msg}"],
        )

    for node in tree.body:
        start_line = int(getattr(node, "lineno", 1))
        end_line = int(getattr(node, "end_lineno", start_line))
        text = source_segment(lines, start_line, end_line)
        symbols: list[str] = []
        call_symbols: list[str] = []
        preferred: str | None = None

        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                symbols.extend(target_names(target))
            value = node.value
            section = classify_section(text, symbols)
            for symbol in symbols:
                assignments.append(
                    DeckAssignment(name=symbol, value=literal_value(value), section=section, line=start_line)
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(node.name)
            preferred = classify_section(node.name)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = call_name(node.value.func)
            if name:
                call_symbols.append(name)
            keyword_values = {
                keyword.arg: literal_value(keyword.value)
                for keyword in node.value.keywords
                if keyword.arg is not None
            }
            section = classify_section(text, call_symbols)
            if name:
                calls.append(DeckCall(function=name, section=section, line=start_line, keywords=keyword_values))

        if not symbols and not call_symbols:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    name = call_name(child.func)
                    if name:
                        call_symbols.append(name)
                elif isinstance(child, ast.Name):
                    symbols.append(child.id)

        section = classify_section(text, [*symbols, *call_symbols], preferred=preferred)
        sections.append(
            DeckIRSection(
                name=section,
                start_line=start_line,
                end_line=end_line,
                symbols=sorted(set(symbols)),
                calls=sorted(set(call_symbols)),
                text_preview=text[:240],
            )
        )

    if not sections and lines:
        warnings.append("no_top_level_ast_sections_found")
    return DeckSourceIR(source_path=source_path, sections=sections, assignments=assignments, calls=calls, parse_warnings=warnings)


def parse_devsim_deck_file(path: Path) -> DeckSourceIR:
    return parse_devsim_deck_source(path.read_text(encoding="utf-8"), source_path=str(path.resolve()))


def line_offsets(source: str) -> list[int]:
    offsets = [0]
    total = 0
    for line in source.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def node_span(source: str, node: ast.AST) -> tuple[int, int] | None:
    if not all(hasattr(node, attr) for attr in ["lineno", "col_offset", "end_lineno", "end_col_offset"]):
        return None
    offsets = line_offsets(source)
    start = offsets[int(node.lineno) - 1] + int(node.col_offset)
    end = offsets[int(node.end_lineno) - 1] + int(node.end_col_offset)
    return start, end


def patch_candidates(patch: dict[str, Any]) -> set[str]:
    path = str(patch.get("deck_path") or patch.get("request_path") or patch.get("path") or "")
    parts = [part for part in re.split(r"[.\[\]]+", path) if part]
    candidates = {normalized_name(part) for part in parts}
    if parts:
        candidates.add(normalized_name(parts[-1]))
    for key in ["name", "target", "source_mutation"]:
        if patch.get(key):
            candidates.add(normalized_name(str(patch[key])))
    return {candidate for candidate in candidates if candidate and candidate not in {"geometry", "mesh", "doping", "physics_models", "model"}}


def patch_section(patch: dict[str, Any]) -> str | None:
    path = str(patch.get("deck_path") or "")
    head = path.split(".", 1)[0] if "." in path else ""
    if head == "physics_models":
        return "model"
    if head in SECTION_KEYWORDS:
        return head
    return None


def python_literal(value: Any) -> str:
    if isinstance(value, float):
        if value == 0:
            return "0.0"
        return f"{value:.12g}"
    if isinstance(value, (int, bool)) or value is None:
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(python_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{python_literal(key)}: {python_literal(val)}" for key, val in value.items()) + "}"
    return repr(value)


def match_name(name: str, candidates: set[str]) -> bool:
    normalized = normalized_name(name)
    return normalized in candidates or any(normalized.endswith(f"_{candidate}") for candidate in candidates)


class _PatchLocator(ast.NodeVisitor):
    def __init__(self, source: str, patch: dict[str, Any]):
        self.source = source
        self.patch = patch
        self.candidates = patch_candidates(patch)
        path = str(patch.get("deck_path") or patch.get("request_path") or patch.get("path") or "")
        parts = [part for part in re.split(r"[.\[\]]+", path) if part]
        self.leaf_candidate = normalized_name(parts[-1]) if parts else ""
        self.preferred_section = patch_section(patch)
        self.matches: list[tuple[int, ast.AST, str]] = []

    def score(self, node: ast.AST, names: list[str], reason: str) -> None:
        if not any(match_name(name, self.candidates) for name in names):
            return
        start_line = int(getattr(node, "lineno", 1))
        end_line = int(getattr(node, "end_lineno", start_line))
        text = source_segment(self.source.splitlines(), start_line, end_line)
        section = classify_section(text, names, preferred=self.preferred_section)
        priority = 0
        if self.preferred_section and section == self.preferred_section:
            priority += 20
        if self.leaf_candidate and any(match_name(name, {self.leaf_candidate}) for name in names):
            priority += 30
        priority += 10 if reason in {"assignment", "dict_key", "call_keyword"} else 0
        self.matches.append((priority, node, reason))

    def visit_Assign(self, node: ast.Assign) -> None:
        names: list[str] = []
        for target in node.targets:
            names.extend(target_names(target))
        self.score(node.value, names, "assignment")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.score(node.value, target_names(node.target), "assignment")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key_node, value_node in zip(node.keys, node.values):
            key = literal_value(key_node) if key_node is not None else None
            if isinstance(key, str):
                self.score(value_node, [key], "dict_key")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        for keyword in node.keywords:
            if keyword.arg and keyword.arg != "name":
                self.score(keyword.value, [keyword.arg], "call_keyword")
        name_literals = [literal_value(arg) for arg in node.args]
        name_literals.extend(literal_value(keyword.value) for keyword in node.keywords if keyword.arg in {"name", "parameter", "model"})
        if any(isinstance(value, str) and match_name(value, self.candidates) for value in name_literals):
            for keyword in node.keywords:
                if keyword.arg in {"value", "init", "default"}:
                    self.matches.append((25, keyword.value, "named_parameter_value"))
        self.generic_visit(node)


def appended_patch_block(patches: list[dict[str, Any]]) -> str:
    lines = ["", "", "# actsoft semantic deck patch fallback"]
    for patch in patches:
        path = str(patch.get("deck_path") or patch.get("request_path") or "patched_value")
        variable = normalized_name(path.split(".")[-1] if "." in path else path) or "patched_value"
        lines.append(f"{variable} = {python_literal(patch.get('value'))}")
    return "\n".join(lines)


def apply_semantic_deck_patch(
    source: str,
    patches: list[dict[str, Any]] | dict[str, Any],
    *,
    source_path: str | None = None,
) -> DeckPatchResult:
    patch_list = [patches] if isinstance(patches, dict) else list(patches)
    replacements: list[tuple[int, int, str, dict[str, Any], str]] = []
    unapplied: list[dict[str, Any]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return DeckPatchResult(
            source_path=source_path,
            unapplied_patches=patch_list,
            unified_diff="",
            patched_source=source,
            ir=parse_devsim_deck_source(source, source_path=source_path),
            applied_patches=[],
        ).model_copy(update={"unapplied_patches": [{**patch, "reason": f"syntax_error:{exc.lineno}:{exc.msg}"} for patch in patch_list]})

    occupied: set[tuple[int, int]] = set()
    for patch in patch_list:
        locator = _PatchLocator(source, patch)
        locator.visit(tree)
        locator.matches.sort(key=lambda item: item[0], reverse=True)
        chosen: tuple[int, ast.AST, str] | None = None
        for match in locator.matches:
            span = node_span(source, match[1])
            if span and span not in occupied:
                chosen = match
                break
        if chosen is None:
            unapplied.append({**patch, "reason": "no_matching_deck_symbol"})
            continue
        _, node, reason = chosen
        span = node_span(source, node)
        if span is None:
            unapplied.append({**patch, "reason": "matched_symbol_without_source_span"})
            continue
        occupied.add(span)
        replacements.append((span[0], span[1], python_literal(patch.get("value")), patch, reason))

    patched = source
    applied: list[dict[str, Any]] = []
    for start, end, text, patch, reason in sorted(replacements, key=lambda item: item[0], reverse=True):
        original = patched[start:end]
        patched = patched[:start] + text + patched[end:]
        applied.append(
            {
                "deck_path": patch.get("deck_path"),
                "request_path": patch.get("request_path"),
                "value": patch.get("value"),
                "reason": reason,
                "original": original,
                "replacement": text,
            }
        )

    if unapplied:
        patched += appended_patch_block(unapplied)
        for patch in unapplied:
            applied.append(
                {
                    "deck_path": patch.get("deck_path"),
                    "request_path": patch.get("request_path"),
                    "value": patch.get("value"),
                    "reason": "fallback_append",
                    "original": None,
                    "replacement": normalized_name(str(patch.get("deck_path") or patch.get("request_path") or "patched_value")),
                }
            )

    diff = "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=source_path or "deck.py",
            tofile=(source_path or "deck.py") + ".patched",
        )
    )
    return DeckPatchResult(
        source_path=source_path,
        applied_patches=applied,
        unapplied_patches=unapplied,
        unified_diff=diff,
        patched_source=patched,
        ir=parse_devsim_deck_source(patched, source_path=source_path),
    )


def write_semantic_deck_patch_artifacts(
    source_path: Path,
    patches: list[dict[str, Any]] | dict[str, Any],
    output_dir: Path,
) -> DeckPatchResult:
    source = source_path.read_text(encoding="utf-8")
    result = apply_semantic_deck_patch(source, patches, source_path=str(source_path.resolve()))
    output_dir.mkdir(parents=True, exist_ok=True)
    patched_path = output_dir / f"{source_path.stem}.patched.py"
    diff_path = output_dir / f"{source_path.stem}.patch.diff"
    ir_path = output_dir / f"{source_path.stem}.deck_ir.json"
    result_path = output_dir / f"{source_path.stem}.semantic_patch.json"
    patched_path.write_text(result.patched_source or source, encoding="utf-8")
    diff_path.write_text(result.unified_diff, encoding="utf-8")
    if result.ir:
        ir_path.write_text(json.dumps(result.ir.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    result.patched_source_path = str(patched_path.resolve())
    result.diff_path = str(diff_path.resolve())
    result.ir_path = str(ir_path.resolve())
    result_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    return result
