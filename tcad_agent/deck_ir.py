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
    context: str | None = None
    resolved_function: str | None = None
    control_flow: list[str] = Field(default_factory=list)


class DeckImport(BaseModel):
    module: str | None = None
    name: str
    alias: str | None = None
    line: int


class DeckFunction(BaseModel):
    name: str
    start_line: int
    end_line: int
    args: list[str] = Field(default_factory=list)
    defaults: dict[str, Any] = Field(default_factory=dict)
    calls: list[str] = Field(default_factory=list)
    assignments: list[str] = Field(default_factory=list)
    control_flow: list[str] = Field(default_factory=list)


class DeckSemanticBinding(BaseModel):
    name: str
    section: str
    line: int
    source_kind: str
    value: Any = None
    expression: str | None = None
    context: str | None = None
    function: str | None = None
    call_function: str | None = None
    control_flow: list[str] = Field(default_factory=list)


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
    imports: list[DeckImport] = Field(default_factory=list)
    functions: list[DeckFunction] = Field(default_factory=list)
    semantic_bindings: list[DeckSemanticBinding] = Field(default_factory=list)
    external_symbols: list[str] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)


class DeckPatchResult(BaseModel):
    schema_version: str = "actsoft.tcad.deck_semantic_patch.v1"
    source_path: str | None = None
    patched_source_path: str | None = None
    diff_path: str | None = None
    ir_path: str | None = None
    applied_patches: list[dict[str, Any]] = Field(default_factory=list)
    unapplied_patches: list[dict[str, Any]] = Field(default_factory=list)
    verified_patches: list[dict[str, Any]] = Field(default_factory=list)
    unverified_patches: list[dict[str, Any]] = Field(default_factory=list)
    all_patches_verified: bool = False
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


def expression_text(source: str, node: ast.AST | None) -> str | None:
    if node is None:
        return None
    span = node_span(source, node)
    if not span:
        return None
    return source[span[0] : span[1]]


def resolved_call_name(name: str | None, aliases: dict[str, str]) -> str | None:
    if not name:
        return None
    head, _, tail = name.partition(".")
    if head in aliases:
        return f"{aliases[head]}.{tail}" if tail else aliases[head]
    return name


def is_devsim_like_call(name: str | None) -> bool:
    if not name:
        return False
    leaf = name.rsplit(".", 1)[-1]
    return (
        name.startswith("devsim.")
        or leaf.startswith(("create_", "add_", "set_", "get_", "node_", "edge_", "element_"))
        or leaf in {"solve", "rampbias", "set_parameter", "node_model", "edge_model", "contact_equation"}
    )


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


class _DeckIRCollector(ast.NodeVisitor):
    def __init__(self, source: str):
        self.source = source
        self.lines = source.splitlines()
        self.assignments: list[DeckAssignment] = []
        self.calls: list[DeckCall] = []
        self.imports: list[DeckImport] = []
        self.functions: list[DeckFunction] = []
        self.semantic_bindings: list[DeckSemanticBinding] = []
        self.external_symbols: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.context_stack: list[str] = []
        self.control_stack: list[str] = []
        self.function_stack: list[DeckFunction] = []

    def context(self) -> str | None:
        return ".".join(self.context_stack) if self.context_stack else None

    def current_function(self) -> str | None:
        return self.function_stack[-1].name if self.function_stack else None

    def text_for(self, node: ast.AST) -> str:
        start_line = int(getattr(node, "lineno", 1))
        end_line = int(getattr(node, "end_lineno", start_line))
        return source_segment(self.lines, start_line, end_line)

    def add_binding(
        self,
        *,
        name: str,
        node: ast.AST,
        source_kind: str,
        value_node: ast.AST | None = None,
        symbols: list[str] | None = None,
        call_function: str | None = None,
    ) -> None:
        line = int(getattr(node, "lineno", 1))
        text = self.text_for(node)
        section = classify_section(text, [name, *(symbols or [])])
        value = literal_value(value_node) if value_node is not None else None
        binding = DeckSemanticBinding(
            name=name,
            section=section,
            line=line,
            source_kind=source_kind,
            value=value,
            expression=expression_text(self.source, value_node),
            context=self.context(),
            function=self.current_function(),
            call_function=call_function,
            control_flow=list(self.control_stack),
        )
        self.semantic_bindings.append(binding)
        if isinstance(value_node, ast.Name) and value is None:
            self.external_symbols.add(value_node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            asname = alias.asname or alias.name.split(".", 1)[0]
            self.aliases[asname] = alias.name
            self.imports.append(DeckImport(module=None, name=alias.name, alias=alias.asname, line=int(node.lineno)))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            asname = alias.asname or alias.name
            full_name = f"{module}.{alias.name}" if module else alias.name
            self.aliases[asname] = full_name
            self.imports.append(DeckImport(module=module or None, name=alias.name, alias=alias.asname, line=int(node.lineno)))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        args = [arg.arg for arg in node.args.args]
        defaults: dict[str, Any] = {}
        default_offset = len(args) - len(node.args.defaults)
        function = DeckFunction(
            name=node.name,
            start_line=int(node.lineno),
            end_line=int(getattr(node, "end_lineno", node.lineno)),
            args=args,
        )
        self.functions.append(function)
        self.context_stack.append(node.name)
        self.function_stack.append(function)
        for index, default in enumerate(node.args.defaults):
            arg_name = args[default_offset + index]
            defaults[arg_name] = literal_value(default)
            self.add_binding(
                name=arg_name,
                node=default,
                source_kind="function_default",
                value_node=default,
                symbols=[node.name, arg_name],
            )
        function.defaults = defaults
        self.generic_visit(node)
        self.function_stack.pop()
        self.context_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_For(self, node: ast.For) -> None:
        self.control_stack.append("for")
        if self.function_stack and "for" not in self.function_stack[-1].control_flow:
            self.function_stack[-1].control_flow.append("for")
        self.generic_visit(node)
        self.control_stack.pop()

    def visit_While(self, node: ast.While) -> None:
        self.control_stack.append("while")
        if self.function_stack and "while" not in self.function_stack[-1].control_flow:
            self.function_stack[-1].control_flow.append("while")
        self.generic_visit(node)
        self.control_stack.pop()

    def visit_If(self, node: ast.If) -> None:
        self.control_stack.append("if")
        if self.function_stack and "if" not in self.function_stack[-1].control_flow:
            self.function_stack[-1].control_flow.append("if")
        self.generic_visit(node)
        self.control_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        names: list[str] = []
        for target in node.targets:
            names.extend(target_names(target))
        section = classify_section(self.text_for(node), names)
        for name in names:
            self.assignments.append(DeckAssignment(name=name, value=literal_value(node.value), section=section, line=int(node.lineno)))
            self.add_binding(name=name, node=node, source_kind="assignment", value_node=node.value, symbols=names)
            if self.function_stack and name not in self.function_stack[-1].assignments:
                self.function_stack[-1].assignments.append(name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        names = target_names(node.target)
        section = classify_section(self.text_for(node), names)
        for name in names:
            self.assignments.append(DeckAssignment(name=name, value=literal_value(node.value), section=section, line=int(node.lineno)))
            self.add_binding(name=name, node=node, source_kind="assignment", value_node=node.value, symbols=names)
            if self.function_stack and name not in self.function_stack[-1].assignments:
                self.function_stack[-1].assignments.append(name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = call_name(node.func)
        resolved = resolved_call_name(name, self.aliases)
        if name:
            symbols = [name]
            if resolved and resolved != name:
                symbols.append(resolved)
            keyword_values = {
                keyword.arg: literal_value(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            section = classify_section(self.text_for(node), symbols)
            self.calls.append(
                DeckCall(
                    function=name,
                    resolved_function=resolved,
                    section=section,
                    line=int(getattr(node, "lineno", 1)),
                    keywords=keyword_values,
                    context=self.context(),
                    control_flow=list(self.control_stack),
                )
            )
            if self.function_stack and name not in self.function_stack[-1].calls:
                self.function_stack[-1].calls.append(name)
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                self.add_binding(
                    name=keyword.arg,
                    node=keyword.value,
                    source_kind="devsim_call_keyword" if is_devsim_like_call(resolved or name) else "call_keyword",
                    value_node=keyword.value,
                    symbols=[name, resolved or "", keyword.arg],
                    call_function=resolved or name,
                )
            named = next((literal_value(keyword.value) for keyword in node.keywords if keyword.arg in {"name", "parameter", "model"}), None)
            if isinstance(named, str):
                for keyword in node.keywords:
                    if keyword.arg in {"value", "init", "default"}:
                        self.add_binding(
                            name=named,
                            node=keyword.value,
                            source_kind="named_devsim_parameter",
                            value_node=keyword.value,
                            symbols=[name, resolved or "", named],
                            call_function=resolved or name,
                        )
        self.generic_visit(node)


def parse_devsim_deck_source(source: str, *, source_path: str | None = None) -> DeckSourceIR:
    lines = source.splitlines()
    warnings: list[str] = []
    sections: list[DeckIRSection] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return DeckSourceIR(
            source_path=source_path,
            parse_warnings=[f"syntax_error:{exc.lineno}:{exc.msg}"],
        )

    collector = _DeckIRCollector(source)
    collector.visit(tree)

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
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(node.name)
            preferred = classify_section(node.name)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = call_name(node.value.func)
            if name:
                call_symbols.append(name)

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
    return DeckSourceIR(
        source_path=source_path,
        sections=sections,
        assignments=collector.assignments,
        calls=collector.calls,
        imports=collector.imports,
        functions=collector.functions,
        semantic_bindings=collector.semantic_bindings,
        external_symbols=sorted(collector.external_symbols),
        parse_warnings=warnings,
    )


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
        candidates.add(normalized_name(".".join(parts)))
        candidates.add(normalized_name("_".join(parts)))
        for index in range(len(parts)):
            candidates.add(normalized_name("_".join(parts[index:])))
    if parts:
        candidates.add(normalized_name(parts[-1]))
    for key in ["name", "target", "source_mutation"]:
        if patch.get(key):
            candidates.add(normalized_name(str(patch[key])))
    candidates.update(semantic_aliases_for_patch(patch, candidates))
    return {candidate for candidate in candidates if candidate and candidate not in {"geometry", "mesh", "doping", "physics_models", "model"}}


def semantic_aliases_for_patch(patch: dict[str, Any], candidates: set[str]) -> set[str]:
    combined = " ".join(
        [
            *(sorted(candidates)),
            normalized_name(str(patch.get("deck_path") or "")),
            normalized_name(str(patch.get("request_path") or "")),
            normalized_name(str(patch.get("target") or "")),
            normalized_name(str(patch.get("source_mutation") or "")),
        ]
    )
    aliases: set[str] = set()
    if any(token in combined for token in ["doping", "donor", "acceptor", "implant"]):
        aliases.update(
            {
                "netdoping",
                "net_doping",
                "donor",
                "donors",
                "acceptor",
                "acceptors",
                "doping",
                "concentration",
                "n_doping",
                "p_doping",
                "profile",
                "dose",
            }
        )
    if "field_plate" in combined or "fieldplate" in combined:
        aliases.update({"fieldplate", "field_plate", "field_plate_length", "field_plate_length_um", "plate_length"})
    if "guard_ring" in combined or "guardring" in combined:
        aliases.update({"guardring", "guard_ring", "guard_ring_spacing", "guard_ring_spacing_um", "termination_spacing"})
    if "junction" in combined:
        aliases.update({"junction", "junction_depth", "junction_depth_um", "junction_spacing", "junction_spacing_um"})
    if "oxide" in combined or "tox" in combined:
        aliases.update({"tox", "tox_nm", "oxide", "oxide_thickness", "oxide_thickness_nm", "gate_oxide", "gate_oxide_thickness"})
    if "lifetime" in combined or "srh" in combined:
        aliases.update(
            {
                "lifetime",
                "carrier_lifetime",
                "carrier_lifetime_s",
                "electron_lifetime",
                "electron_lifetime_s",
                "hole_lifetime",
                "hole_lifetime_s",
                "srh_lifetime",
                "taun",
                "taup",
            }
        )
    if "trap" in combined:
        aliases.update({"trap", "traps", "trap_density", "trap_density_cm2", "interface_trap", "interface_trap_density"})
    if "trench" in combined or "corner_radius" in combined:
        aliases.update({"trench", "trench_radius", "corner_radius", "corner_radius_um", "trench_corner_radius"})
    return aliases


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


def literal_numeric_value(node: ast.AST) -> float | None:
    value = literal_value(node)
    if isinstance(value, bool):
        return None
    return finite_float(value)


def operation_name(patch: dict[str, Any]) -> str:
    return str(patch.get("operation") or patch.get("op") or "set").strip().lower().replace("-", "_")


def patch_numeric_value(original: float | None, patch: dict[str, Any]) -> Any:
    operation = operation_name(patch)
    raw_value = patch.get("value")
    numeric_value = finite_float(raw_value)
    if operation in {"set", "replace"}:
        return raw_value
    if original is None or numeric_value is None:
        return raw_value
    if operation in {"scale", "multiply", "mul"}:
        return original * numeric_value
    if operation in {"add", "offset", "delta"}:
        return original + numeric_value
    if operation in {"subtract", "sub"}:
        return original - numeric_value
    if operation in {"percent", "pct", "relative_percent"}:
        return original * (1.0 + numeric_value / 100.0)
    return raw_value


def replacement_for_node(source: str, node: ast.AST, patch: dict[str, Any]) -> tuple[str, str, str | None]:
    """Return replacement text, original text, and optional inner replacement reason."""
    span = node_span(source, node)
    original = source[span[0] : span[1]] if span else ""
    direct_numeric = literal_numeric_value(node)
    if direct_numeric is not None or operation_name(patch) in {"set", "replace"}:
        if not isinstance(node, ast.Call):
            return python_literal(patch_numeric_value(direct_numeric, patch)), original, None

    if isinstance(node, ast.Call):
        numeric_children: list[ast.AST] = [
            arg for arg in node.args if literal_numeric_value(arg) is not None
        ]
        numeric_children.extend(
            keyword.value
            for keyword in node.keywords
            if keyword.arg not in {"name", "parameter", "model", "region", "device"}
            and literal_numeric_value(keyword.value) is not None
        )
        if numeric_children:
            child = numeric_children[0]
            child_span = node_span(source, child)
            if span and child_span:
                child_original = source[child_span[0] : child_span[1]]
                child_numeric = literal_numeric_value(child)
                child_replacement = python_literal(patch_numeric_value(child_numeric, patch))
                relative_start = child_span[0] - span[0]
                relative_end = child_span[1] - span[0]
                return (
                    original[:relative_start] + child_replacement + original[relative_end:],
                    original,
                    f"preserved_call_wrapper:{child_original}->{child_replacement}",
                )

    return python_literal(patch.get("value")), original, None


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
        self.full_path_candidate = normalized_name("_".join(parts)) if parts else ""
        self.preferred_section = patch_section(patch)
        self.matches: list[tuple[int, ast.AST, str]] = []
        self.path_stack: list[str] = []

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
        if self.full_path_candidate and any(normalized_name(name) == self.full_path_candidate for name in names):
            priority += 80
        if self.leaf_candidate and any(match_name(name, {self.leaf_candidate}) for name in names):
            priority += 30
        priority += 160 if reason == "function_default" else 0
        priority += 25 if reason == "loop_iterable" else 0
        priority += 10 if reason in {"assignment", "dict_key", "call_keyword"} else 0
        self.matches.append((priority, node, reason))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        args = list(node.args.args)
        default_offset = len(args) - len(node.args.defaults)
        self.path_stack.append(node.name)
        for index, default in enumerate(node.args.defaults):
            arg_name = args[default_offset + index].arg
            names = [arg_name, ".".join([*self.path_stack, arg_name]), "_".join([*self.path_stack, arg_name])]
            self.score(default, names, "function_default")
        self.generic_visit(node)
        self.path_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_For(self, node: ast.For) -> None:
        names = target_names(node.target)
        if names:
            self.score(node.iter, names, "loop_iterable")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        names: list[str] = []
        for target in node.targets:
            names.extend(target_names(target))
        self.score(node.value, names, "assignment")
        if isinstance(node.value, ast.Dict) and names:
            self.path_stack.append(names[0])
            self.visit(node.value)
            self.path_stack.pop()
        else:
            self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        names = target_names(node.target)
        self.score(node.value, names, "assignment")
        if isinstance(node.value, ast.Dict) and names:
            self.path_stack.append(names[0])
            self.visit(node.value)
            self.path_stack.pop()
        else:
            self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key_node, value_node in zip(node.keys, node.values):
            key = literal_value(key_node) if key_node is not None else None
            if isinstance(key, str):
                self.path_stack.append(key)
                path_name = ".".join(self.path_stack)
                names = [key, path_name, "_".join(self.path_stack)]
                self.score(value_node, names, "dict_key")
                self.visit(value_node)
                self.path_stack.pop()
            else:
                self.visit(value_node)

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
        replacement, _original, inner_reason = replacement_for_node(source, node, patch)
        replacements.append((span[0], span[1], replacement, patch, reason if inner_reason is None else f"{reason}:{inner_reason}"))

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
                "effective": True,
                "semantic_status": "verified_existing_binding",
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
                    "effective": False,
                    "semantic_status": "unverified_fallback_append",
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
    verified = [patch for patch in applied if patch.get("effective")]
    unverified = [
        {**patch, "semantic_status": "unverified_fallback_append"}
        for patch in unapplied
    ]
    return DeckPatchResult(
        source_path=source_path,
        applied_patches=applied,
        unapplied_patches=unapplied,
        verified_patches=verified,
        unverified_patches=unverified,
        all_patches_verified=not unapplied and len(verified) == len(patch_list),
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
