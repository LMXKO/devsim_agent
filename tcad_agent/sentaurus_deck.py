from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SentaurusDeckAssignment(BaseModel):
    key: str
    value: str
    line: int
    block_index: int | None = None
    section_path: list[str] = Field(default_factory=list)
    raw: str = ""


class SentaurusDeckBlock(BaseModel):
    index: int
    name: str
    header: str = ""
    start_line: int
    end_line: int
    depth: int
    path: list[str] = Field(default_factory=list)
    assignments: list[SentaurusDeckAssignment] = Field(default_factory=list)


class SentaurusDeckIR(BaseModel):
    schema_version: str = "actsoft.tcad.sentaurus_deck_ir.v1"
    source_path: str
    sections: list[SentaurusDeckBlock] = Field(default_factory=list)
    set_variables: list[SentaurusDeckAssignment] = Field(default_factory=list)
    assignments: list[SentaurusDeckAssignment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def sentaurus_section_index(ir: SentaurusDeckIR) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for section in ir.sections:
        key = "/".join(section.path) if section.path else section.name
        output.setdefault(key, []).append(
            {
                "start_line": section.start_line,
                "end_line": section.end_line,
                "depth": section.depth,
                "assignment_keys": [assignment.key for assignment in section.assignments[:12]],
            }
        )
    return output


def parse_block_header(prefix: str) -> tuple[str, str]:
    cleaned = prefix.strip()
    if not cleaned:
        return "{record}", ""
    match = re.match(r"([A-Za-z_][\w:.-]*)", cleaned)
    if not match:
        return "{record}", cleaned
    return match.group(1), cleaned[match.end() :].strip()


def unquoted_brace_events(line: str) -> list[tuple[int, str]]:
    events: list[tuple[int, str]] = []
    in_quote: str | None = None
    escaped = False
    for index, token in enumerate(line):
        if escaped:
            escaped = False
            continue
        if token == "\\":
            escaped = True
            continue
        if token in {"'", '"'}:
            if in_quote == token:
                in_quote = None
            elif in_quote is None:
                in_quote = token
            continue
        if in_quote is None and token in "{}":
            events.append((index, token))
    return events


def count_unquoted_char(value: str, char: str) -> int:
    count = 0
    in_quote: str | None = None
    escaped = False
    for token in value:
        if escaped:
            escaped = False
            continue
        if token == "\\":
            escaped = True
            continue
        if token in {"'", '"'}:
            if in_quote == token:
                in_quote = None
            elif in_quote is None:
                in_quote = token
            continue
        if token == char and in_quote is None:
            count += 1
    return count


def matching_open_paren(text: str, close_index: int) -> int | None:
    depth = 0
    in_quote: str | None = None
    escaped = False
    for index in range(close_index, -1, -1):
        token = text[index]
        if escaped:
            escaped = False
            continue
        if token == "\\":
            escaped = True
            continue
        if token in {"'", '"'}:
            if in_quote == token:
                in_quote = None
            elif in_quote is None:
                in_quote = token
            continue
        if in_quote is not None:
            continue
        if token == ")":
            depth += 1
        elif token == "(":
            depth -= 1
            if depth == 0:
                return index
    return None


def block_header_before_open(prefix: str) -> tuple[str, str]:
    left = prefix.rstrip()
    if not left:
        return "{record}", ""
    if left.endswith(")"):
        open_index = matching_open_paren(left, len(left) - 1)
        if open_index is not None:
            prefix = left[:open_index].rstrip()
            name, prefix_header = parse_block_header(prefix)
            if name != "{record}":
                suffix = left[open_index:].strip()
                header = " ".join(part for part in [prefix_header, suffix] if part)
                return name, header
    if count_unquoted_char(left, "(") > count_unquoted_char(left, ")"):
        match = re.search(r"([A-Za-z_][\w:.-]*)\s*$", left)
        if match:
            return match.group(1), left[match.start(1) :].strip()
    return parse_block_header(left)


ASSIGNMENT_RE = re.compile(
    r"(?P<key>[A-Za-z_][\w:.-]*)\s*=\s*(?P<value>\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'|[^\s{}()]+)"
)
SET_RE = re.compile(r"^\s*set\s+(?P<key>[A-Za-z_][\w:.-]*)\s+(?P<value>.+?)\s*(?:#.*)?$")
DEFINE_RE = re.compile(r"^\s*#\s*define\s+(?P<key>[A-Za-z_][\w:.-]*)\s+(?P<value>.+?)\s*$")
FLOW_CALL_RE = re.compile(r"^\s*(?P<name>Quasistationary|Transient|ACCoupled)\s*\(", re.IGNORECASE)


def parse_assignments(raw: str, line_number: int, path: list[str], block_index: int | None) -> list[SentaurusDeckAssignment]:
    assignments: list[SentaurusDeckAssignment] = []
    for match in ASSIGNMENT_RE.finditer(raw):
        assignments.append(
            SentaurusDeckAssignment(
                key=match.group("key"),
                value=match.group("value"),
                line=line_number,
                block_index=block_index,
                section_path=path,
                raw=match.group(0),
            )
        )
    return assignments


def parse_sentaurus_deck_text(text: str, *, source_path: str = "") -> SentaurusDeckIR:
    lines = text.splitlines()
    sections: list[SentaurusDeckBlock] = []
    assignments: list[SentaurusDeckAssignment] = []
    set_variables: list[SentaurusDeckAssignment] = []
    stack: list[int] = []
    warnings: list[str] = []

    for line_index, line in enumerate(lines, start=1):
        opened_logical_flow = False
        flow_match = FLOW_CALL_RE.match(line)
        if flow_match and not any(token == "{" for _, token in unquoted_brace_events(line)):
            name = flow_match.group("name")
            open_logical_flow_block(sections, stack, name=name, header=line.strip(), line_index=line_index)
            opened_logical_flow = True

        set_match = SET_RE.match(line)
        define_match = DEFINE_RE.match(line)
        if set_match or define_match:
            match = set_match or define_match
            set_variables.append(
                SentaurusDeckAssignment(
                    key=match.group("key"),
                    value=match.group("value").strip(),
                    line=line_index,
                    block_index=None,
                    section_path=[],
                    raw=line.strip(),
                )
            )

        last_index = 0
        for brace_index, brace in unquoted_brace_events(line):
            if brace == "}":
                current_path = sections[stack[-1]].path if stack else []
                current_index = stack[-1] if stack else None
                segment = line[last_index:brace_index]
                line_assignments = parse_assignments(segment, line_index, current_path, current_index)
                assignments.extend(line_assignments)
                if current_index is not None:
                    sections[current_index].assignments.extend(line_assignments)
                if not stack:
                    warnings.append(f"unmatched closing brace at line {line_index}")
                else:
                    block_index = stack.pop()
                    sections[block_index].end_line = line_index
                last_index = brace_index + 1
                continue

            segment = line[last_index:brace_index]
            inline_flow = FLOW_CALL_RE.search(segment)
            if inline_flow and not (stack and same_token(sections[stack[-1]].name, inline_flow.group("name"))):
                flow_block = open_logical_flow_block(
                    sections,
                    stack,
                    name=inline_flow.group("name"),
                    header=segment[inline_flow.start() :].strip(),
                    line_index=line_index,
                )
                flow_assignments = parse_assignments(
                    segment[inline_flow.start() :],
                    line_index,
                    flow_block.path,
                    flow_block.index,
                )
                if flow_assignments:
                    assignments.extend(flow_assignments)
                    flow_block.assignments.extend(flow_assignments)

            name, header = block_header_before_open(segment)
            if name == "{record}" and stack and sections[stack[-1]].name.lower() in {"quasistationary", "transient", "accoupled"}:
                last_index = brace_index + 1
                continue
            parent_path = sections[stack[-1]].path if stack else []
            block = SentaurusDeckBlock(
                index=len(sections),
                name=name,
                header=header,
                start_line=line_index,
                end_line=line_index,
                depth=len(stack),
                path=[*parent_path, name],
            )
            sections.append(block)
            header_assignments = parse_assignments(header, line_index, block.path, block.index)
            if header_assignments:
                sections[block.index].assignments.extend(header_assignments)
                assignments.extend(header_assignments)
            stack.append(block.index)
            last_index = brace_index + 1

        if last_index < len(line):
            current_path = sections[stack[-1]].path if stack else []
            current_index = stack[-1] if stack else None
            line_assignments = parse_assignments(line[last_index:], line_index, current_path, current_index)
            assignments.extend(line_assignments)
            if current_index is not None:
                sections[current_index].assignments.extend(line_assignments)
        if opened_logical_flow and stack and sections[stack[-1]].start_line == line_index:
            sections[stack[-1]].assignments.extend(
                parse_assignments(line, line_index, sections[stack[-1]].path, stack[-1])
            )

    for block_index in stack:
        sections[block_index].end_line = len(lines)
        warnings.append(f"unclosed block {sections[block_index].name} starting at line {sections[block_index].start_line}")

    return SentaurusDeckIR(
        source_path=source_path,
        sections=sections,
        set_variables=set_variables,
        assignments=assignments,
        warnings=warnings,
    )


def parse_sentaurus_deck_file(path: Path) -> SentaurusDeckIR:
    return parse_sentaurus_deck_text(path.read_text(encoding="utf-8", errors="replace"), source_path=str(path))


def open_logical_flow_block(
    sections: list[SentaurusDeckBlock],
    stack: list[int],
    *,
    name: str,
    header: str,
    line_index: int,
) -> SentaurusDeckBlock:
    parent_path = sections[stack[-1]].path if stack else []
    block = SentaurusDeckBlock(
        index=len(sections),
        name=name,
        header=header.strip(),
        start_line=line_index,
        end_line=line_index,
        depth=len(stack),
        path=[*parent_path, name],
    )
    sections.append(block)
    stack.append(block.index)
    return block


def normalize_path(path: list[str] | str | None) -> list[str]:
    if path is None:
        return []
    if isinstance(path, str):
        return [part for part in re.split(r"[./]", path) if part]
    return [str(part) for part in path if str(part)]


def same_token(left: str, right: str) -> bool:
    return left.strip().lower() == right.strip().lower()


def path_matches(block_path: list[str], requested_path: list[str]) -> bool:
    if not requested_path:
        return True
    normalized_block = [part.lower() for part in block_path]
    normalized_requested = [part.lower() for part in requested_path]
    if len(normalized_block) < len(normalized_requested):
        return False
    return normalized_block[: len(normalized_requested)] == normalized_requested


def unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def block_assignment_map(block: SentaurusDeckBlock) -> dict[str, str]:
    return {assignment.key.lower(): unquote(assignment.value) for assignment in block.assignments}


def block_matches_selector(block: SentaurusDeckBlock, selector: dict[str, Any] | None) -> bool:
    if not selector:
        return True
    values = block_assignment_map(block)
    for key, expected in selector.items():
        actual = values.get(str(key).lower())
        if actual is None or actual != str(expected):
            return False
    return True


def find_target_blocks(
    ir: SentaurusDeckIR,
    *,
    section_path: list[str] | str | None,
    selector: dict[str, Any] | None = None,
) -> list[SentaurusDeckBlock]:
    requested = normalize_path(section_path)
    blocks = [
        block
        for block in ir.sections
        if path_matches(block.path, requested) and block_matches_selector(block, selector)
    ]
    if selector and requested:
        under_section = [
            block
            for block in ir.sections
            if path_matches(block.path, requested)
        ]
        selected: list[SentaurusDeckBlock] = []
        for block in ir.sections:
            if not any(path_matches(block.path, parent.path) for parent in under_section):
                continue
            if block_matches_selector(block, selector):
                selected.append(block)
        return selected or blocks
    return blocks


def format_value(value: Any, *, existing_value: str | None = None) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.12g}"
    text = str(value)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text
    if text.startswith("@") and text.endswith("@"):
        return text
    if existing_value and existing_value.strip().startswith('"'):
        return json.dumps(text)
    if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", text):
        return text
    if re.fullmatch(r"[A-Za-z_][\w:./@+-]*", text):
        return text
    return json.dumps(text)


def line_indent(line: str, fallback: str = "  ") -> str:
    match = re.match(r"\s*", line)
    indent = match.group(0) if match else ""
    return indent or fallback


def replace_assignment_on_line(line: str, key: str, value: Any) -> tuple[str, bool, str | None]:
    for match in ASSIGNMENT_RE.finditer(line):
        if not same_token(match.group("key"), key):
            continue
        existing = match.group("value")
        replacement = f"{match.group('key')}={format_value(value, existing_value=existing)}"
        return line[: match.start()] + replacement + line[match.end() :], True, existing
    return line, False, None


def apply_set_variable(lines: list[str], ir: SentaurusDeckIR, *, variable: str, value: Any) -> tuple[list[str], dict[str, Any]]:
    record: dict[str, Any] = {
        "operation": "sentaurus_set_variable",
        "variable": variable,
        "applied": False,
        "verified": False,
    }
    for assignment in ir.set_variables:
        if not same_token(assignment.key, variable):
            continue
        line = lines[assignment.line - 1]
        formatted = format_value(value, existing_value=assignment.value)
        if line.lstrip().startswith("#"):
            lines[assignment.line - 1] = re.sub(
                rf"^(\s*#\s*define\s+{re.escape(assignment.key)}\s+).*$",
                rf"\g<1>{formatted}",
                line,
            )
        else:
            lines[assignment.line - 1] = re.sub(
                rf"^(\s*set\s+{re.escape(assignment.key)}\s+).*$",
                rf"\g<1>{formatted}",
                line,
            )
        record.update({"applied": True, "verified": True, "line": assignment.line, "old_value": assignment.value, "value": value})
        return lines, record
    record["error"] = "set/#define variable not found"
    return lines, record


def apply_section_assignment(
    lines: list[str],
    ir: SentaurusDeckIR,
    *,
    section_path: list[str] | str | None,
    parameter: str,
    value: Any,
    selector: dict[str, Any] | None = None,
    insert_if_missing: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    record: dict[str, Any] = {
        "operation": "sentaurus_upsert_assignment" if insert_if_missing else "sentaurus_update_assignment",
        "section_path": normalize_path(section_path),
        "selector": selector or {},
        "parameter": parameter,
        "value": value,
        "applied": False,
        "verified": False,
    }
    blocks = find_target_blocks(ir, section_path=section_path, selector=selector)
    if not blocks:
        record["error"] = "target section/selector not found"
        return lines, record
    block = blocks[0]
    for assignment in block.assignments:
        if not same_token(assignment.key, parameter):
            continue
        new_line, changed, old_value = replace_assignment_on_line(lines[assignment.line - 1], parameter, value)
        if not changed:
            record["error"] = "assignment line could not be rewritten"
            return lines, record
        lines[assignment.line - 1] = new_line
        record.update(
            {
                "applied": True,
                "verified": True,
                "line": assignment.line,
                "old_value": old_value,
                "block_path": block.path,
            }
        )
        return lines, record

    if not insert_if_missing:
        record["error"] = "parameter not found in target section"
        return lines, record

    formatted = f"{parameter}={format_value(value)}"
    if block.start_line == block.end_line:
        line = lines[block.start_line - 1]
        insert_at = line.rfind("}")
        if insert_at < 0:
            record["error"] = "inline target block has no closing brace"
            return lines, record
        separator = "" if line[:insert_at].endswith((" ", "\t")) else " "
        lines[block.start_line - 1] = line[:insert_at] + f"{separator}{formatted} " + line[insert_at:]
        record.update({"applied": True, "verified": True, "line": block.start_line, "inserted": True, "block_path": block.path})
        return lines, record

    close_index = max(block.end_line - 1, block.start_line)
    base_indent = line_indent(lines[block.start_line - 1])
    insert_indent = base_indent + "  "
    lines.insert(close_index, f"{insert_indent}{formatted}")
    record.update({"applied": True, "verified": True, "line": close_index + 1, "inserted": True, "block_path": block.path})
    return lines, record


def apply_model_line(
    lines: list[str],
    ir: SentaurusDeckIR,
    *,
    section_path: list[str] | str | None,
    model: str,
) -> tuple[list[str], dict[str, Any]]:
    record: dict[str, Any] = {
        "operation": "sentaurus_add_model",
        "section_path": normalize_path(section_path),
        "model": model,
        "applied": False,
        "verified": False,
    }
    blocks = find_target_blocks(ir, section_path=section_path)
    if not blocks:
        record["error"] = "target section not found"
        return lines, record
    block = blocks[0]
    if any(model.strip() in lines[index - 1] for index in range(block.start_line, block.end_line + 1)):
        record.update({"applied": True, "verified": True, "already_present": True, "block_path": block.path})
        return lines, record
    close_index = max(block.end_line - 1, block.start_line)
    base_indent = line_indent(lines[block.start_line - 1])
    lines.insert(close_index, f"{base_indent}  {model.strip()}")
    record.update({"applied": True, "verified": True, "line": close_index + 1, "inserted": True, "block_path": block.path})
    return lines, record


def apply_sentaurus_semantic_patch_text(text: str, patch: dict[str, Any], *, source_path: str = "") -> tuple[str, dict[str, Any], SentaurusDeckIR]:
    before_lines = text.splitlines()
    lines = list(before_lines)
    ir = parse_sentaurus_deck_text(text, source_path=source_path)
    operation = str(patch.get("operation") or "")
    if operation == "sentaurus_set_variable":
        variable = str(patch.get("variable") or patch.get("parameter") or "")
        lines, record = apply_set_variable(lines, ir, variable=variable, value=patch.get("value"))
    elif operation in {"sentaurus_update_assignment", "sentaurus_upsert_assignment"}:
        parameter = str(patch.get("parameter") or patch.get("key") or "")
        lines, record = apply_section_assignment(
            lines,
            ir,
            section_path=patch.get("section_path"),
            parameter=parameter,
            value=patch.get("value"),
            selector=patch.get("selector") if isinstance(patch.get("selector"), dict) else None,
            insert_if_missing=operation == "sentaurus_upsert_assignment" or bool(patch.get("insert_if_missing")),
        )
    elif operation == "sentaurus_add_model":
        lines, record = apply_model_line(
            lines,
            ir,
            section_path=patch.get("section_path"),
            model=str(patch.get("model") or patch.get("value") or ""),
        )
    else:
        record = {"operation": operation, "applied": False, "verified": False, "error": "unsupported Sentaurus semantic operation"}

    after_text = "\n".join(lines) + ("\n" if text.endswith("\n") or lines else "")
    if record.get("applied") and after_text == text and not record.get("already_present"):
        record.update({"applied": False, "verified": False, "error": "semantic patch produced no content change"})
    if record.get("applied"):
        record["diff"] = "\n".join(
            difflib.unified_diff(
                before_lines,
                after_text.splitlines(),
                fromfile=f"a/{source_path}",
                tofile=f"b/{source_path}",
                lineterm="",
            )
        )
    patched_ir = parse_sentaurus_deck_text(after_text, source_path=source_path)
    record["round_trip_verified"] = not patched_ir.warnings
    record["round_trip_warnings"] = patched_ir.warnings
    record["source_section_index"] = sentaurus_section_index(ir)
    record["patched_section_index"] = sentaurus_section_index(patched_ir)
    record["patch_lineage"] = [
        {
            "operation": record.get("operation"),
            "line": record.get("line"),
            "block_path": record.get("block_path"),
            "section_path": record.get("section_path"),
            "parameter": record.get("parameter") or record.get("variable") or record.get("model"),
            "applied": record.get("applied"),
            "verified": record.get("verified"),
            "round_trip_verified": record.get("round_trip_verified"),
        }
    ]
    return after_text, record, patched_ir
