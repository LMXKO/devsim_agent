from __future__ import annotations

import argparse
import json
import shlex
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from tcad_agent.llm import LLMClient, LLMConfig


ALLOWED_NEXT_TOOL_MODULES = {
    "tcad_agent.tools.pn_junction_iv",
    "tcad_agent.tools.result_judge",
    "tcad_agent.tools.llm_diagnose",
}


class DiagnosisStatus(str, Enum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class ChatClient(Protocol):
    config: LLMConfig

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        ...


class LLMDiagnosisResult(BaseModel):
    status: DiagnosisStatus
    state_path: str
    output_path: str | None = None
    run_id: str | None = None
    model: str | None = None
    quality_status: str | None = None
    reason: str | None = None
    prompt_context: dict[str, Any] | None = None
    raw_response: str | None = None
    parsed_response: dict[str, Any] | None = None
    recommended_next_action: str | None = None


class DiagnosisContext(BaseModel):
    state_path: str
    run_id: str | None = None
    tool_status: str | None = None
    quality_status: str | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    quality_report: dict[str, Any] | None = None
    final_summary: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    log_excerpts: list[dict[str, str]] = Field(default_factory=list)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def text_tail(text: str, limit: int) -> str:
    return text[-limit:] if len(text) > limit else text


def read_tail(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    return text_tail(path.read_text(encoding="utf-8", errors="replace"), limit)


def default_output_path(state_path: Path) -> Path:
    return state_path.parent / "llm_diagnosis.json"


def compact_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": attempt.get("index"),
        "status": attempt.get("status"),
        "step_v": attempt.get("step_v"),
        "returncode": attempt.get("returncode"),
        "failure_class": attempt.get("failure_class"),
        "failure_reason": attempt.get("failure_reason"),
        "summary_path": attempt.get("summary_path"),
        "stderr_tail": text_tail(attempt.get("stderr_tail") or "", 1200),
    }


def failed_attempt_log_path(state: dict[str, Any], attempt: dict[str, Any]) -> Path:
    index = int(attempt["index"])
    return (
        Path(state["run_dir"])
        / "attempt_runs"
        / "pn_junction"
        / f"attempt_{index:03d}"
        / "devsim.log"
    )


def collect_log_excerpts(state: dict[str, Any], max_chars_per_log: int) -> list[dict[str, str]]:
    excerpts: list[dict[str, str]] = []
    for attempt in state.get("attempts", []):
        if attempt.get("status") == "failed":
            path = failed_attempt_log_path(state, attempt)
            text = read_tail(path, max_chars_per_log)
            if text:
                excerpts.append(
                    {
                        "label": f"attempt_{int(attempt['index']):03d}_devsim_log",
                        "path": str(path),
                        "text_tail": text,
                    }
                )

    final_log = (((state.get("final_summary") or {}).get("artifacts") or {}).get("log"))
    if final_log:
        path = Path(final_log)
        text = read_tail(path, max_chars_per_log)
        if text:
            excerpts.append(
                {
                    "label": "successful_attempt_devsim_log",
                    "path": str(path),
                    "text_tail": text,
                }
            )
    return excerpts


def build_context(state_path: Path, max_log_chars: int = 4000) -> DiagnosisContext:
    state = load_json(state_path)
    quality_report = state.get("quality_report") or {}
    return DiagnosisContext(
        state_path=str(state_path),
        run_id=state.get("run_id"),
        tool_status=state.get("status"),
        quality_status=quality_report.get("status"),
        request=state.get("request") or {},
        checkpoint=state.get("checkpoint") or {},
        quality_report=quality_report or None,
        final_summary=state.get("final_summary"),
        attempts=[compact_attempt(attempt) for attempt in state.get("attempts", [])],
        log_excerpts=collect_log_excerpts(state, max_log_chars),
    )


def should_skip(context: DiagnosisContext, force: bool) -> bool:
    return not force and context.quality_status == "passed"


def build_messages(context: DiagnosisContext) -> tuple[str, str]:
    system = (
        "你是 TCAD 仿真诊断助手。"
        "你负责为自主 TCAD agent 诊断 DEVSIM PN junction IV sweep 的执行结果。"
        "请保持保守：区分数值完成和物理可信。"
        "只返回简洁 JSON，不要使用 markdown 代码块；自然语言字段请使用中文。"
    )
    user = {
        "task": "诊断 PN junction IV 仿真运行",
        "required_response_schema": {
            "diagnosis": "中文简短说明发生了什么",
            "risk_level": "low | medium | high",
            "recommended_next_action": "给 TCAD agent 的中文具体下一步动作",
            "next_tool_command": "下一步可执行 shell 命令，或 null",
            "rationale": ["中文简短理由"],
            "follow_up_checks": ["下一步应运行的确定性检查"],
        },
        "context": context.model_dump(mode="json"),
    }
    return system, json.dumps(user, ensure_ascii=False, indent=2)


def parse_json_response(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return normalize_response(parsed)


def normalize_response(parsed: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(parsed)
    next_tool_command = normalized.get("next_tool_command")
    if isinstance(next_tool_command, str) and next_tool_command.strip().lower() in {
        "",
        "none",
        "null",
    }:
        normalized["next_tool_command"] = None
    elif isinstance(next_tool_command, str) and not is_allowed_next_tool_command(
        next_tool_command
    ):
        normalized["rejected_next_tool_command"] = next_tool_command
        normalized["next_tool_command"] = None
    return normalized


def is_allowed_next_tool_command(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if len(tokens) < 3:
        return False
    executable = Path(tokens[0]).name
    return (
        executable == "python3.11"
        and tokens[1] == "-m"
        and tokens[2] in ALLOWED_NEXT_TOOL_MODULES
    )


def diagnose_state(
    state_path: Path,
    output_path: Path | None = None,
    force: bool = False,
    max_log_chars: int = 4000,
    client: ChatClient | None = None,
) -> LLMDiagnosisResult:
    context = build_context(state_path, max_log_chars=max_log_chars)
    actual_output = output_path or default_output_path(state_path)

    if should_skip(context, force):
        result = LLMDiagnosisResult(
            status=DiagnosisStatus.SKIPPED,
            state_path=str(state_path),
            output_path=str(actual_output),
            run_id=context.run_id,
            quality_status=context.quality_status,
            reason="quality_report.status 已通过；除非使用 --force，否则跳过 LLM 诊断。",
            recommended_next_action="接受当前结果产物，并进入下一项 TCAD 任务。",
        )
        write_json(actual_output, result.model_dump(mode="json"))
        return result

    chat_client = client or LLMClient()
    system, user = build_messages(context)
    raw_response = chat_client.chat(system=system, user=user, temperature=0.1)
    parsed = parse_json_response(raw_response)
    recommended = None
    if parsed:
        recommended = parsed.get("recommended_next_action")

    result = LLMDiagnosisResult(
        status=DiagnosisStatus.COMPLETED,
        state_path=str(state_path),
        output_path=str(actual_output),
        run_id=context.run_id,
        model=chat_client.config.model,
        quality_status=context.quality_status,
        prompt_context=context.model_dump(mode="json"),
        raw_response=raw_response,
        parsed_response=parsed,
        recommended_next_action=recommended,
    )
    write_json(actual_output, result.model_dump(mode="json"))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM diagnosis for TCAD tool state.")
    parser.add_argument("--state", type=Path, required=True, help="Path to tool state.json.")
    parser.add_argument("--output", type=Path, default=None, help="Output diagnosis JSON path.")
    parser.add_argument("--force", action="store_true", help="Diagnose even when quality passed.")
    parser.add_argument("--max-log-chars", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = diagnose_state(
            state_path=args.state,
            output_path=args.output,
            force=args.force,
            max_log_chars=args.max_log_chars,
        )
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if result.status != DiagnosisStatus.FAILED else 1)
    except Exception as exc:
        result = LLMDiagnosisResult(
            status=DiagnosisStatus.FAILED,
            state_path=str(args.state),
            output_path=str(args.output) if args.output else None,
            reason=str(exc),
        )
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
