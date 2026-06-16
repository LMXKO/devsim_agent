from __future__ import annotations

import json
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote

from tcad_agent.llm_health import check_llm_health
from tcad_agent.run_queue import (
    cancel_item,
    default_queue_db_path,
    pause_item,
    recover_stale_items,
    resume_item,
    run_queue_worker,
)
from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.web_app import (
    WebAppConfig,
    WorkerController,
    activity_has_artifacts,
    activity_has_process,
    approve_item_confirmation,
    collect_execution_activity,
    collect_recent_experiment_activity,
    collect_web_state_data,
    enqueue_mission_from_payload,
    artifact_content_type,
    int_from_payload,
    llm_settings_response,
    recover_owner_running_items,
    render_app_html,
    reject_item_confirmation,
    resolve_artifact_path,
    save_llm_settings_from_payload,
    save_sentaurus_settings_from_payload,
    sentaurus_settings_response,
)


CONFIG = WebAppConfig(
    root=PROJECT_ROOT / "runs",
    queue_db_path=default_queue_db_path(),
    worker_stop_file=PROJECT_ROOT / "runs" / "tcad_web_worker.stop",
)
WORKER = WorkerController(CONFIG)
recover_owner_running_items(CONFIG.queue_db_path, owner=CONFIG.worker_owner)
LAST_LLM_STATUS: dict[str, Any] | None = None


async def read_body(receive: Any) -> bytes:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] != "http.request":
            continue
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


async def send_response(send: Any, status: HTTPStatus, content_type: str, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status.value,
            "headers": [
                (b"content-type", content_type.encode("utf-8")),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def send_json(send: Any, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
    await send_response(
        send,
        status,
        "application/json; charset=utf-8",
        json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"),
    )


async def send_html(send: Any, text: str) -> None:
    await send_response(send, HTTPStatus.OK, "text/html; charset=utf-8", text.encode("utf-8"))


async def send_error(send: Any, status: HTTPStatus, message: str) -> None:
    await send_json(send, {"status": "failed", "failure_reason": message}, status)


async def payload_from_request(receive: Any) -> dict[str, Any]:
    raw = await read_body(receive)
    if not raw:
        return {}
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def collect_state() -> dict[str, Any]:
    data = collect_web_state_data(
        CONFIG.root,
        queue_db_path=CONFIG.queue_db_path,
        index_db_path=CONFIG.index_db_path,
        rebuild=CONFIG.rebuild_index,
    )
    if LAST_LLM_STATUS:
        data["llm_status"] = LAST_LLM_STATUS
    data["worker_status"] = WORKER.status()
    activity = collect_execution_activity(data.get("queue_items") or [])
    if not activity or not activity_has_artifacts(activity) or not activity_has_process(activity):
        activity.extend(collect_recent_experiment_activity(data.get("experiment_records") or [], limit=4))
    data["activity"] = activity
    return data


def match_item_action(path: str) -> tuple[str, str] | None:
    parts = [part for part in path.split("/") if part]
    if len(parts) == 4 and parts[0] == "api" and parts[1] == "items":
        return unquote(parts[2]), parts[3]
    return None


async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
    global LAST_LLM_STATUS
    if scope["type"] != "http":
        await send_error(send, HTTPStatus.NOT_FOUND, "unsupported scope")
        return
    method = scope.get("method", "GET").upper()
    path = scope.get("path", "/")
    try:
        if method == "GET" and path in {"/", "/index.html"}:
            await send_html(send, render_app_html())
            return
        if method == "GET" and path == "/api/state":
            await send_json(send, collect_state())
            return
        if method == "GET" and path == "/api/settings/llm":
            await send_json(send, llm_settings_response())
            return
        if method == "GET" and path == "/api/settings/sentaurus":
            await send_json(send, sentaurus_settings_response(settings_path=CONFIG.sentaurus_settings_path))
            return
        if method == "GET" and path == "/api/worker/status":
            await send_json(send, WORKER.status())
            return
        if method == "GET" and path == "/api/artifact":
            query = parse_qs((scope.get("query_string") or b"").decode("utf-8"))
            raw_path = (query.get("path") or [""])[0]
            if not raw_path:
                await send_error(send, HTTPStatus.BAD_REQUEST, "path is required")
                return
            artifact_path = resolve_artifact_path(raw_path)
            await send_response(send, HTTPStatus.OK, artifact_content_type(artifact_path), artifact_path.read_bytes())
            return
        if method != "POST":
            await send_error(send, HTTPStatus.NOT_FOUND, f"unknown route: {path}")
            return

        payload = await payload_from_request(receive)
        if path == "/api/missions":
            await send_json(send, enqueue_mission_from_payload(CONFIG, payload), HTTPStatus.CREATED)
            return
        if path == "/api/worker/run-once":
            result = run_queue_worker(
                CONFIG.queue_db_path,
                owner=str(payload.get("owner") or CONFIG.worker_owner),
                concurrency=int_from_payload(payload, "concurrency", 1, minimum=1),
                lease_seconds=float(payload.get("lease_seconds") or 7200.0),
                max_items=int_from_payload(payload, "max_items", 1, minimum=1),
            )
            await send_json(send, result.model_dump(mode="json"))
            return
        if path == "/api/worker/start":
            await send_json(
                send,
                WORKER.start(
                    concurrency=int_from_payload(payload, "concurrency", 1, minimum=1),
                    lease_seconds=float(payload.get("lease_seconds") or 7200.0),
                    poll_interval_seconds=float(payload.get("poll_interval_seconds") or 5.0),
                    max_loops=int(payload["max_loops"]) if payload.get("max_loops") not in {None, ""} else None,
                    max_idle_loops=int(payload["max_idle_loops"])
                    if payload.get("max_idle_loops") not in {None, ""}
                    else None,
                ),
            )
            return
        if path == "/api/worker/stop":
            await send_json(send, WORKER.stop())
            return
        if path == "/api/recover":
            await send_json(send, recover_stale_items(CONFIG.queue_db_path))
            return
        if path == "/api/llm/check":
            LAST_LLM_STATUS = check_llm_health().model_dump(mode="json")
            await send_json(send, LAST_LLM_STATUS)
            return
        if path == "/api/settings/llm":
            LAST_LLM_STATUS = None
            await send_json(send, save_llm_settings_from_payload(payload))
            return
        if path == "/api/settings/sentaurus":
            await send_json(send, save_sentaurus_settings_from_payload(payload, settings_path=CONFIG.sentaurus_settings_path))
            return
        item_action = match_item_action(path)
        if item_action:
            queue_id, action = item_action
            if action == "pause":
                await send_json(send, pause_item(CONFIG.queue_db_path, queue_id).model_dump(mode="json"))
                return
            if action == "resume":
                await send_json(send, resume_item(CONFIG.queue_db_path, queue_id).model_dump(mode="json"))
                return
            if action == "cancel":
                await send_json(send, cancel_item(CONFIG.queue_db_path, queue_id).model_dump(mode="json"))
                return
            if action == "approve":
                await send_json(send, approve_item_confirmation(CONFIG, queue_id))
                return
            if action == "reject":
                await send_json(send, reject_item_confirmation(CONFIG, queue_id))
                return
        await send_error(send, HTTPStatus.NOT_FOUND, f"unknown route: {path}")
    except Exception as exc:
        await send_error(send, HTTPStatus.BAD_REQUEST, str(exc))
