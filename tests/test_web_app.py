from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tcad_agent.llm import LLMConfig, load_persisted_llm_settings
from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.run_queue import QueueDaemonResult, QueueStatus, claim_next_items, enqueue_run, get_item, list_items, pause_item
from tcad_agent.web_app import (
    SEMICONDUCTOR_TEST_CASES,
    WebAppConfig,
    WorkerController,
    activity_has_artifacts,
    activity_has_process,
    approve_item_confirmation,
    collect_execution_activity,
    collect_recent_experiment_activity,
    compact_conclusion,
    compact_result,
    enqueue_mission_from_payload,
    llm_settings_response,
    mission_request_from_payload,
    preview_artifact,
    reject_item_confirmation,
    render_app_html,
    save_llm_settings_from_payload,
)


class FakeConclusionClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.config = type("Config", (), {"model": "fake-conclusion"})()
        self.payload: dict[str, object] | None = None

    def chat(self, system: str, user: str, *, temperature: float = 0.1) -> str:
        self.payload = json.loads(user)
        return json.dumps(self.response, ensure_ascii=False)


class WebAppTest(unittest.TestCase):
    def test_mission_request_from_payload_defaults_to_llm_execution(self) -> None:
        request = mission_request_from_payload({"goal_text": "做 MOSFET Id-Vg"})

        self.assertEqual(request["goal_text"], "做 MOSFET Id-Vg")
        self.assertTrue(request["execute"])
        self.assertTrue(request["use_llm_decomposer"])
        self.assertTrue(request["allow_llm_fallback"])
        self.assertEqual(request["max_cycles"], 12)

    def test_enqueue_mission_from_payload_writes_queue_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = WebAppConfig(
                root=root,
                queue_db_path=root / "queue.sqlite",
                worker_stop_file=root / "worker.stop",
            )

            item = enqueue_mission_from_payload(
                config,
                {
                    "goal_text": "做 diode breakdown 并给结论",
                    "execute": False,
                    "priority": 7,
                    "max_cycles": 2,
                },
            )
            rows = list_items(config.queue_db_path)

        self.assertEqual(item["tool_name"], "mission_agent")
        self.assertEqual(item["priority"], 7)
        self.assertEqual(rows[0]["request"]["goal_text"], "做 diode breakdown 并给结论")
        self.assertFalse(rows[0]["request"]["execute"])

    def test_render_app_html_contains_workbench_api_hooks(self) -> None:
        html = render_app_html()

        self.assertIn("TCAD Mission", html)
        self.assertIn('id="activity"', html)
        self.assertIn('id="missionForm"', html)
        self.assertIn("/api/missions", html)
        self.assertIn("/api/worker/start", html)
        self.assertIn("/api/worker/stop", html)
        self.assertIn("/api/artifact", html)
        self.assertIn('id="missionActionBtn"', html)
        self.assertIn('id="clearActivityBtn"', html)
        self.assertIn('id="settingsBtn"', html)
        self.assertIn('id="settingsModal"', html)
        self.assertIn("/api/settings/llm", html)
        self.assertIn("function openSettings", html)
        self.assertIn("function saveSettings", html)
        self.assertIn("留空保存为空", html)
        self.assertNotIn('id="runOnceBtn"', html)
        self.assertNotIn('id="startWorkerBtn"', html)
        self.assertNotIn('id="stopWorkerBtn"', html)
        self.assertNotIn('id="llmBtn"', html)
        self.assertIn("tcadMission.clearBefore", html)
        self.assertIn("submitAndStartMission", html)
        self.assertIn("JSON 明细", html)
        self.assertIn("function scrollToLatest", html)
        self.assertIn("function preserveScrollPosition", html)
        self.assertIn("function handleTranscriptScroll", html)
        self.assertIn("function eventDisplayStatus", html)
        self.assertIn("function outputRiskStatus", html)
        self.assertIn("function conclusionBlock", html)
        self.assertIn("conclusion-card", html)
        self.assertIn("conclusion-image", html)
        self.assertIn("function revealNextActivityEvent", html)
        self.assertIn("pendingActivity", html)
        self.assertIn("syncActivity(filteredActivity", html)
        self.assertIn("工具执行失败", html)
        self.assertIn("quality-failed", html)
        self.assertIn('id="latestJumpBtn"', html)
        self.assertIn("scrollToLatest({force: true})", html)
        self.assertIn("function noticeBlock", html)
        self.assertIn("function decisionBlock", html)
        self.assertIn("Agent 判断", html)
        self.assertIn("decision-card", html)
        self.assertIn('class="action-stack"', html)
        self.assertIn('class="advanced-menu"', html)
        self.assertIn("<summary>选项</summary>", html)
        self.assertIn('<details class="example-menu">', html)
        self.assertIn("<summary>例子</summary>", html)
        self.assertIn("case-title", html)
        self.assertIn("case-desc", html)
        self.assertIn('id="caseRail"', html)
        self.assertIn("MOSCAP 曲线偏移", html)
        self.assertLess(html.index('id="goalText"'), html.index('id="caseRail"'))
        self.assertLess(html.index('id="caseRail"'), html.index('id="missionActionBtn"'))

    def test_llm_settings_blank_api_key_overwrites_existing_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "llm_settings.json"

            first = save_llm_settings_from_payload(
                {
                    "base_url": "llm.local:8000/v1",
                    "model": "unit-model",
                    "api_key": "unit-placeholder",
                },
                settings_path=settings_path,
            )
            second = save_llm_settings_from_payload(
                {
                    "base_url": "llm.local:7382/v1",
                    "model": "unit-model-2",
                    "api_key": "",
                },
                settings_path=settings_path,
            )
            persisted = load_persisted_llm_settings(settings_path)
            config = LLMConfig.from_env(settings_path=settings_path)
            response = llm_settings_response(settings_path=settings_path)

        self.assertTrue(first["api_key_set"])
        self.assertEqual(persisted["api_key"], "")
        self.assertEqual(second["base_url"], "http://llm.local:7382/v1")
        self.assertFalse(second["api_key_set"])
        self.assertEqual(config.api_key, "")
        self.assertEqual(config.model, "unit-model-2")
        self.assertFalse(response["api_key_set"])

    def test_llm_settings_can_clear_url_model_and_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "llm_settings.json"

            response = save_llm_settings_from_payload(
                {
                    "base_url": "",
                    "model": "",
                    "api_key": "",
                },
                settings_path=settings_path,
            )
            persisted = load_persisted_llm_settings(settings_path)
            config = LLMConfig.from_env(settings_path=settings_path)

        self.assertEqual(response["status"], "unconfigured")
        self.assertEqual(response["base_url"], "")
        self.assertEqual(response["model"], "")
        self.assertFalse(response["api_key_set"])
        self.assertEqual(persisted["base_url"], "")
        self.assertEqual(persisted["model"], "")
        self.assertEqual(persisted["api_key"], "")
        self.assertEqual(config.base_url, "")
        self.assertEqual(config.model, "")
        self.assertEqual(config.api_key, "")

    def test_worker_start_recovers_orphaned_web_worker_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = WebAppConfig(
                root=root,
                queue_db_path=root / "queue.sqlite",
                worker_stop_file=root / "worker.stop",
                worker_owner="tcad_web_worker",
            )
            enqueue_mission_from_payload(
                config,
                {
                    "queue_id": "q_orphan",
                    "goal_text": "跑 MOSFET Id-Vg",
                    "execute": False,
                    "max_attempts": 2,
                },
            )
            claim_next_items(config.queue_db_path, owner="tcad_web_worker", limit=1, lease_seconds=7200)
            controller = WorkerController(config)

            with patch(
                "tcad_agent.web_app.run_queue_daemon",
                return_value=QueueDaemonResult(
                    db_path=str(config.queue_db_path),
                    owner="tcad_web_worker",
                    stopped_by="idle",
                ),
            ):
                status = controller.start(poll_interval_seconds=0, max_idle_loops=1)
                if controller.thread:
                    controller.thread.join(timeout=2)
                final_status = controller.status()
            item = get_item(config.queue_db_path, "q_orphan")

        self.assertEqual(status["owner"], "tcad_web_worker")
        self.assertEqual(final_status["last_recovery"], {"recovered": 1, "failed": 0})
        self.assertEqual(item.status, QueueStatus.QUEUED)
        self.assertIsNone(item.lease_owner)

    def test_approve_confirmation_patches_request_and_resumes_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = WebAppConfig(
                root=root,
                queue_db_path=root / "queue.sqlite",
                worker_stop_file=root / "worker.stop",
            )
            cancel_file = root / "agents" / "agent_wait" / "cancel.requested"
            cancel_file.parent.mkdir(parents=True, exist_ok=True)
            cancel_file.write_text("cancel", encoding="utf-8")
            enqueue_run(
                config.queue_db_path,
                queue_id="q_wait",
                tool_name="autonomous_devsim_agent",
                request={"goal_text": "需要确认", "agent_id": "agent_wait", "cancel_file": str(cancel_file)},
            )
            pause_item(config.queue_db_path, "q_wait")

            approved = approve_item_confirmation(config, "q_wait")
            item = get_item(config.queue_db_path, "q_wait")
            cancel_removed = not cancel_file.exists()

        self.assertEqual(approved["status"], QueueStatus.QUEUED.value)
        self.assertTrue(item.request["resume"])
        self.assertTrue(item.request["allow_user_confirmation_actions"])
        self.assertTrue(cancel_removed)

    def test_reject_confirmation_cancels_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = WebAppConfig(
                root=root,
                queue_db_path=root / "queue.sqlite",
                worker_stop_file=root / "worker.stop",
            )
            enqueue_run(
                config.queue_db_path,
                queue_id="q_reject",
                tool_name="autonomous_devsim_agent",
                request={"goal_text": "需要确认", "agent_root": str(root / "agents"), "agent_id": "agent_reject"},
            )
            pause_item(config.queue_db_path, "q_reject")

            rejected = reject_item_confirmation(config, "q_reject")
            item = get_item(config.queue_db_path, "q_reject")
            cancel_written = (root / "agents" / "agent_reject" / "cancel.requested").exists()

        self.assertEqual(rejected["status"], QueueStatus.CANCELLED.value)
        self.assertEqual(item.status, QueueStatus.CANCELLED)
        self.assertTrue(cancel_written)

    def test_compact_result_keeps_artifacts_attempts_cases_and_preview(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "runs") as tmp:
            root = Path(tmp)
            csv_path = root / "curve.csv"
            png_path = root / "curve.png"
            csv_path.write_text("voltage,current\n0,0\n1,1e-6\n", encoding="utf-8")
            png_path.write_bytes(b"png")

            compact = compact_result(
                {
                    "status": "completed",
                    "attempts": [
                        {
                            "index": 1,
                            "status": "completed",
                            "step_v": 0.1,
                            "command": ["python3.11", "-m", "tcad_agent.tools.pn_junction_iv"],
                            "stdout_tail": "loading UMFPACK\ncompleted",
                            "stderr_tail": "",
                        }
                    ],
                    "cases": [{"index": 1, "status": "completed", "metric_value": 0.01}],
                    "checkpoint": {"completed_attempts": 1, "current_step_v": 0.1},
                    "final_summary": {
                        "artifacts": {"csv": str(csv_path), "plot": str(png_path)},
                        "points": 2,
                    },
                    "quality_report": {
                        "status": "passed",
                        "metrics": {"points": 2},
                    },
                }
            )

        self.assertEqual(compact["attempts"][0]["step_v"], 0.1)
        self.assertEqual(compact["attempts"][0]["command"][0], "python3.11")
        self.assertIn("UMFPACK", compact["attempts"][0]["stdout_tail"])
        self.assertEqual(compact["cases"][0]["metric_value"], 0.01)
        self.assertEqual(compact["checkpoint"]["completed_attempts"], 1)
        self.assertIn("plot", compact["final_summary"]["artifacts"])
        self.assertIn("csv", compact["final_summary"]["artifact_previews"])
        self.assertIn("voltage,current", compact["final_summary"]["artifact_previews"]["csv"]["preview"])

    def test_compact_result_summarizes_recent_records_without_step_quality_leakage(self) -> None:
        compact = compact_result(
            {
                "index": {"tool_name": "experiment_index", "status": "completed", "records_indexed": 143},
                "recent_records": [
                    {"kind": "mosfet_2d_id_sweep", "status": "completed", "quality_status": "suspicious"},
                    {"kind": "tool_convergence", "status": "failed", "quality_status": "failed"},
                ],
            }
        )

        encoded = json.dumps(compact, ensure_ascii=False)
        self.assertIn("recent_records_summary", compact)
        self.assertNotIn('"recent_records"', encoded)
        self.assertNotIn('"quality_status"', encoded)
        self.assertEqual(compact["recent_records_summary"]["quality_counts"]["failed"], 1)
        self.assertEqual(compact["index"]["records_indexed"], 143)

    def test_semiconductor_engineering_test_cases_are_real_mission_templates(self) -> None:
        self.assertGreaterEqual(len(SEMICONDUCTOR_TEST_CASES), 14)
        ids = {case["id"] for case in SEMICONDUCTOR_TEST_CASES}
        self.assertIn("mosfet_idvg_split", ids)
        self.assertIn("diode_bv_leakage", ids)
        self.assertIn("mosfet_output_kink_debug", ids)
        self.assertIn("mesh_vs_model_signoff", ids)
        self.assertIn("ldmos_bv_ron_tradeoff", ids)
        self.assertIn("igbt_turnoff_tail", ids)
        self.assertIn("gan_hemt_current_collapse", ids)
        self.assertIn("bjt_gummel_gain", ids)
        self.assertIn("finfet_dibl_cv", ids)
        self.assertIn("soi_finfet_variability", ids)
        self.assertNotIn("pn_doping_unit_regression", ids)
        self.assertNotIn("photodiode_iv", ids)
        for case in SEMICONDUCTOR_TEST_CASES:
            self.assertIn("业务任务", case["goal"])
            self.assertGreaterEqual(case["max_cycles"], 12)
            self.assertTrue(case["expected_outputs"])
            self.assertTrue(
                any(marker in case["goal"] for marker in ["帮我", "我想", "客户", "项目", "麻烦", "请", "你"]),
                case["goal"],
            )

    def test_preview_artifact_rejects_paths_outside_runs(self) -> None:
        self.assertIsNone(preview_artifact("/tmp/not_under_runs.csv"))

    def test_compact_conclusion_extracts_key_markdown_sections(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "runs") as tmp:
            conclusion_path = Path(tmp) / "conclusion.md"
            plot_path = Path(tmp) / "idvd.png"
            plot_path.write_bytes(b"png")
            (Path(tmp) / "state.json").write_text(
                json.dumps({"final_summary": {"artifacts": {"idvd_plot": str(plot_path)}}}),
                encoding="utf-8",
            )
            conclusion_path.write_text(
                "\n".join(
                    [
                        "# TCAD Conclusion: mosfet_unit",
                        "",
                        "## Engineering decision",
                        "",
                        "- Accept the stable Id-Vd baseline for discussion.",
                        "- Do not sign off high-field kink until mesh convergence is checked.",
                        "",
                        "## Key Metrics",
                        "",
                        "- `idvd_final_current_a`: `2.98`",
                        "- `relative_delta`: `0.014`",
                        "",
                        "## Recommended Next Steps",
                        "",
                        "- Run x_divisions convergence around the high-Vd segment.",
                    ]
                ),
                encoding="utf-8",
            )
            client = FakeConclusionClient(
                {
                    "title": "MOSFET 讨论版结论",
                    "blocks": [
                        {
                            "type": "text",
                            "label": "判断",
                            "content": "稳定 Id-Vd baseline 可以进入项目讨论，高场 kink 仍需网格复核。",
                        },
                        {
                            "type": "bullets",
                            "label": "关键数值",
                            "items": ["idvd_final_current_a = 2.98", "relative_delta = 0.014"],
                        },
                        {
                            "type": "image",
                            "label": "Id-Vd 曲线",
                            "path": str(plot_path),
                            "caption": "用于说明高 Vd 段形状。",
                        },
                    ],
                }
            )

            summary = compact_conclusion(conclusion_path, client=client)

        self.assertEqual(summary["title"], "MOSFET 讨论版结论")
        self.assertFalse(summary["fallback_used"])
        self.assertEqual(summary["model"], "fake-conclusion")
        self.assertEqual(summary["blocks"][0]["type"], "text")
        self.assertIn("baseline", summary["blocks"][0]["content"])
        self.assertEqual(summary["blocks"][1]["type"], "bullets")
        self.assertIn("idvd_final_current_a", summary["blocks"][1]["items"][0])
        self.assertEqual(summary["blocks"][2]["type"], "image")
        self.assertEqual(summary["blocks"][2]["path"], str(plot_path))
        self.assertIsNotNone(client.payload)
        self.assertIn("image_candidates", client.payload)

    def test_collect_execution_activity_includes_mission_intermediate_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "mission_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "tool_name": "tcad_mission_agent",
                        "status": "planned",
                        "mission_id": "mission_unit",
                        "goal_text": "做 MOS C-V",
                        "created_at": "2026-06-04T00:00:00Z",
                        "updated_at": "2026-06-04T00:00:01Z",
                        "checkpoint": {
                            "goal_decomposer": "deterministic",
                            "goal_decomposer_fallback_used": False,
                            "goal_decomposition": {
                                "status": "completed",
                                "steps": [
                                    {
                                        "index": 1,
                                        "kind": "run_supervisor",
                                        "title": "Run MOS C-V",
                                        "depends_on": [],
                                    }
                                ],
                            },
                            "controller_cycles": [
                                {
                                    "cycle": 1,
                                    "created_at": "2026-06-04T00:00:02Z",
                                    "observation": {
                                        "step_index": 1,
                                        "soft_failure_count": 0,
                                        "blocked_goal_steps": [],
                                        "pending_goal_kinds": ["generate_conclusion"],
                                    },
                                    "decision": {
                                        "action": "continue",
                                        "reason_zh": "当前步骤未发现阻塞风险，继续执行下一步。",
                                    },
                                }
                            ],
                        },
                        "steps": [
                            {
                                "index": 1,
                                "kind": "run_supervisor",
                                "status": "completed",
                                "reason": "execute primary TCAD action",
                                "result": {
                                    "status": "completed",
                                    "state_path": str(root / "mos_state.json"),
                                    "quality_report": {
                                        "status": "passed",
                                        "metrics": {"final_capacitance_f_per_cm2": 1.2e-7},
                                    },
                                },
                                "created_at": "2026-06-04T00:00:01Z",
                                "updated_at": "2026-06-04T00:00:02Z",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            activity = collect_execution_activity(
                [
                    {
                        "queue_id": "q_unit",
                        "tool_name": "mission_agent",
                        "status": "completed",
                        "request": {"goal_text": "做 MOS C-V"},
                        "attempts": 1,
                        "max_attempts": 1,
                        "created_at": "2026-06-04T00:00:00Z",
                        "updated_at": "2026-06-04T00:00:03Z",
                        "result_state_path": str(state_path),
                    }
                ]
            )

        titles = [event["title"] for event in activity]
        self.assertIn("任务计划", titles)
        self.assertIn("任务步骤 1：执行 TCAD 主任务", titles)
        self.assertIn("Agent 决策 1", titles)
        step_event = next(event for event in activity if event["title"] == "任务步骤 1：执行 TCAD 主任务")
        self.assertEqual(step_event["output"]["quality_report"]["status"], "passed")
        decision_event = next(event for event in activity if event["title"] == "Agent 决策 1")
        self.assertEqual(decision_event["output"]["action"], "continue")
        self.assertEqual(decision_event["output"]["agent_decision"]["action_label"], "继续执行")

    def test_collect_execution_activity_shows_goal_decomposition_in_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "mission_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "tool_name": "tcad_mission_agent",
                        "status": "running",
                        "mission_id": "mission_decompose_unit",
                        "goal_text": "做 MOSFET Id-Vg",
                        "created_at": "2026-06-05T00:00:00Z",
                        "updated_at": "2026-06-05T00:00:01Z",
                        "checkpoint": {
                            "goal_decomposer": "llm",
                            "goal_decomposition_status": "running",
                        },
                        "steps": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            activity = collect_execution_activity(
                [
                    {
                        "queue_id": "q_decompose",
                        "tool_name": "mission_agent",
                        "status": "running",
                        "request": {"goal_text": "做 MOSFET Id-Vg"},
                        "attempts": 1,
                        "max_attempts": 2,
                        "created_at": "2026-06-05T00:00:00Z",
                        "updated_at": "2026-06-05T00:00:01Z",
                        "result_state_path": str(state_path),
                    }
                ]
            )

        event = next(item for item in activity if item["title"] == "任务步骤 0：拆解任务")
        self.assertEqual(event["status"], "running")
        self.assertIn("等待 LLM", event["detail"])
        self.assertEqual(event["output"]["agent_decision"]["action_label"], "拆解任务")

    def test_collect_execution_activity_surfaces_replan_reason_for_suspicious_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "mission_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "tool_name": "tcad_mission_agent",
                        "status": "completed",
                        "mission_id": "mission_replan_unit",
                        "created_at": "2026-06-05T00:00:00Z",
                        "updated_at": "2026-06-05T00:00:05Z",
                        "checkpoint": {
                            "controller_cycles": [
                                {
                                    "cycle": 2,
                                    "created_at": "2026-06-05T00:00:03Z",
                                    "observation": {
                                        "step_index": 2,
                                        "soft_failure_count": 1,
                                        "goal_status_counts": {"soft_failed": 1},
                                        "pending_goal_kinds": ["generate_conclusion", "run_tool_convergence"],
                                        "primary_tcad_record": {
                                            "experiment_id": "mosfet_unit_002",
                                            "kind": "mosfet_2d_id_sweep",
                                            "status": "completed",
                                            "quality_status": "suspicious",
                                        },
                                    },
                                    "decision": {
                                        "action": "replan",
                                        "reason_zh": "存在软失败或阻塞步骤，下一步让 agent 重新诊断并调整计划。",
                                        "next_action": "run_supervisor",
                                    },
                                },
                                {
                                    "cycle": 4,
                                    "created_at": "2026-06-05T00:00:05Z",
                                    "observation": {
                                        "step_index": 4,
                                        "soft_failure_count": 1,
                                        "primary_tcad_record": {
                                            "experiment_id": "mosfet_unit_002",
                                            "kind": "mosfet_2d_id_sweep",
                                            "quality_status": "suspicious",
                                        },
                                    },
                                    "decision": {
                                        "action": "continue_with_risk",
                                        "reason_zh": "有非阻塞风险，但仍可继续生成带风险说明的工程结论。",
                                    },
                                },
                            ]
                        },
                        "steps": [
                            {
                                "index": 2,
                                "kind": "run_supervisor",
                                "status": "completed",
                                "reason": "execute goal-decomposition supervisor step",
                                "result": {"status": "completed", "quality_report": {"status": "suspicious"}},
                                "created_at": "2026-06-05T00:00:02Z",
                                "updated_at": "2026-06-05T00:00:03Z",
                            },
                            {
                                "index": 4,
                                "kind": "agent_replan",
                                "status": "completed",
                                "reason": "diagnose execution issues and adapt the mission plan",
                                "result": {"status": "completed"},
                                "created_at": "2026-06-05T00:00:04Z",
                                "updated_at": "2026-06-05T00:00:05Z",
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            activity = collect_execution_activity(
                [
                    {
                        "queue_id": "q_replan",
                        "tool_name": "mission_agent",
                        "status": "completed",
                        "request": {"goal_text": "检查 MOSFET 可疑质量"},
                        "attempts": 1,
                        "max_attempts": 1,
                        "created_at": "2026-06-05T00:00:00Z",
                        "updated_at": "2026-06-05T00:00:06Z",
                        "result_state_path": str(state_path),
                    }
                ]
            )

        replan_event = next(event for event in activity if event["title"] == "Agent 决策 2")
        risk_event = next(event for event in activity if event["title"] == "Agent 决策 4")
        self.assertEqual(replan_event["output"]["agent_decision"]["action_label"], "重新编排")
        self.assertIn("发现主仿真质量可疑，触发重新编排", replan_event["detail"])
        self.assertIn("主仿真质量：可疑", replan_event["output"]["agent_decision"]["observations"])
        self.assertIn("软失败次数：1", replan_event["output"]["agent_decision"]["observations"])
        self.assertEqual(risk_event["output"]["agent_decision"]["action_label"], "带风险继续")
        self.assertIn("带风险说明", risk_event["detail"])

    def test_collect_execution_activity_includes_conclusion_summary_on_goal_step(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "runs") as tmp:
            root = Path(tmp)
            conclusion_path = root / "conclusion.md"
            conclusion_path.write_text(
                "\n".join(
                    [
                        "# TCAD Conclusion: unit",
                        "",
                        "## Engineering decision",
                        "- Accept the result for project discussion.",
                        "",
                        "## Recommended Next Steps",
                        "- Run one finer mesh before signoff.",
                    ]
                ),
                encoding="utf-8",
            )
            conclusion_path.with_suffix(".web_summary.json").write_text(
                json.dumps(
                    {
                        "title": "项目讨论结论",
                        "blocks": [
                            {
                                "type": "text",
                                "label": "判断",
                                "content": "Accept the result for project discussion.",
                            },
                            {"type": "bullets", "label": "下一步", "items": ["Run one finer mesh before signoff."]},
                        ],
                        "path": str(conclusion_path),
                    }
                ),
                encoding="utf-8",
            )
            state_path = root / "mission_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "tool_name": "tcad_mission_agent",
                        "status": "completed",
                        "mission_id": "mission_unit",
                        "created_at": "2026-06-04T00:00:00Z",
                        "updated_at": "2026-06-04T00:00:02Z",
                        "checkpoint": {"conclusion_path": str(conclusion_path)},
                        "steps": [
                            {
                                "index": 1,
                                "kind": "generate_conclusion",
                                "status": "completed",
                                "reason": "execute goal-decomposition engineering conclusion step",
                                "result": {"status": "completed", "conclusion_path": str(conclusion_path)},
                                "created_at": "2026-06-04T00:00:01Z",
                                "updated_at": "2026-06-04T00:00:02Z",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            activity = collect_execution_activity(
                [
                    {
                        "queue_id": "q_conclusion",
                        "tool_name": "mission_agent",
                        "status": "completed",
                        "request": {"goal_text": "生成工程结论"},
                        "attempts": 1,
                        "max_attempts": 1,
                        "created_at": "2026-06-04T00:00:00Z",
                        "updated_at": "2026-06-04T00:00:03Z",
                        "result_state_path": str(state_path),
                    }
                ]
            )

        self.assertFalse([event for event in activity if event["title"] == "工程结论"])
        step_event = next(event for event in activity if event["title"] == "任务步骤 1：生成工程结论")
        summary = step_event["output"]["conclusion_summary"]
        self.assertEqual(summary["title"], "项目讨论结论")
        self.assertIn("结果可以进入项目讨论", summary["blocks"][0]["content"])
        self.assertIn("签核前再跑一档更细网格", summary["blocks"][1]["items"][0])

    def test_collect_execution_activity_falls_back_to_checkpoint_conclusion_event(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "runs") as tmp:
            root = Path(tmp)
            conclusion_path = root / "conclusion.md"
            conclusion_path.write_text("# TCAD 工程结论：unit\n\n## 建议下一步\n\n- 继续复核。", encoding="utf-8")
            state_path = root / "mission_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "tool_name": "tcad_mission_agent",
                        "status": "completed",
                        "mission_id": "mission_unit",
                        "updated_at": "2026-06-04T00:00:02Z",
                        "checkpoint": {"conclusion_path": str(conclusion_path)},
                        "steps": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            activity = collect_execution_activity(
                [
                    {
                        "queue_id": "q_conclusion",
                        "tool_name": "mission_agent",
                        "status": "completed",
                        "request": {"goal_text": "生成工程结论"},
                        "attempts": 1,
                        "max_attempts": 1,
                        "updated_at": "2026-06-04T00:00:03Z",
                        "result_state_path": str(state_path),
                    }
                ]
            )

        conclusion_events = [event for event in activity if event["title"] == "工程结论"]
        self.assertEqual(len(conclusion_events), 1)
        self.assertIn("conclusion_summary", conclusion_events[0]["output"])

    def test_collect_execution_activity_uses_soft_failed_goal_status(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "runs") as tmp:
            root = Path(tmp)
            state_path = root / "mission_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "tool_name": "tcad_mission_agent",
                        "status": "completed",
                        "mission_id": "mission_soft",
                        "created_at": "2026-06-04T00:00:00Z",
                        "updated_at": "2026-06-04T00:00:02Z",
                        "checkpoint": {
                            "goal_step_statuses": {
                                "2": {
                                    "status": "soft_failed",
                                    "mission_step_index": 5,
                                    "kind": "run_tool_convergence",
                                    "result": {
                                        "status": "failed",
                                        "quality_status": "failed",
                                    },
                                }
                            }
                        },
                        "steps": [
                            {
                                "index": 5,
                                "kind": "run_tool_convergence",
                                "status": "completed",
                                "reason": "execute goal-decomposition tool convergence study before accepting TCAD evidence",
                                "result": {
                                    "status": "failed",
                                    "quality_report": {
                                        "status": "failed",
                                        "issues": [
                                            {
                                                "code": "too_few_completed_convergence_cases",
                                                "severity": "error",
                                                "message": "At least two completed tool convergence cases are required.",
                                            }
                                        ],
                                        "recommended_next_action": "rerun failed convergence cases before trusting the result",
                                    },
                                },
                                "created_at": "2026-06-04T00:00:01Z",
                                "updated_at": "2026-06-04T00:00:02Z",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            activity = collect_execution_activity(
                [
                    {
                        "queue_id": "q_soft",
                        "tool_name": "mission_agent",
                        "status": "completed",
                        "request": {"goal_text": "验证 soft failed"},
                        "attempts": 1,
                        "max_attempts": 1,
                        "created_at": "2026-06-04T00:00:00Z",
                        "updated_at": "2026-06-04T00:00:03Z",
                        "result_state_path": str(state_path),
                    }
                ]
            )

        step_event = next(event for event in activity if event["title"] == "任务步骤 5：执行收敛验证")
        self.assertEqual(step_event["status"], "soft_failed")
        issue = step_event["output"]["quality_report"]["issues"][0]
        self.assertIn("至少需要两个", issue["message"])
        self.assertIn("先重跑失败", step_event["output"]["quality_report"]["recommended_next_action"])

    def test_collect_recent_experiment_activity_extracts_state_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "runs") as tmp:
            root = Path(tmp)
            csv_path = root / "iv.csv"
            state_path = root / "state.json"
            csv_path.write_text("v,i\n0,0\n", encoding="utf-8")
            state_path.write_text(
                json.dumps(
                    {
                        "tool_name": "pn_junction_iv_sweep",
                        "status": "completed",
                        "run_id": "hist_unit",
                        "quality_report": {"status": "passed", "metrics": {"points": 1}},
                        "final_summary": {"artifacts": {"csv": str(csv_path)}, "points": 1},
                    }
                ),
                encoding="utf-8",
            )

            activity = collect_recent_experiment_activity(
                [
                    {
                        "experiment_id": "hist_unit",
                        "kind": "pn_junction_iv_sweep",
                        "status": "completed",
                        "state_path": str(state_path),
                    }
                ]
            )

        self.assertEqual(activity[0]["title"], "pn_junction_iv_sweep result")
        self.assertIn("artifact_previews", activity[0]["output"]["final_summary"])
        self.assertTrue(activity_has_artifacts(activity))
        self.assertFalse(activity_has_process(activity))

    def test_activity_has_process_detects_tcad_stdout_and_commands(self) -> None:
        self.assertTrue(
            activity_has_process(
                [
                    {
                        "output": {
                            "attempts": [
                                {
                                    "command": ["python3.11", "-m", "tcad_agent.tools.mos_capacitor_cv"],
                                    "stdout_tail": "loading UMFPACK",
                                }
                            ]
                        }
                    }
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
