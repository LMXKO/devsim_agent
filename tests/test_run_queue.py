from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tcad_agent.run_queue import (
    QueueStatus,
    cancel_item,
    claim_next_items,
    default_runner_registry,
    enqueue_run,
    get_item,
    heartbeat_item,
    list_items,
    pause_item,
    recover_owner_running_items,
    recover_stale_items,
    resume_item,
    run_queue_daemon,
    run_queue_worker,
)


class RunQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "queue.sqlite"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def expire_running_item(self, queue_id: str) -> None:
        connection = sqlite3.connect(self.db)
        try:
            connection.execute(
                "UPDATE run_queue SET lease_expires_at = ? WHERE queue_id = ?",
                ("2000-01-01T00:00:00Z", queue_id),
            )
            connection.commit()
        finally:
            connection.close()

    def test_enqueue_and_list_items(self) -> None:
        item = enqueue_run(
            self.db,
            queue_id="q_a",
            tool_name="supervisor",
            request={"goal_text": "做 MOS C-V"},
            tags=["mos"],
            priority=3,
        )

        rows = list_items(self.db)

        self.assertEqual(item.status, QueueStatus.QUEUED)
        self.assertEqual(rows[0]["queue_id"], "q_a")
        self.assertEqual(rows[0]["request"]["goal_text"], "做 MOS C-V")
        self.assertEqual(rows[0]["tags"], ["mos"])

    def test_claim_respects_priority_and_lease(self) -> None:
        enqueue_run(self.db, queue_id="q_low", tool_name="fake", request={}, priority=1)
        enqueue_run(self.db, queue_id="q_high", tool_name="fake", request={}, priority=10)

        first = claim_next_items(self.db, owner="worker_a", limit=1, lease_seconds=60)
        second = claim_next_items(self.db, owner="worker_b", limit=1, lease_seconds=60)

        self.assertEqual(first[0].queue_id, "q_high")
        self.assertEqual(first[0].lease_owner, "worker_a")
        self.assertEqual(second[0].queue_id, "q_low")
        self.assertEqual(get_item(self.db, "q_high").status, QueueStatus.RUNNING)

    def test_pause_resume_cancel(self) -> None:
        enqueue_run(self.db, queue_id="q_pause", tool_name="fake", request={})

        paused = pause_item(self.db, "q_pause")
        resumed = resume_item(self.db, "q_pause")
        cancelled = cancel_item(self.db, "q_pause")

        self.assertEqual(paused.status, QueueStatus.PAUSED)
        self.assertEqual(resumed.status, QueueStatus.QUEUED)
        self.assertEqual(cancelled.status, QueueStatus.CANCELLED)

    def test_heartbeat_extends_owner_lease(self) -> None:
        enqueue_run(self.db, queue_id="q_beat", tool_name="fake", request={})
        claimed = claim_next_items(self.db, owner="worker_a", limit=1, lease_seconds=1)[0]

        updated = heartbeat_item(self.db, claimed.queue_id, owner="worker_a", lease_seconds=60)

        self.assertEqual(updated.status, QueueStatus.RUNNING)
        self.assertEqual(updated.lease_owner, "worker_a")
        self.assertIn("heartbeat_at", updated.checkpoint)

    def test_recover_stale_items_requeues_then_fails_after_attempt_limit(self) -> None:
        enqueue_run(self.db, queue_id="q_stale", tool_name="fake", request={}, max_attempts=2)
        claim_next_items(self.db, owner="worker_a", limit=1, lease_seconds=1)
        self.expire_running_item("q_stale")

        first_recovery = recover_stale_items(self.db)
        claim_next_items(self.db, owner="worker_b", limit=1, lease_seconds=1)
        self.expire_running_item("q_stale")
        second_recovery = recover_stale_items(self.db)
        item = get_item(self.db, "q_stale")

        self.assertEqual(first_recovery, {"recovered": 1, "failed": 0})
        self.assertEqual(second_recovery, {"recovered": 0, "failed": 1})
        self.assertEqual(item.status, QueueStatus.FAILED)
        self.assertIn("lease expired", item.failure_reason)

    def test_recover_owner_running_items_only_recovers_matching_owner(self) -> None:
        enqueue_run(self.db, queue_id="q_owned", tool_name="fake", request={}, max_attempts=2)
        enqueue_run(self.db, queue_id="q_other", tool_name="fake", request={}, max_attempts=2)
        claim_next_items(self.db, owner="web_owner", limit=1, lease_seconds=3600)
        claim_next_items(self.db, owner="other_owner", limit=1, lease_seconds=3600)

        recovery = recover_owner_running_items(self.db, owner="web_owner")
        owned = get_item(self.db, "q_owned")
        other = get_item(self.db, "q_other")

        self.assertEqual(recovery, {"recovered": 1, "failed": 0})
        self.assertEqual(owned.status, QueueStatus.QUEUED)
        self.assertIsNone(owned.lease_owner)
        self.assertIn("owner_recovered_at", owned.checkpoint)
        self.assertEqual(other.status, QueueStatus.RUNNING)
        self.assertEqual(other.lease_owner, "other_owner")

    def test_worker_executes_registered_tool_and_records_result(self) -> None:
        state_path = self.root / "state.json"
        enqueue_run(self.db, queue_id="q_run", tool_name="fake", request={"x": 2})

        result = run_queue_worker(
            self.db,
            owner="worker_a",
            registry={"fake": lambda request: {"status": "completed", "state_path": str(state_path), "x": request["x"]}},
        )
        item = get_item(self.db, "q_run")

        self.assertEqual(result.completed, 1)
        self.assertEqual(item.status, QueueStatus.COMPLETED)
        self.assertEqual(item.result["x"], 2)
        self.assertEqual(item.result_state_path, str(state_path))

    def test_default_mission_runner_passes_llm_decomposer_flags(self) -> None:
        fake_state = Mock()
        fake_state.model_dump.return_value = {"status": "planned", "mission_id": "mission_unit"}

        with patch("tcad_agent.mission_agent.run_mission_agent", return_value=fake_state) as runner:
            result = default_runner_registry()["mission_agent"](
                {
                    "goal_text": "做 MOSFET Id-Vg 并给工程结论",
                    "execute": False,
                    "max_cycles": 1,
                    "use_llm_decomposer": True,
                    "allow_llm_fallback": False,
                }
            )

        self.assertEqual(result["mission_id"], "mission_unit")
        kwargs = runner.call_args.kwargs
        self.assertTrue(kwargs["use_llm_decomposer"])
        self.assertFalse(kwargs["allow_llm_fallback"])

    def test_default_mission_runner_is_agent_first_without_explicit_flag(self) -> None:
        fake_state = Mock()
        fake_state.model_dump.return_value = {"status": "planned", "mission_id": "mission_default"}

        with patch("tcad_agent.mission_agent.run_mission_agent", return_value=fake_state) as runner:
            result = default_runner_registry()["mission_agent"](
                {
                    "goal_text": "做 MOSFET Id-Vg 并给工程结论",
                    "execute": False,
                    "max_cycles": 1,
                }
            )

        self.assertEqual(result["mission_id"], "mission_default")
        kwargs = runner.call_args.kwargs
        self.assertTrue(kwargs["use_llm_decomposer"])

    def test_default_autonomous_devsim_agent_runner_is_registered(self) -> None:
        fake_state = Mock()
        fake_state.model_dump.return_value = {"status": "planned", "agent_id": "agent_unit"}

        with patch("tcad_agent.autonomous_devsim_agent.run_autonomous_devsim_agent", return_value=fake_state) as runner:
            result = default_runner_registry()["autonomous_devsim_agent"](
                {
                    "goal_text": "自主跑 PN IV，失败时修复并给结论",
                    "execute": False,
                    "max_steps": 1,
                    "use_llm": False,
                }
            )

        self.assertEqual(result["agent_id"], "agent_unit")
        self.assertEqual(runner.call_args.args[0].goal_text, "自主跑 PN IV，失败时修复并给结论")
        self.assertIn("runner_registry", runner.call_args.kwargs)

    def test_default_agent_soak_runner_is_registered(self) -> None:
        fake_state = Mock()
        fake_state.model_dump.return_value = {"status": "completed", "soak_id": "soak_unit"}

        with patch("tcad_agent.agent_soak.run_agent_soak", return_value=fake_state) as runner:
            result = default_runner_registry()["agent_soak"](
                {
                    "goal_text": "长时间自主跑 PN IV，失败时继续修复并给结论",
                    "execute": False,
                    "max_steps": 2,
                    "autonomous_request": {"use_llm": False},
                }
            )

        self.assertEqual(result["soak_id"], "soak_unit")
        self.assertEqual(runner.call_args.args[0].goal_text, "长时间自主跑 PN IV，失败时继续修复并给结论")
        self.assertIn("runner_registry", runner.call_args.kwargs)

    def test_default_sentaurus_preflight_and_replay_runners_are_registered(self) -> None:
        registry = default_runner_registry()

        preflight = registry["sentaurus_preflight"]({"output_path": str(self.root / "preflight.json")})
        replay = registry["sentaurus_replay"]({"output_dir": str(self.root / "replay")})

        self.assertEqual(preflight["status"], "blocked")
        self.assertEqual(preflight["tool_name"], "sentaurus_preflight")
        self.assertFalse(preflight["ready_to_execute_real_sentaurus"])
        self.assertEqual(replay["status"], "failed")
        self.assertEqual(replay["tool_name"], "sentaurus_replay")
        self.assertTrue((self.root / "replay" / "sentaurus_replay_state.json").exists())

    def test_worker_pauses_autonomous_agent_waiting_for_user(self) -> None:
        enqueue_run(
            self.db,
            queue_id="q_waiting_agent",
            tool_name="autonomous_devsim_agent",
            request={"goal_text": "需要确认的 deck patch", "execute": True},
        )

        result = run_queue_worker(
            self.db,
            owner="worker_a",
            registry={"autonomous_devsim_agent": lambda request: {"status": "waiting_for_user", "agent_dir": str(self.root / "agent")}},
        )
        item = get_item(self.db, "q_waiting_agent")

        self.assertEqual(result.claimed, 1)
        self.assertEqual(item.status, QueueStatus.PAUSED)
        self.assertEqual(item.checkpoint["paused_reason"], "waiting_for_user")
        self.assertEqual(item.result["status"], "waiting_for_user")

    def test_autonomous_agent_queue_request_gets_control_paths(self) -> None:
        seen: dict[str, object] = {}
        enqueue_run(
            self.db,
            queue_id="q_control_agent",
            tool_name="autonomous_devsim_agent",
            request={"goal_text": "自主执行", "execute": False},
        )

        def fake_agent(request: dict[str, object]) -> dict[str, object]:
            seen.update(request)
            return {"status": "planned", "agent_dir": str(self.root / "runs" / "autonomous_devsim_agent" / "q_control_agent")}

        run_queue_worker(self.db, owner="worker_a", registry={"autonomous_devsim_agent": fake_agent})

        self.assertEqual(seen["queue_id"], "q_control_agent")
        self.assertEqual(seen["agent_id"], "q_control_agent")
        self.assertTrue(str(seen["cancel_file"]).endswith("q_control_agent/cancel.requested"))
        self.assertTrue(str(seen["heartbeat_path"]).endswith("q_control_agent/heartbeat.json"))

    def test_agent_soak_queue_request_gets_control_paths(self) -> None:
        seen: dict[str, object] = {}
        enqueue_run(
            self.db,
            queue_id="q_control_soak",
            tool_name="agent_soak",
            request={"goal_text": "长跑自主执行", "execute": False, "max_steps": 2},
        )

        def fake_soak(request: dict[str, object]) -> dict[str, object]:
            seen.update(request)
            return {"status": "completed", "soak_dir": str(self.root / "runs" / "agent_soak" / "q_control_soak")}

        run_queue_worker(self.db, owner="worker_a", registry={"agent_soak": fake_soak})

        self.assertEqual(seen["queue_id"], "q_control_soak")
        self.assertEqual(seen["soak_id"], "q_control_soak")
        self.assertTrue(str(seen["cancel_file"]).endswith("q_control_soak/cancel.requested"))
        self.assertTrue(str(seen["heartbeat_path"]).endswith("q_control_soak/agent_soak_heartbeat.json"))

    def test_cancel_autonomous_agent_writes_cancel_file(self) -> None:
        enqueue_run(
            self.db,
            queue_id="q_cancel_agent",
            tool_name="autonomous_devsim_agent",
            request={"goal_text": "自主执行", "agent_root": str(self.root / "agents"), "agent_id": "agent_cancel"},
        )

        cancelled = cancel_item(self.db, "q_cancel_agent")
        cancel_file = self.root / "agents" / "agent_cancel" / "cancel.requested"

        self.assertEqual(cancelled.status, QueueStatus.CANCELLED)
        self.assertTrue(cancel_file.exists())

    def test_cancel_agent_soak_writes_cancel_file(self) -> None:
        enqueue_run(
            self.db,
            queue_id="q_cancel_soak",
            tool_name="agent_soak",
            request={"goal_text": "长跑自主执行", "soak_root": str(self.root / "soaks"), "soak_id": "soak_cancel"},
        )

        cancelled = cancel_item(self.db, "q_cancel_soak")
        cancel_file = self.root / "soaks" / "soak_cancel" / "cancel.requested"

        self.assertEqual(cancelled.status, QueueStatus.CANCELLED)
        self.assertTrue(cancel_file.exists())

    def test_worker_marks_unknown_tool_failed(self) -> None:
        enqueue_run(self.db, queue_id="q_unknown", tool_name="missing", request={})

        result = run_queue_worker(self.db, owner="worker_a", registry={})
        item = get_item(self.db, "q_unknown")

        self.assertEqual(result.failed, 1)
        self.assertEqual(item.status, QueueStatus.FAILED)
        self.assertIn("unknown queued tool", item.failure_reason)

    def test_worker_fails_when_budget_is_exhausted_before_execution(self) -> None:
        calls: list[dict[str, object]] = []
        enqueue_run(self.db, queue_id="q_budget", tool_name="fake", request={}, budget_seconds=0)

        result = run_queue_worker(
            self.db,
            owner="worker_a",
            registry={"fake": lambda request: calls.append(request)},
        )
        item = get_item(self.db, "q_budget")

        self.assertEqual(result.failed, 1)
        self.assertEqual(calls, [])
        self.assertEqual(item.status, QueueStatus.FAILED)
        self.assertIn("budget_seconds exhausted", item.failure_reason)

    def test_daemon_processes_items_across_loops(self) -> None:
        calls: list[int] = []
        enqueue_run(self.db, queue_id="q_one", tool_name="fake", request={"x": 1}, priority=1)
        enqueue_run(self.db, queue_id="q_two", tool_name="fake", request={"x": 2}, priority=2)

        result = run_queue_daemon(
            self.db,
            owner="daemon_a",
            concurrency=1,
            poll_interval_seconds=0,
            max_idle_loops=1,
            registry={"fake": lambda request: {"status": "completed", "x": calls.append(request["x"]) or request["x"]}},
        )

        self.assertEqual(result.completed, 2)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.stopped_by, "idle")
        self.assertGreaterEqual(result.loops, 3)
        self.assertEqual(calls, [2, 1])
        self.assertEqual(get_item(self.db, "q_one").status, QueueStatus.COMPLETED)
        self.assertEqual(get_item(self.db, "q_two").status, QueueStatus.COMPLETED)

    def test_daemon_stops_before_work_when_stop_file_exists(self) -> None:
        stop_file = self.root / "stop"
        stop_file.write_text("stop", encoding="utf-8")
        enqueue_run(self.db, queue_id="q_stop", tool_name="fake", request={})

        result = run_queue_daemon(
            self.db,
            owner="daemon_stop",
            poll_interval_seconds=0,
            max_loops=3,
            stop_file=stop_file,
            registry={"fake": lambda request: {"status": "completed"}},
        )

        self.assertEqual(result.stopped_by, "stop_file")
        self.assertEqual(result.loops, 0)
        self.assertEqual(get_item(self.db, "q_stop").status, QueueStatus.QUEUED)


if __name__ == "__main__":
    unittest.main()
