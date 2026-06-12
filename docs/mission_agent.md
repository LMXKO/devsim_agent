# TCAD Mission Agent

`tcad_agent.tools.mission_agent` is the long-horizon outer loop for the project goal:

> AI long-term autonomous driving of TCAD to complete user-specified tasks.

It sits above the supervisor and run queue. The supervisor chooses and executes one TCAD action; the mission agent manages a longer mission across multiple durable steps.

## What It Does

Each mission writes:

```text
runs/missions/<mission_id>/mission_state.json
```

The state records:

- user goal;
- stored goal decomposition;
- per-goal-step status in `checkpoint.goal_step_statuses`;
- durable mission steps;
- refreshed experiment-memory snapshots;
- supervisor run checkpoints;
- tool-convergence checkpoint when the decomposed goal requests convergence;
- repair plan path when the latest result is failed or suspicious;
- conclusion path when a result is accepted;
- whether the mission is waiting for user confirmation.

## Decision Loop

The mission runs a conservative step-DAG executor around the stored goal decomposition:

1. rebuild experiment index as infrastructure before executing goal steps;
2. find the next goal-decomposition step whose dependencies are completed or skipped;
3. map supported step kinds to mission actions:
   - `query_history` reads indexed experiment history;
   - `run_supervisor` delegates primary TCAD action selection/execution;
   - `run_tool_convergence` executes convergence over a tool request and refreshes the index;
   - `agent_replan` diagnoses failures with the configured LLM and applies a small plan patch when execution issues appear;
   - `run_repair_executor` skips accepted results or expands into repair-plan generation plus repair execution;
   - `generate_conclusion` writes a conclusion from the repaired state or latest accepted evidence;
   - `ask_user` stops in `waiting_for_user`;
4. record each goal step as `completed`, `skipped`, `soft_failed`, `failed`, or `waiting_for_user`;
5. stop when every goal-decomposition step is terminal.

This is intentionally conservative. Deterministic and optional LLM-backed goal decomposition both feed the same validated step-DAG executor.

Goal steps with `stop_on_failure=false` use soft-failure semantics: the validation or repair problem is recorded in `checkpoint.soft_failures`, dependencies may continue, and the mission can still generate a conclusion from the current primary TCAD state. This makes the loop behave more like Codex/Claude-style agents: report the failed check, continue with a viable path, and only stop for user input when the blocked step is actually required.

When LLM decomposition is enabled, the mission also enables LLM-backed replanning. A new execution issue creates an `agent_replan` mission step. The replanner receives the current goal, DAG, statuses, primary TCAD evidence, and latest failure context; it returns an issue family, Chinese strategy, recommended actions, plus optional `mark_soft_failed`, `skip_goal_steps`, and `append_steps` patch. If the LLM is unavailable or returns invalid JSON, a deterministic fallback classifies schema/field-alias, solver-convergence, physical-quality, repair-exhaustion, or generic execution issues and continues from the safest current evidence.

## LLM Decomposition

Use `--use-llm` to ask the configured OpenAI-compatible model to generate the goal-decomposition DAG before execution:

```bash
python3.11 -m tcad_agent.tools.mission_agent \
  --mission-id mission_llm \
  --goal "优化 MOSFET，让漏电低且 Ion/Ioff 达标，失败时自动修复，最后给结论" \
  --use-llm \
  --execute
```

By default, invalid LLM output or a failed model call falls back to the deterministic decomposer. Use `--no-llm-fallback` when a mission should fail instead of running a fallback plan.

The LLM endpoint is documented in [llm_config.md](llm_config.md). Public builds are unconfigured by default; set an OpenAI-compatible `/v1` endpoint through environment variables or the web settings dialog.

## Plan Only

```bash
python3.11 -m tcad_agent.tools.mission_agent \
  --mission-id mission_plan \
  --goal "完成一个 MOS C-V 并生成结论"
```

## Execute

```bash
python3.11 -m tcad_agent.tools.mission_agent \
  --mission-id mission_mos_cv \
  --goal "完成一个 MOS C-V 从 -0.5V 到 0.5V，并给工程结论" \
  --execute \
  --max-cycles 8 \
  --supervisor-max-cycles 3
```

## Resume

```bash
python3.11 -m tcad_agent.tools.mission_agent \
  --mission-id mission_mos_cv \
  --goal "完成一个 MOS C-V 从 -0.5V 到 0.5V，并给工程结论" \
  --resume \
  --execute
```

## Relationship To The Run Queue

Use the run queue when multiple missions should be scheduled or recovered by workers:

```bash
python3.11 -m tcad_agent.tools.run_queue enqueue \
  --tool supervisor \
  --goal "做 2D MOSFET Id-Vg gate_start 0V gate_stop 1V"
```

The mission agent is the single-mission brain; the run queue is the multi-run scheduler.

For browser-based operation, start the workbench:

```bash
python3.11 -m uvicorn tcad_agent.asgi_web:app --host 127.0.0.1 --port 8766 --no-access-log
```

The page queues long-running `agent_soak` items by default and still allows API callers to force `mission_agent` when needed. Worker controls are exposed without requiring command-line interaction for normal use.
