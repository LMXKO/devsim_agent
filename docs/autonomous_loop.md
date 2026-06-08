# Autonomous TCAD Loop

`tcad_agent.tools.autonomous_loop` is the first top-level long-running executor. It does not replace the lower-level TCAD tool. It orchestrates it:

1. run the PN-junction seed tool for a diode/SBD reverse-leakage task;
2. inspect the tool `quality_report`;
3. optionally ask the configured LLM to diagnose suspicious or failed runs;
4. build a constrained follow-up request with `strategy_executor`;
5. checkpoint the loop state;
6. repeat until a result passes quality checks or `--max-cycles` is reached.

## Deterministic Command

Start without LLM calls:

```bash
python3.11 -m tcad_agent.tools.autonomous_loop \
  --loop-id diode_seed_auto_extreme \
  --start 0.0 \
  --stop -5.0 \
  --step 5.0 \
  --min-step 1.25 \
  --max-attempts 3 \
  --max-cycles 3 \
  --no-llm
```

This is the default and is useful for reproducible automation. A suspicious large-step reverse sweep is expected to create a follow-up request with a narrower voltage range.

## LLM-Assisted Command

Use the OpenAI-compatible model endpoint for diagnosis before planning the follow-up:

```bash
python3.11 -m tcad_agent.tools.autonomous_loop \
  --loop-id diode_seed_auto_extreme_llm \
  --start 0.0 \
  --stop -5.0 \
  --step 5.0 \
  --min-step 1.25 \
  --max-attempts 3 \
  --max-cycles 3 \
  --use-llm
```

The model is advisory. The loop still uses deterministic safety rails:

- passed quality reports stop the loop;
- arbitrary shell commands suggested by the model are rejected;
- follow-up commands are converted into validated `PNJunctionIVRequest` objects;
- inherited request values such as `min_step` are normalized before execution.

## Resume

Resume a loop from its checkpoint:

```bash
python3.11 -m tcad_agent.tools.autonomous_loop \
  --loop-id diode_seed_auto_extreme \
  --resume
```

The loop reads:

```text
runs/autonomous_loop/<loop_id>/loop_state.json
```

If a cycle was interrupted after the lower-level PN tool created its own `state.json`, the loop requests the PN tool with `resume=True`.

## Output Layout

Loop-level state:

```text
runs/autonomous_loop/<loop_id>/
  loop_state.json
```

TCAD tool runs still live under:

```text
runs/agent_tools/pn_junction_iv/<run_id>/
  state.json
  strategy_plan.json
  llm_diagnosis.json
  attempt_runs/
```

## Loop State Contract

`loop_state.json` records:

- `status`: `running`, `completed`, or `failed`;
- `cycles`: one record per TCAD tool run;
- `checkpoint.pending_request`: the next request to run after a suspicious or failed cycle;
- `final_state_path`: accepted tool `state.json`, present only after success;
- `final_quality_report`: accepted quality report;
- `failure_reason`: reason the loop stopped without an accepted result.

This file is the handoff point for a future planner: the planner can start a loop, go away, inspect the checkpoint later, and decide whether to continue, branch, or report results.
