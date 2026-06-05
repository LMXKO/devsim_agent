# PN Junction IV Tool

This tool is the first agent-callable TCAD execution interface. It wraps the DEVSIM PN junction runner with:

- structured request validation;
- isolated subprocess execution;
- failure classification;
- automatic convergence retry with smaller bias steps;
- persistent `state.json` checkpoints;
- deterministic result quality judging;
- final machine-readable summaries.

## Command

```bash
python3.11 -m tcad_agent.tools.pn_junction_iv --stop 0.5 --step 0.1
```

Useful options:

```bash
python3.11 -m tcad_agent.tools.pn_junction_iv \
  --run-id agent_smoke \
  --start 0.0 \
  --stop 0.5 \
  --step 0.1 \
  --min-step 0.025 \
  --max-attempts 3 \
  --quality-max-abs-current-a 1.0
```

Parameterized device options:

```bash
python3.11 -m tcad_agent.tools.pn_junction_iv \
  --run-id param_agent_smoke \
  --stop 0.2 \
  --step 0.1 \
  --length-um 0.2 \
  --junction-um 0.08 \
  --p-doping-cm3 1e17 \
  --n-doping-cm3 2e17 \
  --temperature-k 325 \
  --contact-spacing-um 0.002 \
  --junction-spacing-um 0.00002
```

These fields are written to the runner `summary.json` under `parameters`.

## Retry Example

This command intentionally starts with a very large bias step:

```bash
python3.11 -m tcad_agent.tools.pn_junction_iv \
  --run-id agent_extreme \
  --stop 5.0 \
  --step 5.0 \
  --min-step 1.25 \
  --max-attempts 3
```

Observed local behavior:

- attempt 1: `step=5.0 V`, failed, classified as `convergence`;
- attempt 2: `step=2.5 V`, failed, classified as `convergence`;
- attempt 3: `step=1.25 V`, completed.

## Resume

Resume reads the existing checkpoint:

```bash
python3.11 -m tcad_agent.tools.pn_junction_iv \
  --run-id agent_extreme \
  --resume \
  --stop 5.0 \
  --step 5.0 \
  --min-step 1.25 \
  --max-attempts 3
```

If the run is already complete, the tool returns the saved state without launching a new DEVSIM attempt.

## Output Layout

Tool-level run directory:

```text
runs/agent_tools/pn_junction_iv/<run_id>/
  state.json
  attempt_runs/
    pn_junction/
      attempt_001/
      attempt_002/
      attempt_003/
```

Each successful attempt writes:

- `iv_sweep.csv`;
- `iv_curve.png`;
- `device_tecplot.dat`;
- `devsim.log`;
- `summary.json`.

Failed attempts keep their `devsim.log` and the tool records stdout/stderr tails in `state.json`.

## Quality Report

Completed runner output is not automatically treated as acceptable. The tool adds `quality_report` to `state.json`:

```text
quality_report.status: passed | suspicious | failed
quality_report.issues: deterministic warnings or errors
quality_report.metrics: extracted IV and retry metrics
quality_report.recommended_next_action: next action for the agent
```

Current deterministic checks:

- IV CSV exists and can be parsed;
- values are finite, not `NaN` or `inf`;
- voltage sweep is monotonic increasing;
- point count is at least `--quality-min-points`;
- absolute current stays below `--quality-max-abs-current-a`;
- convergence retry count is at most `--quality-max-convergence-failures`;
- expected artifacts are present.

Default quality policy:

```text
quality_min_points: 3
quality_max_abs_current_a: 1.0
quality_max_convergence_failures: 0
```

Example outcomes verified locally:

- `--stop 0.5 --step 0.1`: `quality_report.status = passed`;
- `--stop 5.0 --step 5.0 --min-step 1.25 --max-attempts 3`: `quality_report.status = suspicious`.

## Failure Classes

- `validation`: bad request arguments;
- `convergence`: DEVSIM solver convergence failure;
- `timeout`: subprocess exceeded `timeout_seconds`;
- `output_missing`: runner returned success but did not produce `summary.json`;
- `runner_error`: Python/DEVSIM exception not classified as convergence;
- `unknown`: fallback for unrecognized failures.

Only `convergence` currently triggers automatic smaller-step retry.

## Agent Contract

An agent should treat the tool response as the source of truth:

- `status`: `completed` or `failed`;
- `quality_report.status`: result quality after deterministic judging;
- `attempts`: full attempt history;
- `checkpoint`: current resumable state;
- `next_action`: recommended next action;
- `final_summary`: produced only after success.

For long-running workflows, the agent should store the `run_id` and call with `--resume` after interruptions.

## Result Judge CLI

The quality judge can also be run independently:

```bash
python3.11 -m tcad_agent.tools.result_judge \
  --summary runs/agent_tools/pn_junction_iv/quality_smoke/attempt_runs/pn_junction/attempt_001/summary.json \
  --state runs/agent_tools/pn_junction_iv/quality_smoke/state.json
```

## LLM Diagnosis CLI

After the deterministic quality report marks a run as `suspicious` or `failed`, ask the configured OpenAI-compatible model for a strategy diagnosis:

```bash
python3.11 -m tcad_agent.tools.llm_diagnose \
  --state runs/agent_tools/pn_junction_iv/quality_extreme/state.json
```

The diagnosis is written to:

```text
runs/agent_tools/pn_junction_iv/<run_id>/llm_diagnosis.json
```

The LLM response is advisory. The deterministic `quality_report` remains the source of truth for whether artifacts are accepted.

If the model proposes `next_tool_command`, the diagnosis parser only accepts whitelisted `python3.11 -m tcad_agent.tools.*` commands. Arbitrary shell commands are rejected.

## Strategy Executor

Use the strategy executor to convert a suspicious or failed run into a constrained follow-up request:

```bash
python3.11 -m tcad_agent.tools.strategy_executor \
  --state runs/agent_tools/pn_junction_iv/quality_extreme/state.json
```

The dry-run writes:

```text
runs/agent_tools/pn_junction_iv/<run_id>/strategy_plan.json
```

Add `--execute` to run the planned follow-up:

```bash
python3.11 -m tcad_agent.tools.strategy_executor \
  --state runs/agent_tools/pn_junction_iv/quality_extreme/state.json \
  --execute
```

Verified local behavior:

- source run: `quality_extreme`, `quality_report.status = suspicious`;
- strategy: narrow voltage range from `5.0 V` to `0.5 V`;
- follow-up run: `quality_extreme_followup_001`;
- follow-up quality: `passed`.

The executor remains constrained:

- passed runs are skipped;
- arbitrary LLM shell commands are ignored;
- only `pn_junction_iv` follow-up requests are generated;
- deterministic quality rules can override advisory model text.

## Autonomous Loop

The higher-level loop chains this tool, optional LLM diagnosis, and strategy planning over multiple cycles:

```bash
python3.11 -m tcad_agent.tools.autonomous_loop \
  --loop-id auto_extreme \
  --stop 5.0 \
  --step 5.0 \
  --min-step 1.25 \
  --max-attempts 3 \
  --max-cycles 3 \
  --no-llm
```

It writes:

```text
runs/autonomous_loop/<loop_id>/loop_state.json
```

See [autonomous_loop.md](autonomous_loop.md) for the loop-level checkpoint contract.
