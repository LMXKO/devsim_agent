# PN Junction IV Seed Tool

This low-level tool wraps the DEVSIM PN junction runner. Public examples should use it as a seed path for the diode/SBD breakdown and reverse-leakage category, while higher-level missions should prefer `diode_breakdown_leakage_sweep` for BV-oriented work. It provides:

- structured request validation;
- isolated subprocess execution;
- failure classification;
- automatic convergence retry with smaller bias steps;
- persistent `state.json` checkpoints;
- deterministic result quality judging;
- final machine-readable summaries.

## Command

```bash
python3.11 -m tcad_agent.tools.pn_junction_iv --start 0 --stop -1 --step 0.25
```

Useful options:

```bash
python3.11 -m tcad_agent.tools.pn_junction_iv \
  --run-id diode_seed_smoke \
  --start 0.0 \
  --stop -1.0 \
  --step 0.25 \
  --min-step 0.0625 \
  --max-attempts 3 \
  --quality-max-abs-current-a 1.0
```

Parameterized device options:

```bash
python3.11 -m tcad_agent.tools.pn_junction_iv \
  --run-id diode_seed_param_smoke \
  --start 0.0 \
  --stop -1.0 \
  --step 0.25 \
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
  --run-id diode_seed_extreme \
  --start 0.0 \
  --stop -5.0 \
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
  --run-id diode_seed_extreme \
  --resume \
  --start 0.0 \
  --stop -5.0 \
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

Seed outcomes verified locally:

- `--start 0 --stop -1 --step 0.25`: reverse-leakage seed run should complete;
- `--start 0 --stop -5 --step 5.0 --min-step 1.25 --max-attempts 3`: intentionally difficult reverse sweep should exercise retry and quality warnings.

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

## Agent Follow-Up

For multi-step repair, mutation refinement, objective checks, and reporting, run this tool through the project-level autonomous agent:

```bash
python3.11 -m tcad_agent.tools.autonomous_devsim_agent \
  --goal "自主跑 PN IV，发现曲线或收敛问题就修复，最后给工程结论" \
  --initial-tool-name pn_junction_iv_sweep \
  --initial-request-json '{"start":0,"stop":0.5,"step":0.1,"run_id":"pn_auto_001"}' \
  --execute
```

The direct PN tool remains the source of truth for attempt history and curve artifacts; the autonomous agent owns cross-run decisions, repair lineage, objective/Pareto checks, reports, and dashboards.
