# Parameter Sweep

`tcad_agent.tools.parameter_sweep` runs multiple `TaskSpec` variants and summarizes the best case by an objective metric.

It is intentionally built above `task_runner`: each case still gets its own `task.json`, `task_run_state.json`, autonomous loop checkpoint, quality report, and DEVSIM artifacts.

## Dry Run

Generate cases without executing DEVSIM:

```bash
python3.11 -m tcad_agent.tools.parameter_sweep \
  --sweep-id diode_leakage_plan \
  --text "diode/SBD reverse leakage 从 0V 扫到 -5V 步长 0.5V，目标是 -5V 漏电最小，max_attempts 3 max_cycles 2" \
  --axis parameters.p_doping_cm3=1e16,1e17,1e18 \
  --no-llm
```

## Execute

Run the sweep:

```bash
python3.11 -m tcad_agent.tools.parameter_sweep \
  --sweep-id diode_leakage_sweep \
  --text "diode/SBD reverse leakage 从 0V 扫到 -5V 步长 0.5V，目标是 -5V 漏电最小，max_attempts 3 max_cycles 2" \
  --axis parameters.p_doping_cm3=1e17,2e17,5e17 \
  --execute \
  --no-llm
```

By default the objective is:

```text
minimize abs(final_quality_report.metrics.final_total_current_a)
```

Change it with:

```bash
--objective-metric final_quality_report.metrics.max_abs_current_a
--direction minimize
```

Use `--raw-objective` if signed values should not be converted to absolute values.

## Output Layout

```text
runs/sweeps/<sweep_id>/
  base_task.json
  sweep_state.json
  summary.csv
  tasks/
    <sweep_id>_case_001/
      task.json
      task_run_state.json
  autonomous_loop/
  agent_tools/
```

`sweep_state.json` records all cases and `best_case`. `summary.csv` is a compact table for later analysis.

## Axes

Axis syntax:

```text
--axis path=value1,value2,value3
```

Supported path prefixes:

- `sweep.*`
- `parameters.*`
- `mesh.*`
- `quality.*`
- `execution.*`

Multiple `--axis` flags create a Cartesian product.

## Natural-Language Sweep Planning

Let the deterministic parser infer one common axis:

```bash
python3.11 -m tcad_agent.tools.parameter_sweep \
  --sweep-planner deterministic \
  --sweep-id diode_p_doping_auto \
  --text "扫描 P 区掺杂从 1e16 到 1e18，做 diode/SBD reverse leakage 到 -5V，目标是漏电最小" \
  --execute \
  --no-llm
```

Use the LLM sweep planner for freer language:

```bash
python3.11 -m tcad_agent.tools.parameter_sweep \
  --sweep-planner llm \
  --sweep-id diode_lifetime_llm \
  --text "请围绕 lifetime 和掺杂做 3 个点的 diode/SBD leakage sweep，反偏到 -5V，目标是漏电最小且曲线可信" \
  --execute \
  --no-llm
```

With `--sweep-planner llm`, the planner writes:

```text
runs/sweep_plans/<sweep_id>/sweep_plan_result.json
```
