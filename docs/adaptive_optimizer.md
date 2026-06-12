# Adaptive Optimizer

`tcad_agent.adaptive_optimizer` turns fixed parameter sweeps into a checkpointed optimization loop.

It currently supports one numeric axis. Each round runs a normal `parameter_sweep`, records observations, chooses the best completed case, and proposes new points around that best value. This keeps every TCAD execution fully traceable through existing task, sweep, and tool state files.

## Dry Run

Plan the first round without executing DEVSIM:

```bash
python3.11 -m tcad_agent.adaptive_optimizer \
  --optimize-id diode_p_doping_opt_plan \
  --text "diode/SBD reverse leakage 从 0V 扫到 -5V 步长 0.5V，优化 p 区掺杂让漏电最小，max_attempts 3 max_cycles 2" \
  --axis parameters.p_doping_cm3 \
  --min-value 1e16 \
  --max-value 1e18 \
  --scale log \
  --initial-points 3 \
  --max-rounds 2 \
  --no-llm
```

## Execute

Run an adaptive optimization:

```bash
python3.11 -m tcad_agent.adaptive_optimizer \
  --optimize-id diode_p_doping_opt \
  --text "diode/SBD reverse leakage 从 0V 扫到 -5V 步长 0.5V，优化 p 区掺杂让漏电最小，max_attempts 3 max_cycles 2" \
  --axis parameters.p_doping_cm3 \
  --min-value 1e16 \
  --max-value 1e18 \
  --scale log \
  --initial-points 3 \
  --max-rounds 2 \
  --execute \
  --no-llm
```

By default the objective is:

```text
minimize abs(final_quality_report.metrics.final_total_current_a)
```

Change it with:

```bash
--objective-metric final_quality_report.metrics.final_total_current_a
--direction maximize
--raw-objective
```

## Checkpoint Layout

```text
runs/optimizations/<optimize_id>/
  base_task.json
  optimization_state.json
  sweeps/
    <optimize_id>_round_001/
      sweep_state.json
      summary.csv
      tasks/
      agent_tools/
```

If the same `--optimize-id` is used again without `--overwrite`, the optimizer resumes from `optimization_state.json` and only adds missing rounds up to the requested `--max-rounds`.

## Strategy

- `--scale log` uses geometric midpoints and is the better default for doping and lifetime values.
- `--scale linear` uses arithmetic midpoints and is suitable for geometry, temperature windows, and voltage-like quantities.
- Each refinement round proposes points between the current best value and its nearest evaluated neighbors.
