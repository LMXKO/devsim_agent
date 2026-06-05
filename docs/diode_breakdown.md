# Diode Breakdown And Leakage Tool

`tcad_agent.tools.diode_breakdown` is an agent-callable reverse-bias diode leakage / breakdown workflow.

It reuses the PN junction DEVSIM runner and PN IV tool as the underlying execution path, then adds reverse-bias-specific metrics, quality checks, checkpoint state, and conclusion support.

## Run A Reverse Leakage Sweep

```bash
python3.11 -m tcad_agent.tools.diode_breakdown \
  --run-id diode_reverse \
  --stop -5.0 \
  --step 0.5 \
  --breakdown-current-a 1e-6 \
  --leakage-voltage-v -1.0
```

The tool writes:

```text
runs/agent_tools/diode_breakdown/<run_id>/
  state.json
  conclusion.md
  inner_agent_tools/
    pn_junction_iv/<run_id>_pn_reverse/
      state.json
      attempt_runs/
```

## Extracted Metrics

The `quality_report.metrics` section includes:

- `leakage_abs_current_at_target_a`
- `leakage_voltage_used_v`
- `max_reverse_abs_current_a`
- `breakdown_current_threshold_a`
- `breakdown_voltage_at_threshold_v`
- `breakdown_detected`
- `reverse_abs_current_gain`
- `reverse_current_shape_violations`

If `breakdown_detected` is false, the simulated reverse-bias range did not cross the configured current threshold. The leakage result can still be valid; extend `--stop` to a more negative voltage if BV extraction is required.

## Quality Checks

The tool marks results as failed when no reverse-bias points are present or required artifacts are missing.

It marks results as suspicious when:

- leakage exceeds `--quality-max-leakage-abs-current-a`;
- reverse current exceeds `--quality-max-abs-current-a`;
- reverse current shape is non-monotonic as `|Vreverse|` increases;
- convergence failures were needed before success;
- `--require-breakdown` is set but the threshold was not reached.

## Reports

Generate a conclusion-oriented report:

```bash
python3.11 -m tcad_agent.tools.experiment_conclusion \
  --state runs/agent_tools/diode_breakdown/diode_reverse/state.json
```

The experiment index records this tool under kind `diode_breakdown_leakage_sweep`.
