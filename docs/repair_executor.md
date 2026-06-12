# TCAD Repair Executor

`tcad_agent.repair_executor` closes the loop after `repair_strategy`.

It reads a suspicious or failed run `state.json`, builds a repair plan, selects the highest-priority executable action, applies the action patch to the original request, launches the corresponding TCAD tool again, then repeats until:

- the repaired run has `quality_report.status = passed` and its post-repair physical benchmark passes;
- a repair action requires user confirmation;
- the repair budget is exhausted;
- the runner fails.

## Command

Plan the next repair without running TCAD:

```bash
python3.11 -m tcad_agent.repair_executor \
  --state runs/agent_tools/pn_junction_iv/quality_extreme/state.json
```

Execute the repair loop:

```bash
python3.11 -m tcad_agent.repair_executor \
  --state runs/agent_tools/pn_junction_iv/quality_extreme/state.json \
  --execute \
  --max-rounds 3
```

Allow sensitive actions such as geometry or unit repair:

```bash
python3.11 -m tcad_agent.repair_executor \
  --state runs/agent_tools/pn_junction_iv/bad_geometry/state.json \
  --execute \
  --allow-user-confirmation-actions
```

## Supported Tool Targets

The executor can rerun:

- `pn_junction_iv_sweep`;
- `mos_capacitor_cv_sweep`;
- `diode_breakdown_leakage_sweep`;
- `mosfet_2d_id_sweep`.
- `extended_device_sweep`.

After each executed repair, the executor runs `physical_benchmark`. If benchmark status is `suspicious` or `failed`, it writes a benchmark-augmented state with the benchmark checks folded into `quality_report.issues`, then uses that state as the next repair input.

When a repair action includes a deck mutation, the executor compares baseline and mutation curves, writes `baseline_mutation_overlay.svg`, records `mutation_effect_analysis`, and stores the recommended next target. The autonomous DEVSIM agent can consume that analysis to generate the next finer request/deck patch instead of blindly applying another rule.

## Mission Agent Integration

`mission_agent` now uses this executor automatically:

1. detect suspicious/failed latest result;
2. generate `repair_plan.json`;
3. if no sensitive action blocks execution, run `repair_executor`;
4. generate the final conclusion from the repaired state.
