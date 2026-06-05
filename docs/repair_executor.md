# TCAD Repair Executor

`tcad_agent.tools.repair_executor` closes the loop after `repair_strategy`.

It reads a suspicious or failed run `state.json`, builds a repair plan, selects the highest-priority executable action, applies the action patch to the original request, launches the corresponding TCAD tool again, then repeats until:

- the repaired run has `quality_report.status = passed`;
- a repair action requires user confirmation;
- the repair budget is exhausted;
- the runner fails.

## Command

Plan the next repair without running TCAD:

```bash
python3.11 -m tcad_agent.tools.repair_executor \
  --state runs/agent_tools/pn_junction_iv/quality_extreme/state.json
```

Execute the repair loop:

```bash
python3.11 -m tcad_agent.tools.repair_executor \
  --state runs/agent_tools/pn_junction_iv/quality_extreme/state.json \
  --execute \
  --max-rounds 3
```

Allow sensitive actions such as geometry or unit repair:

```bash
python3.11 -m tcad_agent.tools.repair_executor \
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

## Mission Agent Integration

`mission_agent` now uses this executor automatically:

1. detect suspicious/failed latest result;
2. generate `repair_plan.json`;
3. if no sensitive action blocks execution, run `repair_executor`;
4. generate the final conclusion from the repaired state.
