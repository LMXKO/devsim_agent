# MOS Capacitor C-V

`tcad_agent.examples.mos_capacitor.run` runs a 1D DEVSIM MOS capacitor quasistatic C-V sweep.

The device contains:

- metal gate contact;
- oxide region;
- oxide/silicon interface;
- p-type silicon substrate;
- substrate contact.

The runner solves the electrostatic potential and extracts gate charge. Capacitance is computed by finite difference:

```text
Cgate = dQgate / dVgate
```

## Smoke Run

```bash
python3.11 -m tcad_agent.examples.mos_capacitor.run \
  --run-id moscap_smoke \
  --run-root runs/moscap_smoke \
  --start -0.5 \
  --stop 0.5 \
  --step 0.25 \
  --oxide-thickness-nm 5 \
  --substrate-doping-cm3 1e17
```

## Outputs

```text
runs/moscap_smoke/mos_capacitor/moscap_smoke/
  cv_sweep.csv
  cv_curve.png
  device_tecplot.dat
  devsim.log
  summary.json
```

`summary.json` contains:

- voltage range;
- min/max gate charge;
- min/max/final capacitance;
- artifact paths.

This is the second TCAD task family after PN junction IV. The next step is to wrap it as an agent-callable tool and add TaskSpec planner routing.

## Agent Tool

Run the checkpointed agent-callable wrapper:

```bash
python3.11 -m tcad_agent.tools.mos_capacitor_cv \
  --run-id moscap_tool_smoke \
  --run-root runs/agent_tools_moscap \
  --start -0.5 \
  --stop 0.5 \
  --step 0.25 \
  --min-step 0.125 \
  --max-attempts 2 \
  --oxide-thickness-nm 5 \
  --substrate-doping-cm3 1e17
```

The tool writes:

```text
runs/agent_tools_moscap/mos_capacitor_cv/<run_id>/
  state.json
  attempt_runs/
```

`state.json` records:

- request parameters;
- attempts and failure class;
- checkpoint;
- final summary;
- quality report.

On convergence failure, the tool retries with a smaller gate-bias step down to `--min-step`.

## Quality Checks

The tool marks results as suspicious or failed when:

- required artifacts are missing;
- capacitance is non-finite or non-positive;
- voltage span looks like a unit mistake;
- oxide thickness, substrate doping, temperature, or geometry are outside sanity ranges;
- extracted capacitance exceeds the analytic oxide-capacitance estimate by a wide margin.
