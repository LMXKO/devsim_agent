# TCAD Agent Plan

## Mission

Build an AI agent that can autonomously drive open-source TCAD tools over long horizons to complete semiconductor device simulation tasks.

The agent should not simply call a simulator once. It should plan, execute, inspect, adjust, and continue until the task is complete or a well-defined stopping condition is reached.

## Recommended First Toolchain

- Simulator: DEVSIM
- Language interface: Python
- Data processing: NumPy, pandas
- Plotting: matplotlib or Plotly
- Optimization: Optuna
- Persistence: SQLite or DuckDB
- Orchestration: a small state machine first, LangGraph later if needed

## Core Loop

```text
1. Parse user task
2. Create a task plan
3. Generate or edit TCAD Python deck
4. Run simulator in an isolated run directory
5. Capture logs, output files, curves, and metadata
6. Extract metrics
7. Judge progress against the task objective
8. Diagnose failures or poor results
9. Modify deck, mesh, model, solver settings, or sweep parameters
10. Save checkpoint and repeat
```

## First Milestone

Create a minimal closed loop around the seven-category showcase, starting with executable MOSCAP, MOSFET, and diode/SBD paths:

- generate or select a DEVSIM Python deck;
- run C-V, Id-Vg/Id-Vd, or reverse-leakage/BV sweeps;
- save curve data;
- plot the generated curve;
- save solver logs;
- summarize the result.

Current MOSCAP command:

```bash
python3.11 -m tcad_agent.tools.mos_capacitor_cv --start -0.5 --stop 0.5 --step 0.25
```

Current MOSFET command:

```bash
python3.11 -m tcad_agent.tools.mosfet_2d_id --sweep-type both --gate-start 0 --gate-stop 1.2 --gate-step 0.1
```

Current diode/SBD breakdown command:

```bash
python3.11 -m tcad_agent.tools.diode_breakdown --start 0 --stop -5 --step 0.5 --breakdown-current-a 1e-6
```

These tools add explicit convergence-failure classification, smaller sweep-step retry, persistent run state, and checkpoint resume.

It also adds deterministic result quality judging:

```text
passed | suspicious | failed
```

Current LLM-assisted diagnosis command:

```bash
python3.11 -m tcad_agent.tools.llm_diagnose \
  --state runs/agent_tools/pn_junction_iv/quality_extreme/state.json
```

Current constrained strategy command:

```bash
python3.11 -m tcad_agent.tools.strategy_executor \
  --state runs/agent_tools/pn_junction_iv/quality_extreme/state.json \
  --execute
```

Verified local loop:

```text
quality_extreme: suspicious
  -> LLM diagnosis: high risk
  -> strategy_executor: narrow stop voltage to 0.5 V
  -> quality_extreme_followup_001: passed
```

Next improvement: add a higher-level task runner that chains tool execution, quality judging, LLM diagnosis, and strategy execution until a stopping condition is reached.

Current verification command:

```bash
python3.11 -m unittest tests.test_result_judge tests.test_llm_diagnose tests.test_strategy_executor
```

## Suggested Repository Layout

```text
tcad_agent/
  agents/
    planner.py
    deck_writer.py
    log_analyzer.py
    result_judge.py
  tools/
    devsim_runner.py
    metrics.py
    plotter.py
    sweep.py
  storage/
    run_store.py
  examples/
    pn_junction/
    mos_cap/
    mosfet_2d/
  benchmarks/
    tasks.yaml
  runs/
```

## Benchmark Tasks

### Task 1: MOS Capacitor / Capacitance

Objective: sweep gate voltage and extract Cox, Cmin, flat-band shift, and fixed-charge plausibility.

Success criteria:

- C-V sweep completes;
- curve and extracted metrics are saved;
- Cox is compared with an analytic oxide-capacitance estimate;
- final summary labels calibration or fixed-charge gaps.

### Task 2: MOSFET Id-Vg / Id-Vd / DIBL

Objective: sweep gate and drain bias to obtain Id-Vg, Id-Vd, Vth, SS, Ion/Ioff, gm, and DIBL evidence.

Success criteria:

- curves are generated;
- threshold-related metrics are estimated;
- low/high drain Id-Vg cases are kept distinct for DIBL;
- failed bias points are retried with smaller steps or restart from a saved state.

### Task 3: Diode / SBD Breakdown

Objective: run reverse leakage/BV and Schottky/SBD barrier-calibration tasks with explicit breakdown and model-coupling evidence.

Success criteria:

- reverse IV or calibration curve is produced;
- leakage and BV/barrier metrics are extracted;
- reverse-bias continuation failures are classified;
- high-field or thermionic-contact coupling limits are explicit.

### Task 4: LDMOS / IGBT Power Devices

Objective: run a physics-coupled LDMOS/power-MOSFET BV/Ron baseline, then keep IGBT turn-off as a planned transient-promotion workflow.

Success criteria:

- power MOSFET/LDMOS states carry peak-field, Ron-component, and impact-ionization evidence;
- BV/Ron or tail-current metrics are named;
- high-voltage continuation and transient initialization are specified;
- final conclusion does not claim signoff without a real runner.

### Task 5: GaN / AlGaN HEMT

Objective: route output, BV, 2DEG, and current-collapse goals into a planned heterojunction workflow.

Success criteria:

- polarization, trap, self-heating, and field-plate model gaps are recorded;
- stress/recovery and dynamic-Ron metrics are defined;
- planned-only status is visible to the user.

### Task 6: BJT Gummel / Output

Objective: run BJT Gummel/output with a `physics_1d` transport baseline, then correlate it against public DEVSIM-backed BJT runner coverage.

Success criteria:

- beta, Early voltage, and leakage metrics are named;
- base-emitter and collector-bias continuation steps are defined;
- public-source promotion steps are recorded.

### Task 7: FinFET / SOI Variability

Objective: define 3D/advanced-geometry, density-gradient, capacitance, DIBL, and variability campaign evidence.

Success criteria:

- quantum-correction and 3D geometry gaps are visible;
- nominal-first and mesh-reuse strategy is defined;
- distribution-level signoff replaces single-point claims.

## Long-Running Requirements

- Every action should write a checkpoint.
- Every simulator run should have a unique run directory.
- Agent state should include task, plan, current deck path, run history, metrics, failure history, and next action.
- The system should support resume from the latest checkpoint.
- The final answer should include generated files, plots, metrics, and unresolved limitations.
