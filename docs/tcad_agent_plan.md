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

Create a minimal closed loop for a PN junction task:

- generate a DEVSIM Python script;
- run a voltage sweep;
- save IV data;
- plot the IV curve;
- save solver logs;
- summarize the result.

Current runnable command:

```bash
python3.11 -m tcad_agent.examples.pn_junction.run --stop 0.5 --step 0.1
```

Current agent-callable tool:

```bash
python3.11 -m tcad_agent.tools.pn_junction_iv --stop 0.5 --step 0.1
```

The tool adds explicit convergence-failure classification, smaller sweep-step retry, persistent run state, and checkpoint resume.

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

### Task 1: PN Junction IV

Objective: generate an IV curve over a requested voltage range and report turn-on behavior.

Success criteria:

- simulation finishes;
- voltage-current table is produced;
- IV plot is produced;
- final summary includes bias range, current trend, and any convergence retries.

### Task 2: MOS Capacitor Sweep

Objective: sweep gate voltage and extract a target capacitance or surface-potential trend.

Success criteria:

- deck is generated;
- sweep completes;
- curve and extracted metrics are saved.

### Task 3: MOSFET DC Characteristics

Objective: sweep gate and drain bias to obtain Id-Vg and Id-Vd curves.

Success criteria:

- curves are generated;
- threshold-related metrics are estimated;
- failed bias points are retried with smaller steps.

### Task 4: Parameter Optimization

Objective: adjust geometry, doping, or oxide thickness to meet a target device metric.

Success criteria:

- all trials are logged;
- best candidate is selected;
- optimization reasoning is summarized.

### Task 5: Convergence Repair

Objective: recover from intentionally difficult solver settings or bias steps.

Success criteria:

- failure is classified;
- at least one recovery strategy is attempted;
- final status is clear and reproducible.

## Long-Running Requirements

- Every action should write a checkpoint.
- Every simulator run should have a unique run directory.
- Agent state should include task, plan, current deck path, run history, metrics, failure history, and next action.
- The system should support resume from the latest checkpoint.
- The final answer should include generated files, plots, metrics, and unresolved limitations.
