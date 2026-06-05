# TCAD Supervisor

`tcad_agent.tools.supervisor` is the first long-running top-level controller.

It is intentionally conservative: it uses deterministic routing today, persists every decision, and can be resumed later. The goal is to provide the durable control loop that future LLM planning can plug into.

For multi-run scheduling, lease-based worker control, pause/resume/cancel, and stale recovery, put supervisor goals into the durable run queue described in [run_queue.md](run_queue.md).

## What It Does

Each run writes:

```text
runs/supervisor/<supervisor_id>/supervisor_state.json
```

The state records:

- user goal text;
- refreshed experiment index summary;
- recent experiment records;
- planned and completed actions;
- tool requests and results;
- checkpoint;
- next action.

Currently supported routed actions:

- rebuild/query experiment index;
- run PN junction IV through the task runner;
- run MOS capacitor C-V through `mos_capacitor_cv`;
- run diode reverse leakage / breakdown through `diode_breakdown`;
- run 2D MOSFET Id-Vg / Id-Vd through `mosfet_2d_id`;
- run Schottky IV calibration through `schottky_iv_calibration` when the goal asks for calibration, fitting, or a trusted/measured curve;
- run PN mesh convergence checks through `mesh_convergence`;
- generate Markdown reports;
- generate static HTML dashboards;
- generate engineering conclusions;
- generate TCAD repair plans;
- ask the user when the goal is ambiguous.

## Plan Only

```bash
python3.11 -m tcad_agent.tools.supervisor \
  --supervisor-id sup_plan \
  --goal "做一个 MOS C-V 从 -0.5V 到 0.5V"
```

This writes a planned action without running TCAD.

## Execute

```bash
python3.11 -m tcad_agent.tools.supervisor \
  --supervisor-id sup_mos_cv \
  --goal "做 MOS C-V 从 -0.5V 到 0.5V 步长 0.25V 氧化层 5nm 衬底掺杂 1e17" \
  --execute \
  --max-cycles 3
```

The supervisor first refreshes `runs/experiment_index.sqlite`, then chooses and executes the next action. After running a TCAD tool, it refreshes the experiment index again so the new result is immediately searchable.

Example routed goals:

```bash
python3.11 -m tcad_agent.tools.supervisor \
  --supervisor-id sup_mosfet2d \
  --goal "做 2D MOSFET Id-Vg gate_start 0V gate_stop 1V gate_step 0.25V drain_voltage 0.05V" \
  --execute \
  --max-cycles 3

python3.11 -m tcad_agent.tools.supervisor \
  --supervisor-id sup_breakdown \
  --goal "做 PN 二极管 breakdown 从 0V 到 -5V step 0.5V breakdown_current 1e-6" \
  --execute \
  --max-cycles 3

python3.11 -m tcad_agent.tools.supervisor \
  --supervisor-id sup_mesh \
  --goal "对 PN IV 做 mesh convergence relative_tolerance 0.05" \
  --execute \
  --max-cycles 3

python3.11 -m tcad_agent.tools.supervisor \
  --supervisor-id sup_schottky_cal \
  --goal "校准 Schottky diode 到可信曲线 trusted_schottky_iv.csv，并用 DEVSIM 复核" \
  --execute \
  --max-cycles 3
```

Schottky calibration goals are routed before generic Schottky IV templates, so "calibrate", "fit", "校准", "拟合", "可信曲线", "目标曲线", and "实测曲线" invoke the calibration tool rather than a one-off IV sweep.

## Resume

```bash
python3.11 -m tcad_agent.tools.supervisor \
  --supervisor-id sup_mos_cv \
  --goal "做 MOS C-V 从 -0.5V 到 0.5V 步长 0.25V" \
  --resume \
  --execute
```

## Current Limits

The supervisor is not yet a full research agent. It still needs:

- LLM-backed multi-step decomposition;
- richer action selection over sweeps and optimizers;
- budget and queue management;
- user-confirmation policies for expensive or ambiguous actions;
- multi-day scheduling and automatic continuation.

It is nevertheless the first durable outer loop for the project: it can receive a goal, inspect history, choose a TCAD tool, execute it, checkpoint the result, and refresh long-term experiment memory.
