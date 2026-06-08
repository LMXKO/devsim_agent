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
- run MOS capacitor C-V through `mos_capacitor_cv`;
- run diode reverse leakage / breakdown through `diode_breakdown`;
- run 2D MOSFET Id-Vg / Id-Vd through `mosfet_2d_id`;
- run Schottky IV through `extended_device_sweep` with `fidelity=devsim_1d`;
- run BJT and power MOSFET/LDMOS `physics_1d` executable evidence paths through `extended_device_sweep`;
- route planned GaN HEMT, FinFET/GAA, SiC diode, and IGBT templates to explicit implementation/capability-boundary steps;
- run Schottky IV calibration through `schottky_iv_calibration` when the goal asks for calibration, fitting, or a trusted/measured curve;
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
  --supervisor-id sup_moscap \
  --goal "MOSCAP C-V 从 -2V 到 2V，tox 5nm，P-sub 1e17，判断 Cox 和平带偏移" \
  --execute \
  --max-cycles 3

python3.11 -m tcad_agent.tools.supervisor \
  --supervisor-id sup_mosfet_dibl \
  --goal "做 2D MOSFET 低/高 Vd 的 Id-Vg split，提取 Vth、SS、Ion/Ioff 和 DIBL" \
  --execute \
  --max-cycles 3

python3.11 -m tcad_agent.tools.supervisor \
  --supervisor-id sup_breakdown \
  --goal "做 diode/SBD reverse leakage 和 breakdown，从 0V 到 -5V step 0.5V breakdown_current 1e-6" \
  --execute \
  --max-cycles 3

python3.11 -m tcad_agent.tools.supervisor \
  --supervisor-id sup_schottky_cal \
  --goal "校准 Schottky/SBD 到可信曲线 trusted_schottky_iv.csv，并用 DEVSIM 复核" \
  --execute \
  --max-cycles 3

python3.11 -m tcad_agent.tools.supervisor \
  --supervisor-id sup_ldmos \
  --goal "LDMOS BV 和 Ron tradeoff，检查 impact ionization、场峰值和 Ron 分解" \
  --execute \
  --max-cycles 3

python3.11 -m tcad_agent.tools.supervisor \
  --supervisor-id sup_gan_hemt \
  --goal "GaN HEMT 输出特性、BV 和 current collapse 风险，列出 polarization/trap/self-heating 缺口" \
  --execute \
  --max-cycles 3

python3.11 -m tcad_agent.tools.supervisor \
  --supervisor-id sup_bjt \
  --goal "BJT Gummel plot、beta、Early voltage 和 collector leakage 评估" \
  --execute \
  --max-cycles 3

python3.11 -m tcad_agent.tools.supervisor \
  --supervisor-id sup_finfet \
  --goal "FinFET/GAA DIBL、Cgg/Cgd 和 variability campaign 签核计划" \
  --execute \
  --max-cycles 3
```

Schottky calibration goals are routed before generic Schottky IV templates, so "calibrate", "fit", "校准", "拟合", "可信曲线", "目标曲线", and "实测曲线" invoke the calibration tool rather than a one-off IV sweep.

Specialized device routing uses `device_templates` as a capability boundary. `executable` templates can produce TCAD evidence, `compact_baseline` templates run only as planning evidence with explicit warnings, and `planned` industrial templates produce a user-confirmation/implementation action instead of a surrogate simulation.

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
