# ActSoft TCAD Agent

ActSoft TCAD Agent is a research prototype for natural-language driven TCAD work. The goal is:

> A TCAD simulation engineer describes the task in natural language; an agent plans, runs, diagnoses, repairs, resumes, and finally summarizes the TCAD work over a long horizon.

The current public version focuses on open-source DEVSIM workflows. It does not include proprietary TCAD software, unlicensed software, private model gateways, API keys, or local run artifacts.

## What It Does

- Accepts semiconductor simulation tasks from a lightweight web UI.
- Parses natural-language engineering intent into device family, analyses, metrics, evidence requirements, risk level, and executable request hints.
- Converts natural language into a structured `TCADSpec` containing geometry, materials, models, bias hints, constraints, evidence policy, missing inputs, and signoff workflow.
- Decomposes natural-language goals into durable mission steps.
- Runs an agent-first long-horizon observe/diagnose/plan/act policy with a risk ledger, replan budget, and missing-evidence tracking.
- Provides a long-running `autonomous_devsim_agent` runtime that repeatedly chooses tools, runs DEVSIM-backed tasks, inspects state/metrics/artifacts/logs/curves/deck diffs, repairs suspicious results, benchmarks evidence, evaluates objectives, and writes conclusions.
- Supports cooperative long-run control with agent heartbeat files, cancel tokens, subprocess termination, queue pause/resume, user-confirmation approval, and autonomous run timeline dashboards.
- Runs agent-callable TCAD tools with checkpoints and run state.
- Classifies failures such as convergence, schema mismatch, physical-quality risk, and repair exhaustion.
- Repairs selected failures with an agent policy that can inspect curve diagnostics, deck patch lineage, physical benchmarks, and deterministic fallback actions.
- Parses user-provided DEVSIM Python decks into a source IR, locates geometry/model/bias/mesh/doping sections, applies semantic deck patches, emits diffs, and can execute the patched/user deck directly.
- Compares baseline and mutation curves with shape features, leakage/BV brackets, field peaks, tradeoff checks, and overlay artifacts.
- Supports deck mutation schemas for field plates, drift doping, lifetime, guard rings, junction depth, oxide thickness, implant dose, trench corner radius, trap density, and region-specific lifetime.
- Scores physical credibility with unit, curve-shape, model-coupling, convergence, and golden/measured evidence checks.
- Builds a signoff evidence pack that gates quality, artifacts, structured deck/spec, benchmark, convergence, golden/measured comparison, and capability boundary.
- Displays process logs, plots, metrics, quality checks, deck patch lineage, agent reasoning, replanning decisions, and engineering conclusions in the page.
- Supports queue/worker recovery so long runs can be resumed after interruption.

## Current Simulation Scope

The showcased examples are aligned to the seven public TCAD source categories:

- MOS capacitor C-V and flat-band / fixed-charge review;
- 2D MOSFET Id-Vg, Id-Vd, DIBL, and convergence evidence;
- diode/SBD reverse leakage, BV, and Schottky calibration;
- LDMOS / power-MOSFET BV/Ron with `physics_1d` field/avalanche evidence, plus IGBT transient-promotion planning;
- GaN / AlGaN HEMT output, BV, polarization/trap, and current-collapse planning;
- BJT Gummel/output, beta, Early-voltage, output-family, and collector-leakage evidence;
- FinFET/GAA or SOI variability, DIBL, capacitance, and quantum-correction planning;
- parameter sweep, multi-dimensional optimization, convergence checks, benchmark checks, and report generation.

This is not a sign-off TCAD replacement. Treat results as automation evidence that still needs engineering review, model calibration, and mesh/physics validation.

The capability catalog is deliberately split into three levels:

- `executable`: a real runnable TCAD-backed path exists and can produce engineering evidence;
- `compact_baseline`: a deterministic compact/planning route exists, but it is not final TCAD signoff evidence;
- `planned`: the device template is known, but the runner, quality rules, and benchmark evidence must be implemented before execution.

Executable templates also expose `tcad_fidelity` and `signoff_workflow`. Current executable evidence paths are PN 1D drift-diffusion, MOS C-V, diode reverse IV/BV, 2D MOSFET Id sweeps, Schottky 1D thermionic-contact IV, BJT Gummel/output `physics_1d`, and power MOSFET/LDMOS/GaN/FinFET/SOI/SiC/IGBT-style routes where the public implementation has an executable or physics-backed workflow. Advanced workflow claims remain gated by runner fidelity, quality rules, benchmark evidence, and capability audit records.

## Architecture

```text
Natural-language task
  -> autonomous DEVSIM agent runtime
  -> agent-first goal decomposer
  -> engineering-intent parser
  -> mission agent
  -> long-horizon control policy
  -> run queue / worker
  -> agent-first supervisor
  -> TCAD tool runner
  -> quality, metrics, curve diagnostics, convergence, repair
  -> deck IR / semantic patch / mutation overlay
  -> physical credibility assessment
  -> checkpoint / replan / resume
  -> engineering conclusion
  -> web UI
```

The default control path is agent-first where an LLM is configured, with deterministic planners and validators used as safety fallbacks. Without an LLM, the same checkpoints, schemas, quality gates, and deterministic fallback actions still run. With an OpenAI-compatible model configured, the agent can use it for freer goal decomposition, diagnosis, repair-action selection, replanning, and conclusion summarization.

### Agent-Driven Repair Loop

The long-running DEVSIM agent and repair loop are designed around structured agent decisions rather than only fixed rules:

- `autonomous_devsim_agent` is the direct long-horizon runtime for “run DEVSIM, observe, decide, repair, benchmark, report, continue”.
- It exposes dynamic tool schemas to configured OpenAI-compatible models, while still accepting structured JSON actions as a fallback.
- It can ingest user DEVSIM Python decks, apply semantic deck patches, emit diffs, run objective/Pareto checks, write heartbeat/cancel state, and render an autonomous timeline dashboard.
- `mission_agent` decomposes the goal and routes work through the supervisor, convergence checks, golden-curve comparison, physical benchmark, repair planning, and repair execution.
- `supervisor` can let an agent override the deterministic next action, while rejecting unsupported tool kinds or shell commands.
- `repair_agent` observes the run state, quality issues, metrics, curve diagnostics, deck mutations, physical benchmark context, and recent repair case memory before choosing one next action.
- `repair_executor` applies the selected request/deck patch, records the agent observation, hypothesis, tool plan, safety review, benchmark result, mutation-effect analysis, and next target.
- High-risk geometry/process/model changes require confirmation unless explicitly allowed.

Each repair attempt can produce `deck_patch_history.json`, `tcad_deck_ir.json`, semantic patch diffs, patched source decks, `baseline_mutation_overlay.svg`, physical benchmark evidence, and a case-memory record for future agent context.

## Install

Use Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

DEVSIM is installed through Python dependencies. No Sentaurus, Silvaco, or other proprietary simulator is bundled.

## Start The Web UI

```bash
python3.11 -m uvicorn tcad_agent.asgi_web:app --host 127.0.0.1 --port 8766 --no-access-log
```

Open:

```text
http://127.0.0.1:8766/
```

The normal workflow is intentionally minimal: type a natural-language TCAD task at the bottom, submit it, and watch the autonomous agent steps, tool calls, logs, plots, metrics, deck patch lineage, replanning decisions, and final conclusion appear above.

## Configure An LLM

Public builds are unconfigured by default. You can configure a model in the web UI with the Settings button, or through environment variables:

```bash
export ACTSOFT_LLM_BASE_URL="http://localhost:8000/v1"
export ACTSOFT_LLM_MODEL="your-chat-model"
export ACTSOFT_LLM_API_KEY=""
export ACTSOFT_LLM_TIMEOUT_SECONDS="60"
```

`ACTSOFT_LLM_BASE_URL` should point to the OpenAI-compatible API root ending at `/v1`. Do not include `/chat/completions`.

Saved web settings live in `runs/llm_settings.json`, which is ignored by git.

## CLI Examples

List the seven public TCAD source categories:

```bash
python3.11 -m tcad_agent.tools.device_templates sources --kind categories
```

Run a MOS capacitor C-V task:

```bash
python3.11 -m tcad_agent.tools.mos_capacitor_cv --start -0.5 --stop 0.5 --step 0.25
```

Run a 2D MOSFET Id-Vg / Id-Vd task:

```bash
python3.11 -m tcad_agent.tools.mosfet_2d_id \
  --sweep-type both \
  --gate-start 0 \
  --gate-stop 1.2 \
  --gate-step 0.1 \
  --drain-voltage 0.05 \
  --drain-start 0 \
  --drain-stop 1.0 \
  --drain-step 0.1 \
  --idvd-gate-voltage 1.2
```

Run a diode/SBD reverse leakage and BV task:

```bash
python3.11 -m tcad_agent.tools.diode_breakdown \
  --start 0 \
  --stop -5 \
  --step 0.5 \
  --breakdown-current-a 1e-6 \
  --require-breakdown
```

Route extended seven-category templates:

```bash
python3.11 -m tcad_agent.tools.device_templates route \
  --goal "GaN HEMT 输出特性和 current collapse 风险"

python3.11 -m tcad_agent.tools.device_templates route \
  --goal "LDMOS BV 和 Ron tradeoff，检查 impact ionization、场峰值和 Ron 分解"

python3.11 -m tcad_agent.tools.device_templates route \
  --goal "BJT Gummel plot、beta 和 Early voltage 提取"

python3.11 -m tcad_agent.tools.device_templates route \
  --goal "FinFET DIBL 和 Cgg/Cgd variability 签核计划"
```

Run a long-horizon mission:

```bash
python3.11 -m tcad_agent.tools.mission_agent \
  --goal "帮我看一下 2D NMOS 的线性区和饱和区 Id-Vg，提取 Vth、SS、Ion/Ioff 和 DIBL，失败时自动修复并给结论" \
  --use-llm \
  --execute
```

Run the autonomous DEVSIM agent runtime directly:

```bash
python3.11 -m tcad_agent.tools.autonomous_devsim_agent \
  --goal "自主跑 PN IV，发现曲线或收敛问题就修复，最后给工程结论" \
  --initial-tool-name pn_junction_iv_sweep \
  --initial-request-json '{"start":0,"stop":0.5,"step":0.1,"run_id":"pn_auto_001"}' \
  --execute
```

Plan or execute a repair with the agent policy:

```bash
python3.11 -m tcad_agent.tools.repair_executor \
  --state path/to/state.json \
  --use-agent-policy \
  --max-rounds 3

python3.11 -m tcad_agent.tools.repair_agent --state path/to/state.json
```

Generate an engineer-readable dashboard or report from a sweep/optimization state:

```bash
python3.11 -m tcad_agent.tools.experiment_dashboard --state path/to/optimization_state.json
python3.11 -m tcad_agent.tools.experiment_report --state path/to/optimization_state.json
```

Run the queue worker through the web app when you want browser-based long jobs.

## Repository Layout

```text
tcad_agent/
  examples/       DEVSIM-backed runnable device examples
  tools/          agent-callable CLI tools
  web_app.py      lightweight browser workbench
  autonomous_devsim_agent.py
  mission_agent.py
  supervisor.py
  run_queue.py
  engineering_intent.py
  long_horizon_agent.py
  deck_ir.py
  deck_writer.py
  curve_diagnostics.py
  repair_agent.py
  repair_strategy.py
  repair_executor.py
  physical_quality.py
  physical_benchmark.py
docs/             design notes and tool documentation
tests/            unit and integration-style tests
requirements.txt  Python dependencies
```

Generated data is written under `runs/` and should not be committed.

## Test

```bash
python3.11 -m pytest -q
```

Some tests execute DEVSIM examples and may take longer than pure unit tests.

## Open-Source Hygiene

Before publishing, keep these out of the repository:

- `runs/`, local SQLite databases, generated plots, logs, and simulator output;
- `.env`, local LLM settings, API keys, private endpoint URLs, and private model names;
- IDE metadata such as `.idea/`;
- simulator binaries, proprietary decks, licensed PDK files, and confidential process data.

If you publish from a real git repository with existing commits, scan the entire git history as well as the current tree.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Roadmap

- Broaden executable TCAD coverage for planned industrial device templates.
- Continue improving agent repair playbooks with richer measured/golden-curve grounding and more realistic Pareto policies.
- Add richer experiment search, comparison, and repair-case retrieval across long-running missions.
- Add adapters for user-provided licensed commercial TCAD installations without bundling proprietary content.
- Improve conclusion generation so final reports are concise, evidence-linked, and useful to device engineers.
