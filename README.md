# ActSoft TCAD Agent

ActSoft TCAD Agent is a research prototype for natural-language driven TCAD work. The goal is:

> A TCAD simulation engineer describes the task in natural language; an agent plans, runs, diagnoses, repairs, resumes, and finally summarizes the TCAD work over a long horizon.

The current public version focuses on open-source DEVSIM workflows and a local adapter for user-owned Sentaurus installations. It does not include proprietary TCAD software, unlicensed software, private model gateways, API keys, licenses, PDKs, commercial model files, private decks, or local run artifacts.

## What It Does

- Accepts semiconductor simulation tasks from a lightweight web UI.
- Parses natural-language engineering intent into device family, analyses, metrics, evidence requirements, risk level, and executable request hints.
- Converts natural language into a structured `TCADSpec` containing geometry, materials, models, bias hints, constraints, evidence policy, missing inputs, and signoff workflow.
- Decomposes natural-language goals into durable mission steps.
- Runs an agent-first long-horizon observe/diagnose/plan/act policy with a risk ledger, replan budget, and missing-evidence tracking.
- Provides a long-running `autonomous_devsim_agent` runtime that repeatedly chooses tools, runs DEVSIM-backed tasks, inspects state/metrics/artifacts/logs/curves/deck diffs, repairs suspicious results, benchmarks evidence, evaluates objectives, and writes conclusions.
- Provides a `sentaurus_run` external runner that can clone a user-owned Sentaurus project into a controlled run workspace, apply verified semantic file patches, execute configured local commands, parse logs, ingest extracted CSV curves, benchmark the result, and hand the state back to the autonomous agent.
- Supports cooperative long-run control with agent heartbeat files, cancel tokens, subprocess termination, queue pause/resume, user-confirmation approval, and autonomous run timeline dashboards.
- Runs agent-callable TCAD tools with checkpoints and run state.
- Classifies failures such as convergence, schema mismatch, physical-quality risk, and repair exhaustion.
- Repairs selected failures with an agent policy that can inspect curve diagnostics, deck patch lineage, physical benchmarks, mutation-effect overlays, and deterministic fallback actions.
- Parses user-provided DEVSIM Python decks into a source IR with imports, functions, control flow, DEVSIM call aliases, semantic bindings, geometry/model/bias/mesh/doping sections, and verified vs unverified patch lineage.
- Applies path-aware semantic deck patches to assignments, nested dictionaries, DEVSIM named parameters, function defaults, loop-driven bias values, and common unit-wrapper calls, then emits patched source plus unified diffs.
- Compares baseline and mutation curves with shape features, leakage/BV brackets, field peaks, tradeoff checks, and overlay artifacts.
- Supports deck mutation schemas for field plates, drift doping, lifetime, guard rings, junction depth, oxide thickness, implant dose, trench corner radius, trap density, and region-specific lifetime.
- Scores physical credibility with unit, curve-shape, model-coupling, convergence, and golden/measured evidence checks.
- Aligns golden/measured curves with unit normalization, interpolated x-grid matching, log-domain error metrics, and a calibration recommendation artifact.
- Builds a signoff evidence pack that gates quality, artifacts, structured deck/spec, benchmark, convergence, golden/measured comparison, and capability boundary.
- Generates ranked agent experiment-design candidates from signoff gaps, curve evidence, benchmark warnings, and deck mutations, then can execute the highest-value next experiment.
- Displays a minimal agent cockpit with the active hypothesis, pending next experiment, deck patch lineage, golden/measured calibration summary, process logs, plots, metrics, quality checks, replanning decisions, and engineering conclusions.
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
  -> autonomous TCAD agent runtime
  -> agent-first goal decomposer
  -> engineering-intent parser
  -> mission agent
  -> long-horizon control policy
  -> run queue / worker
  -> agent-first supervisor
  -> TCAD tool runner / local Sentaurus adapter
  -> quality, metrics, curve diagnostics, convergence, repair
  -> deck IR / semantic patch / mutation overlay
  -> physical credibility assessment
  -> checkpoint / replan / resume
  -> engineering conclusion
  -> web UI
```

The default control path is agent-first where an LLM is configured, with deterministic planners and validators used as safety fallbacks. Without an LLM, the same checkpoints, schemas, quality gates, candidate-experiment planning, and deterministic fallback actions still run. With an OpenAI-compatible model configured, the agent can use it for freer goal decomposition, diagnosis, experiment design, repair-action selection, replanning, and conclusion summarization.

### Agent-Driven Repair Loop

The long-running DEVSIM agent and repair loop are designed around structured agent decisions rather than only fixed rules:

- `autonomous_devsim_agent` is the direct long-horizon runtime for “run DEVSIM, observe, decide, repair, benchmark, report, continue”.
- It exposes dynamic tool schemas to configured OpenAI-compatible models, while still accepting structured JSON actions as a fallback.
- It can ingest user DEVSIM Python decks, apply verified semantic deck patches, emit diffs, run objective/Pareto checks, write heartbeat/cancel state, and render an autonomous timeline dashboard.
- `mission_agent` decomposes the goal and routes work through the supervisor, convergence checks, golden-curve comparison, physical benchmark, repair planning, and repair execution.
- `supervisor` can let an agent override the deterministic next action, while rejecting unsupported tool kinds or shell commands.
- `repair_agent` observes the run state, quality issues, metrics, curve diagnostics, deck mutations, physical benchmark context, and recent repair case memory before choosing one next action.
- `repair_executor` applies the selected request/deck patch, records the agent observation, hypothesis, tool plan, safety review, benchmark result, mutation-effect analysis, and next target.
- `autonomous_devsim_agent` can turn a successful baseline-vs-mutation overlay into the next finer request/deck patch and execute it within the configured refinement budget.
- When enabled, `autonomous_devsim_agent` can also run `plan_experiment_design`, which ranks convergence, golden/measured correlation, repair, and mutation candidates from the latest evidence instead of following a single fixed rule.
- For Sentaurus states, enabled experiment design first runs `sentaurus_patch_planner`: it reads the copied/user deck, maps the natural-language goal to verified semantic patch candidates, records validation diffs, and can execute the selected low/medium-risk candidate as the next `sentaurus_run`.
- After a Sentaurus patch run, `sentaurus_mutation_effect_analyzer` compares baseline vs patched metrics/curves, flags BV/Ron/field/leakage tradeoffs, writes overlay artifacts, and decides whether to continue, switch direction, or pause for Pareto/constraint review.
- `sentaurus_patch_refiner` consumes that decision: `continue_refine` becomes a smaller verified follow-up patch, while `switch_target`/`reject_candidate` ask the planner for a different verified target instead of repeating the same edit.
- Each patched Sentaurus state writes `sentaurus_lineage_archive.json`, a compact multi-run trail with patch fields, effect decisions, key metrics, overlays, Pareto front, and best entry.
- Every autonomous step updates `checkpoint.agent_hypothesis_tree` with the current hypothesis, expected observation, stop condition, evidence used, verdict, and fallback alternatives.
- High-risk geometry/process/model changes require confirmation unless explicitly allowed.

Each repair attempt can produce `deck_patch_history.json`, `tcad_deck_ir.json`, semantic patch diffs, patched source decks, `baseline_mutation_overlay.svg`, physical benchmark evidence, golden/measured calibration artifacts, and a case-memory record for future agent context.

## Install

Use Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

DEVSIM is installed through Python dependencies. No Sentaurus, Silvaco, or other proprietary simulator is bundled.

## Local Sentaurus Adapter

The Sentaurus path is an adapter for a licensed installation that already exists on your machine or cluster. The repository never stores Synopsys binaries, license strings, commercial PDKs, process decks, calibrated model files, or private simulation outputs.

What the adapter does:

- reads a local runtime profile JSON outside the repository;
- copies the selected project into `runs/sentaurus/<run_id>/project`;
- parses common Sentaurus command-file blocks into a deck IR, applies explicit text/regex/JSON or semantic deck patches, and writes `sentaurus_patch.diff`;
- executes configured flow steps such as `sdevice`, `svisual`, `inspect`, or a site wrapper script;
- captures stdout/stderr, scans logs for license/convergence/mesh/fatal issues, collects `.log`, `.plt`, `.tdr`, and `.csv` artifacts, and extracts metrics from CSV curves;
- writes `sentaurus_state.json` so the same autonomous benchmark, objective/Pareto, report, and next-action logic can continue.

The public sources used to set this boundary are conservative: Synopsys describes Sentaurus Device, Workbench, and Visual as device simulation, workflow/execution, and visualization/xy-data tools; public training mirrors show command-file execution and extraction patterns. If a real project needs a new operation or parser, add it from real project evidence or public/official documentation first rather than guessing.

Reference links used for this adapter boundary:

- [Synopsys TCAD](https://www.synopsys.com/manufacturing/tcad.html)
- [Sentaurus Device training mirror](https://ghzphy.github.io/Sentaurus_Training/sd/sd_1.html)
- [Sentaurus quasistationary sweep training mirror](https://ghzphy.github.io/Sentaurus_Training/sd/sd_8.html)
- [Sentaurus Visual batch/Python training mirror](https://ghzphy.github.io/Sentaurus_Training/sv/sv_6.html)

Example profile, saved outside git such as `~/.actsoft/sentaurus_profile.json`:

```json
{
  "profile_id": "local_sentaurus",
  "sentaurus_home": "/opt/synopsys/sentaurus",
  "commands": {
    "sdevice": "/opt/synopsys/sentaurus/bin/sdevice",
    "svisual": "/opt/synopsys/sentaurus/bin/svisual"
  },
  "allowed_project_roots": ["/Users/me/tcad_projects"],
  "run_root": "/Users/me/tcad_runs/actsoft_sentaurus",
  "env": {
    "STROOT": "/opt/synopsys/sentaurus"
  },
  "default_flow": ["sdevice"],
  "curve_globs": ["*.csv", "*_extract.csv", "*_iv.csv"],
  "artifact_globs": ["*.log", "*_des.log", "*.plt", "*_des.plt", "*.tdr", "*_des.tdr", "*.csv"]
}
```

Keep license variables in your shell, site module, keychain, or private profile outside this repo. `runtime_profile` stored in state files records only profile id, command names, allowed roots, and environment variable keys, not their values.

Run one Sentaurus baseline directly:

```bash
python3.11 -m tcad_agent.tools.sentaurus_run \
  --goal "Run BV baseline and extract IV/field curve" \
  --project /Users/me/tcad_projects/ldmos_case \
  --profile ~/.actsoft/sentaurus_profile.json \
  --flow sdevice \
  --deck-file device.cmd \
  --timeout-seconds 7200
```

Run the long-horizon agent with natural language plus Sentaurus context:

```bash
python3.11 -m tcad_agent.tools.autonomous_devsim_agent \
  --goal "用 Sentaurus 跑这个 LDMOS 项目，降低漏电，同时不要牺牲 BV/Ron，必要时提出下一轮 deck patch" \
  --sentaurus-project-path /Users/me/tcad_projects/ldmos_case \
  --sentaurus-profile-path ~/.actsoft/sentaurus_profile.json \
  --sentaurus-request-json '{"flow":["sdevice"],"deck_files":["device.cmd"],"timeout_seconds":7200}' \
  --enable-experiment-design \
  --execute
```

Plan Sentaurus deck patches without launching the simulator:

```bash
python3.11 -m tcad_agent.tools.sentaurus_patch_planner \
  --goal "Ramp reverse BV to 1200V, reduce step size if convergence is difficult" \
  --project /Users/me/tcad_projects/ldmos_case \
  --deck-file device.cmd \
  --output /tmp/sentaurus_patch_plan.json
```

The planner currently covers continuation/Math controls, BV `Goal` voltage updates, drift doping, lifetime, trap density, field plate, guard ring, oxide thickness, implant dose, junction depth, trench corner radius, and region-specific lifetime variables when those variables are present in the deck. It validates each proposed semantic patch against the current deck before selecting it. Geometry/process/model classes are marked high risk and require confirmation.

Compare a baseline Sentaurus state against a patched run:

```bash
python3.11 -m tcad_agent.tools.sentaurus_mutation_effect \
  --baseline /tmp/sentaurus_base/sentaurus_state.json \
  --mutation /tmp/sentaurus_patch/sentaurus_state.json \
  --goal "降低漏电，同时不要牺牲 BV/Ron/field peak" \
  --candidate-json '{"candidate_id":"device.cmd:lifetime:LIFETIME_SCALE"}' \
  --output /tmp/sentaurus_mutation_effect.json
```

Refine the next Sentaurus patch from that effect evidence:

```bash
python3.11 -m tcad_agent.tools.sentaurus_patch_refiner \
  --state /tmp/sentaurus_patch/sentaurus_state.json \
  --goal "继续降低漏电，但不要牺牲 BV/Ron/field peak" \
  --output /tmp/sentaurus_patch_refinement.json
```

Archive multi-run Sentaurus patch lineage and Pareto status:

```bash
python3.11 -m tcad_agent.tools.sentaurus_lineage \
  --state /tmp/sentaurus_patch/sentaurus_state.json \
  --output /tmp/sentaurus_lineage_archive.json
```

For curve-aware diagnosis, configure your Sentaurus Visual/Inspect or site wrapper to export a numeric CSV into the copied project. Recommended columns are explicit and unit-bearing, for example `voltage_v,current_a,electric_field_v_per_cm`. Without a CSV, the run can still capture logs/artifacts, but benchmark status remains limited because the agent cannot inspect curve shape, BV brackets, leakage interval, or field peak.

Patch format is deliberately explicit. For raw text compatibility:

```json
[
  {
    "file": "device.cmd",
    "operation": "replace_text",
    "pattern": "set DRIFT_DOPING 1e15",
    "replacement": "set DRIFT_DOPING 8e14",
    "reason": "test lower drift doping as a BV/leakage direction"
  }
]
```

For common Sentaurus command files, semantic patch v1 can target `set`/`#define` variables, assignments inside blocks such as `File`, `Electrode`, `Physics`, `Math`, `Solve`, selector-matched records such as an electrode with `Name="drain"`, and model-line insertion:

```json
[
  {
    "file": "device.cmd",
    "operation": "sentaurus_set_variable",
    "variable": "DRIFT_DOPING",
    "value": "8e14"
  },
  {
    "file": "device.cmd",
    "operation": "sentaurus_update_assignment",
    "section_path": ["Electrode"],
    "selector": {"Name": "drain"},
    "parameter": "Voltage",
    "value": -1200
  },
  {
    "file": "device.cmd",
    "operation": "sentaurus_upsert_assignment",
    "section_path": ["Math"],
    "parameter": "Iterations",
    "value": 80
  }
]
```

Each parsed deck emits a `sentaurus_deck_ir_*.json` artifact with sections, assignments, variables, and warnings. Unsupported or unmatched patches are recorded as unverified and should not be executed as trusted mutations. Geometry/process/model edits should remain behind user confirmation unless you have a validated site-specific patch schema.

When Sentaurus is not installed, validate the agent-side contract against the public-syntax fixture corpus:

```bash
python3.11 -m tcad_agent.tools.sentaurus_contract \
  --all-fixtures \
  --fixtures-root tcad_agent/examples/sentaurus_fixtures

python3.11 -m tcad_agent.tools.sentaurus_contract \
  --project tcad_agent/examples/sentaurus_fixtures/power_diode_bv \
  --run-fake-e2e \
  --output-root /tmp/actsoft_sentaurus_contract_smoke
```

The fake backend in this contract harness only validates process control, artifact collection, log parsing, deck IR, semantic patches, and CSV schema. It is explicitly marked `interface_contract_only` and is not a Sentaurus physics substitute.

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

Validate long-run behavior:

```bash
python3.11 -m tcad_agent.tools.long_run_validation --validation-id smoke_longrun

python3.11 -m tcad_agent.tools.long_run_validation \
  --suite autonomous_e2e \
  --validation-id autonomous_e2e
```

The `autonomous_e2e` suite checks confirmation gates, cancellation, repair/report output, multi-round mutation refinement, queue approval/resume, and interrupted-worker recovery. For real overnight LLM/DEVSIM runs, use `--mode real --use-llm --real-agent-request-json ...`; generated evidence stays under `runs/long_run_validation/<validation_id>/`.

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

- Run real overnight soak tests with private user decks, configured LLMs, and DEVSIM to harden the new experiment-design loop.
- Promote remaining `physics_1d` first-pass evidence into mesh-resolved public runners where open examples are available.
- Add richer experiment search, comparison, and repair-case retrieval across long-running missions.
- Add adapters for user-provided licensed commercial TCAD installations without bundling proprietary content.
- Improve conclusion generation so final reports are concise, evidence-linked, and useful to device engineers.
