# ActSoft TCAD Agent

ActSoft TCAD Agent is a research prototype for long-running, natural-language-driven TCAD work.

The target workflow is simple: a device engineer describes a task, then an agent plans, runs, diagnoses, repairs, refines, resumes, and summarizes TCAD evidence over a long horizon.

The public repository focuses on open-source DEVSIM workflows plus a local adapter for user-owned Sentaurus installations. It does not include proprietary TCAD software, licenses, PDKs, commercial model files, private decks, API keys, private model gateways, or local simulation artifacts.

## Core Capabilities

- Natural-language task routing into structured TCAD specs and executable requests.
- Top-level goal routing that turns broad "AI agent operates DEVSIM/Sentaurus" requests into an autonomous mission plan.
- Agent-first `autonomous_devsim_agent` loop with decision ledger, hypothesis tree, dynamic toolbelt, and guardrail fallback.
- Compiled mission specs that turn natural-language goals into objectives, constraints, mutations, stop conditions, and validation plans.
- DEVSIM-backed examples for PN, diode/BV, MOS C-V, MOSFET Id, Schottky, BJT, power-device planning, and related sweeps.
- DEVSIM-backed Power MOSFET/LDMOS 2D field-plate runner plus 1D drift/BV baseline, runner contracts, mesh/field artifacts, and signoff-gap evidence.
- Industrial runner registry for agent-callable Power MOSFET, GaN HEMT, SiC diode, and IGBT routes with explicit maturity/signoff boundaries.
- External industrial runner contracts for user-owned Sentaurus workspaces; missing software/license/PDK stays outside git and becomes an explicit gate.
- User DEVSIM deck ingestion, source IR extraction, semantic patching, diffs, and guarded execution.
- Curve diagnostics for leakage windows, BV brackets, field peaks, knees, overlays, and mutation effects.
- Multi-objective/Pareto evaluation with machine-readable continue/review/reject decisions.
- Agent experiment design from benchmark gaps, curve evidence, deck mutations, and signoff evidence.
- Agent memory, recovery classification, curve-guided patch execution, and soak daemon lifecycle state.
- Power MOSFET 2D signoff evidence workflow for baseline, benchmark, convergence, optional golden correlation, and signoff gate.
- Live public evidence lookup with hard pause gates, plus runner-promotion work packages for new simulator/device operations.
- Run queue, heartbeat, cancel token, approval pause/resume, and interruption recovery.
- Minimal web cockpit for natural-language tasks, progress, artifacts, patch lineage, and conclusions.
- Local Sentaurus adapter for licensed user environments, with verified semantic deck patches and lineage.

## Agent Policy

The runtime is agent-first when an OpenAI-compatible LLM is configured, but deterministic validators and fallbacks remain active.

Every autonomous run creates a `public_evidence_dossier` before patch/signoff planning. It records matched public sources, convergence playbooks, model/metric expectations, and guardrails. If a simulator operation is not covered by local deck evidence plus public evidence, the agent should perform live lookup or pause instead of guessing.

High-risk geometry/process/model changes stay confirmation-gated unless explicitly allowed.

## Install

Use Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Start The Web UI

The web UI is the simplest entry point: type a natural-language TCAD goal, press Send, and the queue runs it through the long-duration `agent_soak` agent wrapper by default.

```bash
python3.11 -m uvicorn tcad_agent.asgi_web:app --host 127.0.0.1 --port 8766 --no-access-log
```

Open:

```text
http://127.0.0.1:8766/
```

## Configure An LLM

The project is unconfigured by default. Configure a model in the web UI settings, or use environment variables:

```bash
export ACTSOFT_LLM_BASE_URL="http://localhost:8000/v1"
export ACTSOFT_LLM_MODEL="your-chat-model"
export ACTSOFT_LLM_API_KEY=""
export ACTSOFT_LLM_TIMEOUT_SECONDS="60"
```

Saved web settings are written under `runs/` and are ignored by git.

## Quick Commands

Run an autonomous DEVSIM task:

```bash
python3.11 -m tcad_agent.tools.autonomous_devsim_agent \
  --goal "Run PN IV, repair suspicious curves, benchmark evidence, and summarize" \
  --initial-tool-name pn_junction_iv_sweep \
  --initial-request-json '{"start":0,"stop":0.5,"step":0.1,"run_id":"pn_auto_001"}' \
  --execute
```

Run a long-horizon mission:

```bash
python3.11 -m tcad_agent.tools.mission_agent \
  --goal "Analyze 2D NMOS Id-Vg/Id-Vd, extract Vth/SS/Ion-Ioff/DIBL, repair failures, and conclude" \
  --use-llm \
  --execute
```

Route public device templates:

```bash
python3.11 -m tcad_agent.tools.device_templates route \
  --goal "LDMOS BV and Ron tradeoff with field peak review"
```

Route a broad autonomous-agent goal:

```bash
python3.11 -m tcad_agent.tools.agent_goal_router \
  --goal "AI 长时间自主操作 DEVSIM/Sentaurus 完成功率器件优化任务"
```

Fetch public evidence and build an industrial runner-promotion work package:

```bash
python3.11 -m tcad_agent.tools.public_evidence_lookup --live --goal "GaN HEMT BV current collapse" --template-id gan_hemt_id_bv
python3.11 -m tcad_agent.tools.industrial_runner_promotion --goal "GaN HEMT BV current collapse" --template-id gan_hemt_id_bv
```

Run the promoted Power MOSFET/LDMOS 2D field-plate runner:

```bash
python3.11 -m tcad_agent.tools.extended_device_sweep --device-type power_mosfet_bv_ron --fidelity devsim_2d_field_plate
```

Build the Power MOSFET/LDMOS signoff evidence gate:

```bash
python3.11 -m tcad_agent.tools.power_mosfet_signoff --run-id ldmos_gate_001
```

Gate an external Sentaurus industrial runner without committing commercial assets:

```bash
python3.11 -m tcad_agent.tools.industrial_external_runner \
  --goal "GaN HEMT BV current collapse" \
  --template-id gan_hemt_id_bv \
  --project /path/to/user_owned_sentaurus_project \
  --profile ~/.actsoft/sentaurus_profile.json
```

Validate long-run behavior:

```bash
python3.11 -m tcad_agent.tools.long_run_validation --suite autonomous_e2e --validation-id autonomous_e2e
```

Run a long-duration autonomous soak:

```bash
python3.11 -m tcad_agent.tools.agent_soak --goal "AI 长时间自主操作 DEVSIM，优化 Power MOSFET BV/Ron/leakage/field peak" --duration-hours 0.5 --max-steps 40 --step-slice 4 --execute
```

Run the same goal through the queue-backed soak daemon:

```bash
python3.11 -m tcad_agent.tools.agent_soak_daemon --goal "AI 长时间自主操作 DEVSIM，优化 Power MOSFET BV/Ron/leakage/field peak" --duration-hours 1 --max-steps 80 --execute
```

Run only the natural-language Power MOSFET marathon:

```bash
python3.11 -m tcad_agent.tools.long_run_validation --suite autonomous_e2e --scenario-id natural_language_power_marathon
```

Generate a dashboard or report:

```bash
python3.11 -m tcad_agent.tools.experiment_dashboard --state path/to/state.json
python3.11 -m tcad_agent.tools.experiment_report --state path/to/state.json
```

## Sentaurus Adapter

Sentaurus support is an adapter for a licensed installation that already exists on your machine or cluster. Keep the profile outside git.

Example:

```bash
python3.11 -m tcad_agent.tools.autonomous_devsim_agent \
  --goal "Use Sentaurus to reduce LDMOS leakage without sacrificing BV/Ron/field peak" \
  --sentaurus-project-path /Users/me/tcad_projects/ldmos_case \
  --sentaurus-profile-path ~/.actsoft/sentaurus_profile.json \
  --sentaurus-request-json '{"flow":["sdevice"],"deck_files":["device.cmd"],"timeout_seconds":7200}' \
  --enable-experiment-design \
  --execute
```

The adapter can copy a project into a controlled run workspace, apply verified semantic patches, run configured local commands, parse logs, ingest CSV curves, compare baseline vs mutation, build patch lineage, and return state to the autonomous agent.

When Sentaurus is not installed, validate only the agent-side contract:

```bash
python3.11 -m tcad_agent.tools.sentaurus_contract \
  --all-fixtures \
  --fixtures-root tcad_agent/examples/sentaurus_fixtures
```

The fake backend is interface validation only. It is not a Sentaurus physics substitute.

## Tests

```bash
python3.11 -m unittest
```

Some tests execute DEVSIM examples and may take longer than pure unit tests.

## Repository Layout

```text
tcad_agent/
  tools/                       CLI tools
  examples/                    runnable examples and public fixtures
  autonomous_devsim_agent.py   long-running agent runtime
  mission_agent.py             goal decomposition and orchestration
  sentaurus.py                 local Sentaurus adapter
  sentaurus_*                  Sentaurus deck, patch, effect, lineage modules
  curve_diagnostics.py         curve shape and mutation-effect analysis
  engineering_objectives.py    constraints, objectives, Pareto decisions
docs/                          focused design and tool notes
tests/                         unit and integration-style tests
```

Generated data is written under `runs/` and should not be committed.

## More Documentation

- [Autonomous DEVSIM Agent](docs/autonomous_devsim_agent.md)
- [Web Workbench](docs/web_app.md)
- [Run Queue](docs/run_queue.md)
- [Sentaurus Adapter](docs/sentaurus_adapter.md)
- [Public TCAD Sources](docs/tcad_public_sources.md)
- [Engineering Objectives](docs/engineering_objectives.md)
- [Long Run Validation](docs/long_run_validation.md)
- [Agent Soak](docs/agent_soak.md)

## Publishing Hygiene

Before pushing or publishing, keep these out of the repository:

- `runs/`, SQLite databases, plots, logs, and simulator outputs;
- `.env`, local LLM settings, API keys, private endpoint URLs, and private model names;
- simulator binaries, license strings, proprietary decks, PDK files, commercial models, and confidential process data;
- IDE metadata and local machine state.

If publishing from a repository with prior commits, scan git history as well as the current tree.

## License

MIT. See [LICENSE](LICENSE).
