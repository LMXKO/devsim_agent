# ActSoft TCAD Agent

ActSoft TCAD Agent is a research prototype for natural-language driven TCAD work. The goal is:

> A TCAD simulation engineer describes the task in natural language; an agent plans, runs, diagnoses, repairs, resumes, and finally summarizes the TCAD work over a long horizon.

The current public version focuses on open-source DEVSIM workflows. It does not include proprietary TCAD software, unlicensed software, private model gateways, API keys, or local run artifacts.

## What It Does

- Accepts semiconductor simulation tasks from a lightweight web UI.
- Parses natural-language engineering intent into device family, analyses, metrics, evidence requirements, risk level, and executable request hints.
- Decomposes natural-language goals into durable mission steps.
- Runs a long-horizon observe/diagnose/plan/act policy with a risk ledger, replan budget, and missing-evidence tracking.
- Runs agent-callable TCAD tools with checkpoints and run state.
- Classifies failures such as convergence, schema mismatch, physical-quality risk, and repair exhaustion.
- Retries selected failures with TCAD-specific repair strategies such as smaller bias steps.
- Scores physical credibility with unit, curve-shape, model-coupling, convergence, and golden/measured evidence checks.
- Displays process logs, plots, metrics, quality checks, replanning decisions, and engineering conclusions in the page.
- Supports queue/worker recovery so long runs can be resumed after interruption.

## Current Simulation Scope

The implemented examples are intentionally practical rather than broad:

- PN junction forward/reverse IV;
- diode leakage and breakdown-style reverse sweep;
- MOS capacitor C-V;
- 2D MOSFET Id-Vg and Id-Vd;
- Schottky diode compact and DEVSIM-backed calibration paths;
- compact baseline routes for BJT, JFET, power MOSFET, and photodiode tasks;
- planned industrial templates for FinFET/GAA, SiC power diode, GaN HEMT, and IGBT workflows;
- parameter sweep, multi-dimensional optimization, convergence checks, benchmark checks, and report generation.

This is not a sign-off TCAD replacement. Treat results as automation evidence that still needs engineering review, model calibration, and mesh/physics validation.

## Architecture

```text
Natural-language task
  -> goal decomposer
  -> engineering-intent parser
  -> mission agent
  -> long-horizon control policy
  -> run queue / worker
  -> supervisor
  -> TCAD tool runner
  -> quality, metrics, convergence, repair
  -> physical credibility assessment
  -> checkpoint / replan / resume
  -> engineering conclusion
  -> web UI
```

LLM use is optional. Without an LLM, deterministic planners and validators still run. With an OpenAI-compatible model configured, the agent can use it for freer goal decomposition, diagnosis, replanning, and conclusion summarization.

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

The normal workflow is: type a natural-language TCAD task at the bottom, submit it, and watch the mission steps, tool calls, logs, plots, metrics, replanning decisions, and final conclusion appear above.

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

Run a PN junction IV sweep:

```bash
python3.11 -m tcad_agent.tools.pn_junction_iv --stop 0.5 --step 0.1
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

Run a long-horizon mission:

```bash
python3.11 -m tcad_agent.tools.mission_agent \
  --goal "帮我看一下 2D NMOS 的线性区和饱和区 Id-Vg，提取 Vth、SS、Ion/Ioff 和 DIBL，失败时自动修复并给结论" \
  --use-llm \
  --execute
```

Run the queue worker through the web app when you want browser-based long jobs.

## Repository Layout

```text
tcad_agent/
  examples/       DEVSIM-backed runnable device examples
  tools/          agent-callable CLI tools
  web_app.py      lightweight browser workbench
  mission_agent.py
  supervisor.py
  run_queue.py
  engineering_intent.py
  long_horizon_agent.py
  repair_strategy.py
  physical_quality.py
docs/             design notes and tool documentation
tests/            unit and integration-style tests
requirements.txt  Python dependencies
```

Generated data is written under `runs/` and should not be committed.

## Test

```bash
python3.11 -m unittest discover
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

- Broaden TCAD task coverage and add stronger physical-quality checks.
- Add more robust mesh, continuation, solver, and model-switch repair strategies.
- Add richer experiment search and comparison across long-running missions.
- Add adapters for user-provided licensed commercial TCAD installations without bundling proprietary content.
- Improve conclusion generation so final reports are concise, evidence-linked, and useful to device engineers.
