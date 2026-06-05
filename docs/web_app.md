# TCAD Mission Workbench

`tcad_agent.tools.web_app` starts a local browser UI for long-running autonomous TCAD work. It is the interactive page entrypoint for:

- submitting a natural-language TCAD mission;
- choosing LLM-backed or deterministic mission decomposition;
- watching the execution timeline;
- starting the local queue worker automatically when a mission is sent;
- pausing, resuming, and cancelling queued items;
- checking the configured OpenAI-compatible LLM endpoint;
- reviewing indexed experiments, physical benchmarks, and quality status.

The layout is intentionally minimal: the top area is a Codex-like execution transcript, and the bottom composer is the task input. The transcript is populated from durable state, not console text, so it survives refreshes and process restarts.

The composer uses a Codex-like single action button: `Send` enqueues the mission and starts the worker, then switches to `Stop` while the worker is running. The top `清空` button only clears the current browser tab's visible transcript. It does not delete queued missions, checkpoints, run directories, curve artifacts, or indexed experiment history; new TCAD activity after the clear point will continue to appear.

The transcript shows:

- queue item status, attempts, result state path, and failures;
- mission decomposition steps;
- mission step outputs;
- supervisor action outputs;
- compact tool quality metrics, issues, artifacts, and next action;
- inline curve plots for generated PNG/SVG artifacts;
- links and short previews for CSV, JSON, log, Markdown, and text artifacts;
- terminal-style process blocks with the exact tool command, stdout tail, stderr tail, solver loading messages, convergence failures, and retry attempts;
- compact checkpoint, attempt, and case summaries so bias-step retry and convergence calculation progress is visible;
- prominent quality/failure notices before lower-level artifacts and process logs;
- Chinese labels for mission steps, supervisor actions, common failure notices, and quality status;
- quality-aware status pills such as `质量失败` or `需复核` when a completed step contains failed/suspicious quality reports;
- folded JSON details for raw structured output, so verbose state does not dominate the transcript;
- conclusion/report paths when generated.

The transcript follows the latest visible event only while the user is already near the bottom. Scrolling upward pauses auto-follow so earlier TCAD output can be inspected without being pulled back down; mission submission and the compact `最新` button force the view back to the newest event.

The compact `例子` menu floats above the `Send` button and contains natural-language semiconductor-engineering test cases such as MOS C-V fixed-charge debug, tox/Qf corner review, 2D MOSFET Vth/DIBL/SS triage, Id-Vd kink debug, diode BV/leakage spec signoff, Schottky golden-curve and temperature-corner calibration, mesh/model signoff, and existing bad-run repair. Each example shows a short title plus expected outputs. The canonical list is in `docs/semiconductor_engineering_test_cases.md`.

Run:

```bash
python3.11 -m tcad_agent.tools.web_app \
  --host 127.0.0.1 \
  --port 8765
```

If the stdlib web server cannot bind a local port in a restricted runtime, use the ASGI entrypoint:

```bash
python3.11 -m uvicorn tcad_agent.asgi_web:app \
  --host 127.0.0.1 \
  --port 8766 \
  --no-access-log
```

Open:

```text
http://127.0.0.1:8765
```

## Mission Flow

The page posts mission requests to `/api/missions`. Those requests are queued as `mission_agent` items with:

```json
{
  "goal_text": "做 2D MOSFET Id-Vg，提取 Vth、SS、Ion/Ioff，最后给工程结论",
  "execute": true,
  "use_llm_decomposer": true,
  "allow_llm_fallback": true,
  "max_cycles": 12,
  "supervisor_max_cycles": 3
}
```

The in-page worker controls call the same durable queue worker used by `tcad_agent.tools.run_queue`. Work remains checkpointed in `runs/run_queue.sqlite` and mission states remain under `runs/missions`.

## Covered TCAD Paths

The workbench routes through the same mission, supervisor, and queue runners used by tests:

- PN junction IV;
- MOS capacitor C-V;
- 2D MOSFET Id-Vg / Id-Vd;
- diode breakdown and leakage;
- Schottky IV calibration and convergence;
- extended compact devices such as BJT, JFET, power MOSFET BV/Ron, and photodiode IV;
- physical benchmarks;
- parameter sweeps, adaptive optimization, multidimensional optimization;
- engineering objective and Pareto evaluation;
- experiment report and engineering conclusion generation.

## LLM Health

The page shows configured LLM status without blocking on a network call. Press `LLM Check` to make a live OpenAI-compatible chat-completions request to the configured endpoint.
