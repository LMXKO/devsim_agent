# Agent Soak

`tcad_agent.agent_soak` is the long-duration wrapper for `autonomous_devsim_agent`.

It runs the same autonomous agent in step slices, then persists a soak-level checkpoint, heartbeat, decision counts, latest agent state, and minimal cockpit after each cycle. This is the entry point for "let the AI operate DEVSIM for a long time" validation.

The web UI posts natural-language missions to `agent_soak` by default. The user-facing flow is intentionally small: enter the TCAD goal, press Send, then watch the queue item, soak cycles, autonomous decisions, artifacts, and cockpit links in the transcript.

The soak layer now adds the long-running agent scaffolding around the inner DEVSIM agent:

- compiles the natural-language goal into a mission spec with objectives, constraints, allowed deck mutations, stop conditions, validation plan, and risk gates;
- retrieves prior agent memory records before the run and appends a compact memory record when the run reaches a terminal state;
- classifies failures into recovery families, records retry decisions, and patches the next autonomous request when a safe retry is available;
- reads curve shape, baseline-vs-mutation effect, and Pareto decision evidence to recommend and execute the next deck-patch direction;
- writes lifecycle events for start, resume, cycle, recovery, curve guidance, and memory writeback.

If an inner autonomous cycle returns `completed` but the new `curve_guidance` contains an actionable `next_patch_hint`, the soak does not stop immediately. It starts another slice, lets `plan_guidance_patch` build the next request/deck patch, executes that patch through the original runner, and then compares the patched curve back against the source state. The default cap is one automatic curve-guided patch; use `--max-curve-guided-patches` to raise it or `--no-auto-curve-guidance` to only record the recommendation.

Run a short real LLM/DEVSIM soak:

```bash
python3.11 -m tcad_agent.agent_soak \
  --goal "AI 长时间自主操作 DEVSIM，优化 Power MOSFET BV/Ron/leakage/field peak" \
  --soak-id power_mosfet_soak_001 \
  --duration-hours 0.5 \
  --max-steps 40 \
  --step-slice 4 \
  --execute \
  --require-capability-audit \
  --enable-experiment-design \
  --initial-tool-name extended_device_sweep \
  --initial-request-json '{"device_type":"power_mosfet_bv_ron","fidelity":"devsim_2d_field_plate","evidence_level":"tcad_executable","run_id":"power_mosfet_soak_001","start":0,"stop":-20,"step":10}'
```

Resume the same soak:

```bash
python3.11 -m tcad_agent.agent_soak \
  --goal "AI 长时间自主操作 DEVSIM，优化 Power MOSFET BV/Ron/leakage/field peak" \
  --soak-id power_mosfet_soak_001 \
  --resume \
  --duration-hours 0.5 \
  --max-steps 80 \
  --step-slice 4 \
  --execute
```

Cancel is file-based. Create the cancel file shown in `agent_soak_state.json`, or pass your own path with `--cancel-file`.

Run the same mission through the queue-backed daemon:

```bash
python3.11 -m tcad_agent.agent_soak_daemon \
  --goal "AI 长时间自主操作 DEVSIM，优化 Power MOSFET BV/Ron/leakage/field peak" \
  --daemon-id power_mosfet_daemon_001 \
  --duration-hours 1 \
  --max-steps 80 \
  --execute
```

Artifacts are written under `runs/agent_soak/<soak_id>/` by default:

- `agent_soak_state.json`;
- `agent_soak_heartbeat.json`;
- nested `autonomous_devsim_agent_state.json`;
- `agent_cockpit.html`;
- per-cycle cockpit snapshots under `cockpit/`.

Important state fields:

- `mission_spec`: compiled goal, intent, objectives, constraints, mutation vocabulary, validation plan, and risk gates;
- `agent_memory_context`: compact records retrieved before the run;
- `recovery_events`: failure family, retry/pause decision, request patch, and next action;
- `curve_guidance`: shape/effect/Pareto-driven next patch hint;
- `curve_guided_patch_runs`: number of executed guidance-driven patches seen by the inner autonomous agent;
- `lifecycle_events`: short chronological events for the web transcript and daemon heartbeat;
- `memory_record_path`: JSONL memory file written under `runs/` unless `--memory-path` or `ACTSOFT_AGENT_MEMORY_PATH` is set.

LLM endpoint settings are read from environment variables or `runs/llm_settings.json`. The `runs/` directory is ignored by git; do not commit endpoint URLs, model names, API keys, simulator licenses, PDKs, private decks, or commercial model files.
