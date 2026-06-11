# Agent Soak

`tcad_agent.tools.agent_soak` is the long-duration wrapper for `autonomous_devsim_agent`.

It runs the same autonomous agent in step slices, then persists a soak-level checkpoint, heartbeat, decision counts, latest agent state, and minimal cockpit after each cycle. This is the entry point for "let the AI operate DEVSIM for a long time" validation.

Run a short real LLM/DEVSIM soak:

```bash
python3.11 -m tcad_agent.tools.agent_soak \
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
python3.11 -m tcad_agent.tools.agent_soak \
  --goal "AI 长时间自主操作 DEVSIM，优化 Power MOSFET BV/Ron/leakage/field peak" \
  --soak-id power_mosfet_soak_001 \
  --resume \
  --duration-hours 0.5 \
  --max-steps 80 \
  --step-slice 4 \
  --execute
```

Cancel is file-based. Create the cancel file shown in `agent_soak_state.json`, or pass your own path with `--cancel-file`.

Artifacts are written under `runs/agent_soak/<soak_id>/` by default:

- `agent_soak_state.json`;
- `agent_soak_heartbeat.json`;
- nested `autonomous_devsim_agent_state.json`;
- `agent_cockpit.html`;
- per-cycle cockpit snapshots under `cockpit/`.

LLM endpoint settings are read from environment variables or `runs/llm_settings.json`. The `runs/` directory is ignored by git; do not commit endpoint URLs, model names, API keys, simulator licenses, PDKs, private decks, or commercial model files.
