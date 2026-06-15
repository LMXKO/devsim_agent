# Public DEVSIM User-Deck Acceptance

This directory contains a small, public, local DEVSIM Python deck for validating the user-deck path:

```text
natural-language goal -> deck IR -> semantic patch -> direct deck execution -> benchmark/report artifacts
```

Run the deck directly:

```bash
python3.11 tcad_agent/examples/user_deck_acceptance/pn_diode_acceptance_deck.py
```

Run it through the autonomous agent with a verified semantic patch in deterministic regression mode:

```bash
python3.11 -m tcad_agent.autonomous_devsim_agent \
  --goal "读取公开 PN diode DEVSIM deck，把 N 区掺杂调低后运行并输出验收证据" \
  --source-deck-path tcad_agent/examples/user_deck_acceptance/pn_diode_acceptance_deck.py \
  --deck-patches-json '[{"deck_path":"doping.n_doping_cm3","request_path":"n_doping_cm3","value":8e17}]' \
  --execute \
  --allow-user-confirmation-actions \
  --no-llm \
  --max-steps 6
```

Run the strict live LLM acceptance gate:

```bash
python3.11 -m tcad_agent.long_run_validation \
  --suite autonomous_e2e \
  --scenario-id public_user_deck_live_llm_acceptance \
  --validation-id public_user_deck_live_llm_acceptance \
  --use-llm \
  --no-llm-fallback
```

The live gate requires local LLM configuration through environment variables or `runs/llm_settings.json`. It fails if the model is unavailable, returns an invalid action, or any deterministic fallback is used.

Generated artifacts are written under `runs/user_deck_acceptance/` and the agent state is written under `runs/autonomous_devsim_agent/`.
