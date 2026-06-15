# Long-Run Validation

`tcad_agent.long_run_validation` runs unattended validations of the long-horizon execution stack.

The original fast queue smoke suite exercises:

- run queue enqueue;
- queue daemon polling;
- executable tool dispatch;
- physical benchmark generation;
- experiment index rebuild;
- durable `validation_state.json`.

Run:

```bash
python3.11 -m tcad_agent.long_run_validation \
  --validation-id smoke_longrun
```

By default it queues Schottky/SBD, BJT Gummel/output, and power MOSFET/LDMOS extended-device runs aligned to the seven public TCAD source categories. BJT and power MOSFET/LDMOS use `fidelity=physics_1d`, so the unattended regression checks the upgraded executable coverage rather than only compact baselines. Custom queue items can be supplied with `--queue-goals-json`.

## Autonomous E2E Suite

The autonomous E2E suite validates the agent behavior contract around long-running DEVSIM work:

- semantic deck patch confirmation before executing unverified source edits;
- cancel-token handling at agent step boundaries;
- suspicious curve observation, repair execution, benchmark, report, and dashboard output;
- baseline-vs-mutation curve comparison followed by two finer mutation-refinement rounds;
- Sentaurus baseline run, verified patch planning, mutation-effect analysis, effect-driven patch refinement, second patched run, lineage archive, and Pareto/best-entry evidence using the public-syntax fake contract;
- a natural-language "AI 长时间自主操作 DEVSIM/Sentaurus" marathon that routes to Power MOSFET/LDMOS, runs the DEVSIM 2D field-plate runner, plans and executes the `power_mosfet_signoff` evidence pack, writes a minimal cockpit, and proves resume/cancel boundaries;
- deterministic public DEVSIM user-deck acceptance that ingests a Python deck, applies a verified semantic patch, executes the patched deck, and benchmarks the resulting artifacts;
- public real-style user-deck corpus acceptance covering function-wrapped config, package imports with overrides, and multi-sweep bias decks;
- explicit live-LLM public user-deck acceptance that requires a configured OpenAI-compatible model, disables deterministic fallback, and fails unless the model decision ledger proves every agent step came from the LLM;
- explicit live-LLM user-deck soak that slices the same mission across multiple `agent_soak` cycles and verifies resume state, heartbeat, cockpit, model decisions, and zero fallback;
- queue pause, approval, resume, and explicit unverified-patch approval;
- worker interruption recovery for queued long-run agent items.

Run the deterministic E2E harness:

```bash
python3.11 -m tcad_agent.long_run_validation \
  --suite autonomous_e2e \
  --validation-id autonomous_e2e
```

This mode uses lightweight local runners so it can run in CI while still asserting the same durable artifacts the production loop depends on: `autonomous_devsim_agent_state.json`, `heartbeat.json`, semantic deck diffs, report/dashboard files, mutation-refinement plans, Sentaurus patch/refinement work packages, overlays, lineage archives, queue rows, and `validation_state.json`.

Run only the natural-language marathon:

```bash
python3.11 -m tcad_agent.long_run_validation \
  --suite autonomous_e2e \
  --scenario-id natural_language_power_marathon \
  --validation-id nl_power_marathon
```

Run only the public user-deck acceptance scenario:

```bash
python3.11 -m tcad_agent.long_run_validation \
  --suite autonomous_e2e \
  --scenario-id public_user_deck_acceptance \
  --validation-id public_user_deck_acceptance
```

Run the public user-deck corpus scenario:

```bash
python3.11 -m tcad_agent.long_run_validation \
  --suite autonomous_e2e \
  --scenario-id public_user_deck_corpus_acceptance \
  --validation-id public_user_deck_corpus_acceptance
```

Run the curve-driven next-patch decision scenario:

```bash
python3.11 -m tcad_agent.long_run_validation \
  --suite autonomous_e2e \
  --scenario-id public_curve_decision_eval \
  --validation-id public_curve_decision_eval
```

This scenario compares public baseline/mutation curve fixtures, writes overlay SVGs and mutation-effect artifacts, then verifies that the agent chooses refine, switch, Pareto review, or curve-shape repair as appropriate.

Run the true live-LLM user-deck acceptance scenario:

```bash
export ACTSOFT_LLM_BASE_URL="http://localhost:8000/v1"
export ACTSOFT_LLM_MODEL="your-chat-model"
export ACTSOFT_LLM_API_KEY=""

python3.11 -m tcad_agent.long_run_validation \
  --suite autonomous_e2e \
  --scenario-id public_user_deck_live_llm_acceptance \
  --validation-id public_user_deck_live_llm_acceptance \
  --use-llm \
  --no-llm-fallback
```

This scenario is not part of the default deterministic E2E suite. It is a live acceptance gate: no configured model, failed model call, invalid model action, or any deterministic fallback makes the scenario fail.

Run the true live-LLM curve-decision scenario:

```bash
python3.11 -m tcad_agent.long_run_validation \
  --suite autonomous_e2e \
  --scenario-id public_curve_decision_live_llm_eval \
  --validation-id public_curve_decision_live_llm_eval \
  --use-llm \
  --no-llm-fallback
```

This live gate requires the model to return valid JSON decisions for every curve case, with raw responses recorded and fallback count fixed at zero.

Run the true live-LLM curve-decision agent loop:

```bash
python3.11 -m tcad_agent.long_run_validation \
  --suite autonomous_e2e \
  --scenario-id public_curve_decision_live_llm_agent_loop \
  --validation-id public_curve_decision_live_llm_agent_loop \
  --use-llm \
  --no-llm-fallback
```

This scenario starts from a state that already has `mutation_effect_analysis`, then requires the autonomous agent to call the curve-decision planner, convert the model decision into a guidance patch, execute the next runner request, benchmark the refined state, and stop without fallback.

Run the sliced live-LLM soak scenario:

```bash
python3.11 -m tcad_agent.long_run_validation \
  --suite autonomous_e2e \
  --scenario-id public_user_deck_live_llm_soak \
  --validation-id public_user_deck_live_llm_soak \
  --use-llm \
  --no-llm-fallback
```

The soak scenario uses `agent_soak` with small step slices so the mission crosses resume boundaries before completion. For longer local runs, pass a `public_user_deck_live_llm_soak` object through `--real-agent-request-json`, for example `{"public_user_deck_live_llm_soak":{"duration_hours":1,"max_steps":40,"step_slice":4}}`.

The Sentaurus E2E scenario is deliberately an interface and agent-control validation. It uses the public fixture and fake CSV/log artifacts to prove the agent can run baseline -> patch planner -> patched run -> curve/effect analyzer -> patch refiner -> patched run -> lineage/Pareto -> final benchmark. It does not simulate proprietary Sentaurus physics.

Run everything:

```bash
python3.11 -m tcad_agent.long_run_validation \
  --suite all \
  --validation-id full_longrun_regression
```

## Real LLM/DEVSIM Soak

For an overnight real-tool validation, pass a real autonomous-agent request. The request is merged into the default real scenario, so it can point at a real user deck, real run roots, objectives, constraints, and LLM settings.

```bash
python3.11 -m tcad_agent.long_run_validation \
  --suite autonomous_e2e \
  --mode real \
  --use-llm \
  --agent-max-steps 40 \
  --validation-id overnight_real_devsim \
  --real-agent-request-json '{
    "goal_text": "长时间自主操作真实 DEVSIM deck，修复收敛/物理质量问题并输出工程结论",
    "source_deck_path": "/path/to/user_devsim_deck.py",
    "deck_patches": [],
    "allow_user_confirmation_actions": false,
    "generate_report": true,
    "generate_dashboard": true
  }'
```

The real scenario is considered valid when the agent either completes or stops at an explicit confirmation gate. A failed/cancelled real run fails the validation and keeps the state, heartbeat, logs, and artifacts under `runs/long_run_validation/<validation_id>/`.

For a repo-local public user-deck acceptance sample, use:

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

For the same sample under strict live LLM control, remove `--no-llm` and add `--no-llm-fallback`. A configured model must choose each action; otherwise the run fails instead of silently falling back.
