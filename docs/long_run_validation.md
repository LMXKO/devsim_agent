# Long-Run Validation

`tcad_agent.tools.long_run_validation` runs unattended validations of the long-horizon execution stack.

The original fast queue smoke suite exercises:

- run queue enqueue;
- queue daemon polling;
- executable tool dispatch;
- physical benchmark generation;
- experiment index rebuild;
- durable `validation_state.json`.

Run:

```bash
python3.11 -m tcad_agent.tools.long_run_validation \
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
- queue pause, approval, resume, and explicit unverified-patch approval;
- worker interruption recovery for queued long-run agent items.

Run the deterministic E2E harness:

```bash
python3.11 -m tcad_agent.tools.long_run_validation \
  --suite autonomous_e2e \
  --validation-id autonomous_e2e
```

This mode uses lightweight local runners so it can run in CI while still asserting the same durable artifacts the production loop depends on: `autonomous_devsim_agent_state.json`, `heartbeat.json`, semantic deck diffs, report/dashboard files, mutation-refinement plans, Sentaurus patch/refinement work packages, overlays, lineage archives, queue rows, and `validation_state.json`.

Run only the natural-language marathon:

```bash
python3.11 -m tcad_agent.tools.long_run_validation \
  --suite autonomous_e2e \
  --scenario-id natural_language_power_marathon \
  --validation-id nl_power_marathon
```

The Sentaurus E2E scenario is deliberately an interface and agent-control validation. It uses the public fixture and fake CSV/log artifacts to prove the agent can run baseline -> patch planner -> patched run -> curve/effect analyzer -> patch refiner -> patched run -> lineage/Pareto -> final benchmark. It does not simulate proprietary Sentaurus physics.

Run everything:

```bash
python3.11 -m tcad_agent.tools.long_run_validation \
  --suite all \
  --validation-id full_longrun_regression
```

## Real LLM/DEVSIM Soak

For an overnight real-tool validation, pass a real autonomous-agent request. The request is merged into the default real scenario, so it can point at a real user deck, real run roots, objectives, constraints, and LLM settings.

```bash
python3.11 -m tcad_agent.tools.long_run_validation \
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
