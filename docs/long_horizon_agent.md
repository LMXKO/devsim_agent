# Long-Horizon Agent Policy

`tcad_agent.long_horizon_agent` provides the persistent control policy for long TCAD missions.

The policy consumes:

- the user's original goal;
- `checkpoint.engineering_intent`;
- the latest mission observation;
- soft failures and blocked goal steps;
- current replan budget;
- physical benchmark and repair state;
- tool-convergence status and evidence matrix.

It produces:

- action: continue, replan, repair_or_verify, continue_with_risk, ask_user;
- Chinese reason text for UI display;
- risk level;
- required and missing evidence;
- risk-ledger updates;
- remaining replan budget.

The policy is deliberately conservative:

- planned verification and repair steps run before generic replanning;
- optional tool-convergence failures can trigger replanning early so the agent can mark them as non-blocking and continue with risk;
- if replan budget is exhausted and a required goal step remains blocked, the agent asks the user;
- if the mission reaches conclusion with missing evidence, it continues only with explicit risk notation.
