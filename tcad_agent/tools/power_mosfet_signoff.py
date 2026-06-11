from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.power_mosfet_signoff import PowerMOSFETSignoffRequest, run_power_mosfet_signoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Power MOSFET/LDMOS 2D signoff evidence workflow.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--baseline-request-json", default=None)
    parser.add_argument("--reference-curve-path", type=Path, default=None)
    parser.add_argument("--convergence-value", type=float, action="append", default=[])
    parser.add_argument("--convergence-relative-tolerance", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--no-convergence", action="store_true")
    parser.add_argument("--no-execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        baseline = json.loads(args.baseline_request_json) if args.baseline_request_json else {}
        state = run_power_mosfet_signoff(
            PowerMOSFETSignoffRequest(
                run_id=args.run_id,
                run_root=args.run_root or PowerMOSFETSignoffRequest.model_fields["run_root"].default,
                execute=not args.no_execute,
                baseline_request=baseline,
                run_convergence=not args.no_convergence,
                convergence_values=args.convergence_value or PowerMOSFETSignoffRequest.model_fields["convergence_values"].default_factory(),
                convergence_relative_tolerance=args.convergence_relative_tolerance,
                reference_curve_path=args.reference_curve_path,
                timeout_seconds=args.timeout_seconds,
            )
        )
        print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if state.status in {"completed", "planned"} else 1)
    except Exception as exc:
        print(json.dumps({"tool_name": "power_mosfet_signoff", "status": "failed", "failure_reason": str(exc)}, indent=2, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()

