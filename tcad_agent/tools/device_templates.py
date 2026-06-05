from __future__ import annotations

import argparse
import json

from tcad_agent.device_templates import TemplateSupport, list_device_templates, route_device_goal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List and route TCAD device task templates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List known device templates.")
    list_parser.add_argument("--support", choices=[item.value for item in TemplateSupport], default=None)

    route_parser = subparsers.add_parser("route", help="Route a natural-language goal to a device template.")
    route_parser.add_argument("--goal", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "list":
        support = TemplateSupport(args.support) if args.support else None
        output = {"templates": list_device_templates(support=support)}
    elif args.command == "route":
        output = route_device_goal(args.goal).model_dump(mode="json")
    else:
        raise ValueError(f"unknown command: {args.command}")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
