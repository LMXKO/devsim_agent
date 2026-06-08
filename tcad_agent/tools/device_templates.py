from __future__ import annotations

import argparse
import json

from tcad_agent.device_templates import TemplateSupport, list_device_templates, route_device_goal
from tcad_agent.public_sources import list_public_tcad_categories, list_public_tcad_sources, validate_public_tcad_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List and route TCAD device task templates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List known device templates.")
    list_parser.add_argument("--support", choices=[item.value for item in TemplateSupport], default=None)

    route_parser = subparsers.add_parser("route", help="Route a natural-language goal to a device template.")
    route_parser.add_argument("--goal", required=True)

    sources_parser = subparsers.add_parser("sources", help="List public TCAD source categories and references.")
    sources_parser.add_argument("--kind", choices=["categories", "sources", "all"], default="all")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "list":
        support = TemplateSupport(args.support) if args.support else None
        output = {"templates": list_device_templates(support=support)}
    elif args.command == "route":
        output = route_device_goal(args.goal).model_dump(mode="json")
    elif args.command == "sources":
        output = {"registry_errors": validate_public_tcad_registry()}
        if args.kind in {"categories", "all"}:
            output["categories"] = list_public_tcad_categories()
        if args.kind in {"sources", "all"}:
            output["sources"] = list_public_tcad_sources()
    else:
        raise ValueError(f"unknown command: {args.command}")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
