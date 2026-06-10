from __future__ import annotations

import csv
import json
from pathlib import Path


def read_manifest(project_dir: Path) -> dict:
    path = project_dir / "actsoft_sentaurus_contract.json"
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def fake_rows(manifest: dict) -> list[dict[str, str]]:
    backend = manifest.get("fake_backend") if isinstance(manifest.get("fake_backend"), dict) else {}
    rows = backend.get("curve_rows") if isinstance(backend.get("curve_rows"), list) else None
    if rows:
        return [{str(key): str(value) for key, value in row.items()} for row in rows if isinstance(row, dict)]
    return [
        {"voltage_v": "0", "current_a": "1e-12", "electric_field_v_per_cm": "1e4"},
        {"voltage_v": "-10", "current_a": "1e-9", "electric_field_v_per_cm": "2e5"},
        {"voltage_v": "-20", "current_a": "1e-6", "electric_field_v_per_cm": "8e5"},
    ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_dir = Path.cwd()
    manifest = read_manifest(project_dir)
    backend = manifest.get("fake_backend") if isinstance(manifest.get("fake_backend"), dict) else {}
    log_name = str(backend.get("log_name") or "contract_des.log")
    csv_name = str(backend.get("csv_name") or "sentaurus_contract_extract.csv")
    plt_name = str(backend.get("plt_name") or "contract_des.plt")
    (project_dir / log_name).write_text(
        "ACTSOFT SENTaurus CONTRACT FAKE BACKEND\n"
        "interface_contract_only=true\n"
        "This output validates agent IO only and is not a Sentaurus physics result.\n"
        "Sentaurus Device finished\n",
        encoding="utf-8",
    )
    (project_dir / plt_name).write_text(
        "ACTSOFT interface placeholder for artifact collection only.\n",
        encoding="utf-8",
    )
    write_csv(project_dir / csv_name, fake_rows(manifest))
    print(json.dumps({"status": "completed", "interface_contract_only": True, "csv": csv_name}, ensure_ascii=False))


if __name__ == "__main__":
    main()

