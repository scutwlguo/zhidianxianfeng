from __future__ import annotations

import json
import re
from pathlib import Path


KG_FILE_RE = re.compile(r"^REDD_House(\d+)_stats\.json$")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    kg_dir = project_root() / "data" / "kg_export" / "REDD"
    if not kg_dir.exists():
        raise FileNotFoundError(f"Directory not found: {kg_dir}")

    changed = 0
    for path in sorted(kg_dir.glob("REDD_House*_stats.json")):
        house_match = KG_FILE_RE.match(path.name)
        if not house_match:
            continue

        expected_name = f"用户{house_match.group(1)}"
        with path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)

        user = payload.get("user")
        if not isinstance(user, dict):
            print(f"[SKIP] missing user object: {path.name}")
            continue

        old_name = user.get("name")
        if old_name == expected_name:
            continue

        user["name"] = expected_name
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")

        changed += 1
        print(f"[FIX] {path.name}: {old_name!r} -> {expected_name!r}")

    print(f"Done. Updated {changed} kg_export JSON files.")


if __name__ == "__main__":
    main()
