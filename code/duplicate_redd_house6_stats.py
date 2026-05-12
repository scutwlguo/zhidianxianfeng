from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


COPY_COUNT = 28
SOURCE_HOUSE_ID = 6
FIRST_TARGET_HOUSE_ID = 7
KG_FILE_RE = re.compile(r"^REDD_House(\d+)_stats\.json$")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_copy_jobs(root: Path) -> list[tuple[Path, Path]]:
    data_root = root / "data"
    source_specs = [
        data_root / "用电行为分析_json" / "REDD" / "REDD_House6_stats",
        data_root / "all_datasets" / "REDD" / "REDD_House6_stats",
        data_root / "kg_export" / "REDD" / "REDD_House6_stats.json",
    ]

    jobs: list[tuple[Path, Path]] = []
    for source in source_specs:
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        parent = source.parent
        suffix = source.suffix
        for house_id in range(
            FIRST_TARGET_HOUSE_ID,
            FIRST_TARGET_HOUSE_ID + COPY_COUNT,
        ):
            target_name = f"REDD_House{house_id}_stats{suffix}"
            jobs.append((source, parent / target_name))

    return jobs


def copy_one(source: Path, target: Path, overwrite: bool, dry_run: bool) -> None:
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target already exists: {target}\n"
                "Use --overwrite if you really want to replace it."
            )
        if not dry_run:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

    action = "DRY-RUN" if dry_run else "COPY"
    print(f"[{action}] {source} -> {target}")

    if dry_run:
        return

    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
        update_kg_user_name(target)


def update_kg_user_name(path: Path) -> None:
    house_match = KG_FILE_RE.match(path.name)
    if not house_match:
        return

    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    user = payload.get("user")
    if not isinstance(user, dict):
        return

    user["name"] = f"用户{house_match.group(1)}"

    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Duplicate REDD_House6_stats in three REDD data directories "
            "as REDD_House7_stats through REDD_House34_stats."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned copy operations without creating files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing target files or folders.",
    )
    args = parser.parse_args()

    jobs = build_copy_jobs(project_root())
    for source, target in jobs:
        copy_one(source, target, overwrite=args.overwrite, dry_run=args.dry_run)

    print(f"Done. Prepared {len(jobs)} copy operations.")


if __name__ == "__main__":
    main()
