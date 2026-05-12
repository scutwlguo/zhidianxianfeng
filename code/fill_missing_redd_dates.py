from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HOUSE_RE = re.compile(r"^REDD_House(\d+)_stats$")


@dataclass(frozen=True)
class DateRoot:
    relative_dir: Path
    suffix: str
    update_json_date: bool = False


@dataclass(frozen=True)
class CopyJob:
    source: Path
    target: Path
    target_date: date
    house_dir: str
    update_json_date: bool


DATE_ROOTS = [
    DateRoot(
        Path("data") / "用电行为分析_json" / "REDD",
        ".json",
        update_json_date=True,
    ),
    DateRoot(
        Path("data") / "all_datasets" / "REDD",
        ".xlsx",
    ),
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def dated_files(house_dir: Path, suffix: str) -> dict[date, Path]:
    files: dict[date, Path] = {}
    for file_path in house_dir.glob(f"*{suffix}"):
        if file_path.is_file() and DATE_RE.match(file_path.stem):
            files[parse_date(file_path.stem)] = file_path
    return files


def nearest_existing_date(missing_day: date, existing_days: list[date]) -> date:
    # If two dates are equally close, prefer the previous day.
    return min(
        existing_days,
        key=lambda existing_day: (
            abs((existing_day - missing_day).days),
            existing_day > missing_day,
        ),
    )


def build_jobs(root: Path) -> list[CopyJob]:
    jobs: list[CopyJob] = []
    for date_root in DATE_ROOTS:
        base_dir = root / date_root.relative_dir
        if not base_dir.exists():
            raise FileNotFoundError(f"Directory not found: {base_dir}")

        house_dirs = [
            path
            for path in base_dir.iterdir()
            if path.is_dir() and HOUSE_RE.match(path.name)
        ]
        for house_dir in sorted(house_dirs, key=lambda path: int(HOUSE_RE.match(path.name).group(1))):
            files_by_date = dated_files(house_dir, date_root.suffix)
            if not files_by_date:
                print(f"[SKIP] No dated {date_root.suffix} files in {house_dir}")
                continue

            existing_days = sorted(files_by_date)
            for day in iter_days(existing_days[0], existing_days[-1]):
                if day in files_by_date:
                    continue

                source_day = nearest_existing_date(day, existing_days)
                jobs.append(
                    CopyJob(
                        source=files_by_date[source_day],
                        target=house_dir / f"{day:%Y-%m-%d}{date_root.suffix}",
                        target_date=day,
                        house_dir=house_dir.name,
                        update_json_date=date_root.update_json_date,
                    )
                )
    return jobs


def copy_json_with_date(source: Path, target: Path, target_date: date, house_dir: str) -> None:
    with source.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    payload["date"] = f"{target_date:%Y-%m-%d}"

    user = payload.get("user")
    if isinstance(user, dict):
        user["house_dir"] = house_dir
        house_match = HOUSE_RE.match(house_dir)
        if house_match:
            user["user_name"] = f"用户{house_match.group(1)}"

    with target.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def run_job(job: CopyJob, dry_run: bool, overwrite: bool) -> None:
    if job.target.exists() and not overwrite:
        raise FileExistsError(
            f"Target already exists: {job.target}\n"
            "Use --overwrite if you really want to replace it."
        )

    action = "DRY-RUN" if dry_run else "COPY"
    print(f"[{action}] {job.source.name} -> {job.target}")

    if dry_run:
        return

    if job.target.exists() and overwrite:
        job.target.unlink()

    if job.update_json_date:
        copy_json_with_date(job.source, job.target, job.target_date, job.house_dir)
    else:
        shutil.copy2(job.source, job.target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fill missing date files in REDD_House*_stats folders by copying "
            "the nearest existing date file within the same house folder."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually create missing files. Without this flag, only prints a dry run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace target files if they already exist.",
    )
    args = parser.parse_args()

    jobs = build_jobs(project_root())
    dry_run = not args.apply

    for job in jobs:
        run_job(job, dry_run=dry_run, overwrite=args.overwrite)

    mode = "dry-run" if dry_run else "created"
    print(f"Done. {mode}: {len(jobs)} missing date files.")


if __name__ == "__main__":
    main()
