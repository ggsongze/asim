from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gspo_houston_bandit import PROJECT_ROOT, RESULT_DIR, HoustonGSPOBandit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect 8-zone Houston observation windows for GSPO training.")
    parser.add_argument(
        "--count",
        type=int,
        default=8,
        help="Number of prompt states to collect. Use 0 to collect the whole run that matches the filters.",
    )
    parser.add_argument(
        "--skip-start",
        type=int,
        default=0,
        help="Starting skip_valid_steps value.",
    )
    parser.add_argument(
        "--skip-step",
        type=int,
        default=12,
        help="Increment in valid timesteps between samples.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=RESULT_DIR / "houston_gspo_dataset.jsonl",
        help="Output JSONL dataset path.",
    )
    parser.add_argument(
        "--request-mode",
        type=str,
        default="step_action",
        choices=("step_action", "workday_policy"),
        help="Prompt format: single-step action or compact workday closed-loop policy.",
    )
    parser.add_argument(
        "--window-start",
        type=str,
        default="",
        help="Optional control window start in HH:MM, e.g. 06:30.",
    )
    parser.add_argument(
        "--window-end",
        type=str,
        default="",
        help="Optional control window end in HH:MM, e.g. 19:00.",
    )
    parser.add_argument(
        "--weekday-only",
        action="store_true",
        help="Keep only Monday-Friday samples.",
    )
    parser.add_argument(
        "--day-start-only",
        action="store_true",
        help="Keep only the first valid control step of each day.",
    )
    parser.add_argument(
        "--min-max-occupancy",
        type=float,
        default=0.0,
        help="Keep only samples whose maximum zone occupancy is at least this value.",
    )
    parser.add_argument(
        "--min-net-grid-kwh",
        type=float,
        default=0.0,
        help="Keep only samples whose payload net_grid_kwh is at least this value.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum number of sample attempts. Defaults to count * 200.",
    )
    return parser.parse_args()


def row_passes_filters(
    row: dict[str, Any],
    *,
    min_max_occupancy: float,
    min_net_grid_kwh: float,
) -> bool:
    payload = row.get("payload", {})
    zones = payload.get("zones", [])
    max_occupancy = max((float(zone.get("occupancy", 0.0)) for zone in zones), default=0.0)
    net_grid_kwh = float(payload.get("net_grid_kwh", 0.0))
    return (
        max_occupancy >= float(min_max_occupancy)
        and net_grid_kwh >= float(min_net_grid_kwh)
    )


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    bandit = HoustonGSPOBandit(
        include_forecast=True,
        control_window_start=args.window_start or None,
        control_window_end=args.window_end or None,
        weekday_only=args.weekday_only,
        request_mode=args.request_mode,
    )
    seen_dates: set[str] = set()

    def combined_row_filter(row: dict[str, Any]) -> bool:
        if not row_passes_filters(
            row,
            min_max_occupancy=args.min_max_occupancy,
            min_net_grid_kwh=args.min_net_grid_kwh,
        ):
            return False
        if not args.day_start_only:
            return True
        day_key = str(row["wallclock"]).split(" ", 1)[0]
        if day_key in seen_dates:
            return False
        seen_dates.add(day_key)
        return True

    collected = bandit.collect_states(
        count=(None if int(args.count) <= 0 else int(args.count)),
        skip_start=args.skip_start,
        skip_step=args.skip_step,
        row_filter=combined_row_filter,
    )
    rows = list(collected["rows"])
    valid_steps_seen = int(collected["valid_steps_seen"])

    if int(args.count) > 0 and len(rows) < int(args.count):
        raise RuntimeError(
            f"Collected only {len(rows)} samples after scanning {valid_steps_seen} valid steps; "
            f"filters may be too strict (min_max_occupancy={args.min_max_occupancy}, "
            f"min_net_grid_kwh={args.min_net_grid_kwh})."
        )

    with args.output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = args.output_path.with_suffix(".summary.json")
    summary = {
        "dataset_path": str(args.output_path),
        "count": len(rows),
        "zone_ids": list(bandit.zone_ids),
        "skip_start": args.skip_start,
        "skip_step": args.skip_step,
        "attempts": valid_steps_seen,
        "valid_steps_seen": valid_steps_seen,
        "window_start": args.window_start or None,
        "window_end": args.window_end or None,
        "weekday_only": bool(args.weekday_only),
        "day_start_only": bool(args.day_start_only),
        "request_mode": args.request_mode,
        "min_max_occupancy": args.min_max_occupancy,
        "min_net_grid_kwh": args.min_net_grid_kwh,
        "first_wallclock": rows[0]["wallclock"] if rows else None,
        "last_wallclock": rows[-1]["wallclock"] if rows else None,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
