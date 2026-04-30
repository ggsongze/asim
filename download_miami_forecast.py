#!/usr/bin/env python3
"""Download Miami forecast model runs from Open-Meteo single-runs API.

Matches Houston format: each run_time has lead_hours 1-6.
Then converts raw to label_h6 wide format for the ForecastBundleReader.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.request
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd

WEATHER_DIR = Path("/home/songze/asim/weather")

SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
LATITUDE = 25.7617
LONGITUDE = -80.1918
TIMEZONE = "America/New_York"
MODEL = "ncep_hrrr_conus"
START_DATE = "2025-06-01"
END_DATE = "2025-09-30"
FORECAST_HORIZON_HOURS = 6
API_FORECAST_HOURS = FORECAST_HORIZON_HOURS + 1
RUN_CYCLE_HOURS_UTC = list(range(0, 24, 1))  # every 1 hour (HRRR supports hourly runs)

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "snow_depth",
    "weathercode",
    "pressure_msl",
    "surface_pressure",
    "cloudcover",
    "cloudcover_low",
    "cloudcover_mid",
    "cloudcover_high",
    "visibility",
    "evapotranspiration",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
]


def request_json(url: str, params: dict, retries: int = 3, timeout: int = 45) -> dict:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    full_url = f"{url}?{query}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(full_url, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed after {retries} retries")


def build_run_schedule(start_date: str, end_date: str) -> list[datetime]:
    start_day = date.fromisoformat(start_date) - timedelta(days=1)
    end_day = date.fromisoformat(end_date)
    schedule = []
    current = start_day
    while current <= end_day:
        for hour in RUN_CYCLE_HOURS_UTC:
            schedule.append(datetime(current.year, current.month, current.day, hour))
        current += timedelta(days=1)
    return schedule


def fetch_single_run(run_time_utc: datetime) -> list[dict]:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(HOURLY_VARS),
        "models": MODEL,
        "run": run_time_utc.strftime("%Y-%m-%dT%H:%M"),
        "forecast_hours": API_FORECAST_HOURS,
        "timezone": "GMT",
    }
    try:
        payload = request_json(SINGLE_RUNS_URL, params)
    except Exception as e:
        return []

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return []

    rows = []
    from zoneinfo import ZoneInfo
    local_tz = ZoneInfo(TIMEZONE)

    for i, t_str in enumerate(times):
        target_utc = datetime.fromisoformat(t_str)
        lead_hours = int((target_utc - run_time_utc).total_seconds() / 3600)
        if lead_hours < 1 or lead_hours > FORECAST_HORIZON_HOURS:
            continue

        run_local = (run_time_utc.replace(tzinfo=ZoneInfo("UTC"))).astimezone(local_tz)
        target_local = (target_utc.replace(tzinfo=ZoneInfo("UTC"))).astimezone(local_tz)

        row = {
            "source_api": "single_runs_api",
            "model": MODEL,
            "run_time_utc": run_time_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "run_time": run_local.strftime("%Y-%m-%d %H:%M:%S"),
            "target_time_utc": target_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "target_time": target_local.strftime("%Y-%m-%d %H:%M:%S"),
            "lead_hours": lead_hours,
        }
        for var in HOURLY_VARS:
            vals = hourly.get(var, [])
            row[var] = vals[i] if i < len(vals) and vals[i] is not None else ""
        rows.append(row)
    return rows


def raw_to_label_h6(raw_path: Path, label_path: Path):
    """Convert raw long format to Houston-style wide format (label_h6)."""
    df = pd.read_csv(raw_path)
    if df.empty:
        print("  WARNING: empty raw file, skipping label_h6 conversion")
        return

    value_cols = [c for c in HOURLY_VARS if c in df.columns]

    # Pivot: each run_time becomes one row, columns become var_t_plus_1h ... var_t_plus_6h
    pivoted_rows = []
    for run_time, group in df.groupby("run_time_utc"):
        group = group.sort_values("lead_hours")
        row = {
            "source_api": "single_runs_api",
            "model": MODEL,
            "run_time_utc": run_time,
            "run_time": group.iloc[0]["run_time"],
        }
        for _, r in group.iterrows():
            lead = int(r["lead_hours"])
            suffix = f"_t_plus_{lead}h"
            row[f"target_time_utc{suffix}"] = r["target_time_utc"]
            row[f"target_time{suffix}"] = r["target_time"]
            for var in value_cols:
                row[f"{var}{suffix}"] = r[var]
        pivoted_rows.append(row)

    out_df = pd.DataFrame(pivoted_rows)
    out_df.to_csv(label_path, index=False)
    print(f"  label_h6: {len(out_df)} rows, {len(out_df.columns)} columns")


def main():
    schedule = build_run_schedule(START_DATE, END_DATE)
    print(f"Total run_times to fetch: {len(schedule)}")

    all_rows = []
    for i, run_time in enumerate(schedule):
        if i % 50 == 0:
            print(f"  Fetching {i}/{len(schedule)}: {run_time}...")
        rows = fetch_single_run(run_time)
        all_rows.extend(rows)
        if i % 10 == 0 and i > 0:
            time.sleep(0.5)  # Rate limiting

    # Write raw CSV
    tag = f"miami_{START_DATE.replace('-','_')}_{END_DATE.replace('-','_')}"
    raw_path = WEATHER_DIR / f"{tag}_hourly_model_runs_api_raw.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(raw_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved raw: {len(all_rows)} rows to {raw_path}")
    else:
        print("WARNING: No data collected!")
        return

    # Convert to label_h6 format
    label_path = WEATHER_DIR / f"{tag}_hourly_model_runs_api_label_h6.csv"
    raw_to_label_h6(raw_path, label_path)
    print(f"Saved label_h6: {label_path}")


if __name__ == "__main__":
    main()
