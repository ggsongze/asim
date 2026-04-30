#!/usr/bin/env python3
"""Download Miami weather data from Open-Meteo APIs matching Houston format.

Downloads:
1. Historical weather (archive API) - actual observations
2. Historical forecast model runs (previous runs API) - what forecast was available at each time
"""
from __future__ import annotations

import csv
import io
import json
import time
import urllib.request
from datetime import datetime, timedelta, date
from pathlib import Path

WEATHER_DIR = Path("/home/songze/asim/weather")

# Miami coordinates
LATITUDE = 25.7617
LONGITUDE = -80.1918
TIMEZONE = "America/New_York"
LOCAL_STANDARD_OFFSET_HOURS = -5.0

START_DATE = "2025-06-01"
END_DATE = "2025-09-30"

# --- Historical Weather (Archive API) ---
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "surface_pressure",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "visibility",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "weather_code",
    "wet_bulb_temperature_2m",
    "total_column_integrated_water_vapour",
    "is_day",
    "sunshine_duration",
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm",
    "soil_temperature_100_to_255cm",
]

# --- Forecast Model Runs (Previous Runs API) ---
PREVIOUS_RUNS_API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_VARS = [
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


def fetch_json(url: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read())
        except Exception as e:
            print(f"  Retry {attempt+1}/{max_retries}: {e}")
            time.sleep(5)
    raise RuntimeError(f"Failed to fetch {url}")


def download_historical_weather():
    """Download actual historical weather observations."""
    print("=== Downloading historical weather (archive API) ===")
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{ARCHIVE_API_URL}?{query}"
    print(f"  URL: {url[:120]}...")
    data = fetch_json(url)

    hourly = data["hourly"]
    times = hourly["time"]
    n = len(times)
    print(f"  Got {n} hourly records")

    out_path = WEATHER_DIR / f"miami_{START_DATE.replace('-','_')}_{END_DATE.replace('-','_')}_historical_weather_api.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["source_api", "time_utc", "time_local", "time_local_standard"] + HOURLY_VARS
        writer.writerow(header)
        for i in range(n):
            t_utc = times[i].replace("T", " ") + ":00"
            dt_utc = datetime.fromisoformat(times[i])
            dt_local = dt_utc + timedelta(hours=LOCAL_STANDARD_OFFSET_HOURS)
            # EDT in summer = UTC-4, but we use standard offset for consistency
            t_local = dt_local.strftime("%Y-%m-%d %H:%M:%S")
            t_local_std = (dt_utc + timedelta(hours=LOCAL_STANDARD_OFFSET_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
            row = ["historical_weather_api", t_utc, t_local, t_local_std]
            for var in HOURLY_VARS:
                val = hourly[var][i]
                row.append("" if val is None else val)
            writer.writerow(row)

    print(f"  Saved to {out_path}")
    return out_path


def download_forecast_model_runs():
    """Download historical forecast model runs (what was predicted at each point in time)."""
    print("=== Downloading forecast model runs (previous runs API) ===")

    # Download in monthly chunks to avoid API limits
    all_rows = []
    start = date.fromisoformat(START_DATE)
    end = date.fromisoformat(END_DATE)

    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=30), end)
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": current.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": ",".join(FORECAST_VARS),
            "timezone": "UTC",
            "models": "ncep_hrrr_conus",
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{PREVIOUS_RUNS_API_URL}?{query}"
        print(f"  Fetching {current} to {chunk_end}...")

        try:
            data = fetch_json(url)
        except Exception as e:
            print(f"  ERROR: {e}, skipping chunk")
            current = chunk_end + timedelta(days=1)
            continue

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        print(f"    Got {len(times)} records")

        for i in range(len(times)):
            t_utc = times[i].replace("T", " ")
            if len(t_utc) == 16:
                t_utc += ":00"
            dt_utc = datetime.fromisoformat(times[i])
            dt_local = dt_utc + timedelta(hours=LOCAL_STANDARD_OFFSET_HOURS)

            row = {
                "source_api": "single_runs_api",
                "model": "ncep_hrrr_conus",
                "run_time_utc": t_utc,
                "run_time": dt_local.strftime("%Y-%m-%d %H:%M:%S"),
                "target_time_utc": t_utc,
                "target_time": dt_local.strftime("%Y-%m-%d %H:%M:%S"),
                "lead_hours": 0,
            }
            for var in FORECAST_VARS:
                val = hourly.get(var, [None] * len(times))[i]
                row[var] = "" if val is None else val
            all_rows.append(row)

        current = chunk_end + timedelta(days=1)
        time.sleep(1)  # Rate limiting

    # Write raw CSV
    raw_path = WEATHER_DIR / f"miami_{START_DATE.replace('-','_')}_{END_DATE.replace('-','_')}_hourly_model_runs_api_raw.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(raw_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  Saved {len(all_rows)} rows to {raw_path}")
    else:
        print("  WARNING: No forecast data retrieved!")

    # Also create the label_h6 format (6-hour ahead forecast labels)
    label_path = WEATHER_DIR / f"miami_{START_DATE.replace('-','_')}_{END_DATE.replace('-','_')}_hourly_model_runs_api_label_h6.csv"
    if all_rows:
        with open(label_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  Saved label_h6 to {label_path}")

    return raw_path, label_path


def main():
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = download_historical_weather()
    raw_path, label_path = download_forecast_model_runs()
    print("\n=== Done ===")
    print(f"Historical weather: {hist_path}")
    print(f"Forecast raw: {raw_path}")
    print(f"Forecast label_h6: {label_path}")


if __name__ == "__main__":
    main()
