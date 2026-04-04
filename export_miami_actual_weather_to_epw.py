from __future__ import annotations

import csv
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests


ROOT = Path(__file__).resolve().parent
WEATHER_DIR = ROOT / "weather"
WEATHER_DIR.mkdir(parents=True, exist_ok=True)


ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"

LOCATION_NAME = "miami"
CITY = "Miami"
STATE = "Florida"
COUNTRY = "USA"
LATITUDE = 25.7617
LONGITUDE = -80.1918
LOCAL_CIVIL_TIMEZONE = "America/New_York"
LOCAL_STANDARD_OFFSET_HOURS = -5.0
WMO_ID = "722020"

START_DATE = "2025-06-01"
END_DATE = "2025-09-30"

# Fetch one extra day at the end so the EPW can include the final hour-24 record.
EPW_FETCH_END_DATE = (date.fromisoformat(END_DATE) + timedelta(days=1)).isoformat()

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

CSV_COLUMNS = [
    "source_api",
    "time_utc",
    "time_local",
    "time_local_standard",
    *HOURLY_VARS,
]

SIGMA = 5.6697e-8
SOLAR_CONSTANT = 1367.0


def request_archive_weather() -> tuple[dict, list[dict[str, object]]]:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": EPW_FETCH_END_DATE,
        "timezone": "GMT",
        "wind_speed_unit": "ms",
        "hourly": ",".join(HOURLY_VARS),
    }
    response = requests.get(ARCHIVE_API_URL, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()

    hourly = payload["hourly"]
    time_utc_series = [
        datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        for ts in hourly["time"]
    ]
    civil_tz = ZoneInfo(LOCAL_CIVIL_TIMEZONE)
    standard_tz = timezone(timedelta(hours=LOCAL_STANDARD_OFFSET_HOURS))

    rows: list[dict[str, object]] = []
    for idx, time_utc in enumerate(time_utc_series):
        row: dict[str, object] = {
            "source_api": "historical_weather_api",
            "time_utc": time_utc.replace(tzinfo=None),
            "time_local": time_utc.astimezone(civil_tz).replace(tzinfo=None),
            "time_local_standard": time_utc.astimezone(standard_tz).replace(tzinfo=None),
        }
        for field in HOURLY_VARS:
            row[field] = hourly[field][idx]
        rows.append(row)
    return payload, rows


def clamp(value: float | int | None, lower: float, upper: float) -> float | int | None:
    if value is None:
        return None
    return min(max(value, lower), upper)


def build_csv_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    start_ts = datetime.combine(date.fromisoformat(START_DATE), time.min)
    end_ts = datetime.combine(date.fromisoformat(END_DATE) + timedelta(days=1), time.min)

    csv_rows: list[dict[str, object]] = []
    for row in rows:
        local_time = row["time_local"]
        if not isinstance(local_time, datetime):
            continue
        if start_ts <= local_time < end_ts:
            csv_row = dict(row)
            for key in ("time_utc", "time_local", "time_local_standard"):
                csv_row[key] = csv_row[key].isoformat(sep=" ")
            csv_rows.append(csv_row)
    return csv_rows


def to_epw_record_time(local_standard_ts: datetime) -> tuple[date, int]:
    if local_standard_ts.hour == 0:
        return (local_standard_ts - timedelta(days=1)).date(), 24
    return local_standard_ts.date(), local_standard_ts.hour


def solar_geometry(local_standard_ts: datetime) -> tuple[int, float]:
    midpoint = local_standard_ts - timedelta(minutes=30)
    day_of_year = midpoint.timetuple().tm_yday
    decimal_hour = midpoint.hour + midpoint.minute / 60.0 + midpoint.second / 3600.0

    b_angle = math.radians((360.0 / 364.0) * (day_of_year - 81))
    equation_of_time = (
        9.87 * math.sin(2.0 * b_angle)
        - 7.53 * math.cos(b_angle)
        - 1.5 * math.sin(b_angle)
    )
    local_standard_meridian = 15.0 * LOCAL_STANDARD_OFFSET_HOURS
    solar_time = decimal_hour + (
        4.0 * (LONGITUDE - local_standard_meridian) + equation_of_time
    ) / 60.0
    hour_angle = math.radians(15.0 * (solar_time - 12.0))
    declination = math.radians(
        23.45 * math.sin(math.radians(360.0 * (284 + day_of_year) / 365.0))
    )
    latitude = math.radians(LATITUDE)

    cos_zenith = (
        math.sin(latitude) * math.sin(declination)
        + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle)
    )
    cos_zenith = max(min(cos_zenith, 1.0), 0.0)
    return day_of_year, cos_zenith


def extraterrestrial_radiation(local_standard_ts: datetime) -> tuple[int, int]:
    day_of_year, cos_zenith = solar_geometry(local_standard_ts)
    direct_normal = SOLAR_CONSTANT * (
        1.0 + 0.033 * math.cos(math.radians(360.0 * day_of_year / 365.0))
    )
    if cos_zenith <= 0.0:
        return 0, 0
    horizontal = direct_normal * cos_zenith
    return int(round(horizontal)), int(round(direct_normal))


def calculate_horizontal_ir(
    dry_bulb_c: float | None,
    dew_point_c: float | None,
    opaque_sky_cover: int,
) -> int:
    if dry_bulb_c is None or dew_point_c is None:
        return 9999

    dry_bulb_k = dry_bulb_c + 273.15
    dew_point_k = dew_point_c + 273.15
    clear_sky_emissivity = 0.787 + 0.764 * math.log(dew_point_k / 273.15)
    sky_emissivity = clear_sky_emissivity * (
        1.0
        + 0.0224 * opaque_sky_cover
        - 0.0035 * opaque_sky_cover**2
        + 0.00028 * opaque_sky_cover**3
    )
    horizontal_ir = sky_emissivity * SIGMA * dry_bulb_k**4
    return int(round(clamp(horizontal_ir, 0.0, 9999.0)))


def compute_days_since_last_snowfall(working: list[dict[str, object]]) -> list[int]:
    last_snow_date: date | None = None
    days_since: list[int] = []
    for row in working:
        snowfall = row["snowfall"]
        epw_date = row["epw_date"]
        if not isinstance(epw_date, date):
            days_since.append(99)
            continue
        if isinstance(snowfall, (int, float)) and snowfall > 0.0:
            last_snow_date = epw_date
            days_since.append(0)
            continue
        if last_snow_date is None:
            days_since.append(99)
            continue
        days_since.append(min((epw_date - last_snow_date).days, 99))
    return days_since


def build_epw_rows(rows: list[dict[str, object]]) -> list[list[object]]:
    start_day = date.fromisoformat(START_DATE)
    end_day = date.fromisoformat(END_DATE)

    working: list[dict[str, object]] = []
    for row in rows:
        local_standard = row["time_local_standard"]
        if not isinstance(local_standard, datetime):
            continue
        epw_date, epw_hour = to_epw_record_time(local_standard)
        if start_day <= epw_date <= end_day:
            enriched = dict(row)
            enriched["epw_date"] = epw_date
            enriched["epw_hour"] = epw_hour
            working.append(enriched)

    working.sort(key=lambda item: (item["epw_date"], item["epw_hour"]))

    expected_rows = ((end_day - start_day).days + 1) * 24
    if len(working) < expected_rows:
        # Pad missing hours at the end with last available row
        while len(working) < expected_rows:
            last = dict(working[-1])
            last_dt = datetime.combine(last["epw_date"], time.min) + timedelta(hours=last["epw_hour"])
            next_dt = last_dt + timedelta(hours=1)
            next_date, next_hour = to_epw_record_time(next_dt)
            last["epw_date"] = next_date
            last["epw_hour"] = next_hour
            last["time_local_standard"] = next_dt
            working.append(last)
    working = working[:expected_rows]

    days_since_last_snowfall = compute_days_since_last_snowfall(working)
    epw_rows: list[list[object]] = []

    for row, days_since_snowfall in zip(working, days_since_last_snowfall):
        cloud_cover = row["cloud_cover"]
        total_sky_cover = int(
            round(clamp((cloud_cover if isinstance(cloud_cover, (int, float)) else 99.0) / 10.0, 0.0, 10.0))
        )

        cloud_cover_low = row["cloud_cover_low"]
        cloud_cover_mid = row["cloud_cover_mid"]
        if isinstance(cloud_cover_low, (int, float)) and isinstance(cloud_cover_mid, (int, float)):
            opaque_sky_cover = int(
                round(clamp((cloud_cover_low + cloud_cover_mid) / 10.0, 0.0, float(total_sky_cover)))
            )
        else:
            opaque_sky_cover = total_sky_cover

        visibility = row["visibility"]
        visibility_km = (
            round(clamp(visibility / 1000.0, 0.0, 9999.0), 1)
            if isinstance(visibility, (int, float))
            else 9999
        )

        extraterrestrial_horizontal, extraterrestrial_direct = extraterrestrial_radiation(
            row["time_local_standard"]
        )
        horizontal_ir = calculate_horizontal_ir(
            row["temperature_2m"] if isinstance(row["temperature_2m"], (int, float)) else None,
            row["dew_point_2m"] if isinstance(row["dew_point_2m"], (int, float)) else None,
            opaque_sky_cover,
        )

        precipitable_water = row["total_column_integrated_water_vapour"]
        if isinstance(precipitable_water, (int, float)):
            precipitable_water_value = int(round(clamp(precipitable_water, 0.0, 999.0)))
        else:
            precipitable_water_value = 999

        albedo = 999
        liquid_precipitation_depth = row["rain"]
        if isinstance(liquid_precipitation_depth, (int, float)):
            liquid_precipitation_depth_value = round(
                clamp(liquid_precipitation_depth, 0.0, 999.0),
                1,
            )
            liquid_precipitation_quantity = 1.0 if liquid_precipitation_depth_value > 0.0 else 0.0
        else:
            liquid_precipitation_depth_value = 999
            liquid_precipitation_quantity = 99

        snow_depth = row["snow_depth"]
        if isinstance(snow_depth, (int, float)):
            snow_depth_cm = int(round(clamp(snow_depth * 100.0, 0.0, 999.0)))
        else:
            snow_depth_cm = 999

        epw_date = row["epw_date"]
        epw_rows.append(
            [
                epw_date.year,
                epw_date.month,
                epw_date.day,
                row["epw_hour"],
                0,
                "",
                round(row["temperature_2m"], 1) if isinstance(row["temperature_2m"], (int, float)) else 99.9,
                round(row["dew_point_2m"], 1) if isinstance(row["dew_point_2m"], (int, float)) else 99.9,
                int(round(clamp(row["relative_humidity_2m"], 0.0, 110.0))) if isinstance(row["relative_humidity_2m"], (int, float)) else 999,
                int(round(clamp(row["surface_pressure"] * 100.0, 31000.0, 120000.0))) if isinstance(row["surface_pressure"], (int, float)) else 999999,
                extraterrestrial_horizontal,
                extraterrestrial_direct,
                horizontal_ir,
                int(round(clamp(row["shortwave_radiation"], 0.0, 9999.0))) if isinstance(row["shortwave_radiation"], (int, float)) else 9999,
                int(round(clamp(row["direct_normal_irradiance"], 0.0, 9999.0))) if isinstance(row["direct_normal_irradiance"], (int, float)) else 9999,
                int(round(clamp(row["diffuse_radiation"], 0.0, 9999.0))) if isinstance(row["diffuse_radiation"], (int, float)) else 9999,
                999999,
                999999,
                999999,
                9999,
                int(round(clamp(row["wind_direction_10m"], 0.0, 360.0))) if isinstance(row["wind_direction_10m"], (int, float)) else 999,
                round(clamp(row["wind_speed_10m"], 0.0, 40.0), 1) if isinstance(row["wind_speed_10m"], (int, float)) else 999,
                total_sky_cover,
                opaque_sky_cover,
                visibility_km,
                99999,
                9,
                "999999999",
                precipitable_water_value,
                ".999",
                snow_depth_cm,
                days_since_snowfall,
                albedo,
                liquid_precipitation_depth_value,
                liquid_precipitation_quantity,
            ]
        )
    return epw_rows


def build_epw_header(elevation_m: float) -> list[str]:
    start_day = date.fromisoformat(START_DATE)
    end_day = date.fromisoformat(END_DATE)
    return [
        f"LOCATION,{CITY},{STATE},{COUNTRY},Open-Meteo Archive API,{WMO_ID},{LATITUDE:.4f},{LONGITUDE:.4f},{LOCAL_STANDARD_OFFSET_HOURS:.1f},{elevation_m:.1f}",
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVING,No,0,0,0",
        "COMMENTS 1,Open-Meteo archive-api historical weather for Miami Florida",
        "COMMENTS 2,Partial-year EPW 2025-06-01 to 2025-09-30 Built only from Historical Weather API fields",
        f"DATA PERIODS,1,1,Data,{start_day.strftime('%A')},{start_day.month}/{start_day.day},{end_day.month}/{end_day.day}",
    ]


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_epw(epw_rows: list[list[object]], path: Path, elevation_m: float) -> None:
    header_lines = build_epw_header(elevation_m)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for line in header_lines:
            handle.write(line + "\n")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(epw_rows)


def load_from_local_csv(csv_path: Path) -> tuple[dict, list[dict[str, object]]]:
    """Load weather data from already-downloaded CSV instead of API call."""
    from zoneinfo import ZoneInfo

    civil_tz = ZoneInfo(LOCAL_CIVIL_TIMEZONE)
    standard_tz = timezone(timedelta(hours=LOCAL_STANDARD_OFFSET_HOURS))
    rows: list[dict[str, object]] = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            time_utc = datetime.fromisoformat(raw["time_utc"].strip()).replace(tzinfo=timezone.utc)
            row: dict[str, object] = {
                "source_api": "historical_weather_api",
                "time_utc": time_utc.replace(tzinfo=None),
                "time_local": time_utc.astimezone(civil_tz).replace(tzinfo=None),
                "time_local_standard": time_utc.astimezone(standard_tz).replace(tzinfo=None),
            }
            for field in HOURLY_VARS:
                val = raw.get(field, "")
                if val == "" or val is None:
                    row[field] = None
                else:
                    try:
                        row[field] = float(val)
                    except (ValueError, TypeError):
                        row[field] = None
            rows.append(row)

    # Estimate elevation (Miami ~2m)
    payload = {"elevation": 2.0}
    return payload, rows


def main() -> None:
    local_csv = WEATHER_DIR / f"{LOCATION_NAME}_{START_DATE.replace('-','_')}_{END_DATE.replace('-','_')}_historical_weather_api.csv"
    if local_csv.exists():
        print(f"Loading from local CSV: {local_csv}")
        payload, raw_rows = load_from_local_csv(local_csv)
    else:
        print("Fetching from Open-Meteo API...")
        payload, raw_rows = request_archive_weather()
    csv_rows = build_csv_rows(raw_rows)
    epw_rows = build_epw_rows(raw_rows)

    window_tag = f"{START_DATE.replace('-', '_')}_{END_DATE.replace('-', '_')}"
    csv_path = WEATHER_DIR / f"{LOCATION_NAME}_{window_tag}_historical_weather_api.csv"
    epw_path = WEATHER_DIR / f"{LOCATION_NAME}_{window_tag}_historical_weather_api.epw"

    write_csv(csv_rows, csv_path)
    write_epw(epw_rows, epw_path, float(payload.get("elevation", 0.0)))

    print(f"Saved CSV: {csv_path}")
    print(f"Saved EPW: {epw_path}")
    print(f"CSV rows: {len(csv_rows):,}, columns: {len(CSV_COLUMNS)}")
    print(f"EPW rows: {len(epw_rows):,}")
    print(f"First CSV local time: {csv_rows[0]['time_local']}")
    print(f"Last CSV local time:  {csv_rows[-1]['time_local']}")
    print(f"First EPW row: {epw_rows[0][0:5]}")
    print(f"Last EPW row:  {epw_rows[-1][0:5]}")


if __name__ == "__main__":
    main()
