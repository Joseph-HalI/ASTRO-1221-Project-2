import argparse
import os
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from skyfield.api import Loader, wgs84, N, W
from skyfield import almanac
from skyfield.framelib import ecliptic_frame


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = "./data"
SKYFIELD_CACHE = "./skyfield-data"   # Where de421.bsp will be cached

# Default observer location (Columbus, OH — change via CLI args or edit here)
DEFAULT_LAT  =  39.9612
DEFAULT_LON  = -82.9988
DEFAULT_ELEV =  288      # metres above sea level

# Supermoon threshold: full/new moon within this many km of perigee
SUPERMOON_PERIGEE_KM = 362_000


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup():
    # Load Skyfield timescale + ephemeris. Downloads de421.bsp if needed.
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SKYFIELD_CACHE, exist_ok=True)

    load = Loader(SKYFIELD_CACHE, verbose=True)
    ts  = load.timescale()
    eph = load("de421.bsp")          # ~17 MB, cached after first download
    return ts, eph


# ---------------------------------------------------------------------------
# Daily lunar data
# ---------------------------------------------------------------------------

def collect_daily_lunar_data(ts, eph, year: int, lat: float, lon: float, elev: float) -> pd.DataFrame:
    """
    For every day of `year`, calculate:
        - Moon phase angle (0–360°)
        - Illumination percentage
        - Phase name  (New / Waxing Crescent / First Quarter / … / Waning Crescent)
        - Moon rise & set times (local UTC offset stored separately)
        - Moon altitude at midnight  (proxy for sky darkness)
        - Distance from Earth centre (km)  — needed for supermoon detection
    """
    print(f"\n[1/4] Collecting daily lunar data for {year}...")

    earth = eph["earth"]
    moon  = eph["moon"]
    sun   = eph["sun"]
    observer = earth + wgs84.latlon(lat, lon, elevation_m=elev)

    records = []
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end   = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    day   = start

    while day < end:
        # Skyfield time at noon UTC for this day
        t = ts.from_datetime(day.replace(hour=12))

        # --- Illumination & phase angle ---
        astrometric_moon = observer.at(t).observe(moon).apparent()
        astrometric_sun  = observer.at(t).observe(sun).apparent()

        # Phase angle: angle Sun–Moon–Earth (0° = new, 180° = full)
        sun_pos  = astrometric_sun.position.km
        moon_pos = astrometric_moon.position.km

        # Vector from moon to sun and moon to earth
        to_sun   = sun_pos - moon_pos
        to_earth = -moon_pos

        cos_angle = np.dot(to_sun, to_earth) / (
            np.linalg.norm(to_sun) * np.linalg.norm(to_earth)
        )
        phase_angle_deg = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))

        # Illumination fraction (0–100%)
        illumination = (1 + math.cos(math.radians(phase_angle_deg))) / 2 * 100

        # Moon distance from Earth centre (km)
        _, _, distance = astrometric_moon.radec()
        distance_km = astrometric_moon.distance().km

        # --- Phase name (based on ecliptic longitude difference) ---
        moon_ecl = astrometric_moon.frame_latlon(ecliptic_frame)
        sun_ecl  = astrometric_sun.frame_latlon(ecliptic_frame)
        elongation = (moon_ecl[1].degrees - sun_ecl[1].degrees) % 360
        phase_name = _phase_name(elongation)

        # --- Rise / Set times ---
        t0 = ts.from_datetime(day.replace(hour=0))
        t1 = ts.from_datetime(day.replace(hour=23, minute=59))

        rise_time, set_time = _find_rise_set(ts, eph, observer, t0, t1)

        records.append({
            "date":           day.strftime("%Y-%m-%d"),
            "phase_angle":    round(phase_angle_deg, 2),
            "illumination":   round(illumination, 2),
            "phase_name":     phase_name,
            "elongation_deg": round(elongation, 2),
            "distance_km":    round(distance_km, 0),
            "rise_utc":       rise_time,
            "set_utc":        set_time,
        })

        day += timedelta(days=1)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    print(f"    → {len(df)} daily records collected.")
    return df


def _phase_name(elongation: float) -> str:
    """Map ecliptic elongation (0–360°) to a phase name."""
    if elongation < 22.5 or elongation >= 337.5:
        return "New Moon"
    elif elongation < 67.5:
        return "Waxing Crescent"
    elif elongation < 112.5:
        return "First Quarter"
    elif elongation < 157.5:
        return "Waxing Gibbous"
    elif elongation < 202.5:
        return "Full Moon"
    elif elongation < 247.5:
        return "Waning Gibbous"
    elif elongation < 292.5:
        return "Third Quarter"
    else:
        return "Waning Crescent"


def _find_rise_set(ts, eph, observer, t0, t1):
    """Return (rise_utc_str, set_utc_str) for the moon on a given day."""
    try:
        f = almanac.risings_and_settings(eph, eph["moon"], observer)
        times, events = almanac.find_discrete(t0, t1, f)
        rise = set_ = None
        for t, e in zip(times, events):
            dt_str = t.utc_strftime("%H:%M")
            if e == 1 and rise is None:
                rise = dt_str
            elif e == 0 and set_ is None:
                set_ = dt_str
        return rise or "N/A", set_ or "N/A"
    except Exception:
        return "N/A", "N/A"


# ---------------------------------------------------------------------------
# Special events
# ---------------------------------------------------------------------------

def collect_special_events(ts, eph, year: int, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify:
        - New moons & Full moons (precise times via Skyfield almanac)
        - Supermoons  (full/new moon when distance < SUPERMOON_PERIGEE_KM)
        - Blue moons  (second full moon in a calendar month)
        - Lunar eclipses (using Skyfield's eclipse finder)
    """
    print("\n[2/4] Detecting special events...")
    events = []

    # --- Precise phase transitions ---
    t0 = ts.utc(year,  1,  1)
    t1 = ts.utc(year, 12, 31, 23, 59)

    phase_times, phase_indices = almanac.find_discrete(t0, t1, almanac.moon_phases(eph))

    # Map index → phase label (Skyfield: 0=New, 1=FirstQ, 2=Full, 3=ThirdQ)
    PHASE_LABELS = {0: "New Moon", 1: "First Quarter", 2: "Full Moon", 3: "Third Quarter"}

    full_moon_months = []   # track months with full moons for blue moon detection

    for t, idx in zip(phase_times, phase_indices):
        label = PHASE_LABELS[idx]
        dt    = t.utc_datetime()
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M UTC")

        # Distance at this moment
        earth = eph["earth"]
        astrometric = earth.at(t).observe(eph["moon"])
        dist_km = astrometric.distance().km

        is_supermoon = (
            label in ("Full Moon", "New Moon")
            and dist_km < SUPERMOON_PERIGEE_KM
        )

        event_type = label
        if is_supermoon:
            event_type = f"Super {label}"

        if label == "Full Moon":
            full_moon_months.append(dt.month)

        events.append({
            "date":       date_str,
            "time_utc":   time_str,
            "event_type": event_type,
            "distance_km": round(dist_km, 0),
            "notes":      f"{'Supermoon — Moon within ' + str(SUPERMOON_PERIGEE_KM) + ' km' if is_supermoon else ''}",
        })

    # --- Blue moons ---
    from collections import Counter
    month_counts = Counter(full_moon_months)
    for month, count in month_counts.items():
        if count >= 2:
            # Find second full moon in that month
            second = [e for e in events
                      if e["event_type"] in ("Full Moon", "Super Full Moon")
                      and datetime.strptime(e["date"], "%Y-%m-%d").month == month]
            if len(second) >= 2:
                second[1]["event_type"] = "Blue Moon"
                second[1]["notes"] = "Second full moon in a calendar month"

    # --- Lunar eclipses ---
    try:
        eclipse_times, eclipse_types = almanac.find_discrete(
            t0, t1, almanac.lunar_eclipses(eph)
        )
        ECLIPSE_LABELS = {0: "Penumbral Lunar Eclipse", 1: "Partial Lunar Eclipse", 2: "Total Lunar Eclipse (Blood Moon)"}
        for t, etype in zip(eclipse_times, eclipse_types):
            dt = t.utc_datetime()
            events.append({
                "date":        dt.strftime("%Y-%m-%d"),
                "time_utc":    dt.strftime("%H:%M UTC"),
                "event_type":  ECLIPSE_LABELS.get(etype, "Lunar Eclipse"),
                "distance_km": None,
                "notes":       "Visibility depends on your location",
            })
    except Exception as e:
        print(f"    ⚠ Eclipse calculation skipped: {e}")

    df = pd.DataFrame(events).sort_values("date").reset_index(drop=True)
    print(f"    → {len(df)} special events found.")
    return df


# ---------------------------------------------------------------------------
# Dark-sky windows
# ---------------------------------------------------------------------------

def find_dark_sky_windows(daily_df: pd.DataFrame, threshold: float = 20.0) -> pd.DataFrame:
    """
    Return contiguous runs of days where illumination < threshold (%).
    These are the best nights for stargazing / astrophotography.
    """
    print(f"\n[3/4] Identifying dark-sky windows (illumination < {threshold}%)...")

    dark = daily_df[daily_df["illumination"] < threshold].copy()
    dark = dark.reset_index()

    if dark.empty:
        print("    → No dark-sky windows found.")
        return pd.DataFrame()

    # Group consecutive dates into windows
    dark["group"] = (dark["date"].diff() > pd.Timedelta(days=1)).cumsum()
    windows = dark.groupby("group").agg(
        window_start=("date", "first"),
        window_end=("date", "last"),
        duration_days=("date", "count"),
        min_illumination=("illumination", "min"),
        avg_illumination=("illumination", "mean"),
    ).reset_index(drop=True)

    windows["avg_illumination"] = windows["avg_illumination"].round(1)
    print(f"    → {len(windows)} dark-sky windows found.")
    return windows


# ---------------------------------------------------------------------------
# Sample user events CSV
# ---------------------------------------------------------------------------

def generate_sample_user_events(year: int):
    """
    Write a template CSV of user events. Edit this file with your own dates!
    Columns: date, event_name, event_type
    """
    path = os.path.join(DATA_DIR, "sample_user_events.csv")
    if os.path.exists(path):
        print(f"\n[4/4] sample_user_events.csv already exists — skipping.")
        return

    print(f"\n[4/4] Generating sample_user_events.csv template...")
    sample = [
        {"date": f"{year}-01-01", "event_name": "New Year's Day",         "event_type": "Holiday"},
        {"date": f"{year}-02-14", "event_name": "Valentine's Day",         "event_type": "Holiday"},
        {"date": f"{year}-03-20", "event_name": "Spring Equinox",          "event_type": "Astronomical"},
        {"date": f"{year}-04-22", "event_name": "Earth Day",               "event_type": "Holiday"},
        {"date": f"{year}-05-15", "event_name": "Planned Stargazing Night","event_type": "Observation"},
        {"date": f"{year}-06-21", "event_name": "Summer Solstice",         "event_type": "Astronomical"},
        {"date": f"{year}-07-04", "event_name": "Independence Day",        "event_type": "Holiday"},
        {"date": f"{year}-08-12", "event_name": "Perseid Meteor Shower Peak","event_type": "Astronomical"},
        {"date": f"{year}-09-22", "event_name": "Autumn Equinox",          "event_type": "Astronomical"},
        {"date": f"{year}-10-31", "event_name": "Halloween",               "event_type": "Holiday"},
        {"date": f"{year}-11-17", "event_name": "Leonid Meteor Shower",    "event_type": "Astronomical"},
        {"date": f"{year}-12-21", "event_name": "Winter Solstice",         "event_type": "Astronomical"},
    ]
    pd.DataFrame(sample).to_csv(path, index=False)
    print(f"    → Saved to {path}. Edit this file to add your own events!")


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(daily_df, special_df, dark_windows_df, year):
    print("\nSaving CSVs...")

    lunar_path   = os.path.join(DATA_DIR, f"lunar_data_{year}.csv")
    special_path = os.path.join(DATA_DIR, f"special_events_{year}.csv")
    windows_path = os.path.join(DATA_DIR, f"dark_sky_windows_{year}.csv")

    daily_df.to_csv(lunar_path)
    special_df.to_csv(special_path, index=False)
    dark_windows_df.to_csv(windows_path, index=False)

    print(f"  ✓ {lunar_path}")
    print(f"  ✓ {special_path}")
    print(f"  ✓ {windows_path}")

    # Quick summary
    print(f"""
========================================
 Collection Summary — {year}
========================================
 Daily records:       {len(daily_df)}
 Special events:      {len(special_df)}
 Dark-sky windows:    {len(dark_windows_df)}
 Output directory:    {os.path.abspath(DATA_DIR)}
========================================
""")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect lunar data using Skyfield.")
    parser.add_argument("--year", type=int, default=datetime.now().year,
                        help="Year to collect data for (default: current year)")
    parser.add_argument("--lat",  type=float, default=DEFAULT_LAT,
                        help=f"Observer latitude  (default: {DEFAULT_LAT})")
    parser.add_argument("--lon",  type=float, default=DEFAULT_LON,
                        help=f"Observer longitude (default: {DEFAULT_LON})")
    parser.add_argument("--elev", type=float, default=DEFAULT_ELEV,
                        help=f"Observer elevation in metres (default: {DEFAULT_ELEV})")
    parser.add_argument("--dark-threshold", type=float, default=20.0,
                        help="Illumination %% threshold for dark-sky windows (default: 20)")
    args = parser.parse_args()

    # Check if files already exist for this year
    lunar_path   = os.path.join(DATA_DIR, f"lunar_data_{args.year}.csv")
    special_path = os.path.join(DATA_DIR, f"special_events_{args.year}.csv")
    windows_path = os.path.join(DATA_DIR, f"dark_sky_windows_{args.year}.csv")

    if all(os.path.exists(p) for p in [lunar_path, special_path, windows_path]):
        print(f"Files already downloaded for {args.year}:")
        print(f"  ✓ {lunar_path}")
        print(f"  ✓ {special_path}")
        print(f"  ✓ {windows_path}")
        print("\nTo regenerate them, delete the files above and rerun the script.")
        return

    print(f"""
Moon Phase Calendar — Data Collection
======================================
Year:      {args.year}
Location:  {args.lat}°N, {args.lon}°E  ({args.elev}m)
Source:    NASA JPL DE421 via Skyfield
""")

    ts, eph = setup()

    daily_df       = collect_daily_lunar_data(ts, eph, args.year, args.lat, args.lon, args.elev)
    special_df     = collect_special_events(ts, eph, args.year, daily_df)
    dark_windows   = find_dark_sky_windows(daily_df, threshold=args.dark_threshold)

    generate_sample_user_events(args.year)
    save_outputs(daily_df, special_df, dark_windows, args.year)

#test string ignore this
if __name__ == "__main__":
    main()