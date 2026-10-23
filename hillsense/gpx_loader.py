"""Load a GPX file into a clean table of GPS points and split it into drives.

Phone GPS loggers are messy: points can be saved out of order, the same
second can appear twice, and elevation is sometimes 0.0 when the fix is bad.
This module fixes those problems and cuts one day of logging into separate
drives wherever there is a long gap in time.

Run it directly to see a summary of each drive:
    python -m hillsense.gpx_loader data/raw/munnar_2026-10-22.gpx
"""

import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

GPX_NAMESPACES = ("http://www.topografix.com/GPX/1/1", "http://www.topografix.com/GPX/1/0")
EARTH_RADIUS_M = 6_371_000
IST = "Asia/Kolkata"


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in metres between two lat/lon points (works on numpy arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def load_gpx(path):
    """Read every track point into a DataFrame sorted by time.

    Works with GPX 1.1 (phone apps) and GPX 1.0 (OpenStreetMap downloads).
    Points without a timestamp are skipped: we can't learn speed from them.
    """
    root = ET.parse(path).getroot()
    ns_uri = root.tag[1:].split("}")[0] if root.tag.startswith("{") else GPX_NAMESPACES[0]
    ns = {"g": ns_uri}
    rows = []
    for pt in root.iterfind(".//g:trkpt", ns):
        ele, time = pt.find("g:ele", ns), pt.find("g:time", ns)
        if time is None:
            continue
        rows.append(
            {
                "time": time.text,
                "lat": float(pt.get("lat")),
                "lon": float(pt.get("lon")),
                "ele": float(ele.text) if ele is not None else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(IST)
    df.loc[df["ele"] == 0.0, "ele"] = np.nan  # 0.0 m means "no elevation fix"
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


def add_motion(df):
    """Add step distance (m), time step (s) and speed (km/h) between points."""
    df = df.copy()
    prev_lat, prev_lon = df["lat"].shift(), df["lon"].shift()
    df["step_m"] = haversine_m(prev_lat, prev_lon, df["lat"], df["lon"]).fillna(0)
    df["dt_s"] = df["time"].diff().dt.total_seconds().fillna(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["speed_kmh"] = np.where(df["dt_s"] > 0, df["step_m"] / df["dt_s"] * 3.6, 0)
    return df


def split_drives(df, gap_minutes=10):
    """Start a new drive whenever the logger was silent for > gap_minutes."""
    df = add_motion(df)
    new_drive = df["dt_s"] > gap_minutes * 60
    df["drive"] = new_drive.cumsum() + 1
    df.loc[new_drive, ["step_m", "dt_s", "speed_kmh"]] = 0
    return df


def summarise(df):
    """One row per drive: times, distance, climb and how noisy the GPS was."""
    out = []
    for drive_id, d in df.groupby("drive"):
        moving = d[(d["speed_kmh"] >= 3.6) & (d["dt_s"] <= 30)]
        out.append(
            {
                "drive": drive_id,
                "start": d["time"].iloc[0].strftime("%H:%M"),
                "end": d["time"].iloc[-1].strftime("%H:%M"),
                "minutes": round((d["time"].iloc[-1] - d["time"].iloc[0]).total_seconds() / 60, 1),
                "points": len(d),
                "raw_km": round(d["step_m"].sum() / 1000, 2),
                "ele_min_m": round(d["ele"].min()) if d["ele"].notna().any() else None,
                "ele_max_m": round(d["ele"].max()) if d["ele"].notna().any() else None,
                "moving_kmh": round(moving["step_m"].sum() / moving["dt_s"].sum() * 3.6, 1),
                "gaps_over_30s": int((d["dt_s"] > 30).sum()),
                "impossible_jumps": int((d["speed_kmh"] > 150).sum()),
            }
        )
    return pd.DataFrame(out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/munnar_2026-10-22.gpx"
    points = split_drives(load_gpx(path))
    print(f"Loaded {len(points)} points from {path}\n")
    pd.set_option("display.width", 200)
    print(summarise(points).to_string(index=False))
