"""Pre-compute everything the Streamlit demo shows, into one small JSON file.

The demo runs on Streamlit Cloud, which only sees what is on GitHub. Our raw
GPS, the road map and the elevation data stay off GitHub (size + privacy), so
we bake just what the demo needs into demo/demo_drives.json:

  * the raw phone GPS and the hillsense track for each of our own drives
  * the live ETA replay: after every stretch, what each method predicted
    for the arrival time, next to the true arrival time

Privacy: points within PRIVACY_RADIUS_M of where each drive starts and ends
are removed, so the public demo does not show where we stayed.

    python -m hillsense.make_demo_data
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hillsense.eta_model import live_trace, load_saved, prepare
from hillsense.gpx_loader import haversine_m, load_gpx, split_drives
from hillsense.kalman import filter_all
from hillsense.map_match import match_drive
from hillsense.road_network import RoadNetwork
from hillsense.trips import trim_to_trips

OWN_GPX = "data/raw/munnar_2026-10-22.gpx"
PRIVACY_RADIUS_M = 500.0
MIN_DEMO_KM = 5.0


def away_from_ends(lat, lon, start, end, radius=PRIVACY_RADIUS_M):
    lat, lon = np.asarray(lat), np.asarray(lon)
    far_start = haversine_m(lat, lon, start[0], start[1]) > radius
    far_end = haversine_m(lat, lon, end[0], end[1]) > radius
    return far_start & far_end


def track(lat, lon, keep, every=1):
    lat, lon = np.asarray(lat)[keep][::every], np.asarray(lon)[keep][::every]
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lat, lon)]


if __name__ == "__main__":
    net = RoadNetwork("data/osm/munnar_drive.graphml")
    model = load_saved()
    passes = prepare(pd.read_csv("data/processed/passes.csv"))
    pub, own = passes[passes["source"] == "public"], passes[passes["source"] == "own"]
    global_kmh = float(np.average(pub["kmh"], weights=pub["dist_m"]))

    logged = split_drives(load_gpx(OWN_GPX))
    trips = filter_all(trim_to_trips(logged))
    drives = []
    for drive_id, d in trips.groupby("drive"):
        d = d.reset_index(drop=True)
        m, road_km, _ = match_drive(net, d)
        if road_km < MIN_DEMO_KM:
            continue
        raw_all = logged[logged["drive"] == drive_id]
        start = (d["lat"].iloc[0], d["lon"].iloc[0])
        end = (d["lat"].iloc[-1], d["lon"].iloc[-1])

        keep_raw = away_from_ends(d["lat"], d["lon"], start, end)
        keep_matched = away_from_ends(m["lat_m"], m["lon_m"], start, end)

        p = own[own["drive"] == drive_id]
        live = live_trace(p, model.predict(p), global_kmh)
        t0 = d["time"].iloc[0]
        replay = [
            {
                "elapsed_min": round(r.elapsed_s / 60, 2),
                "true_arrival_min": round((r.elapsed_s + r.true_left_s) / 60, 2),
                "naive_live_min": round((r.elapsed_s + r.naive_live) / 60, 2),
                "hillsense_min": round((r.elapsed_s + r.hillsense_live) / 60, 2),
            }
            for r in live.itertuples()
        ]
        drives.append({
            "name": f"Drive {drive_id} ({t0:%H:%M} IST, {'uphill' if d['ele'].iloc[-1] > d['ele'].iloc[0] else 'downhill'})",
            "logged_raw_km": round(float(raw_all["step_m"].sum() / 1000), 1),
            "trip_raw_km": round(float(d["step_m"].sum() / 1000), 1),
            "road_km": round(float(road_km), 1),
            "minutes": round(float((d["time"].iloc[-1] - t0).total_seconds() / 60), 1),
            "climb_m": round(float(d["ele"].max() - d["ele"].min())) if d["ele"].notna().any() else None,
            "raw": track(d["lat"], d["lon"], keep_raw, every=3),
            "matched": track(m["lat_m"], m["lon_m"], keep_matched),
            "replay": replay,
        })
        print(f"{drives[-1]['name']}: {drives[-1]['road_km']} km, {len(replay)} replay steps, "
              f"{int(keep_raw.sum())}/{len(keep_raw)} raw points kept after privacy trim")

    Path("demo").mkdir(exist_ok=True)
    out = {"privacy_radius_m": PRIVACY_RADIUS_M, "drives": drives}
    Path("demo/demo_drives.json").write_text(json.dumps(out))
    print(f"-> demo/demo_drives.json ({Path('demo/demo_drives.json').stat().st_size / 1024:.0f} KB)")
