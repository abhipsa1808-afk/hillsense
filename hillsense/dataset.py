"""Turn GPS drives into a training table: one row per road segment driven.

For every drive:
  1. keep only car drives (walks, bike rides and very sparse logs are dropped)
  2. trip detection -> Kalman filter -> HMM map-matching onto OSM roads
  3. cut the matched track into "passes": one continuous stretch on one road
     segment, e.g. "drove 420 m of Munnar-Top Station Highway in 41 s"
  4. attach that segment's features (slope in the direction driven, curves,
     hairpins, road type, how built-up the area is, time of day)

The ETA model then learns: given these features, how fast does traffic go?

    python -m hillsense.dataset     # builds data/processed/passes.csv
"""

import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from hillsense.gpx_loader import load_gpx, split_drives
from hillsense.kalman import filter_all
from hillsense.map_match import match_drive
from hillsense.road_network import RoadNetwork
from hillsense.trips import trim_to_trips

# --- which drives count as car drives -------------------------------------
MIN_P85_KMH = 20.0       # 85% of the time slower than this = walking/cycling
MIN_P98_KMH = 25.0       # never got going = not a car
MAX_MEDIAN_GAP_S = 10.0  # sparser logs can't follow hairpins reliably
MIN_DRIVE_MIN = 3.0

# --- which passes are clean enough to learn from ---------------------------
MIN_PASS_M = 50.0        # very short passes are mostly GPS noise
MIN_PASS_KMH = 5.0       # slower = parked / tea stop, not a driving speed
MAX_PASS_KMH = 100.0     # faster = matching glitch
TOWN_RADIUS_M = 300.0    # junctions counted within this radius = "how built-up"


def smoothed_speed(d, window="30s"):
    s = d.set_index("time")
    v = s["step_m"].rolling(window).sum() / s["dt_s"].rolling(window).sum() * 3.6
    return v.replace([np.inf, -np.inf], np.nan)


def is_car_drive(d):
    """Car-like speeds and a dense enough GPS log."""
    if len(d) < 30:
        return False, "too short"
    minutes = (d["time"].iloc[-1] - d["time"].iloc[0]).total_seconds() / 60
    gap = d["dt_s"][d["dt_s"] > 0].median()
    v = smoothed_speed(d)
    if minutes < MIN_DRIVE_MIN:
        return False, "too short"
    if gap > MAX_MEDIAN_GAP_S:
        return False, f"sparse (a point every {gap:.0f}s)"
    if v.quantile(0.85) < MIN_P85_KMH or v.quantile(0.98) < MIN_P98_KMH:
        return False, f"not a car (85% speed {v.quantile(0.85):.0f} km/h)"
    return True, "car"


def junction_density(net):
    """For each edge: number of real junctions (3+ roads meet) near its middle."""
    neighbours = defaultdict(set)
    for e in net.edges:
        neighbours[e.u].add(e.v)
        neighbours[e.v].add(e.u)
    junctions = np.array([net.node_xy[n] for n, nb in neighbours.items() if len(nb) >= 3])
    out = np.zeros(len(net.edges))
    for i, e in enumerate(net.edges):
        mid = e.xy[len(e.xy) // 2]
        out[i] = np.sum(np.hypot(*(junctions - mid).T) <= TOWN_RADIUS_M)
    return out


def passes_from_match(m, drive_start):
    """Group consecutive matched points on the same edge into passes."""
    m = m.copy()
    m["dt"] = m["time"].diff().dt.total_seconds().fillna(0)
    new_pass = (m["edge"] != m["edge"].shift()) | (m["chain"] != m["chain"].shift())
    m["pass_id"] = new_pass.cumsum()
    rows = []
    for _, g in m.groupby("pass_id"):
        if len(g) < 2:
            continue
        dist, secs = g["step_road_m"].iloc[1:].sum(), g["dt"].iloc[1:].sum()
        if secs <= 0:
            continue
        t0 = g["time"].iloc[0]
        rows.append(
            {
                "edge": int(g["edge"].iloc[0]),
                "dist_m": dist,
                "secs": secs,
                "kmh": dist / secs * 3.6,
                "hour": t0.hour + t0.minute / 60,
                "minutes_into_drive": (t0 - drive_start).total_seconds() / 60,
            }
        )
    return rows


def build(gpx_paths, net, edge_feats, source, verbose=True):
    """All passes from a set of GPX files, plus a log of which drives were used."""
    rows, log = [], []
    for path in gpx_paths:
        trace = Path(path).stem
        try:
            drives = split_drives(load_gpx(path))
        except Exception as err:  # a broken upload shouldn't stop the whole build
            log.append({"source": source, "trace": trace, "drive": None, "used": False, "why": f"unreadable: {err}"})
            continue
        for drive_id, d in drives.groupby("drive"):
            ok, why = is_car_drive(d)
            entry = {"source": source, "trace": trace, "drive": int(drive_id), "used": ok, "why": why,
                     "year": int(d["time"].iloc[0].year)}
            if ok:
                trip = trim_to_trips(d)
                if trip.empty:
                    entry.update(used=False, why="no sustained driving")
                else:
                    f = filter_all(trip)
                    m, road_km, _ = match_drive(net, f.reset_index(drop=True))
                    on_road = len(m)
                    if on_road < 20:
                        entry.update(used=False, why="outside the road map")
                    else:
                        p = passes_from_match(m, f["time"].iloc[0])
                        for r in p:
                            r.update(source=source, trace=trace, drive=int(drive_id), year=entry["year"])
                        rows.extend(p)
                        entry.update(road_km=round(road_km, 2), passes=len(p))
            log.append(entry)
            if verbose:
                print(f"  {source} {trace} drive {drive_id}: {entry['why']}"
                      + (f", {entry.get('passes', 0)} passes, {entry.get('road_km', 0)} km" if entry["used"] else ""))

    passes = pd.DataFrame(rows)
    if passes.empty:
        return passes, pd.DataFrame(log)
    passes = passes.join(edge_feats, on="edge")
    clean = passes["dist_m"].ge(MIN_PASS_M) & passes["kmh"].between(MIN_PASS_KMH, MAX_PASS_KMH)
    return passes[clean].reset_index(drop=True), pd.DataFrame(log)


def load_edge_features(net, path="data/osm/munnar_edge_features.csv"):
    f = pd.read_csv(path).set_index("edge")
    f["junctions_nearby"] = junction_density(net)
    f["hairpins_per_km"] = f["hairpins"] / (f["length_m"] / 1000).clip(lower=0.05)
    return f.drop(columns=["u", "v", "name"])


if __name__ == "__main__":
    net = RoadNetwork("data/osm/munnar_drive.graphml")
    feats = load_edge_features(net)
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    public, log_pub = build(sorted(glob.glob("data/public/munnar/traces/*.gpx")), net, feats, "public")
    own, log_own = build(["data/raw/munnar_2026-10-22.gpx"], net, feats, "own")
    passes = pd.concat([public, own], ignore_index=True)
    log = pd.concat([log_pub, log_own], ignore_index=True)
    passes.to_csv(out / "passes.csv", index=False)
    log.to_csv(out / "drives_log.csv", index=False)

    used = log[log["used"]]
    print(f"\nDrives used: {len(used)} of {len(log)}  (public {int(used['source'].eq('public').sum())}, own {int(used['source'].eq('own').sum())})")
    print("Dropped because:", log[~log["used"]]["why"].str.split(" \\(").str[0].value_counts().to_dict())
    print(f"Passes: {len(passes)}  ({passes['dist_m'].sum() / 1000:.0f} km of road, "
          f"{passes['secs'].sum() / 3600:.1f} h of driving) -> data/processed/passes.csv")
    print(passes.groupby("source").agg(passes=("kmh", "size"), km=("dist_m", lambda x: x.sum() / 1000),
                                       median_kmh=("kmh", "median")).round(1).to_string())
