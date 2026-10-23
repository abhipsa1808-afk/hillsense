"""Describe every road segment the way a driver feels it: steep? twisty? hairpins?

For each road segment (in its direction of travel) we resample the road every
RESAMPLE_M metres, look up the ground height at each point, and compute:

  length_m         how long the segment is
  climb_m          total metres gained going uphill
  descent_m        total metres lost going downhill
  grade_pct        average slope, + = uphill in the direction of travel
  max_grade_pct    steepest 50 m stretch (uphill or downhill, by size)
  turn_deg_per_km  how much the road turns per km (0 = straight)
  hairpins         sharp switchbacks: the road turns > 120 deg within 60 m
  min_radius_m     tightest bend (smaller = slower)
  highway          OSM road class (primary, residential, ...)

These become the inputs for the ETA model: a driver goes slower up steep
grades, through tight bends and around hairpins, and the model learns how much.

Run it directly to build the table for the whole Munnar network:
    python -m hillsense.road_features
"""

import sys

import numpy as np
import pandas as pd

from hillsense.elevation import Elevation
from hillsense.road_network import RoadNetwork

RESAMPLE_M = 10.0
GRADE_WINDOW_M = 50.0
HAIRPIN_WINDOW_M = 60.0
HAIRPIN_TURN_DEG = 120.0
RADIUS_WINDOW_M = 30.0
SMOOTH_M = 90.0             # smooth heights over ~3 DEM pixels before taking slopes
MAX_MISSING_DEM = 0.3       # more of the road than this outside the DEM -> slope unknown
MAX_PLAUSIBLE_GRADE = 35.0  # steeper than this is DEM noise (tree canopy, cuttings)


def resample(xy, cum, step=RESAMPLE_M):
    """Evenly spaced points along a polyline (always includes both ends)."""
    n = max(int(np.ceil(cum[-1] / step)), 1)
    s = np.linspace(0, cum[-1], n + 1)
    return s, np.column_stack([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])])


def smooth(z, step=RESAMPLE_M, window=SMOOTH_M):
    """Moving average along the road. The DEM is 30 m pixels of ground *around*
    the road, so a road cut sideways into a steep hillside picks up the slope of
    the hill. Real roads change height smoothly, so we average that noise out."""
    k = int(window // step) | 1  # odd number of points
    if len(z) < 3 or k < 3:
        return z
    k = min(k, len(z) if len(z) % 2 else len(z) - 1)
    pad = k // 2
    zp = np.pad(z, pad, mode="edge")
    return np.convolve(zp, np.ones(k) / k, mode="valid")


def window_change(values, s, window):
    """For each point: change in `values` over the next `window` metres of road."""
    j = np.searchsorted(s, s + window).clip(max=len(s) - 1)
    return values[j] - values, s[j] - s


def _no_elevation(net, e, ei):
    """Road mostly outside the elevation data: keep shape features, slope unknown."""
    f = edge_features(net, None, ei)
    f.update(climb_m=np.nan, descent_m=np.nan, grade_pct=np.nan, max_grade_pct=np.nan)
    return f


def edge_features(net, dem, ei):
    e = net.edges[ei]
    s, pts = resample(e.xy, e.cum)
    lat, lon = net.to_latlon(pts[:, 0], pts[:, 1])
    z = dem.at(lat, lon) if dem is not None else np.zeros(len(s))
    missing = np.isnan(z)
    if missing.mean() > MAX_MISSING_DEM:
        return _no_elevation(net, e, ei)       # mostly outside the elevation grid
    if missing.any():                         # small gaps: fill along the road
        z = np.interp(s, s[~missing], z[~missing])
    z = smooth(z)

    dz = np.diff(z)
    length = float(e.length)
    feats = {
        "edge": ei,
        "u": e.u,
        "v": e.v,
        "name": e.name,
        "highway": e.highway,
        "length_m": length,
        "climb_m": float(dz[dz > 0].sum()),
        "descent_m": float(-dz[dz < 0].sum()),
        "grade_pct": 100 * float(z[-1] - z[0]) / length if length > 0 else 0.0,
    }

    if length >= GRADE_WINDOW_M:
        dzw, dsw = window_change(z, s, GRADE_WINDOW_M)
        ok = dsw >= GRADE_WINDOW_M * 0.8
        grades = 100 * dzw[ok] / dsw[ok]
        feats["max_grade_pct"] = float(grades[np.argmax(np.abs(grades))]) if ok.any() else feats["grade_pct"]
    else:
        feats["max_grade_pct"] = feats["grade_pct"]

    # heading of each little step along the road, unwrapped so it never jumps 360 deg
    if len(pts) >= 3:
        heading = np.unwrap(np.arctan2(np.diff(pts[:, 1]), np.diff(pts[:, 0])))
        sh = s[:-1]
        turn = np.abs(np.diff(heading)).sum()
        feats["turn_deg_per_km"] = float(np.degrees(turn) / (length / 1000)) if length > 0 else 0.0

        dth, dsw = window_change(heading, sh, HAIRPIN_WINDOW_M)
        sharp = np.degrees(np.abs(dth)) > HAIRPIN_TURN_DEG
        # count separate hairpins, not every point inside one
        feats["hairpins"] = int(np.sum(sharp[1:] & ~sharp[:-1]) + sharp[0])

        dth_r, dsr = window_change(heading, sh, RADIUS_WINDOW_M)
        with np.errstate(divide="ignore", invalid="ignore"):
            radius = np.where(np.abs(dth_r) > 1e-3, dsr / np.abs(dth_r), np.inf)
        feats["min_radius_m"] = float(np.min(radius)) if np.isfinite(radius).any() else 1000.0
    else:
        feats.update(turn_deg_per_km=0.0, hairpins=0, min_radius_m=1000.0)
    feats["min_radius_m"] = min(feats["min_radius_m"], 1000.0)
    return feats


def build_features(net, dem):
    f = pd.DataFrame([edge_features(net, dem, ei) for ei in range(len(net.edges))])
    for col in ("grade_pct", "max_grade_pct"):
        f[col] = f[col].clip(-MAX_PLAUSIBLE_GRADE, MAX_PLAUSIBLE_GRADE)
    return f


if __name__ == "__main__":
    graph = sys.argv[1] if len(sys.argv) > 1 else "data/osm/munnar_drive.graphml"
    net = RoadNetwork(graph)
    dem = Elevation.from_txt()
    f = build_features(net, dem)
    f.to_csv("data/osm/munnar_edge_features.csv", index=False)
    print(f"{len(f)} road segments -> data/osm/munnar_edge_features.csv\n")
    print(f[["length_m", "grade_pct", "max_grade_pct", "turn_deg_per_km", "hairpins", "min_radius_m"]]
          .describe(percentiles=[0.5, 0.9]).round(1).to_string())
    print("\nTwistiest named roads (hairpins per km):")
    named = f[f["name"] != ""].groupby("name").agg(km=("length_m", lambda x: x.sum() / 1000), hairpins=("hairpins", "sum"))
    named = named[named["km"] > 2]
    named["hairpins_per_km"] = named["hairpins"] / named["km"]
    print(named.sort_values("hairpins_per_km", ascending=False).head(5).round(2).to_string())
