"""Snap a GPS track onto real roads with a Hidden Markov Model (from scratch).

Why: on hairpin bends, two loops of the same road can be only 30 m apart.
"Snap to the nearest road" then flips between loops and the distance driven
jumps. An HMM instead asks: which sequence of road positions best explains
ALL the GPS points, given that a car has to actually drive along the roads
from one point to the next?

This follows Newson & Krumm (2009), "Hidden Markov Map Matching Through
Noise and Sparseness", the standard method used in industry:

  hidden states   = candidate positions on nearby roads for each GPS point
  emission score  = how close the GPS point is to that road (Gaussian)
  transition score= how well the driving distance between two candidates
                    matches the straight-line distance between the two GPS
                    points (big detours are unlikely), and the implied speed
                    must be physically possible
  Viterbi         = dynamic programming that finds the best whole path

Points with no road nearby (e.g. a phone indoors at a hotel) are left
unmatched instead of being forced onto a road.

Run it directly to compare raw, Kalman-filtered and road-matched distance:
    python -m hillsense.map_match
"""

import sys

import numpy as np
import pandas as pd

from hillsense.gpx_loader import load_gpx, split_drives
from hillsense.kalman import filter_all, path_km
from hillsense.road_network import RoadNetwork
from hillsense.trips import trim_to_trips

SEARCH_RADIUS_M = 50.0     # only consider roads this close to a GPS point
GPS_SIGMA_M = 12.0         # emission: typical distance from GPS point to true road
BETA_M = 40.0              # transition: tolerance for route vs straight-line mismatch
MIN_STEP_M = 20.0          # thin the track: keep a point once we moved this far
MAX_STEP_S = 60.0          # ...or this much time passed
MAX_SPEED_KMH = 120.0      # route distance needing more than this is impossible
BACKTRACK_TOLERANCE_M = 15.0  # small backwards GPS wobble on the same road is fine


def thin_track(d, x, y):
    """Keep points that moved MIN_STEP_M (or waited MAX_STEP_S) since the last kept one."""
    t = (d["time"] - d["time"].iloc[0]).dt.total_seconds().to_numpy()
    keep = [0]
    for i in range(1, len(d)):
        j = keep[-1]
        if np.hypot(x[i] - x[j], y[i] - y[j]) >= MIN_STEP_M or t[i] - t[j] >= MAX_STEP_S:
            keep.append(i)
    return np.array(keep)


def route_distance(net, a, b, dist_from_end):
    """Driving distance from candidate a to candidate b (np.inf if not reachable)."""
    if a.edge == b.edge:
        forward = b.offset - a.offset
        if forward >= -BACKTRACK_TOLERANCE_M:
            return max(forward, 0.0)
    ea, eb = net.edges[a.edge], net.edges[b.edge]
    to_junction = dist_from_end.get(eb.u, np.inf)
    return (ea.length - a.offset) + to_junction + b.offset


def match_drive(net, d):
    """Run Viterbi over one drive. Returns a table of matched points and the road km."""
    x, y = net.to_local(d["lat_f"].to_numpy(), d["lon_f"].to_numpy())
    idx = thin_track(d, x, y)
    times = (d["time"].iloc[idx] - d["time"].iloc[0]).dt.total_seconds().to_numpy()

    steps = []  # (row index in d, seconds since start, nearby road candidates)
    for k, i in enumerate(idx):
        cands = net.candidates(x[i], y[i], SEARCH_RADIUS_M)
        steps.append((i, times[k], cands))

    chain = []   # current unbroken stretch: (row, candidates, back-pointers, scores)

    def flush(chain):
        """Backtrack the finished chain and return chosen (row, candidate, chain id)."""
        if not chain:
            return []
        best = int(np.argmax(chain[-1][3]))
        out = []
        for row, cands, back, scores in reversed(chain):
            out.append((row, cands[best], chain_id))
            best = back[best] if back is not None else best
        return out[::-1]

    results, chain_id = [], 0
    for row, t, cands in steps:
        if not cands:
            continue  # off-road point: skip it, don't force it onto a road
        emis = np.array([-0.5 * (c.dist / GPS_SIGMA_M) ** 2 for c in cands])
        if not chain:
            chain.append((row, cands, None, emis))
            prev_t = t
            continue
        p_row, p_cands, _, p_scores = chain[-1]
        dt = max(t - prev_t, 1.0)
        gc = float(np.hypot(x[row] - x[p_row], y[row] - y[p_row]))
        cutoff = max(500.0, MAX_SPEED_KMH / 3.6 * dt)
        dijkstra = {}
        trans = np.full((len(p_cands), len(cands)), -np.inf)
        for a_i, a in enumerate(p_cands):
            end_node = net.edges[a.edge].v
            if end_node not in dijkstra:
                dijkstra[end_node] = net.shortest_from(end_node, cutoff)
            for b_i, b in enumerate(cands):
                rd = route_distance(net, a, b, dijkstra[end_node])
                if np.isfinite(rd) and rd / dt * 3.6 <= MAX_SPEED_KMH:
                    trans[a_i, b_i] = -abs(rd - gc) / BETA_M
        total = p_scores[:, None] + trans
        back = np.argmax(total, axis=0)
        scores = total[back, np.arange(len(cands))] + emis
        if not np.isfinite(scores).any():
            results.extend(flush(chain))       # HMM break: no way to drive here
            chain_id += 1
            chain = [(row, cands, None, emis)]  # start a fresh chain
        else:
            chain.append((row, cands, back, scores))
        prev_t = t
    results.extend(flush(chain))

    # distance actually driven along the roads between consecutive matched points
    road_m, rows = 0.0, []
    for k, (row, c, cid) in enumerate(results):
        rd = 0.0
        if k > 0:
            p_row, pc, p_cid = results[k - 1]
            if cid == p_cid:
                dt = max((d["time"].iloc[row] - d["time"].iloc[p_row]).total_seconds(), 1.0)
                dij = net.shortest_from(net.edges[pc.edge].v, max(500.0, MAX_SPEED_KMH / 3.6 * dt))
                rd = route_distance(net, pc, c, dij)
            else:
                # across an HMM break we can't trust any route: count the straight line
                rd = float(np.hypot(c.x - pc.x, c.y - pc.y))
            rd = rd if np.isfinite(rd) else 0.0
            road_m += rd
        lat, lon = net.to_latlon(c.x, c.y)
        rows.append(
            {
                "row": row,
                "time": d["time"].iloc[row],
                "lat_m": lat,
                "lon_m": lon,
                "dist_to_road_m": c.dist,
                "road": net.edges[c.edge].name or net.edges[c.edge].highway,
                "edge": c.edge,
                "chain": cid,
                "step_road_m": rd,
            }
        )
    return pd.DataFrame(rows), road_m / 1000, len(idx)


if __name__ == "__main__":
    gpx = sys.argv[1] if len(sys.argv) > 1 else "data/raw/munnar_2026-10-22.gpx"
    graph = sys.argv[2] if len(sys.argv) > 2 else "data/osm/munnar_drive.graphml"
    net = RoadNetwork(graph)
    points = filter_all(trim_to_trips(split_drives(load_gpx(gpx))))
    rows = []
    for drive_id, d in points.groupby("drive"):
        m, road_km, n_kept = match_drive(net, d.reset_index(drop=True))
        rows.append(
            {
                "drive": drive_id,
                "raw_km": round(path_km(d["lat"], d["lon"]), 2),
                "kalman_km": round(path_km(d["lat_f"], d["lon_f"]), 2),
                "road_km": round(road_km, 2),
                "points_on_road": f"{len(m)}/{n_kept}",
                "median_dist_to_road_m": round(m["dist_to_road_m"].median(), 1) if len(m) else None,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
