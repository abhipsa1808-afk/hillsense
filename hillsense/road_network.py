"""The road network: load OpenStreetMap roads and find the roads near a point.

Reads the .graphml file that osmnx saves, without needing osmnx at runtime.
Every road segment ("edge") is stored as a polyline in metres on a flat local
map, so distances are simple Pythagoras. A grid index makes "which roads are
within 50 m of this GPS point?" fast even with thousands of roads.

Make the file once with:
    python -c "import osmnx as ox; ox.save_graphml(ox.graph_from_bbox(
        bbox=(77.00, 9.98, 77.22, 10.12), network_type='drive'),
        'data/osm/munnar_drive.graphml')"
"""

import heapq
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from hillsense.kalman import to_latlon, to_local_xy

GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
GRID_CELL_M = 100.0


@dataclass
class Edge:
    u: str                 # start junction id
    v: str                 # end junction id
    xy: np.ndarray         # (n, 2) polyline in metres, from u to v
    length: float          # metres along the polyline
    cum: np.ndarray        # distance from u to each polyline vertex
    highway: str           # OSM road class, e.g. "primary", "residential"
    name: str


@dataclass
class Candidate:
    edge: int              # index into RoadNetwork.edges
    offset: float          # metres from the edge's start to the snapped point
    dist: float            # metres from the GPS point to the road
    x: float               # snapped position (local metres)
    y: float


class RoadNetwork:
    def __init__(self, path, lat0=None, lon0=None):
        root = ET.parse(path).getroot()
        keys = {k.get("id"): k.get("attr.name") for k in root.iterfind("g:key", GRAPHML_NS)}

        def attrs(el):
            return {keys[d.get("key")]: d.text for d in el.iterfind("g:data", GRAPHML_NS)}

        graph = root.find("g:graph", GRAPHML_NS)
        nodes = {}
        for n in graph.iterfind("g:node", GRAPHML_NS):
            a = attrs(n)
            nodes[n.get("id")] = (float(a["y"]), float(a["x"]))  # (lat, lon)

        lats = np.array([p[0] for p in nodes.values()])
        lons = np.array([p[1] for p in nodes.values()])
        self.lat0 = float(lats.mean()) if lat0 is None else lat0
        self.lon0 = float(lons.mean()) if lon0 is None else lon0
        self.node_xy = {
            nid: np.array(to_local_xy(lat, lon, self.lat0, self.lon0)) for nid, (lat, lon) in nodes.items()
        }

        self.edges = []
        self.out_edges = defaultdict(list)  # junction id -> indices of edges leaving it
        for e in graph.iterfind("g:edge", GRAPHML_NS):
            a = attrs(e)
            u, v = e.get("source"), e.get("target")
            if "geometry" in a and a["geometry"]:
                coords = np.array(
                    [[float(c) for c in pair.split()] for pair in re.findall(r"[-\d.]+ [-\d.]+", a["geometry"])]
                )  # (lon, lat) pairs
                x, y = to_local_xy(coords[:, 1], coords[:, 0], self.lat0, self.lon0)
                xy = np.column_stack([x, y])
            else:
                xy = np.vstack([self.node_xy[u], self.node_xy[v]])
            seg = np.hypot(*np.diff(xy, axis=0).T)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            highway = (a.get("highway") or "").strip("[]'").split("'")[0]
            self.out_edges[u].append(len(self.edges))
            self.edges.append(Edge(u, v, xy, float(cum[-1]), cum, highway, a.get("name") or ""))

        self._build_grid()

    def _build_grid(self):
        """Put every small piece of every road into 100 m x 100 m buckets."""
        self.grid = defaultdict(list)
        for ei, e in enumerate(self.edges):
            for si in range(len(e.xy) - 1):
                (x1, y1), (x2, y2) = e.xy[si], e.xy[si + 1]
                for gx in range(int(min(x1, x2) // GRID_CELL_M), int(max(x1, x2) // GRID_CELL_M) + 1):
                    for gy in range(int(min(y1, y2) // GRID_CELL_M), int(max(y1, y2) // GRID_CELL_M) + 1):
                        self.grid[(gx, gy)].append((ei, si))

    def candidates(self, x, y, radius):
        """All roads within `radius` metres of (x, y), each snapped to its closest point."""
        best = {}
        r_cells = int(np.ceil(radius / GRID_CELL_M))
        cx, cy = int(x // GRID_CELL_M), int(y // GRID_CELL_M)
        for gx in range(cx - r_cells, cx + r_cells + 1):
            for gy in range(cy - r_cells, cy + r_cells + 1):
                for ei, si in self.grid.get((gx, gy), ()):
                    e = self.edges[ei]
                    p, q = e.xy[si], e.xy[si + 1]
                    d = q - p
                    seg_len2 = d @ d
                    t = 0.0 if seg_len2 == 0 else float(np.clip(((x, y) - p) @ d / seg_len2, 0, 1))
                    sx, sy = p + t * d
                    dist = float(np.hypot(x - sx, y - sy))
                    if dist <= radius and (ei not in best or dist < best[ei].dist):
                        offset = e.cum[si] + t * np.sqrt(seg_len2)
                        best[ei] = Candidate(ei, float(offset), dist, float(sx), float(sy))
        return list(best.values())

    def shortest_from(self, start_node, cutoff):
        """Dijkstra: road distance from one junction to every junction within `cutoff` m."""
        dist = {start_node: 0.0}
        heap = [(0.0, start_node)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, np.inf) or d > cutoff:
                continue
            for ei in self.out_edges[n]:
                e = self.edges[ei]
                nd = d + e.length
                if nd < dist.get(e.v, np.inf) and nd <= cutoff:
                    dist[e.v] = nd
                    heapq.heappush(heap, (nd, e.v))
        return dist

    def point_on_edge(self, ei, offset):
        """(x, y) of the point `offset` metres along edge ei."""
        e = self.edges[ei]
        return np.interp(offset, e.cum, e.xy[:, 0]), np.interp(offset, e.cum, e.xy[:, 1])

    def to_local(self, lat, lon):
        return to_local_xy(lat, lon, self.lat0, self.lon0)

    def to_latlon(self, x, y):
        return to_latlon(x, y, self.lat0, self.lon0)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/osm/munnar_drive.graphml"
    net = RoadNetwork(path)
    total_km = sum(e.length for e in net.edges) / 1000
    by_type = defaultdict(float)
    for e in net.edges:
        by_type[e.highway] += e.length / 1000
    print(f"{len(net.node_xy)} junctions, {len(net.edges)} road segments, {total_km:.0f} km of (one-way) road")
    for kind, km in sorted(by_type.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {kind:<15} {km:7.1f} km")
