"""Download public GPS traces from OpenStreetMap and split them into trips.

OpenStreetMap lets anyone download the GPS points people have uploaded, one
map box (max 0.25 x 0.25 degrees) and 5,000 points at a time. Only traces
whose owners made them "trackable" or "identifiable" keep their timestamps
and point order; the rest come back shuffled for privacy, so we drop them.

Steps:
  1. fetch every page for the region (polite: identifies itself, waits
     between requests, stops at the last page)
  2. group points by the trace they came from (traces can span pages)
  3. write one GPX file per trace into data/public/<region>/traces/

Then each trace goes through the same pipeline as your own drive.

    python -m hillsense.fetch_public            # download + split
    python -m hillsense.fetch_public --split    # only re-split pages already downloaded
"""

import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

API = "https://api.openstreetmap.org/api/0.6/trackpoints"
USER_AGENT = "hillsense/0.1 (student research project on hill-road GPS tracking)"
POINTS_PER_PAGE = 5000
WAIT_S = 2.0
MAX_PAGES = 60

REGIONS = {
    # name: (west, south, east, north) -- each side at most 0.25 degrees
    "munnar": (77.00, 10.00, 77.22, 10.12),
}


def fetch_pages(region, bbox, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for page in range(MAX_PAGES):
        path = out_dir / f"page_{page:03d}.gpx"
        if not path.exists():
            url = f"{API}?bbox={','.join(map(str, bbox))}&page={page}"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                path.write_bytes(r.read())
            time.sleep(WAIT_S)
        n = path.read_text(errors="ignore").count("<trkpt")
        print(f"  {region} page {page}: {n} points")
        if n < POINTS_PER_PAGE:
            break


def split_traces(pages_dir, traces_dir):
    """Merge pages, keep only timestamped traces with an owner URL, one file each."""
    traces = defaultdict(list)  # trace id -> [(time, lat, lon)]
    for page in sorted(pages_dir.glob("page_*.gpx")):
        root = ET.parse(page).getroot()
        ns = {"g": root.tag[1:].split("}")[0]}
        for trk in root.iterfind("g:trk", ns):
            url = trk.findtext("g:url", default="", namespaces=ns)
            if not url:
                continue  # private/anonymous trace: points are shuffled, unusable
            trace_id = url.rstrip("/").split("/")[-1]
            for pt in trk.iterfind(".//g:trkpt", ns):
                t = pt.findtext("g:time", default=None, namespaces=ns)
                if t:
                    traces[trace_id].append((t, pt.get("lat"), pt.get("lon")))

    traces_dir.mkdir(parents=True, exist_ok=True)
    for trace_id, pts in traces.items():
        pts = sorted(set(pts))
        body = "\n".join(
            f'<trkpt lat="{lat}" lon="{lon}"><time>{t}</time></trkpt>' for t, lat, lon in pts
        )
        (traces_dir / f"trace_{trace_id}.gpx").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<gpx version="1.0" xmlns="http://www.topografix.com/GPX/1/0">'
            f"<trk><name>osm trace {trace_id}</name><trkseg>\n{body}\n</trkseg></trk></gpx>\n"
        )
    return {tid: len(p) for tid, p in traces.items()}


if __name__ == "__main__":
    for region, bbox in REGIONS.items():
        base = Path("data/public") / region
        if "--split" not in sys.argv:
            print(f"Downloading {region} {bbox} ...")
            fetch_pages(region, bbox, base / "pages")
        counts = split_traces(base / "pages", base / "traces")
        print(f"{region}: {len(counts)} usable traces, {sum(counts.values())} points")
        for tid, n in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  trace {tid}: {n} points")
