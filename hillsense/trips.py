"""Find when you were actually driving, and trim the rest.

A GPS logger keeps running after you arrive. Indoors, the phone only gets a
weak fix every 10 s to 5 min, and each one can land hundreds of metres away.
No filter can tell "jumped 800 m" from "drove 800 m" with points that sparse,
so instead we detect the trip itself and ignore what happens outside it.

A moment counts as "driving" when, in the surrounding 60-second window:
  * the GPS is healthy: at least MIN_FIXES_PER_MIN fixes (one every ~3 s)
  * the car is moving: median speed above MIN_DRIVE_KMH (faster than walking)

The trip runs from the first driving moment to the last one. Stops in the
middle (a tea break, a traffic jam) stay in, as they are part of the journey.
"""

import numpy as np
import pandas as pd

WINDOW_S = 60
MIN_FIXES_PER_MIN = 20
MIN_DRIVE_KMH = 8.0


def driving_mask(d):
    """True for rows inside a 60 s window of healthy, moving GPS."""
    s = d.set_index("time")
    fixes = s["lat"].rolling(f"{WINDOW_S}s", center=True).count()
    speed = s["speed_kmh"].rolling(f"{WINDOW_S}s", center=True).median()
    return ((fixes >= MIN_FIXES_PER_MIN) & (speed >= MIN_DRIVE_KMH)).to_numpy()


def trim_to_trips(points):
    """Keep each drive only from its first to its last driving moment."""
    kept = []
    for _, d in points.groupby("drive"):
        mask = driving_mask(d)
        if not mask.any():
            continue
        first, last = np.argmax(mask), len(mask) - 1 - np.argmax(mask[::-1])
        trip = d.iloc[first : last + 1].copy()
        trip.iloc[0, trip.columns.get_indexer(["step_m", "dt_s", "speed_kmh"])] = 0
        kept.append(trip)
    if not kept:  # no sustained driving anywhere (e.g. a walk or a parked phone)
        return points.iloc[0:0].copy()
    return pd.concat(kept, ignore_index=True)


if __name__ == "__main__":
    import sys

    from hillsense.gpx_loader import load_gpx, split_drives

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/munnar_2026-10-22.gpx"
    points = split_drives(load_gpx(path))
    trips = trim_to_trips(points)
    for drive_id, d in points.groupby("drive"):
        t = trips[trips["drive"] == drive_id]
        print(
            f"drive {drive_id}: logged {d['time'].iloc[0]:%H:%M}-{d['time'].iloc[-1]:%H:%M}, "
            f"driving {t['time'].iloc[0]:%H:%M}-{t['time'].iloc[-1]:%H:%M} "
            f"({len(d) - len(t)} points trimmed)"
        )
