"""A constant-velocity Kalman filter for phone GPS, written from scratch.

Why: raw phone GPS jumps around by 10-300 m, especially in hills where the
signal bounces off slopes. Adding up those jumps makes the distance you have
driven (and so the distance left) wrong. The filter keeps a running guess of
where the car is and how fast it is moving, and trusts each new GPS reading
only as much as it deserves.

State (in metres, on a flat local map around the start point):
    [x, y, vx, vy]   position east/north and velocity east/north

Each second it does two things:
    predict: "if I kept going at the same velocity, where would I be now?"
    update:  "the GPS says I'm here; blend that with my prediction"

Three extras make it work on real hill data:
    * outlier gating: a reading that is wildly far from the prediction
      (statistically) is ignored, so a 300 m jump can't drag the track
    * speed gating: a reading that would mean driving faster than
      MAX_SPEED_KMH since the last good position is physically impossible
      on a ghat road, so it is ignored too, even after a signal gap
    * reset: after many rejected readings in a row the filter assumes it is
      the one that is lost and starts fresh from the GPS

Run it directly to compare raw vs filtered distance for each drive:
    python -m hillsense.kalman
"""

import sys

import numpy as np
import pandas as pd

from hillsense.gpx_loader import EARTH_RADIUS_M, haversine_m, load_gpx, split_drives

# Tuning knobs (sensible defaults for a phone in a car)
GPS_NOISE_M = 10.0          # typical GPS error, 1 standard deviation
ACCEL_NOISE = 1.5           # how sharply the car can change speed (m/s^2)
GATE_CHI2 = 13.8            # 99.9% cut-off for 2-D readings: above this = outlier
RESET_AFTER_REJECTS = 15    # this many outliers in a row -> trust GPS again
GAP_S = 30                  # silence longer than this counts as a signal gap
MAX_SPEED_KMH = 120         # faster than this on a hill road = bad reading
INIT_POS_SIGMA_M = 100.0    # how unsure we are about the very first fix


def to_local_xy(lat, lon, lat0, lon0):
    """Turn lat/lon into metres east (x) and north (y) of a reference point."""
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def to_latlon(x, y, lat0, lon0):
    """Inverse of to_local_xy."""
    lat = lat0 + np.degrees(y / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(x / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    return lat, lon


class GPSKalmanFilter:
    def __init__(self, gps_noise_m=GPS_NOISE_M, accel_noise=ACCEL_NOISE):
        self.R = np.eye(2) * gps_noise_m**2            # measurement noise
        self.q = accel_noise**2                        # process noise strength
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)  # we only see x, y
        self.x = None                                  # state [x, y, vx, vy]
        self.P = None                                  # uncertainty of the state
        self.rejects_in_a_row = 0
        self.secs_since_good = 0.0                     # time since last accepted reading

    def reset(self, zx, zy):
        self.x = np.array([zx, zy, 0.0, 0.0])
        # start unsure (100 m): if this first fix is bad, the next good
        # readings can still pull us to the right place instead of being rejected
        self.P = np.diag([INIT_POS_SIGMA_M**2, INIT_POS_SIGMA_M**2, 10.0**2, 10.0**2])
        self.rejects_in_a_row = 0
        self.secs_since_good = 0.0

    def gap(self, dt):
        """Signal was lost for dt seconds: we don't know how we moved, so stop
        extrapolating velocity and widen the position uncertainty instead."""
        self.x[2:] = 0.0
        spread = (MAX_SPEED_KMH / 3.6 * dt) / 3            # ~3-sigma = max possible travel
        self.P = np.diag([self.P[0, 0] + spread**2, self.P[1, 1] + spread**2, 10.0**2, 10.0**2])
        self.secs_since_good += dt

    def predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # random-acceleration model: uncertainty grows with dt
        g = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        Q = g @ g.T * self.q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.secs_since_good += dt

    def update(self, zx, zy):
        """Blend in a GPS reading. Returns False if it was rejected as an outlier."""
        z = np.array([zx, zy])
        innovation = z - self.H @ self.x                 # how far GPS is from prediction
        S = self.H @ self.P @ self.H.T + self.R          # expected spread of that gap
        mahalanobis2 = innovation @ np.linalg.solve(S, innovation)
        implied_kmh = np.hypot(*innovation) / max(self.secs_since_good, 1.0) * 3.6
        if mahalanobis2 > GATE_CHI2 or implied_kmh > MAX_SPEED_KMH:
            self.rejects_in_a_row += 1
            if self.rejects_in_a_row >= RESET_AFTER_REJECTS:
                self.reset(zx, zy)                       # we were the ones who were lost
            return False
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain: how much to trust GPS
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.rejects_in_a_row = 0
        self.secs_since_good = 0.0
        return True


def filter_drive(d):
    """Run the filter over one drive. Adds lat_f, lon_f, speed_f_kmh, outlier."""
    d = d.copy()
    lat0, lon0 = d["lat"].iloc[0], d["lon"].iloc[0]
    zx, zy = to_local_xy(d["lat"].to_numpy(), d["lon"].to_numpy(), lat0, lon0)
    dts = d["dt_s"].to_numpy()

    kf = GPSKalmanFilter()
    xs, ys, speeds, outlier = [], [], [], []
    for i in range(len(d)):
        if kf.x is None:
            kf.reset(zx[i], zy[i])
            accepted = True
        else:
            if dts[i] > GAP_S:
                kf.gap(dts[i])
            else:
                kf.predict(max(dts[i], 1e-3))
            accepted = kf.update(zx[i], zy[i])
        xs.append(kf.x[0])
        ys.append(kf.x[1])
        speeds.append(np.hypot(kf.x[2], kf.x[3]) * 3.6)
        outlier.append(not accepted)

    d["lat_f"], d["lon_f"] = to_latlon(np.array(xs), np.array(ys), lat0, lon0)
    d["speed_f_kmh"] = speeds
    d["outlier"] = outlier
    return d


def filter_all(points):
    return pd.concat([filter_drive(d) for _, d in points.groupby("drive")], ignore_index=True)


def path_km(lat, lon):
    lat, lon = np.asarray(lat), np.asarray(lon)
    return haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum() / 1000


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/munnar_2026-10-22.gpx"
    out = filter_all(split_drives(load_gpx(path)))
    rows = []
    for drive_id, d in out.groupby("drive"):
        rows.append(
            {
                "drive": drive_id,
                "raw_km": round(path_km(d["lat"], d["lon"]), 2),
                "filtered_km": round(path_km(d["lat_f"], d["lon_f"]), 2),
                "outliers_rejected": int(d["outlier"].sum()),
                "max_raw_kmh": round(d["speed_kmh"].max()),
                "max_filtered_kmh": round(d["speed_f_kmh"].max()),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
