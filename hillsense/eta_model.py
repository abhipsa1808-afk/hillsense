"""Predict travel time on hill roads, and measure it honestly.

Each row of data/processed/passes.csv is one stretch of one road segment that a
car actually drove, with the time it took. The model predicts the SPEED on that
stretch from the road's features; the ETA is the sum of distance / predicted
speed over the stretches still ahead.

Ways of guessing speed that we compare:
  naive       one average speed for every road (what a simple app does)
  road_type   one average speed per road class (roughly basic navigation)
  no_terrain  gradient-boosted trees WITHOUT slope/curve/hairpin features
  hillsense   gradient-boosted trees WITH all features

Two tests:
  1. pre-trip ETA: predict the whole drive before it starts
  2. live ETA: re-predict the time left after every stretch, like a nav app.
     "naive_live" = km left / your average speed so far.
     "hillsense_live" = terrain model for the road ahead, scaled by how much
     faster or slower than predicted you have been so far (learned cautiously:
     PACE_PRIOR_S, chosen by cross-validation on public drives only).
     We measure both accuracy and stability (how much the arrival time jumps
     between updates -- the problem that started this project).

Evaluation is honest:
  * 5-fold cross-validation grouped by trace (a drive is never in both train and test)
  * a final test on our own Munnar drives, never used for training or tuning

"hour" is deliberately NOT a feature: each public drive happens at one time,
so hour lets the model memorise individual drivers instead of learning roads.

    python -m hillsense.eta_model
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

TERRAIN = ["grade_pct", "max_grade_pct", "climb_per_km", "descent_per_km",
           "turn_deg_per_km", "hairpins_per_km", "min_radius_m"]
CONTEXT = ["highway_code", "junctions_nearby", "length_m"]
ROAD_CLASSES = ["trunk", "primary", "secondary", "tertiary", "unclassified", "residential"]
MIN_DRIVE_KM = 2.0  # ETA % error on tiny drives is meaningless
LIVE_STOP_M = 2000  # stop scoring live ETA in the last 2 km (everyone is right there)
PACE_PRIOR_S = 3600  # how cautiously to trust the driver's pace so far (picked by public CV)
NAIVE_LIVE_WARMUP_S = 120


def prepare(passes):
    p = passes.copy()
    km = (p["length_m"] / 1000).clip(lower=0.05)
    p["climb_per_km"] = p["climb_m"] / km
    p["descent_per_km"] = p["descent_m"] / km
    p["highway_code"] = p["highway"].map({c: i for i, c in enumerate(ROAD_CLASSES)}).fillna(len(ROAD_CLASSES))
    p["drive_key"] = p["source"] + ":" + p["trace"] + ":" + p["drive"].astype(str)
    return p


class Naive:
    def fit(self, X, y, w):
        self.v = np.average(y, weights=w)
        return self

    def predict(self, X):
        return np.full(len(X), self.v)


class RoadType:
    def fit(self, X, y, w):
        df = pd.DataFrame({"c": X["highway_code"].to_numpy(), "y": y, "w": w})
        self.overall = np.average(y, weights=w)
        self.v = df.groupby("c").apply(lambda g: np.average(g["y"], weights=g["w"]), include_groups=False).to_dict()
        return self

    def predict(self, X):
        return X["highway_code"].map(self.v).fillna(self.overall).to_numpy()


class Trees:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y, w):
        cat = [f == "highway_code" for f in self.features]
        self.m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, categorical_features=cat, random_state=0,
        )
        self.m.fit(X[self.features], y, sample_weight=w)
        return self

    def predict(self, X):
        return np.clip(self.m.predict(X[self.features]), 5, 100)


def models():
    return {
        "naive": lambda: Naive(),
        "road_type": lambda: RoadType(),
        "no_terrain": lambda: Trees(CONTEXT),
        "hillsense": lambda: Trees(TERRAIN + CONTEXT),
    }


def drive_errors(test, pred_kmh):
    """Per drive: true total seconds vs predicted (sum of dist / predicted speed)."""
    t = test.assign(pred_secs=test["dist_m"] / (pred_kmh / 3.6))
    d = t.groupby("drive_key").agg(km=("dist_m", lambda x: x.sum() / 1000), true_s=("secs", "sum"),
                                   pred_s=("pred_secs", "sum"))
    d = d[d["km"] >= MIN_DRIVE_KM]
    d["err_pct"] = 100 * (d["pred_s"] - d["true_s"]).abs() / d["true_s"]
    return d


def cross_validate(pub, k=5):
    """Grouped CV on public drives. Returns per-drive errors for each model."""
    out = {name: [] for name in models()}
    groups = pub["trace"]
    for train_idx, test_idx in GroupKFold(n_splits=k).split(pub, groups=groups):
        tr, te = pub.iloc[train_idx], pub.iloc[test_idx]
        for name, make in models().items():
            m = make().fit(tr, tr["kmh"].to_numpy(), tr["dist_m"].to_numpy())
            out[name].append(drive_errors(te, m.predict(te)))
    return {name: pd.concat(parts) for name, parts in out.items()}


def live_trace(d, pred_kmh, global_kmh, pace_prior_s=PACE_PRIOR_S):
    """Re-estimate time left after every stretch of one drive, four ways."""
    d = d.assign(pred_s=d["dist_m"] / (pred_kmh / 3.6)).reset_index(drop=True)
    done_s = d["secs"].cumsum().shift(fill_value=0)
    done_m = d["dist_m"].cumsum().shift(fill_value=0)
    done_pred = d["pred_s"].cumsum().shift(fill_value=0)
    left_true = d["secs"][::-1].cumsum()[::-1]
    left_m = d["dist_m"][::-1].cumsum()[::-1]
    left_pred = d["pred_s"][::-1].cumsum()[::-1]
    rows = []
    for i in range(len(d)):
        if left_m[i] < LIVE_STOP_M:
            break
        your_speed = done_m[i] / done_s[i] if done_s[i] > NAIVE_LIVE_WARMUP_S else global_kmh / 3.6
        pace = (done_s[i] + pace_prior_s) / (done_pred[i] + pace_prior_s)
        rows.append({
            "elapsed_s": done_s[i], "true_left_s": left_true[i],
            "naive_static": left_m[i] / (global_kmh / 3.6),
            "naive_live": left_m[i] / your_speed,
            "hillsense_static": left_pred[i],
            "hillsense_live": left_pred[i] * pace,
        })
    return pd.DataFrame(rows)


LIVE_METHODS = ["naive_static", "naive_live", "hillsense_static", "hillsense_live"]


def live_summary(traces):
    out = {}
    for k in LIVE_METHODS:
        err, jump = [], []
        for t in traces:
            if len(t) < 3:
                continue
            err.append((100 * (t[k] - t["true_left_s"]).abs() / t["true_left_s"]).mean())
            jump.append((t["elapsed_s"] + t[k]).diff().abs().mean() / 60)
        out[k] = {"drives": len(err), "time_left_error_%": np.mean(err), "avg_arrival_jump_min": np.mean(jump)}
    return pd.DataFrame(out).T.round(2)


def summary(errs):
    return pd.DataFrame({
        name: {"drives": len(e), "median_eta_error_%": e["err_pct"].median(),
               "mean_eta_error_%": e["err_pct"].mean()}
        for name, e in errs.items()
    }).T.round(1)


if __name__ == "__main__":
    passes = prepare(pd.read_csv("data/processed/passes.csv"))
    pub, own = passes[passes["source"] == "public"], passes[passes["source"] == "own"]
    print(f"public: {pub['drive_key'].nunique()} drives, {len(pub)} stretches | "
          f"own (held out): {own['drive_key'].nunique()} drives, {len(own)} stretches\n")

    cv = summary(cross_validate(pub))
    print("1) PRE-TRIP ETA -- 5-fold cross-validation on public drives (grouped by trace):")
    print(cv.to_string(), "\n")

    final, own_rows = {}, []
    for name, make in models().items():
        m = make().fit(pub, pub["kmh"].to_numpy(), pub["dist_m"].to_numpy())
        e = drive_errors(own, m.predict(own))
        final[name] = m
        for key, r in e.iterrows():
            own_rows.append({"model": name, "drive": key.split(":")[-1], "km": round(r["km"], 1),
                             "true_min": round(r["true_s"] / 60, 1), "pred_min": round(r["pred_s"] / 60, 1),
                             "err_%": round(r["err_pct"], 1)})
    own_table = pd.DataFrame(own_rows)
    print("   Pre-trip ETA error (%) on our own Munnar drives (trained on public drives only):")
    print(own_table.pivot(index="drive", columns="model", values="err_%")[list(models())].to_string(), "\n")

    # live ETA: cross-validated on public drives, then on our own drives
    live_cv = []
    for train_idx, test_idx in GroupKFold(n_splits=5).split(pub, groups=pub["trace"]):
        tr, te = pub.iloc[train_idx], pub.iloc[test_idx]
        m = Trees(TERRAIN + CONTEXT).fit(tr, tr["kmh"].to_numpy(), tr["dist_m"].to_numpy())
        g = np.average(tr["kmh"], weights=tr["dist_m"])
        live_cv += [live_trace(d, m.predict(d), g) for _, d in te.groupby("drive_key")]
    g_all = np.average(pub["kmh"], weights=pub["dist_m"])
    live_own = [live_trace(d, final["hillsense"].predict(d), g_all) for _, d in own.groupby("drive_key")]
    live_cv_t, live_own_t = live_summary(live_cv), live_summary(live_own)
    print("2) LIVE ETA (re-estimated after every stretch, like a nav app)")
    print("   cross-validated on public drives:")
    print(live_cv_t.to_string())
    print("   on our own Munnar drives (never used for training or tuning):")
    print(live_own_t.to_string(), "\n")

    # what the full model relies on (permutation importance on the public data)
    from sklearn.inspection import permutation_importance
    hs = final["hillsense"]
    imp = permutation_importance(hs.m, pub[hs.features], pub["kmh"], sample_weight=pub["dist_m"],
                                 n_repeats=5, random_state=0)
    importance = pd.Series(imp.importances_mean, index=hs.features).sort_values(ascending=False)
    print("3) What hillsense relies on (drop in R^2 when a feature is shuffled):")
    print(importance.round(3).to_string())

    Path("models").mkdir(exist_ok=True)
    with open("models/eta_hillsense.pkl", "wb") as f:
        pickle.dump(final["hillsense"], f)
    Path("results").mkdir(exist_ok=True)
    cv.to_csv("results/cv_summary.csv")
    own_table.to_csv("results/own_drives_test.csv", index=False)
    importance.to_csv("results/feature_importance.csv", header=["importance"])
    live_cv_t.to_csv("results/live_eta_cv.csv")
    live_own_t.to_csv("results/live_eta_own.csv")
    json.dump({"cv": cv.to_dict(), "own": own_table.to_dict(orient="records"),
               "live_cv": live_cv_t.to_dict(), "live_own": live_own_t.to_dict()},
              open("results/eta_results.json", "w"), indent=2, default=float)
