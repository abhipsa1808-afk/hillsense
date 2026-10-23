"""Make the README figures (privacy-safe: built from demo/demo_drives.json,
which already hides 500 m around the start and end of every drive).

    python -m hillsense.make_figures      # writes docs/*.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAW, HS, ACTUAL, ROAD = "#e4572e", "#1f6fd1", "#444444", "#d9d9d9"
DOCS = Path("docs")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def eta_replay(drive):
    r = pd.DataFrame(drive["replay"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, zoom in zip(axes, [False, True]):
        ax.plot(r.elapsed_min, r.true_arrival_min, color=ACTUAL, ls="--", lw=1.6, label="actual arrival")
        ax.plot(r.elapsed_min, r.naive_live_min, color=RAW, lw=2.2, label="simple: km left ÷ your speed so far")
        ax.plot(r.elapsed_min, r.hillsense_min, color=HS, lw=2.2, label="hillsense")
        ax.set_xlabel("minutes driven")
        style(ax)
        if zoom:
            top = 2 * r.true_arrival_min.max()
            ax.set_ylim(0, top)
            ax.set_title("Zoomed in", fontsize=11)
        else:
            ax.set_title("Full range", fontsize=11)
            ax.set_ylabel("predicted arrival (min after start)")
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(f"Live ETA replay: {drive['name']}, {drive['road_km']} km, {drive['climb_m']} m climb",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(DOCS / "eta_replay.png", dpi=140)
    plt.close(fig)


def tracking(drive, roads=None):
    raw, hs = np.array(drive["raw"]), np.array(drive["matched"])
    fig, ax = plt.subplots(figsize=(10, 6.5))
    if roads is not None:
        for lat, lon in roads:
            ax.plot(lon, lat, color=ROAD, lw=0.6, zorder=1)
    ax.plot(hs[:, 1], hs[:, 0], color=HS, lw=7, alpha=0.4, solid_capstyle="round", label="hillsense (on real roads)", zorder=2)
    ax.plot(raw[:, 1], raw[:, 0], color=RAW, lw=1.1, label="raw phone GPS", zorder=3)
    pad = 0.004
    ax.set_xlim(min(raw[:, 1].min(), hs[:, 1].min()) - pad, max(raw[:, 1].max(), hs[:, 1].max()) + pad)
    ax.set_ylim(min(raw[:, 0].min(), hs[:, 0].min()) - pad, max(raw[:, 0].max(), hs[:, 0].max()) + pad)
    ax.set_aspect(1 / np.cos(np.radians(hs[:, 0].mean())))
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(f"{drive['name']}: raw GPS vs hillsense (500 m around start/end hidden for privacy)", fontsize=11)
    fig.tight_layout()
    fig.savefig(DOCS / "tracking.png", dpi=140)
    plt.close(fig)


def importance():
    imp = pd.read_csv("results/feature_importance.csv", index_col=0)["importance"]
    nice = {
        "highway_code": "road type", "turn_deg_per_km": "how twisty", "junctions_nearby": "how built-up",
        "min_radius_m": "tightest bend", "hairpins_per_km": "hairpins", "climb_per_km": "climb",
        "descent_per_km": "descent", "grade_pct": "average slope", "max_grade_pct": "steepest stretch",
        "length_m": "segment length",
    }
    imp = imp.rename(index=nice).sort_values()
    terrain = {"how twisty", "tightest bend", "hairpins", "climb", "descent", "average slope", "steepest stretch"}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(imp.index, imp.values, color=[HS if i in terrain else "#8aa4c8" for i in imp.index])
    ax.set_xlabel("drop in R² when the feature is shuffled")
    ax.set_title("What slows traffic around Munnar (dark = road-shape features)", fontsize=11)
    style(ax)
    fig.tight_layout()
    fig.savefig(DOCS / "feature_importance.png", dpi=140)
    plt.close(fig)


def road_lines():
    """Background roads near the drive, if the OSM file is available locally."""
    try:
        from hillsense.road_network import RoadNetwork

        net = RoadNetwork("data/osm/munnar_drive.graphml")
        return [net.to_latlon(e.xy[:, 0], e.xy[:, 1]) for e in net.edges]
    except Exception:  # figure still works without the background
        return None


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    demo = json.loads(Path("demo/demo_drives.json").read_text())
    main = max(demo["drives"], key=lambda d: d["road_km"])
    eta_replay(main)
    tracking(main, road_lines())
    importance()
    print("wrote", ", ".join(sorted(p.name for p in DOCS.glob("*.png"))))
