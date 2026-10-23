"""hillsense demo: steadier GPS tracking and ETA on hill roads (Munnar, Kerala).

Run locally:  streamlit run app.py
Data shown here is pre-computed by `python -m hillsense.make_demo_data`.
"""

import json
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).parent
RAW_COLOUR, HILLSENSE_COLOUR = "#e4572e", "#1f6fd1"

st.set_page_config(page_title="hillsense", page_icon="⛰️", layout="wide")


@st.cache_data
def load():
    demo = json.loads((ROOT / "demo/demo_drives.json").read_text())
    res = ROOT / "results"
    return (
        demo,
        pd.read_csv(res / "live_eta_cv.csv", index_col=0),
        pd.read_csv(res / "cv_summary.csv", index_col=0),
        pd.read_csv(res / "feature_importance.csv", index_col=0),
    )


demo, live_cv, pretrip_cv, importance = load()

st.title("⛰️ hillsense")
st.markdown(
    "**Steadier GPS tracking and arrival times on hill roads.** "
    "Driving up to my hotel in Munnar, Google Maps kept changing the time and distance left; "
    "at one point it couldn't find a route at all. hillsense cleans the phone's GPS, snaps it onto "
    "the real roads, and predicts travel time from the shape of the road ahead: curves, hairpins, "
    "climbs and how built-up it is. Built from my own drives plus public OpenStreetMap GPS traces."
)

drives = {d["name"]: d for d in demo["drives"]}
choice = st.selectbox("Pick one of my Munnar drives (22 Oct 2026)", list(drives), index=len(drives) - 1)
d = drives[choice]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw phone GPS, whole log", f"{d['logged_raw_km']} km")
c2.metric("hillsense, on real roads", f"{d['road_km']} km",
          delta=f"{d['road_km'] - d['logged_raw_km']:+.1f} km of GPS noise removed", delta_color="off")
c3.metric("Height change", f"{d['climb_m']} m" if d["climb_m"] is not None else "n/a")
c4.metric("Drive time", f"{d['minutes']:.0f} min")

# ---------------------------------------------------------------- map
st.subheader("1. Tracking: raw GPS vs hillsense")
st.caption(
    f"Red: what the phone recorded. Blue: hillsense (trip detection, Kalman filter, HMM map-matching). "
    f"For privacy, {demo['privacy_radius_m']:.0f} m around the start and end of each drive is hidden."
)
centre = pd.DataFrame(d["matched"], columns=["lat", "lon"]).mean()
m = folium.Map(location=[centre["lat"], centre["lon"]], zoom_start=13, tiles="OpenStreetMap")
folium.PolyLine(d["raw"], color=RAW_COLOUR, weight=2, opacity=0.8, tooltip="raw phone GPS").add_to(m)
folium.PolyLine(d["matched"], color=HILLSENSE_COLOUR, weight=4, opacity=0.9, tooltip="hillsense").add_to(m)
m.fit_bounds(d["matched"])
st_folium(m, height=460, use_container_width=True, returned_objects=[])

# ---------------------------------------------------------------- live ETA replay
st.subheader("2. Live arrival time: does it stay steady?")
st.caption(
    "Replay of the drive. After every stretch of road, each method re-estimates when you'll arrive. "
    "A good ETA is a flat line close to the dashed one. "
    "Red is a simple method (km left ÷ your average speed so far), not Google Maps, which also uses live traffic. "
    "Times are moving time: stops longer than a crawl are left out."
)
r = pd.DataFrame(d["replay"]).rename(columns={
    "elapsed_min": "minutes driven",
    "true_arrival_min": "actual arrival",
    "naive_live_min": "simple: km left ÷ your speed so far",
    "hillsense_min": "hillsense",
})
cols = ["actual arrival", "simple: km left ÷ your speed so far", "hillsense"]
st.line_chart(r, x="minutes driven", y=cols, color=["#555555", RAW_COLOUR, HILLSENSE_COLOUR],
              y_label="predicted arrival (min after start)")

t = st.slider("Scrub through the drive (minutes driven)", 0.0, float(r["minutes driven"].max()),
              0.0, step=0.5)
row = r.iloc[(r["minutes driven"] - t).abs().argmin()]
s1, s2, s3 = st.columns(3)
s1.metric("Actual arrival", f"{row['actual arrival']:.0f} min")
s2.metric("Simple method says", f"{row[cols[1]]:.0f} min",
          delta=f"{row[cols[1]] - row['actual arrival']:+.0f} min off", delta_color="off")
s3.metric("hillsense says", f"{row['hillsense']:.0f} min",
          delta=f"{row['hillsense'] - row['actual arrival']:+.0f} min off", delta_color="off")

# ---------------------------------------------------------------- results
st.subheader("3. Results on drives the model never saw")
left, right = st.columns(2)
with left:
    st.markdown("**Live ETA**: 5-fold cross-validation on public Munnar drives, grouped by drive")
    show = live_cv.loc[["naive_live", "hillsense_live"], ["time_left_error_%", "avg_arrival_jump_min"]]
    show.index = ["Simple: km left ÷ your speed so far", "hillsense"]
    show.columns = ["Error in time left (%)", "Avg arrival-time jump per update (min)"]
    st.dataframe(show.round(2), width="stretch")
    st.markdown("**Before the trip** (median ETA error, %)")
    pre = pretrip_cv[["median_eta_error_%"]].rename(index={
        "naive": "One average speed", "road_type": "Speed per road type",
        "no_terrain": "Trees, no terrain", "hillsense": "hillsense"}, columns={"median_eta_error_%": "Median error (%)"})
    st.dataframe(pre, width="stretch")
with right:
    st.markdown("**What slows traffic around Munnar** (permutation importance)")
    nice = {
        "highway_code": "road type", "turn_deg_per_km": "how twisty", "junctions_nearby": "how built-up",
        "min_radius_m": "tightest bend", "hairpins_per_km": "hairpins", "climb_per_km": "climb",
        "descent_per_km": "descent", "grade_pct": "average slope", "max_grade_pct": "steepest stretch",
        "length_m": "segment length",
    }
    imp = importance.rename(index=nice)
    st.bar_chart(imp, horizontal=True, color=HILLSENSE_COLOUR)
    st.caption("Curves and town traffic matter more than climbs. About 29% of speed differences come down "
               "to the driver, vehicle and traffic on the day, which no road map can know in advance.")

with st.expander("How it works"):
    st.markdown(
        "1. **Trip detection**: keeps only real driving (drops the phone's jumpy fixes once parked indoors).\n"
        "2. **Kalman filter** (from scratch): smooths GPS and rejects physically impossible jumps.\n"
        "3. **HMM map-matching** (from scratch, Newson & Krumm 2009): snaps the track onto OpenStreetMap roads, "
        "so hairpin loops 30 m apart don't get confused.\n"
        "4. **Road features**: slope from Copernicus 30 m elevation data, curvature, hairpins, tightest bend, "
        "junction density, road type.\n"
        "5. **ETA model**: gradient-boosted trees trained on 52 public car drives (866 km of road), tested "
        "with cross-validation grouped by drive and on my own drives, which it never trained on.\n\n"
        "Data: © OpenStreetMap contributors (ODbL) · Copernicus DEM GLO-30 © DLR/Airbus, provided under "
        "COPERNICUS by the European Union and ESA."
    )

st.caption("Built by Abhipsa Rout · github.com/abhipsa1808-afk/hillsense")
