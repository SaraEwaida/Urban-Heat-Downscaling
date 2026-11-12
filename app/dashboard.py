"""Streamlit dashboard: coarse input, prediction, observed temperature and error.

Run:  streamlit run app/dashboard.py -- --config configs/abu_dhabi.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd
import streamlit as st

from uhd import dashboard as D
from uhd.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/abu_dhabi.yaml")
args, _ = parser.parse_known_args()

st.set_page_config(page_title="Urban heat downscaling", layout="wide")


@st.cache_resource(show_spinner="Loading aligned data and predictions...")
def _load(config_path: str):
    cfg = load_config(config_path)
    ds, preds = D.load(cfg)
    return cfg, ds, preds


cfg, ds, preds = _load(args.config)
city = cfg["city"].replace("_", " ").title()
st.title(f"{city}: land surface temperature, 1 km to 30 m")

roles = D.date_roles(ds, cfg)
with st.sidebar:
    t = st.selectbox("Date", range(len(roles)), index=len(roles) - 1,
                     format_func=lambda i: f"{roles[i][0]}  ({roles[i][1]} date)")
    methods = list(preds)
    method = st.radio("Method", methods, index=methods.index("lightgbm") if "lightgbm" in methods else 0,
                      format_func=lambda m: D.METHOD_LABELS.get(m, m))
    stride = st.select_slider("Display resolution", options=[1, 2, 4], value=2,
                              format_func=lambda s: f"{30 * s} m")
    st.caption("All four maps are linked: zoom or pan any one. Grey = masked or water. "
               "The dashed line marks the held-out area.")

scene = ds.attrs.get("scenes", {}).get(roles[t][0], {})
if scene:
    st.caption(f"Landsat {scene['landsat']['id']}  ·  MODIS {scene['modis']['id']}")

st.plotly_chart(D.build_figure(ds, cfg, preds[method][t], t, method, stride), width="stretch")

rows = D.scene_metrics(ds, cfg, preds[method][t], t)
table = pd.DataFrame(rows).rename(columns={"area": "Area", "n": "Pixels", "mae_c": "MAE °C",
                                           "rmse_c": "RMSE °C", "bias_c": "Bias °C"})
st.subheader(f"{D.METHOD_LABELS.get(method, method)} on {roles[t][0]}")
st.dataframe(table.style.format({"Pixels": "{:,}", "MAE °C": "{:.2f}", "RMSE °C": "{:.2f}", "Bias °C": "{:+.2f}"}),
             hide_index=True)
st.caption("Error = prediction minus Landsat; positive is too warm. Scored on clear land pixels with a valid MODIS cell.")
