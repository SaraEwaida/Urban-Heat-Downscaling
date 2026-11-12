"""Milestone 5: dashboard figure and app."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("plotly")

from test_lightgbm import CFG, make_stack  # noqa: E402
from uhd import dashboard as D  # noqa: E402
from uhd.align import save  # noqa: E402
from uhd.baseline import predict_bilinear  # noqa: E402
from uhd.lgbm import save_predictions  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"


def test_figure_scales_and_links():
    ds = make_stack()
    pred = predict_bilinear(ds.lst_coarse.values, ds.attrs["coarse_factor"])
    fig = D.build_figure(ds, CFG, pred[3], 3, "bilinear", stride=2)
    lay = fig.layout
    # one shared temperature scale for input, prediction and observation; separate error scale
    assert [t.coloraxis for t in fig.data] == ["coloraxis", "coloraxis", "coloraxis", "coloraxis2"]
    assert lay.coloraxis2.cmid == 0 and lay.coloraxis2.cmin == -lay.coloraxis2.cmax
    assert lay.coloraxis.colorscale != lay.coloraxis2.colorscale
    # linked maps: all four panels resolve to one shared x axis and one shared y axis
    for axis in ("x", "y"):
        names = [axis] + [f"{axis}{i}" for i in (2, 3, 4)]
        targets = {getattr(lay, f"{axis}axis{n[1:]}").matches or n for n in names}
        assert len(targets) == 1, (axis, targets)
    # held-out boundary drawn on all four panels
    assert {s.xref for s in lay.shapes} == {"x", "x2", "x3", "x4"}


def test_date_roles():
    ds = make_stack()
    cfg = {**CFG, "unet": {"val_dates": ["2023-06-20"]}}
    roles = dict(D.date_roles(ds, cfg))
    assert roles == {"2023-05-03": "training", "2023-05-11": "training",
                     "2023-06-20": "validation", "2023-08-23": "held-out"}


def test_scene_metrics_split_by_area():
    ds = make_stack()
    pred = predict_bilinear(ds.lst_coarse.values, ds.attrs["coarse_factor"])
    rows = D.scene_metrics(ds, CFG, pred[0], 0)
    assert [r["area"] for r in rows] == ["Training area", "Held-out area"]
    assert all(r["n"] > 0 and r["bias_c"] < 0 for r in rows)      # synthetic coarse input is cold


def test_app_runs_and_switches(tmp_path):
    testing = pytest.importorskip("streamlit.testing.v1")
    ds = make_stack()
    ds.attrs["scenes"] = {}
    save(ds, tmp_path / "stack.zarr")
    pred = predict_bilinear(ds.lst_coarse.values, ds.attrs["coarse_factor"]) + 1.0
    save_predictions(ds, pred, "lightgbm", tmp_path / "pred" / "lightgbm.zarr")
    cfg = {"city": "test_city", "bbox": [0, 0, 1, 1], "search": {}, "coarse": {}, "fine": {}, **CFG,
           "paths": {"stack": str(tmp_path / "stack.zarr"), "predictions_dir": str(tmp_path / "pred")}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))

    import sys
    argv, sys.argv = sys.argv, ["dashboard", "--config", str(tmp_path / "cfg.yaml")]
    try:
        at = testing.AppTest.from_file(str(APP), default_timeout=120).run()
        assert not at.exception
        assert at.radio[0].options == ["Bilinear", "LightGBM"] and at.radio[0].value == "lightgbm"
        at.radio[0].set_value("bilinear").run()
        at.selectbox[0].set_value(0).run()
        assert not at.exception and at.subheader[0].value == "Bilinear on 2023-05-03"
    finally:
        sys.argv = argv


def test_error_scale_is_shared_across_methods():
    ds = make_stack()
    bil = predict_bilinear(ds.lst_coarse.values, ds.attrs["coarse_factor"])
    good = ds.lst_ref.values + 0.1                              # a near-perfect method
    preds = {"bilinear": bil, "good": good}
    emax = D.shared_error_range(ds, preds, 0)
    figs = [D.build_figure(ds, CFG, preds[m][0], 0, m, emax=emax) for m in preds]
    assert figs[0].layout.coloraxis2.cmax == figs[1].layout.coloraxis2.cmax == emax
    assert emax > 1.0                                           # set by the worse method, not the better one
