"""Deterministic compute kinds, called in-process (imports the numba engine)."""
from __future__ import annotations

import numpy as np
import pytest

from server.compute import deterministic as D

PAPER = {"preset": "paper"}


@pytest.mark.parametrize("device,expected", [
    ({"preset": "paper", "vg": -2.0}, (3.7037, 2.5979)),
    ({"preset": "paper", "vg": -1.8}, (3.8644, 2.5979)),
    ({"preset": "paper", "vg": -1.8, "light": {"mode": "iph", "iph_pA": 2.63}}, (3.2913, 2.596)),
])
def test_folds_match_validation(device, expected, progress):
    res = D.run_folds({"device": device}, progress)
    assert res["latch"] is True
    assert abs(res["folds"]["V_LU"] - expected[0]) <= 1e-3
    assert abs(res["folds"]["V_LD"] - expected[1]) <= 1e-3


def test_branches_structure(progress):
    res = D.run_branches({"device": PAPER}, progress)
    f = res["folds"]
    hrs, uns, lrs = res["HRS"], res["unstable"], res["LRS"]
    # contiguous split b[:i+1], b[i:j+1], b[j:]
    assert hrs["vd"][-1] == uns["vd"][0] and uns["vd"][-1] == lrs["vd"][0]
    assert np.max(hrs["vd"]) <= f["V_LU"] + 1e-9 and np.min(lrs["vd"]) >= f["V_LD"] - 1e-9
    assert lrs["vd"][-1] <= res["vd_max_V"] + 1.0 + 1e-9
    assert res["full"]["vd"].shape[0] >= hrs["vd"].shape[0] + lrs["vd"].shape[0]
    # fold currents / u by interpolation along u
    assert 1e-12 < f["I_LU"] < 1e-10 and 1e-9 < f["I_LD"] < 1e-7
    assert 0.5 < f["u_LU"] < f["u_LD"] < 0.9
    assert abs(f["window_V"] - (f["V_LU"] - f["V_LD"])) < 1e-12
    # current decomposition: I_D = seed + II + BTBT_j + GIDL + channel + I_PH
    c = hrs["comp"]
    total = c["seed"] + c["ii_total"] + c["btbt_junction"] + c["gidl"] + c["channel"] + c["photo"]
    assert np.allclose(total, hrs["id"], rtol=1e-12, atol=0)
    # quasi-static double sweep: up jumps at V_LU, down jumps at V_LD
    up, dn = res["double_sweep"]["up"], res["double_sweep"]["down"]
    assert up["vd"][0] == 0 and dn["vd"][-1] == 0
    k = np.flatnonzero(np.isclose(up["vd"], f["V_LU"]))
    assert len(k) == 2 and up["id"][k[1]] / up["id"][k[0]] > 1e4
    k = np.flatnonzero(np.isclose(dn["vd"], f["V_LD"]))
    assert len(k) == 2 and dn["id"][k[0]] / dn["id"][k[1]] > 1e4
    assert np.all(np.isfinite(up["id"])) and np.all(np.isfinite(dn["id"]))
    assert len(up["vd"]) <= 2003


def test_branches_no_latch(progress):
    res = D.run_branches({"device": {"preset": "paper", "vg": -0.5}}, progress)
    assert res["latch"] is False
    assert all(v is None for v in res["folds"].values())
    assert len(res["full"]["vd"]) > 10 and len(res["HRS"]["vd"]) == 0
    assert res["warnings"]


def test_branches_vlu_above_sweep(progress):
    res = D.run_branches({"device": PAPER, "sweep": {"vd_max_V": 3.5}}, progress)
    assert any("never latches" in w for w in res["warnings"])
    assert np.nanmax(res["double_sweep"]["up"]["id"]) < 1e-10          # stays on the HRS


def test_photo_power_mode(progress):
    res = D.run_folds({"device": {"preset": "photo", "light": {"power_mW": 3.51}}}, progress)
    assert abs(res["iph_A"] - 2.6325e-12) < 1e-18


def test_charge_balance_three_roots(progress):
    res = D.run_charge_balance({"device": {"preset": "paper", "vg": -2.0}, "vd": 3.2}, progress)
    kinds = [r["kind"] for r in res["roots"]]
    assert kinds == ["stable", "unstable", "stable"]
    u = np.asarray(res["u"])
    for key in ("r", "id", "Q_C", "generation_A", "loss_A", "unit_A", "ii_A", "F_A", "potential"):
        assert len(res[key]) == len(u)
    assert np.all(np.diff(res["Q_C"]) > 0)
    # F = G - L (A) at every row
    assert np.allclose(res["F_A"], res["generation_A"] - res["loss_A"], rtol=1e-6, atol=1e-20)
    # potential zero at the HRS well, barrier at the unstable root
    U = np.asarray(res["potential"])
    assert abs(np.interp(res["roots"][0]["u"], u, U)) < 1e-9
    assert np.interp(res["roots"][1]["u"], u, U) > 10
    assert res["roots"][0]["id"] < res["roots"][1]["id"] < res["roots"][2]["id"]


@pytest.mark.parametrize("vd,n_roots,u_range", [(1.0, 1, (0, 1e-3)), (5.0, 1, (0.95, 1.1))])
def test_charge_balance_monostable(vd, n_roots, u_range, progress):
    res = D.run_charge_balance({"device": PAPER, "vd": vd}, progress)
    assert len(res["roots"]) == n_roots and res["roots"][0]["kind"] == "stable"
    assert u_range[0] <= res["roots"][0]["u"] <= u_range[1]


def test_vg_curve_window(progress):
    res = D.run_vg_curve({"device": PAPER, "vg_min": -4.2, "vg_max": -0.6, "n": 19}, progress)
    w = res["window"]
    assert abs(w["vg_low"] + 3.90) <= 0.010 and abs(w["vg_high"] + 0.815) <= 0.010
    vg, lu = np.asarray(res["vg"]), np.asarray(res["V_LU"], float)
    k = int(np.argmin(np.abs(vg + 2.0)))
    assert abs(vg[k] + 2.0) < 1e-9 and abs(lu[k] - 3.7037) < 1e-3
    assert res["latch"][0] is False and res["latch"][-1] is False
    assert np.isnan(lu[0])


def test_state_row_fallback_matches_engine(progress):
    """The wider r-bracket fallback reproduces S.state wherever S.state succeeds."""
    from server.engine_bridge import S
    p = np.asarray(S.BASE)
    for u in (0.3, 0.6, 0.85):
        a = S.state(u, 3.2, p)
        b = D.state_row(u, 3.2, p)
        assert np.allclose(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError):
        S.state(1.0, 5.0, p)                 # components() is NaN at r = vd - u
    row = D.state_row(1.0, 5.0, p)
    assert np.isfinite(row).all()
    assert abs(D._comp(1.0, row[1], p)[0] - 5.0) < 1e-9     # the solved r reproduces V_D


@pytest.mark.slow
def test_carrier_noise_breakdown():
    from server.compute.validation import carrier_noise_breakdown
    s = {k: v["sd_mV"] for k, v in carrier_noise_breakdown()["subsets"].items()}
    assert abs(s["all"] - 8.03) < 0.1
    for k, e in {"II": 4.6, "BTBT": 2.7, "REC": 4.3, "DIFF": 1.8}.items():
        assert abs(s[k] - e) < 1.0
