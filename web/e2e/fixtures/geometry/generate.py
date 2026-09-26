"""Regenerate authentic browser recordings and independent dimension sweeps.

Run from the repository root: python3 web/e2e/fixtures/geometry/generate.py
This calls the production deterministic solver directly, bypassing HTTP worker transport.
It does not provide independent experimental validation of the extension.
"""
from pathlib import Path
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
from server import params, jsonutil
from server.compute.deterministic import run_branches, state_row, _pvec
from server.geometry_model import constants_from_p, gate_charge_offset, backgate_charge, channel_current

OUT = Path(__file__).resolve().parent

def calculate(geometry, **bias):
    payload = {"device": params.resolve_device({"preset": "paper", "geometry": geometry, **bias}),
               "sweep": copy.deepcopy(params.PRESETS["paper"]["sweep"])}
    return payload, run_branches(payload)

records = {}
for name, geometry in [("reference", {}), ("width-400", {"W_nm": 400}),
                       ("length-400", {"Lg_nm": 400}), ("tsi-30", {"Tsi_nm": 30})]:
    payload, result = calculate(geometry)
    records[name] = (payload, result)
    (OUT / f"{name}.json").write_bytes(jsonutil.dumps({"payload": payload, "result": result}))
    print(name, result["folds"], flush=True)

reference = records["reference"][1]
wide = records["width-400"][1]
width_errors = {}
for branch in ["HRS", "unstable", "LRS", "full"]:
    a = reference[branch]
    b = wide[branch]
    nonzero = a["id"] > 0
    width_errors[branch] = {
        "max_vd_difference_V": float(np.nanmax(np.abs(a["vd"] - b["vd"]))),
        "max_current_ratio_error": float(np.nanmax(np.abs(b["id"][nonzero] / a["id"][nonzero] / 2 - 1))),
    }
assert max(x["max_vd_difference_V"] for x in width_errors.values()) < 1e-8
assert max(x["max_current_ratio_error"] for x in width_errors.values()) < 1e-8
historical = json.loads((ROOT / "web/src/devices/fixtures/export-reference.json").read_text())["result"]
assert reference["folds"] == historical["folds"]

cases = {"Lg_nm": [200, 300, 400, 500, 700], "Tsi_nm": [5, 10, 20, 30, 50, 70],
         "EOT_nm": [10, 14.1, 20], "Tbox_nm": [70, 140, 280],
         "Nbody_cm3": [1e17, params.NA_CM3, 3e17]}
rows = []
for key, values in cases.items():
    for value in values:
        _, result = calculate({key: value})
        row = dict(parameter=key, value=value, latch=result["latch"],
                   VLU=result["folds"]["V_LU"], VLD=result["folds"]["V_LD"],
                   warnings=result["warnings"])
        rows.append(row)
        print(row, flush=True)

lengths = [r for r in rows if r["parameter"] == "Lg_nm"]
silicon = [r for r in rows if r["parameter"] == "Tsi_nm"]
assert all(b["VLU"] > a["VLU"] and b["VLD"] > a["VLD"] for a, b in zip(lengths, lengths[1:]))
assert all(b["VLU"] < a["VLU"] and b["VLD"] < a["VLD"] for a, b in zip(silicon, silicon[1:]))

# Measure the thin-body upturn per nm, rather than just total change.
thin_slope = (silicon[0]["VLU"] - silicon[1]["VLU"]) / 5
thick_slope = (silicon[3]["VLU"] - silicon[4]["VLU"]) / 20
assert thin_slope > thick_slope

storage = []
for g in [{}, {"W_nm": 400}, {"Tbox_nm": 70}, {"Tbox_nm": 280}, {"EOT_nm": 10}, {"EOT_nm": 20}]:
    p = _pvec(params.resolve_device({"geometry": g}))
    row = state_row(0.6, 3.3, p)
    assert row is not None
    storage.append(dict(changed=g, C_body_F=float(constants_from_p(p)[4]),
                        body_capacitive_charge_C=float(row[5]),
                        excess_charge_C=float(row[6]), dopant_charge_C=float(row[7]),
                        I_D_A=float(row[2])))
assert abs(storage[1]["body_capacitive_charge_C"] / storage[0]["body_capacitive_charge_C"] - 2) < 1e-12
assert storage[2]["C_body_F"] > storage[0]["C_body_F"] > storage[3]["C_body_F"]
assert storage[4]["C_body_F"] > storage[0]["C_body_F"] > storage[5]["C_body_F"]

backgate = []
for tbox in [70, 140, 280]:
    device = params.resolve_device({"geometry": {"Tbox_nm": tbox}, "vg": -0.8, "vbg": 1.0})
    p = _pvec(device)
    p_zero = _pvec(params.resolve_device({"geometry": {"Tbox_nm": tbox}, "vg": -0.8, "vbg": 0.0}))
    _, result = calculate({"Tbox_nm": tbox}, vg=-0.8, vbg=1.0)
    row = dict(Tbox_nm=tbox, VG_V=-0.8, VBG_V=1.0,
               delta_front_overdrive_V=device["geometry"]["EOT_nm"] / (tbox + device["geometry"]["Tsi_nm"] / 3),
               C_body_F=float(constants_from_p(p)[4]),
               backgate_charge_C=float(backgate_charge(p)), gate_charge_offset_C=float(gate_charge_offset(p)),
               channel_current_at_u0p6_r2p7_A=float(channel_current(0.6, 2.7, p)),
               zero_backgate_channel_current_A=float(channel_current(0.6, 2.7, p_zero)),
               VLU=result["folds"]["V_LU"], VLD=result["folds"]["V_LD"])
    backgate.append(row)
    print("backgate", row, flush=True)
assert all(a["delta_front_overdrive_V"] > b["delta_front_overdrive_V"] for a, b in zip(backgate, backgate[1:]))
assert all(a["channel_current_at_u0p6_r2p7_A"] > b["channel_current_at_u0p6_r2p7_A"] for a, b in zip(backgate, backgate[1:]))

domain_errors = []
for geometry in [{"Lg_nm": 100}, {"Lg_nm": 150}, {"Nbody_cm3": 1e15}, {"Nbody_cm3": 1e19}]:
    try:
        calculate(geometry)
    except ValueError as exc:
        assert "geometry-domain-unavailable" in str(exc)
        domain_errors.append(dict(geometry=geometry, error=str(exc)))
    else:
        raise AssertionError(f"Expected an explicit geometry domain error for {geometry}")

report = {
    "scope": "Independent invocation of production solver; trends and dimensional consistency, not TCAD/measurement fit validation",
    "reference_geometry": params.GEOMETRY,
    "width_double_errors": width_errors,
    "thin_body_VLU_slope_V_per_nm": thin_slope,
    "moderate_body_VLU_slope_V_per_nm": thick_slope,
    "cases": rows,
    "storage_at_u_0p6_VD_3p3": storage,
    "backgate_at_VG_minus_0p8_VBG_1": backgate,
    "domain_errors": domain_errors,
}
(OUT / "numerical-report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print("Independent checks passed", flush=True)
