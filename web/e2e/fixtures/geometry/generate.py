"""Regenerate authentic browser recordings, independent dimension sweeps and the CSVM check.

Run from the repository root: python3 web/e2e/fixtures/geometry/generate.py
This calls the production deterministic solvers directly, bypassing HTTP worker transport.
It does not provide independent experimental validation of the extension.
Outputs: reference/width-400/length-400/tsi-30.json, numerical-report.json, csvm-check.json.
"""
from pathlib import Path
import copy
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
from server import params, jsonutil
from server.payloads import normalize_device
from server.compute.circuit import run_circuit
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

# the n+ source/drain injection scales with the junction area only (fixed emitter, owner decision D8): shorter L or a
# different Nbody changes V_LU/V_LD only through the base, never through the emitter saturation current
nbody = [r for r in rows if r["parameter"] == "Nbody_cm3"]
assert all(r["latch"] for r in nbody)

domain_errors = []
for geometry in [{"Lg_nm": 100}, {"Lg_nm": 150}, {"Nbody_cm3": 1e15}, {"Nbody_cm3": 2e16}, {"Nbody_cm3": 2e18},
                 {"Nbody_cm3": 1e19}]:
    try:
        calculate(geometry)
    except ValueError as exc:
        assert "geometry-domain-unavailable" in str(exc)
        domain_errors.append(dict(geometry=geometry, error=str(exc)))
    else:
        raise AssertionError(f"Expected an explicit geometry domain error for {geometry}")

# the same limits reach the user before a job is queued (HTTP 422; a circuit names the cell)
input_errors = []
for device in [{"geometry": {"Lg_nm": 100}}, {"geometry": {"Lg_nm": 400, "Nbody_cm3": 3e16}},
               {"geometry": {"EOT_nm": 100, "Tbox_nm": 10, "Tsi_nm": 5}, "vbg": 4.0},
               {"geometry": {"Nbody_cm3": 2e18}}]:
    try:
        normalize_device(copy.deepcopy(device), [])
    except ValueError as exc:
        input_errors.append(dict(device=device, error=str(exc)))
    else:
        raise AssertionError(f"Expected an input error for {device}")

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
    "input_errors": input_errors,
    "emitter_injection_scaling": params.geometry_model_metadata({"geometry": {"Lg_nm": 400}})["emitter_injection_scaling"],
}
(OUT / "numerical-report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print("Independent checks passed", flush=True)

# ---- CSVM at L = 400 nm: the Device tab's current-forcing request (web/src/device/forcing.ts csvmPayload) ----
T = 8e-3
csvm_payload = {
    "bench": "custom", "mode": "deterministic",
    "netlist": {"elements": [
        {"type": "I", "name": "Iin", "nodes": ["0", "drain"], "wave": {"kind": "dc", "value": 1e-9}},
        {"type": "C", "name": "Cdrain", "nodes": ["drain", "0"], "value": 1e-12},
        {"type": "V", "name": "VG", "nodes": ["gate", "0"], "wave": {"kind": "dc", "value": -2.0}},
        {"type": "STL", "name": "X1", "nodes": {"d": "drain", "g": "gate", "s": "0"},
         "device": params.resolve_device({"preset": "paper", "geometry": {"Lg_nm": 400}}), "light_pA": None},
    ]},
    "tran": {"t_stop_s": T, "t_start_save_s": 0, "dt_max_s": T / 3000, "dt_min_s": 1e-15, "method": "BE", "reltol": 1e-3},
    "detect": {"i_threshold_A": 1e-8, "hysteresis": 10},
    "probes": ["V(drain)", "I(X1.d)", "X1.q_b", "I(Cdrain)", "I(Iin)"],
}
t0 = time.perf_counter()
csvm = json.loads(jsonutil.dumps(run_circuit(copy.deepcopy(csvm_payload))))
runtime = time.perf_counter() - t0
run = csvm["runs"][0]
t = np.asarray(run["t"], float)
sig = {s["key"]: np.asarray(s["values"], float) for s in run["signals"]}
ups = [e for e in csvm["events"] if e["kind"] == "latch_up"]
downs = [e for e in csvm["events"] if e["kind"] == "latch_down"]
interval = float(np.mean(np.diff([e["t"] for e in ups]))) if len(ups) > 1 else None
kcl = np.abs(sig["I(Iin)"] - sig["I(Cdrain)"] - sig["I(X1.d)"])
check = {
    "provenance": "Direct production run_circuit solve; no synthesized waveforms or HTTP fixture used for computation",
    "payload": csvm_payload,
    "completed": bool(t[-1] >= T * (1 - 1e-9)),
    "t_reached_s": float(t[-1]),
    "sample_count": int(len(t)),
    "runtime_s": runtime,
    "signal_ranges": {k: {"min": float(np.min(v)), "max": float(np.max(v))} for k, v in sig.items()},
    "all_signals_finite": bool(all(np.all(np.isfinite(v)) for v in sig.values())),
    "events": csvm["events"],
    "summary": csvm["summary"],
    "solver_stats": csvm["solver_stats"],
    "elements": [e for e in csvm["elements"] if e["type"] == "STL"],
    "warnings": csvm["warnings"],
    "maximum_recorded_kcl_residual_A": float(kcl.max()),
    "latch_up_count": len(ups),
    "latch_down_count": len(downs),
    "mean_latch_interval_s": interval,
    "latch_frequency_Hz": 1.0 / interval if interval else None,
    "recorded_kcl_note": "KCL uses returned current traces rounded to seven significant figures; the residual is "
                         "evaluated on those recorded traces, not unrounded Newton states.",
    "maximum_recorded_kcl_residual_over_peak_device_current": float(kcl.max() / np.max(np.abs(sig["I(X1.d)"]))),
    "latch_up_voltage_range_V": [min(e["v_d"] for e in ups), max(e["v_d"] for e in ups)] if ups else None,
    "latch_down_voltage_range_V": [min(e["v_d"] for e in downs), max(e["v_d"] for e in downs)] if downs else None,
}
assert check["completed"] and check["all_signals_finite"] and len(ups) >= 2
(OUT / "csvm-check.json").write_text(json.dumps(check, indent=2), encoding="utf8")
print("csvm", {k: check[k] for k in ("latch_up_count", "latch_down_count", "mean_latch_interval_s", "latch_frequency_Hz",
                                     "latch_up_voltage_range_V", "latch_down_voltage_range_V",
                                     "maximum_recorded_kcl_residual_A", "sample_count")}, check["solver_stats"], flush=True)
