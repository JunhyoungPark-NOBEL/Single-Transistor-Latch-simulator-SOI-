"""Physical identities and numerical domain of the independent Simple Model."""
from __future__ import annotations

import inspect
import numpy as np
import pytest

from server import params
from server import simple_model as sm


def packed(**changes):
    p = np.zeros(3038)
    p[:26] = params.build_p(None)[:26]
    p[26:32] = list(params.GEOMETRY.values())
    p[34] = -1
    p[11] = -3.0
    # Geometry-scaled paper starting values with unfit diffusion beta.
    p[36:49] = [2.3, 200e-9, .86e-15*200/650, 44000*650/200,
                2e-16*200/650, 2.35, 4, .2, .0525, -3.35, 1, .5, 100]
    for key, value in changes.items():
        p[int(key.removeprefix('p'))] = value
    return p


def test_diffusion_beta_is_bias_constant():
    p = packed()
    beta = sm.effective_parameters(p)[0]
    for u, r, vg, vbg in [(.05, .1, -2., 0.), (.45, 1.9, -3., .5), (-.03, 2., -2.5, -.5)]:
        p[11], p[32] = vg, vbg
        z = sm.components(u, r, p)
        assert z[3] / z[6] == pytest.approx(beta, rel=2e-15)


def test_diffusion_geometry_area_cancellation():
    base = sm.effective_parameters(packed())
    for length, doping in [(250., params.NA_CM3), (500., 2*params.NA_CM3), (750., .5*params.NA_CM3)]:
        p = packed(p26=length, p31=doping)
        beta, _, _, _, saturation, *_ = sm.effective_parameters(p)
        assert beta/base[0] == pytest.approx(500/length*params.NA_CM3/doping)
        assert saturation/beta == pytest.approx(base[4]/base[0], rel=2e-15)
    for width, tsi in [(400., 50.), (200., 25.)]:
        p = packed(p27=width, p28=tsi)
        e = sm.effective_parameters(p)
        assert e[0] == base[0]
        assert e[4]/e[0] == pytest.approx(base[4]/base[0]*(width/200)*(tsi/50))


def test_first_order_loss_uses_charge_reservoir_not_contact_voltage():
    p = packed()
    beta, tau, cb, resistance, saturation, *_ = sm.effective_parameters(p)
    bias = sm.effective_parameters(p)[-1]
    u, r = .53, 1.7
    z = sm.components(u, r, p)
    assert z[13] == pytest.approx(cb*(u + z[1]*resistance - bias), rel=2e-15)
    assert z[5] == pytest.approx(z[13]/tau, rel=2e-15)
    assert z[7] == 0
    assert abs(z[13] - cb*u) > .01*abs(z[13])
    assert z[2] == pytest.approx(z[1] - z[3] - z[5] - z[6], rel=1e-13)


def test_isolated_body_charge_decays_with_effective_lifetime():
    p = packed(p40=1e-100, p46=0, p11=-3.35, p32=-3.35)
    out = np.empty(11)
    assert sm.evaluate(.1, 0., p, out)
    tau = sm.effective_parameters(p)[1]
    assert out[2] == pytest.approx(-out[3]/tau, rel=2e-15)
    h = tau/10
    q_next = out[3]/(1+h/tau)
    assert (q_next-out[3])/h == pytest.approx(-q_next/tau, rel=3e-15)


def test_gate_charges_share_reservoir_and_conserve_incremental_charge():
    p = packed()
    first, second = np.empty(11), np.empty(11)
    assert sm.evaluate(.3, 1., p, first)
    initial = sm.effective_parameters(p)
    p[11] += .17
    p[32] -= .08
    assert sm.evaluate(.42, 1.3, p, second)
    cg, cbg, cs = initial[7:10]
    assert second[9] == pytest.approx(cg*(p[11]-second[8]), rel=2e-15)
    assert second[10] == pytest.approx(cbg*(p[32]-second[8]), rel=2e-15)
    difference = (second[3]-first[3])+(second[9]-first[9])+(second[10]-first[10])-cs*(second[8]-first[8])
    assert abs(difference) < 1e-30
    assert second[8] == pytest.approx(.42+second[1]*initial[3])


def test_surface_lifetime_and_passive_capacitance_geometry():
    base = sm.effective_parameters(packed())
    thin = sm.effective_parameters(packed(p28=25.))
    assert thin[1] == pytest.approx(base[1]/1.5)
    thick_box = sm.effective_parameters(packed(p30=280.))
    assert thick_box[8] < base[8]
    assert thick_box[7] == base[7]
    assert all(v > 0 for v in base[7:10])
    assert base[2] == sum(base[7:10])


@pytest.mark.parametrize('reverse,doping', [(.1, 1e16), (1.5, params.NA_CM3), (2.2, 1e18)])
def test_lateral_btbt_is_kane_rate_at_peak_field_over_depletion_volume(reverse, doping):
    """Paper script: G(E_peak) of the abrupt junction times the volume W*Tsi*Wd."""
    p = packed(p31=doping)
    lateral, _ = sm.btbt_currents(reverse, p)
    vbi = sm.VT*np.log(1e20*doping/sm.NI_CM3**2) - .2*(p[11]+3.35)
    assert vbi == pytest.approx(sm.junction_vbi(p))
    width = np.sqrt(2*sm.EPS_SI*(vbi+reverse)/(sm.QE*doping))
    peak = 2*(vbi+reverse)/width
    reference = sm.QE*sm.BB_A*peak**2.5*np.exp(-sm.BB_B/peak)*p[27]*p[28]*1e-14*width*(-np.expm1(-reverse/sm.VT))
    assert lateral == pytest.approx(reference, rel=1e-12, abs=1e-100)


def test_gidl_uses_vertical_field_over_scaled_overlap_volume():
    """GIDL: E=(r-V_G+1.2-Eg)/(3 EOT) over gidl_volume_scale*W*5nm*Wt (paper script)."""
    p = packed(p11=-3.35, p32=0.)
    r = 2.2
    _, gidl = sm.btbt_currents(r, p)
    field = (r + 3.35 + 1.2 - 1.12)/(3*14.1e-7)
    depth = min(np.sqrt(2*sm.EPS_SI*1.12/(sm.QE*7e19)), 50e-7)
    volume = 100*200e-7*5e-7*depth
    reference = sm.QE*volume*sm.BB_A*field**2.5*np.exp(-sm.BB_B/field)*(-np.expm1(-r/sm.VT))
    assert gidl == pytest.approx(reference, rel=1e-12)
    assert depth == pytest.approx(4.55e-7, rel=.01)
    # Bare physical volume with the factor set to 1; zero volume switches GIDL off.
    assert sm.btbt_currents(r, packed(p11=-3.35, p48=1.))[1] == pytest.approx(gidl/100, rel=1e-12)
    assert sm.btbt_currents(r, packed(p11=-3.35, p48=0.))[1] == 0
    # Below the tunnelling onset (E <= 0) there is no GIDL; a more negative gate raises it.
    assert sm.btbt_currents(.05, packed(p11=.5))[1] == 0
    assert sm.btbt_currents(r, packed(p11=-4.))[1] > gidl


def test_accumulation_clamps_gate_coupling_at_flat_band():
    """Paper eq. (2): below V_FB the hole layer screens the gate, bias stops following V_G."""
    above = sm.effective_parameters(packed(p11=-3.0))
    flat = sm.effective_parameters(packed(p11=-3.35))
    below = sm.effective_parameters(packed(p11=-4.0))
    beta, tau, cb, resistance, saturation, vbr, eta, cg, cbg, cs, bias = above
    assert bias == pytest.approx((cg*.35 + cbg*3.35)/cb, rel=1e-12)
    assert flat[-1] == pytest.approx(cbg*3.35/cb, rel=1e-12)
    assert below[-1] == flat[-1]
    # The back gate clamps at its own flat band in the same way.
    assert sm.effective_parameters(packed(p32=-5.))[-1] == sm.effective_parameters(packed(p32=-3.35))[-1]
    assert sm.gate_voltages(packed(p11=-4., p32=-5.)) == (-3.35, -3.35)
    # Junction BTBT keeps the flat-band built-in voltage in accumulation.
    assert sm.junction_vbi(packed(p11=-4.)) == sm.junction_vbi(packed(p11=-3.35))
    assert sm.junction_vbi(packed(p11=-3.)) < sm.junction_vbi(packed(p11=-3.35))
    # Gate charges use the same clamped voltages, so the incremental charge stays conserved.
    p = packed(p11=-4.)
    out = np.empty(11)
    assert sm.evaluate(.3, 1., p, out)
    assert out[9] == pytest.approx(cg*(-3.35-out[8]), rel=1e-12)


def _paper_device():
    """Table I device (W 650 nm, T_ox 13 nm, N_body 3e17) with C_B, R_LRS, I_S at Table I values."""
    geo = dict(Lg_nm=500., W_nm=650., Tsi_nm=50., EOT_nm=13., Tbox_nm=140., Nbody_cm3=3e17)
    simple = dict(beta_ref=2.3*3e17/params.NA_CM3, tau_body_s=2e-7, cb_ref_F=1e-15, r_lrs_ref_ohm=1., is_ref_A=1e-16,
                  vbr_ref_V=2.35, avalanche_eta=4., gamma_fg=.2, gamma_bg=.0525, vfb_V=-3.35,
                  btbt_scale=1., surface_fraction=1., gidl_volume_scale=100.)
    p = np.asarray(params.build_p(dict(model="simple", vg=-3., geometry=geo, simple=simple)))
    beta, tau, cb, resistance, saturation, *_ = sm.effective_parameters(p)
    simple["cb_ref_F"] = .86e-15/cb*1e-15
    simple["r_lrs_ref_ohm"] = 44000./resistance
    simple["is_ref_A"] = 2e-16/saturation*1e-16
    return dict(model="simple", vg=-3., geometry=geo, simple=simple)


def _latch_up(device):
    """(V_LU, I_BTBT/I_gen at the LU fold) from the equilibrium branch."""
    p = np.asarray(params.build_p(device))
    b = sm.branch(p, np.linspace(0, 1.12, 601))
    dv = np.diff(b[:, 0])
    k = np.flatnonzero((dv[:-1] > 0) & (dv[1:] < 0))
    if len(k) == 0:
        return np.nan, np.nan
    z = b[k[0]]
    btbt = z[8] + z[9]
    return z[0], btbt/(z[5] + z[6])


def test_paper_device_latch_up_is_bell_shaped_in_gate_voltage():
    """Fig. 5(a)/(b) of the paper: V_LU peaks near the flat band and BTBT takes over below it."""
    dev = _paper_device()
    e = sm.effective_parameters(np.asarray(params.build_p(dev)))
    assert e[2] == pytest.approx(.86e-15) and e[3] == pytest.approx(44000.) and e[4] == pytest.approx(2e-16)
    vg = [-2.2, -2.6, -3.0, -3.2, -3.4, -3.8, -4.0]
    vlu, frac = zip(*(_latch_up(dict(dev, vg=v)) for v in vg))
    assert np.all(np.isfinite(vlu))
    peak = int(np.argmax(vlu))
    assert vg[peak] == -3.2
    assert vlu[peak] == pytest.approx(2.23, abs=.05)
    assert vlu[0] == pytest.approx(1.76, abs=.05)
    assert vlu[-1] == pytest.approx(1.80, abs=.06)
    assert np.all(np.diff(vlu[:peak+1]) > 0) and np.all(np.diff(vlu[peak:]) < 0)
    assert frac[0] < .01 and .5 < frac[3] < .9 and frac[4] > .7
    # Fig. 5(c): a positive back gate lowers V_LU at V_G = -3 V.
    back = [_latch_up(dict(dev, vbg=v))[0] for v in (0., 1., 2.)]
    assert back[0] == pytest.approx(2.21, abs=.05)
    assert back[0] > back[1] > back[2]


def test_breakdown_domain_is_rejected_without_multiplication_cap():
    p = packed()
    out = np.empty(11)
    assert sm.evaluate(.1, p[41]*(1-1e-6), p, out)
    assert not sm.evaluate(.1, p[41], p, out)
    assert np.isnan(out).all()
    assert not sm.evaluate(.1, -.051, p, out)
    assert sm.evaluate(-.1, -.01, p, out)
    assert out[4] == 0
    assert out[7] == 2


def test_width_scales_all_currents_and_charge_without_moving_voltage():
    a = sm.components(.5, 1.7, packed())
    b = sm.components(.5, 1.7, packed(p27=400.))
    assert b[0] == pytest.approx(a[0], rel=2e-15)
    np.testing.assert_allclose(b[[1,2,3,5,6,8,9,13]], 2*a[[1,2,3,5,6,8,9,13]], rtol=3e-15, atol=0)


def test_equilibrium_branches_have_two_folds_and_correct_length_trend():
    folds = []
    for length in [200., 500.]:
        p = packed(p26=length)
        branch = sm.branch(p, np.linspace(0, 1.12, 601))
        assert len(branch) > 400
        assert np.all(np.isfinite(branch))
        assert np.max(np.abs(branch[:,2])/np.maximum(branch[:,1], 1e-20)) < 1e-7
        dv = np.diff(branch[:,0])
        maxima = np.flatnonzero((dv[:-1] > 0)&(dv[1:] < 0))+1
        minima = np.flatnonzero((dv[:-1] < 0)&(dv[1:] > 0))+1
        assert len(maxima) == len(minima) == 1
        assert maxima[0] < minima[0]
        folds.append(branch[[maxima[0], minima[0]],0])
    assert np.all(folds[0] < folds[1])


def test_no_distributed_srh_or_field_table_dependency():
    source = inspect.getsource(sm)
    assert 'import photo_mean' not in source
    assert 'srh_solve(' not in source
    assert 'Field(' not in source
    assert sm.NREF == params.NA_CM3
    out = np.empty(11)
    assert sm.evaluate(.2, 1.0, packed(), out)
