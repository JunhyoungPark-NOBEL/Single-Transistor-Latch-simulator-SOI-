"""A changed geometry must never reuse the calibrated reference avalanche/noise law."""
import pytest
from server.compute.stochastic import run_hazard, run_sweep_mc, run_vg_curve_stochastic


@pytest.mark.parametrize("compute", [run_hazard, run_sweep_mc, run_vg_curve_stochastic])
@pytest.mark.parametrize("device", [{"geometry": {"Lg_nm": 300}}, {"vbg": 1.0}])
def test_changed_geometry_rejects_reference_noise_kernel(compute, device):
    with pytest.raises(ValueError, match="geometry-stochastic-unavailable"):
        compute({"device": device})
