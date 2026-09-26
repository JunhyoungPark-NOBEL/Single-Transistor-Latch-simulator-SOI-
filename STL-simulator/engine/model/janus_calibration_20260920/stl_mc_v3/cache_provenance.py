"""Fail-closed provenance for cached off-equilibrium charge/rate grids.

Legacy candidate-only metadata is incompatible until a complete numerical
recomputation verifies it or load_or_build writes a newly generated grid.
"""
from pathlib import Path
import hashlib,json,platform
import numpy as np
import scipy,numba

H=Path(__file__).resolve().parent
ROOT=H.parent
SCHEMA=2
COLUMNS=['u','r','ID','birth_per_s','death_per_s','Qgate','Qexcess','Qbackground','F_A','seed_A','BTBT_A']

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest()

def expected(direction,voltage,u,fold):
    paths=[H/'cache_provenance.py',H/'lu_fpt.py',
        ROOT/'idvd_model_v3/mean_model_voltage_v3.py',
        ROOT/'idvd_model_v3/srh_transport.py',
        ROOT/'idvd_model/mean_model.py',
        ROOT/'high_injection_review/current_carrying_transport.py',
        ROOT.parent/'stl_stochastic_research/process_randomness/standard_mean.py',
        ROOT/'model_review/channel_fit/fit_results.json']
    if direction=='LD':paths.append(H/'ld_fpt.py')
    return dict(schema=SCHEMA,direction=direction,
        closure_version='frozen_v3_offequilibrium_free_hole_inventory_unit_event_v1',
        candidate_sha256=sha256(ROOT/'idvd_model_v3/final_candidate.json'),
        dependency_sha256={str(p.relative_to(ROOT.parent)).replace('\\','/'):sha256(p) for p in paths},
        grid=dict(voltage_shape=list(np.shape(voltage)),u_shape=list(np.shape(u)),
            voltage_sha256=array_sha(voltage),u_sha256=array_sha(u),fold_V=float(fold)),
        columns=COLUMNS,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__))

def load_compatible(grid_path,metadata_path,wanted):
    """Return cached arrays only when provenance AND grid bytes agree."""
    if not grid_path.exists() or not metadata_path.exists():return None
    try:
        meta=json.loads(metadata_path.read_text())
        if meta.get('cache_provenance')!=wanted:return None
        if meta.get('grid_file_sha256')!=sha256(grid_path):return None
        with np.load(grid_path) as f:
            voltage=f['VD'].copy();u=f['u'].copy();rows=f['rows'].copy()
        if array_sha(voltage)!=wanted['grid']['voltage_sha256'] or array_sha(u)!=wanted['grid']['u_sha256']:return None
        if rows.shape!=(len(voltage),len(u),len(COLUMNS)) or not np.all(np.isfinite(rows)):return None
        return voltage,u,rows
    except (OSError,ValueError,KeyError):return None

def attach(metadata,grid_path,wanted):
    return dict(metadata,cache_provenance=wanted,grid_file_sha256=sha256(grid_path))
