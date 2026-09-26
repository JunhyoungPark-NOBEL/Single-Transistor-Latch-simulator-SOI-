"""Conditional VG projection. Exact compound FPT at center gates; frozen-state GH response.
No output curve fitted. This is not a dynamic slow-state simulation.
"""
from pathlib import Path
import sys,json,time,hashlib,csv
import numpy as np
from scipy.integrate import cumulative_trapezoid
H=Path(__file__).resolve().parent;R=H.parents[1]
sys.path.insert(0,str(R/'hypothesis_study_20260920'))
sys.path.insert(0,str(R/'claude_crosscheck_20260920/joint_model'))
import gate_mean as m
import conditional_table as ct
import check_escape as ce
D=json.loads((R/'claude_crosscheck_20260920/joint_model/refit_3.json').read_text())
C=json.loads((R/'claude_crosscheck_20260920/joint_model/gate_dynamic_calibration.json').read_text())
BASE=np.r_[D['prediction']['parameters'],C['parameters'][0],.001*C['parameters'][1],-2.,1.]
SIGG=C['parameters'][2];SIGE=.001*C['parameters'][3]
MODEL=m.FastModel(D['NA_cm3'],m.load_transport_table()[0]);ct.m=m;ct.model=MODEL;ct.table=MODEL.table
KERNEL=ce.install_kernel(D['NA_cm3'])
if (H/'gate_avalanche_extended.npz').exists():
 zz=np.load(H/'gate_avalanche_extended.npz');ct.cf.rv=zz['reverse_V'];ct.cf.pmf=zz['probability'];ct.cf.K=int(np.flatnonzero(ct.cf.pmf.sum(axis=0)>0)[-1]);ct.cf.pmf=ct.cf.pmf[:,:ct.cf.K+1]
 KERNEL=dict(kernel='gate_avalanche_extended.npz',maximum_reverse_V=float(ct.cf.rv[-1]),N_per_bias=int(zz['N']))
PROB=(np.arange(10001)+.5)/10001
(H/'fpt_nodes').mkdir(exist_ok=True)

def one(vg,delta_g=0.,fine=False,delta_e=0.):
 p=BASE.copy();p[11]=vg;p[9]+=delta_g;p[10]+=delta_e
 window=.5 if vg>-1.11 else .28
 key=f'vg{vg:+.3f}_g{delta_g:+.6f}_fine{int(fine)}_window{int(window*1000)}'
 if delta_e: key+=f'_e{delta_e:+.9f}'
 path=H/'fpt_nodes'/f'{key}.json'
 if path.exists():return json.loads(path.read_text())
 tic=time.perf_counter();clas=MODEL.classify(p,m.state_grid(601));assert clas is not None
 b,i,j,fold=clas;uf=b[i,17]
 step=.002 if fine else .004
 volts=np.arange(max(fold[0]-window,fold[1]+.001),fold[0]-.001,step)
 ug=np.unique(np.round(np.r_[np.linspace(.1,.9,181),np.linspace(uf-.065,uf+.065,61)],12))
 hazards=[];accepted=[];skipped=[];checks=[]
 for vd in volts:
  rows=np.array([ct.state(u,vd,p) for u in ug])
  try:
   xx,ix,r,bt,ii,death=ct.cf.make_lattice(rows,'LU',.06)
   tm,A,check=ct.cf.backward(r,bt,ii,death,'LU')
   h=1/tm[ix] if np.isfinite(tm[ix]) and tm[ix]>0 else np.nan
   accepted.append(vd);hazards.append(h);checks.append(check)
  except (ValueError,IndexError) as exc:
   skipped.append(dict(VD=float(vd),error=type(exc).__name__))
 V=np.array(accepted);h=np.array(hazards);valid=np.isfinite(h)&(h>=1e-4)
 bad=np.flatnonzero(~valid);begin=int(bad[-1]+1) if len(bad) else 0
 if begin>=len(h):raise ValueError(f'No resolved final hazard for {key}')
 h[:begin]=0.;cum=cumulative_trapezoid(h/.4,V,initial=0)
 qv=np.interp(-np.log1p(-PROB),cum,V,right=fold[0])
 # The absorbing 10 nA threshold is crossed only after the unstable branch.
 # Verify deterministic LU current and direct-channel current remain below it.
 peak=b[i]
 rec=dict(VG=float(vg),delta_phiG=float(delta_g),delta_phiE=float(delta_e),fold_V=float(fold[0]),VLD_fold_V=float(fold[1]),mean_V=float(qv.mean()),SD_V=float(qv.std()),
   fast_shift_V=float(qv.mean()-fold[0]),fold_atom=float(np.exp(-cum[-1])),early_mass_bound_if_monotone=float(abs(V[begin]-V[0])*h[begin]/.4),
   I_at_fold_A=float(peak[1]),channel_at_fold_A=float(peak[16]),threshold_A=1e-8,threshold_above_fold_current=bool(peak[1]<1e-8),
   skipped=skipped,voltage=V.tolist(),hazard=h.tolist(),seconds=time.perf_counter()-tic,
   window_V=window,start_hazard_per_s=float(h[begin]),resolved_start_V=float(V[begin]),monotonic_retained_hazard=bool(np.all(np.diff(h[begin:])>=0)),
   max_backward_residual=float(max(c['normalized_linear_residual'] for c in checks)),quantiles=qv[::10].tolist(),probability=PROB[::10].tolist())
 path.write_text(json.dumps(rec,indent=2));print(json.dumps({k:v for k,v in rec.items() if k not in ['skipped','voltage','hazard','quantiles','probability']}),flush=True)
 return rec

def main():
 gates=np.r_[np.arange(-3.2,-1.39,.2),[-1.3,-1.2,-1.15,-1.1,-1.05,-1.,-.95,-.9]]
 rows=[one(float(round(v,3))) for v in gates]
 validation=[one(v,k*SIGG,True) for v in [-3.2,-2.,-1.,-.95] for k in [-1,1]]
 out=dict(center=rows,validation=validation,kernel=KERNEL,sd_phiG_V=SIGG,sd_phiE_V=SIGE,ramp_V_per_s=.4,
    interpretation='Frozen local-state conditional projection; center exact compound-FPT at each listed VG; not dynamic slow-state simulation; no occupancy bell forcing',
    NA_cm3=D['NA_cm3'],base_parameters=BASE.tolist())
 (H/'gate_fpt_summary.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()


