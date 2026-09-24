from pathlib import Path
"""C2C calibration at VG=-1.8 V dark only: drain-edge state offset (mean) and amplitude (SD) from this device's 400-cycle record.
Frozen states (ramp 1200 V/s << state kinetics), V_LU = escape quantile ~ fold(delta_phi)."""
import json,numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import PchipInterpolator
import setup_photo as S
m=S.m;M=S.MODEL
meas=json.load(open(str(Path(__file__).resolve().parents[1]/'data'/'measured_stats.json')));tgt=[x for x in meas if x['label']=='-1.8V 0.00mW'][0]
def fold_of(vg,iph,dg):
    z=M.classify(S.params(vg,iph,dg),m.state_grid(601));return (np.nan,np.nan) if z is None else (float(z[3][0]),float(z[3][1]))
dgs=np.round(np.arange(-1.5,2.01,.05),3);folds=np.array([fold_of(-1.8,0.,d)[0] for d in dgs]);ok=np.isfinite(folds)
print('fold(delta_phi) at VG=-1.8 dark: latch exists for delta_phi in [%.2f, %.2f]'%(dgs[ok].min(),dgs[ok].max()))
for d in (-1.0,-0.5,0.,0.5,1.0,1.5):print('   delta_phi=%+.1f V -> fold %.3f V'%(d,folds[np.argmin(abs(dgs-d))]))
f=PchipInterpolator(dgs[ok],folds[ok])
x,w=np.polynomial.hermite.hermgauss(40);w=w/np.sqrt(np.pi)
def moments(dg0,sig):
    d=dg0+np.sqrt(2)*sig*x;inside=(d>=dgs[ok].min())&(d<=dgs[ok].max());v=np.where(inside,f(np.clip(d,dgs[ok].min(),dgs[ok].max())),np.nan)
    wv=w[inside]/w[inside].sum();mu=np.sum(wv*v[inside]);sd=np.sqrt(np.sum(wv*(v[inside]-mu)**2));return mu,sd,1-w[inside].sum()
sol=fsolve(lambda q:[moments(q[0],abs(q[1]))[0]-tgt['mean_V'],moments(q[0],abs(q[1]))[1]-tgt['sd_mV']/1e3],[0.05,0.2],xtol=1e-10)
dg0,sig=float(sol[0]),float(abs(sol[1]));mu,sd,lost=moments(dg0,sig)
print('calibrated: delta_phi_G0 = %+.4f V, sigma_phi = %.4f V -> mean %.4f V, SD %.1f mV, weight outside latch range %.2e'%(dg0,sig,mu,sd*1e3,lost))
print('(paper device: offset %+.4f V, sigma_phi %.4f V)'%(S.BASE[9],S.SIGG))
json.dump(dict(delta_phi_G0_V=dg0,sigma_phi_V=sig,target=tgt,model_mean_V=mu,model_sd_mV=sd*1e3,fold_table=dict(delta_phi=dgs[ok].tolist(),fold=folds[ok].tolist())),open('c2c_calibration_m18_dark.json','w'),indent=1)
