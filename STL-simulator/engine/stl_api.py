"""Minimal Python API over the deterministic + stochastic STL model (paper tables) and the photo extension.
Run `python3 stl_api.py` for a smoke test; the numbers printed must match docs/VALIDATION.md."""
import sys,json,numpy as np
from pathlib import Path
H=Path(__file__).resolve().parent;J=H/'model'/'janus_calibration_20260920'
sys.path.insert(0,str(J/'reader_fig3_20260921'/'gate_model'));sys.path.insert(0,str(J/'claude_crosscheck_20260920'/'joint_model'));sys.path.insert(0,str(H/'photo_extension'))
import setup_photo as S            # photo-extended copy of the paper model; p[13]=I_PH, p[14..25] = 0 reproduce the paper model exactly
m=S.m;MODEL=S.MODEL
def branches(vg,iph=0.,dg=0.,de=0.,grid=1201,**kw):
    """Steady-state solutions traced by the body potential: columns 0=V_D, 1=I_D, 16=channel, 17=u (source-body bias), 18=r (drain-junction bias). Returns dict with HRS/LRS arrays and folds (V_LU, V_LD)."""
    z=MODEL.classify(S.params(vg,iph,dg,de,**kw),m.state_grid(grid))
    if z is None:return None
    b,i,j,fold=z;return dict(HRS=b[:i+1],LRS=b[j:],V_LU=float(fold[0]),V_LD=float(fold[1]))
def folds(vg,iph=0.,dg=0.,de=0.,**kw):
    r=branches(vg,iph,dg,de,601,**kw);return (np.nan,np.nan) if r is None else (r['V_LU'],r['V_LD'])
def hazard(vg,iph=0.,dg=0.,de=0.,rate=None,**kw):
    """Compound-jump first-passage hazard h(V_D) below the latch-up fold (II clusters + unit events incl. photo). Cached under photo_extension/photo_nodes."""
    import photo_fpt as F;rec=F.hazard_curve(vg,iph,dg,de,**kw)
    return rec if rate is None else (rec,F.quantiles(rec,rate))
def sweeps(n=100,seed=2026092920,mode='dynamic',rate=.4,dv=.002):
    """Paper-device Monte Carlo sweeps (0->4 V up, 4->0 down) with evolving local states; returns V_LU/V_LD arrays (n each)."""
    import gate_dynamic_compare as g;z,r=g.simulate(n=n,seed=seed,mode=mode,dv=dv,rate=rate);return dict(V_LU=z['transition'][0,:,1],V_LD=z['transition'][1,:,1],records=r['records'])
if __name__=='__main__':
    print('paper model, VG=-2 V dark: folds V_LU/V_LD =',np.round(folds(-2.0),4),'(expected 3.7037 / 2.5979)')
    print('paper model, VG=-1.8 V dark: folds =',np.round(folds(-1.8),4),'(expected 3.8644 / 2.5979)')
    print('photo, VG=-1.8 V, I_PH=2.63 pA: folds =',np.round(folds(-1.8,2.63e-12),4),'(expected 3.2913 / 2.5959)')
    br=branches(-2.0);print('branches: HRS %d pts (V_D %.2f..%.3f), LRS %d pts (V_D %.3f..%.2f), I at 4 V = %.2e A'%(len(br['HRS']),br['HRS'][0,0],br['HRS'][-1,0],len(br['LRS']),br['LRS'][0,0],br['LRS'][-1,0],np.exp(np.interp(4.,br['LRS'][:,0],np.log(br['LRS'][:,1])))))
    rec,q=hazard(-2.0,rate=.4);print('FPT node VG=-2 dark, 0.4 V/s: mean V_LU %.4f, SD %.2f mV (expected ~3.6442 / 6.8 mV with centre states)'%(q.mean(),q.std()*1e3))
    sw=sweeps(n=100);print('dynamic MC 100 sweeps, seed 2026092920: mean V_LU %.4f SD %.1f mV | mean V_LD %.4f SD %.1f mV (expected ~3.63 / 120 | 2.70 / 19)'%(np.nanmean(sw['V_LU']),np.nanstd(sw['V_LU'],ddof=1)*1e3,np.nanmean(sw['V_LD']),np.nanstd(sw['V_LD'],ddof=1)*1e3))
