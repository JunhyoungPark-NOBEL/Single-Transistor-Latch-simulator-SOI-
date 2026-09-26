"""Compound-jump first passage (II clusters + unit events incl. photogeneration) for the photo-extended model,
integrated along a ramp of arbitrary rate. Mirrors gate_fpt.one(); nodes cached under photo_nodes/."""
import json,time,numpy as np
from pathlib import Path
from scipy.integrate import cumulative_trapezoid
import setup_photo as S
m=S.m;M=S.MODEL;ct=S.ct;H=Path(__file__).resolve().parent;(H/'photo_nodes').mkdir(exist_ok=True)
PROB=(np.arange(10001)+.5)/10001
def hazard_curve(vg,iph,dg=0.,de=0.,window=.28,step=.004,dibl=0.,gamma=0.,kappa=0.,ip=0.,Sdec=1.):
    key=f'vg{vg:+.3f}_iph{iph*1e12:.4f}pA_g{dg:+.6f}_e{de:+.9f}_w{int(window*1000)}_s{int(step*1000)}'+(f'_dibl{dibl:.4f}' if dibl else '')+(f'_gamma{gamma:.4f}' if gamma else '')+(f'_kappa{kappa:.4f}' if kappa else '')+(f'_ip{ip*1e12:.4f}pA_S{Sdec:.4f}' if ip else '')
    path=H/'photo_nodes'/f'{key}.json'
    if path.exists():return json.loads(path.read_text())
    tic=time.perf_counter();p=S.params(vg,iph,dg,de,dibl,gamma,kappa,ip,Sdec);clas=M.classify(p,m.state_grid(601))
    if clas is None:
        rec=dict(key=key,VG=vg,I_PH=iph,dg=dg,de=de,fold_V=None,VLD_fold_V=None,voltage=[],hazard=[],seconds=time.perf_counter()-tic);path.write_text(json.dumps(rec));return rec
    b,i,j,fold=clas;uf=b[i,17]
    volts=np.arange(max(fold[0]-window,fold[1]+.001),fold[0]-.001,step)
    ug=np.unique(np.round(np.r_[np.linspace(.1,.9,181),np.linspace(uf-.065,uf+.065,61)],12))
    V=[];Hz=[];skipped=0
    for vd in volts:
        try:
            rows=np.array([S.state(u,vd,p) for u in ug])
            xx,ix,r,bt,ii,death=ct.cf.make_lattice(rows,'LU',.06);tm,A,check=ct.cf.backward(r,bt,ii,death,'LU')
            h=1/tm[ix] if np.isfinite(tm[ix]) and tm[ix]>0 else np.nan
        except (ValueError,IndexError,AssertionError):
            skipped+=1;continue
        V.append(float(vd));Hz.append(float(h))
    rec=dict(key=key,VG=vg,I_PH=iph,dg=dg,de=de,fold_V=float(fold[0]),VLD_fold_V=float(fold[1]),I_at_fold_A=float(b[i,1]),channel_at_fold_A=float(b[i,16]),
             voltage=V,hazard=Hz,skipped=skipped,seconds=time.perf_counter()-tic)
    path.write_text(json.dumps(rec));return rec
def quantiles(rec,rate):
    """Escape quantiles for a ramp of `rate` V/s from a stored hazard curve. Above-fold mass sits at the fold (no post-fold delay in this model)."""
    if rec['fold_V'] is None:return None
    if len(rec['voltage'])<2:return np.full(len(PROB),rec['fold_V'])
    V=np.array(rec['voltage']);h=np.array(rec['hazard']);valid=np.isfinite(h)&(h>=1e-4);bad=np.flatnonzero(~valid);begin=int(bad[-1]+1) if len(bad) else 0
    if begin>=len(h):return np.full(len(PROB),rec['fold_V'])
    h=h.copy();h[:begin]=0.;cum=cumulative_trapezoid(h/rate,V,initial=0)
    return np.interp(-np.log1p(-PROB),cum,V,right=rec['fold_V'])
