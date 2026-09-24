from pathlib import Path
"""Where does the cycle-to-cycle state act?  Four levers, each calibrated at -1.8 V dark (mean via the GIDL offset,
spread via the lever), then the eight conditions and the latch-down spread.  Base: paper tables + high-VD channel
seed (I_p=1.33 pA at -1.8 V, S=0.8 V/dec -> -1.1 V dark 3.42 V).  Frozen states, 1200 V/s (noise part in quadrature)."""
import json,sys,numpy as np
from scipy.optimize import fsolve,brentq
from scipy.interpolate import PchipInterpolator
import setup_photo as S
m=S.m;M=S.MODEL;IP,SD=1.33e-12,0.8
conv=json.load(open('photo_conversion_fit.json'));R=conv['R_A_per_mW']
meas={x['label']:x for x in json.load(open(str(Path(__file__).resolve().parents[1]/'data'/'measured_stats.json')))};tgt=meas['-1.8V 0.00mW']
noise={(-1.8,0.):5.5,(-1.8,1.15):11.6,(-1.8,2.55):25.2,(-1.8,3.51):30.1,(-1.1,0.):5.3,(-1.1,1.15):23.1,(-1.1,2.55):38.6,(-1.1,3.51):36.6}
xq,wq=np.polynomial.hermite.hermgauss(40);wq=wq/np.sqrt(np.pi);t21=np.linspace(-3,3,21);w21=np.exp(-t21*t21/2);w21/=w21.sum()
LEVERS={'H_LOCB local avalanche path on BULK carriers (BJT seed + photo, a=0.3, sat 10 pA), channel and surface GIDL excluded':dict(kind='dlocb'),'H_G gate-edge potential -> GIDL field':dict(kind='dg'),'H_J junction potential -> M and junction BTBT':dict(kind='dj'),
        'H_M global multiplication (M-1) scale':dict(kind='dm'),'H_LOC local avalanche path on the seeds (sat 20 pA)':dict(kind='dloc')}
def folds(vg,iph,dg0,kind,delta):
    kw=dict(dg=dg0)
    if kind=='dg':kw['dg']=dg0+delta
    elif kind=='dj':kw['dj']=delta
    elif kind=='dm':kw['dm']=delta
    elif kind=='dloc':kw.update(aloc=1.,isat=2e-11,dloc=delta)
    elif kind=='dlocb':kw.update(aloc=0.3,isat=1e-11,dloc=delta,bulk=1.)
    z=M.classify(S.params(vg,iph,ip=IP,S=SD,**kw),m.state_grid(601));return (np.nan,np.nan) if z is None else (float(z[3][0]),float(z[3][1]))
out=[]
sel=sys.argv[1:] 
for name,lv in LEVERS.items():
    if sel and lv['kind'] not in sel:continue
    kind=lv['kind'];rng=dict(dg=(-1.5,2.5),dj=(-0.8,0.8),dm=(-2.5,2.5),dloc=(-4.,4.),dlocb=(-4.,4.))[kind];ds=np.linspace(rng[0],rng[1],81)
    def table(dg0):
        fl=np.array([folds(-1.8,0.,dg0,kind,d)[0] for d in ds]);ok=np.isfinite(fl);return PchipInterpolator(ds[ok],fl[ok]),ds[ok].min(),ds[ok].max()
    def mom_t(tab,sig):
        f,lo,hi=tab;d=np.sqrt(2)*sig*xq;ins=(d>=lo)&(d<=hi);v=f(np.clip(d,lo,hi));w=wq[ins]/wq[ins].sum();mu=np.sum(w*v[ins]);return mu,np.sqrt(np.sum(w*(v[ins]-mu)**2))
    def solve_sig(tab):
        try:return brentq(lambda s_:mom_t(tab,s_)[1]-tgt['sd_mV']/1e3,1e-4,(tab[2]-tab[1])/2,xtol=1e-6)
        except ValueError:return (tab[2]-tab[1])/2
    dg0=0.0744
    for it in range(2):
        tab=table(dg0);sig=solve_sig(tab);mu,sd=mom_t(tab,sig);bias=mu-tab[0](0.)
        dg0=brentq(lambda g:folds(-1.8,0.,g,kind,0.)[0]-(tgt['mean_V']-bias),dg0-1.0,dg0+1.0,xtol=1e-5)
    tab=table(dg0);sig=solve_sig(tab);mu,sd=mom_t(tab,sig)
    conds=[]
    for vg in (-1.8,-1.1):
        for P in (0.,1.15,2.55,3.51):
            iph=R*P;fv=np.array([folds(vg,iph,dg0,kind,sig*tj) for tj in t21]);ok=np.isfinite(fv[:,0]);w=w21[ok]/w21[ok].sum()
            m1=float(np.sum(w*fv[ok,0]));s1=float(np.sqrt(np.sum(w*(fv[ok,0]-m1)**2)));m2=float(np.sum(w*fv[ok,1]));s2=float(np.sqrt(np.sum(w*(fv[ok,1]-m2)**2)))
            me=meas['%s %.2fmW'%('-1.8V' if vg==-1.8 else '-1.1V',P)]
            conds.append(dict(VG=vg,P_mW=P,mean_V=m1,sd_mV=1e3*np.sqrt(s1**2+(noise[(vg,P)]*1e-3)**2),VLD_mean_V=m2,VLD_sd_mV=1e3*s2,meas_mean_V=me['mean_V'],meas_sd_mV=me['sd_mV']))
    out.append(dict(hypothesis=name,kind=kind,dg0=dg0,sigma_lever=sig,calib_check=[mu,sd*1e3],conditions=conds))
    c=conds;print('%s\n   calibration: GIDL offset %+.3f, lever sigma %.3f -> mean %.3f sd %.0f mV; sigma(V_LD) at -1.8 dark = %.0f mV'%(name,dg0,sig,mu,sd*1e3,c[0]['VLD_sd_mV']),flush=True)
    print('   -1.8 V: mean %.3f/%.3f/%.3f/%.3f  sigma %3.0f/%3.0f/%3.0f/%3.0f   | measured 3.806/3.500/3.336/3.073, 173/177/119/134'%(c[0]['mean_V'],c[1]['mean_V'],c[2]['mean_V'],c[3]['mean_V'],c[0]['sd_mV'],c[1]['sd_mV'],c[2]['sd_mV'],c[3]['sd_mV']),flush=True)
    print('   -1.1 V: mean %.3f/%.3f/%.3f/%.3f  sigma %3.0f/%3.0f/%3.0f/%3.0f   | measured 3.408/3.273/3.098/2.933, 54/46/43/34'%(c[4]['mean_V'],c[5]['mean_V'],c[6]['mean_V'],c[7]['mean_V'],c[4]['sd_mV'],c[5]['sd_mV'],c[6]['sd_mV'],c[7]['sd_mV']),flush=True)
    json.dump(out,open('hypotheses_%s.json'%kind,'w'),indent=1)
