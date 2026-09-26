from pathlib import Path
"""H_LOC with a gate-drain-field dependent strength. Calibration at -1.8 V dark (GIDL offset for the mean, log-fluctuation sigma
for the spread); the single field constant kappaF is set so that the -1.1 V dark spread is 54 mV. Everything else predicted."""
import json,sys,numpy as np
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
import setup_photo as S
m=S.m;M=S.MODEL;IP,SD=1.33e-12,0.8;ALOC,ISAT=1.,2e-11
R=json.load(open('photo_conversion_fit.json'))['R_A_per_mW'];meas={x['label']:x for x in json.load(open(str(Path(__file__).resolve().parents[1]/'data'/'measured_stats.json')))};tgt=meas['-1.8V 0.00mW']
noise={(-1.8,0.):5.5,(-1.8,1.15):11.6,(-1.8,2.55):25.2,(-1.8,3.51):30.1,(-1.1,0.):5.3,(-1.1,1.15):23.1,(-1.1,2.55):38.6,(-1.1,3.51):36.6}
xq,wq=np.polynomial.hermite.hermgauss(40);wq=wq/np.sqrt(np.pi);t21=np.linspace(-3,3,21);w21=np.exp(-t21*t21/2);w21/=w21.sum()
EDGE=float(sys.argv[1]) if len(sys.argv)>1 else 0.
def folds(vg,iph,dg0,delta,kF):
    z=M.classify(S.params(vg,iph,dg0,ip=IP,S=SD,aloc=ALOC,isat=ISAT,dloc=delta,kappaF=kF,bulk=EDGE),m.state_grid(601));return (np.nan,np.nan) if z is None else (float(z[3][0]),float(z[3][1]))
ds=np.linspace(-4,4,61)
def calibrate(kF,dg0=-0.27):
    def table(dg0):
        fl=np.array([folds(-1.8,0.,dg0,d,kF)[0] for d in ds]);ok=np.isfinite(fl);return PchipInterpolator(ds[ok],fl[ok]),ds[ok].min(),ds[ok].max()
    def mom(tab,sig):
        f,lo,hi=tab;d=np.sqrt(2)*sig*xq;ins=(d>=lo)&(d<=hi);v=f(np.clip(d,lo,hi));w=wq[ins]/wq[ins].sum();mu=np.sum(w*v[ins]);return mu,np.sqrt(np.sum(w*(v[ins]-mu)**2))
    for it in range(2):
        tab=table(dg0);sig=brentq(lambda s_:mom(tab,s_)[1]-tgt['sd_mV']/1e3,1e-3,2.5,xtol=1e-5);mu,sd=mom(tab,sig);bias=mu-tab[0](0.)
        dg0=brentq(lambda g:folds(-1.8,0.,g,0.,kF)[0]-(tgt['mean_V']-bias),dg0-0.8,dg0+0.8,xtol=1e-5)
    tab=table(dg0);sig=brentq(lambda s_:mom(tab,s_)[1]-tgt['sd_mV']/1e3,1e-3,2.5,xtol=1e-5);mu,sd=mom(tab,sig);return dg0,sig,mu,sd
def cond(vg,P,dg0,sig,kF):
    fv=np.array([folds(vg,R*P,dg0,sig*tj,kF) for tj in t21]);ok=np.isfinite(fv[:,0]);w=w21[ok]/w21[ok].sum()
    m1=float(np.sum(w*fv[ok,0]));s1=float(np.sqrt(np.sum(w*(fv[ok,0]-m1)**2)));m2=float(np.sum(w*fv[ok,1]));s2=float(np.sqrt(np.sum(w*(fv[ok,1]-m2)**2)))
    return m1,1e3*np.sqrt(s1**2+(noise[(vg,P)]*1e-3)**2),m2,1e3*s2
def sd11(kF):
    dg0,sig,mu,sd=calibrate(kF);r=cond(-1.1,0.,dg0,sig,kF);print('   kappaF=%.3f: calib dg0=%+.3f sig=%.3f (mean %.3f sd %.0f) -> -1.1 V dark mean %.3f sd %.0f mV'%(kF,dg0,sig,mu,sd*1e3,r[0],r[1]),flush=True);return r[1]-meas['-1.1V 0.00mW']['sd_mV']
kF=float(sys.argv[2]) if len(sys.argv)>2 else 0.
if kF<0:kF=brentq(sd11,0.2,2.5,xtol=0.05)
dg0,sig,mu,sd=calibrate(kF);out=dict(kappaF=kF,dg0=dg0,sigma_log=sig,conditions=[])
print('selected kappaF = %.3f 1/V ; calibration dg0=%+.3f, sigma_log=%.3f'%(kF,dg0,sig),flush=True)
for vg in (-1.8,-1.1):
    for P in (0.,1.15,2.55,3.51):
        m1,s1,m2,s2=cond(vg,P,dg0,sig,kF);me=meas['%s %.2fmW'%('-1.8V' if vg==-1.8 else '-1.1V',P)]
        out['conditions'].append(dict(VG=vg,P_mW=P,I_PH_pA=R*P*1e12,model_mean_V=m1,model_sd_mV=s1,VLD_mean_V=m2,VLD_sd_mV=s2,meas_mean_V=me['mean_V'],meas_sd_mV=me['sd_mV']))
        print('VG=%.1f P=%.2f: model mean %.3f sd %5.1f (V_LD %.3f, sd %4.1f) | measured %.3f %5.1f'%(vg,P,m1,s1,m2,s2,me['mean_V'],me['sd_mV']),flush=True)
json.dump(out,open('verify_hloc_edge%d_kF%.2f.json'%(int(EDGE),kF),'w'),indent=1)
