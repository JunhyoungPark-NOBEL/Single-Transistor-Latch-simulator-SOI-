"""400-cycle Monte Carlo samples at 1200 V/s for the 8 measured conditions.
Per cycle: drain-edge state delta_phi ~ N(delta_phi_G0, sigma_phi) (frozen within the 4 ms sweep), fold from the
photo model, escape voltage drawn from the compound-jump first-passage quantiles (II clusters + BTBT + photo unit events)
of the nearest state node, shifted to that cycle's fold. Channel path with body coupling gamma (device input, set by the
-1.1 V dark mean; no effect at -1.8 V). Same seeds for every condition."""
import json,csv,numpy as np
from scipy.interpolate import PchipInterpolator
import photo_fpt as F,setup_photo as S
cal=json.load(open('c2c_calibration_m18_dark.json'));conv=json.load(open('photo_conversion_fit.json'));R=conv['R_A_per_mW']
dg0,sig=cal['delta_phi_G0_V'],cal['sigma_phi_V'];GAMMA=float(open('gamma_probe.txt').read());RATE=1200.;N=400
x9,w9=np.polynomial.hermite.hermgauss(9)
def fold_of(vg,iph,dg):
    z=S.MODEL.classify(S.params(vg,iph,dg,gamma=GAMMA),S.m.state_grid(601));return np.nan if z is None else float(z[3][0])
conds=[(-1.8,0.),(-1.8,1.15),(-1.8,2.55),(-1.8,3.51),(-1.1,0.),(-1.1,1.15),(-1.1,2.55),(-1.1,3.51)]
rng=np.random.default_rng(20260922);Z=rng.standard_normal(N);U=rng.random(N)   # common random numbers across conditions
out=np.empty((N,8));stats=[]
for k,(vg,P) in enumerate(conds):
    iph=R*P;dgs=dg0+sig*np.linspace(-4.5,4.5,91);fl=np.array([fold_of(vg,iph,d) for d in dgs]);ok=np.isfinite(fl);ftab=PchipInterpolator(dgs[ok],fl[ok])
    nodes=[]
    for xi in x9:
        rec=F.hazard_curve(vg,iph,dg0+np.sqrt(2)*sig*xi,gamma=GAMMA);nodes.append((rec['fold_V'],F.quantiles(rec,RATE)))
    nf=np.array([n[0] for n in nodes]);v=np.empty(N)
    for c in range(N):
        d=dg0+sig*Z[c];f=float(ftab(np.clip(d,dgs[ok].min(),dgs[ok].max())));i=int(np.argmin(abs(nf-f)));q=nodes[i][1]+(f-nf[i]);v[c]=np.interp(U[c],F.PROB,q)
    out[:,k]=v;fine=np.floor(v/.01)*.01+.005
    stats.append(dict(VG=vg,P_mW=P,I_PH_pA=iph*1e12,mean_V=float(v.mean()),sd_mV=float(v.std(ddof=1)*1e3),sd_10mV_grid_mV=float(fine.std(ddof=1)*1e3),fold_center_V=float(ftab(dg0)),
                      p05=float(np.quantile(v,.05)),p95=float(np.quantile(v,.95))))
    print('VG=%.1f P=%.2f mW I_PH=%.2f pA: MC mean %.3f V  SD %5.1f mV (10 mV grid %5.1f)  5-95%% %.2f-%.2f  fold(center) %.3f'%(vg,P,iph*1e12,v.mean(),v.std(ddof=1)*1e3,fine.std(ddof=1)*1e3,stats[-1]['p05'],stats[-1]['p95'],ftab(dg0)),flush=True)
with open('mc_cycles_1200Vps.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['cycle']+['VG=%.1fV %.2fmW'%c for c in conds]);[w.writerow([i+1]+[round(x,5) for x in out[i]]) for i in range(N)]
json.dump(dict(rate_V_per_s=RATE,gamma_body_to_channel=GAMMA,delta_phi_G0_V=dg0,sigma_phi_V=sig,R_pA_per_mW=R*1e12,seed=20260922,stats=stats),open('mc_cycles_1200Vps_stats.json','w'),indent=1)
