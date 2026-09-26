from pathlib import Path
"""Predict the 8 measured conditions with the calibrated (delta_phi_G0, sigma_phi) and the single light->I_PH conversion.
Nothing else changes between conditions except VG and I_PH."""
import json,sys,numpy as np
import photo_fpt as F
cal=json.load(open('c2c_calibration_m18_dark.json'));conv=json.load(open('photo_conversion_fit.json'));R=conv['R_A_per_mW']
dg0,sig=cal['delta_phi_G0_V'],cal['sigma_phi_V'];RATE=float(sys.argv[1]) if len(sys.argv)>1 else 1200.;DIBL=float(sys.argv[2]) if len(sys.argv)>2 else 0.;GAMMA=float(sys.argv[3]) if len(sys.argv)>3 else 0.
meas={x['label']:x for x in json.load(open(str(Path(__file__).resolve().parents[1]/'data'/'measured_stats.json')))}
x,w=np.polynomial.hermite.hermgauss(9);w=w/np.sqrt(np.pi)
rows=[]
for vg in ([-1.8,-1.1] if GAMMA==0 else [-1.1]):
    for P in (0.,1.15,2.55,3.51):
        iph=R*P;lab='%.1fV %.2fmW'%(vg,P);mu=sec=0.;wt=0.;folds=[];fast=[];atoms=[]
        for xi,wi in zip(x,w):
            rec=F.hazard_curve(vg,iph,dg0+np.sqrt(2)*sig*xi,dibl=DIBL,gamma=GAMMA);q=F.quantiles(rec,RATE)
            if q is None:folds.append(np.nan);continue
            mu+=wi*q.mean();sec+=wi*np.mean(q*q);wt+=wi;folds.append(rec['fold_V']);fast.append(q.std());atoms.append(np.mean(q>=rec['fold_V']-1e-9))
        mu/=wt;sd=np.sqrt(sec/wt-mu*mu);folds=np.array(folds);ok=np.isfinite(folds)
        fmu=np.sum(w[ok]*folds[ok])/w[ok].sum();fsd=np.sqrt(np.sum(w[ok]*(folds[ok]-fmu)**2)/w[ok].sum())
        c=folds[4];sens=(folds[5]-folds[3])/(np.sqrt(2)*sig*(x[5]-x[3])) if ok[3] and ok[5] else np.nan
        me=meas['%s %.2fmW'%(('-1.8V' if vg==-1.8 else '-1.1V'),P)]
        r=dict(VG=vg,P_mW=P,I_PH_pA=iph*1e12,model_mean_V=mu,model_sd_mV=sd*1e3,fold_center_V=float(c),fold_mean_V=fmu,fold_sd_mV=fsd*1e3,dfold_dphi=float(sens),
               noise_only_sd_mV=float(np.sqrt(np.sum(w[ok]*np.array(fast)**2)/w[ok].sum())*1e3),weight_without_latch=float(1-wt),meas_mean_V=me['mean_V'],meas_sd_mV=me['sd_mV'])
        rows.append(r);print('VG=%.1f P=%.2f mW (I_PH %.2f pA): model mean %.3f  SD %5.1f mV | fold(center) %.3f dfold/dphi %+.2f  fold-spread %5.1f  noise-only %4.1f  no-latch %.3f | measured %.3f  %5.1f mV'%(vg,P,iph*1e12,mu,sd*1e3,c,sens,fsd*1e3,r['noise_only_sd_mV'],1-wt,me['mean_V'],me['sd_mV']),flush=True)
json.dump(rows,open('predictions_rate%d_dibl%.3f_gamma%.3f.json'%(int(RATE),DIBL,GAMMA),'w'),indent=1)
