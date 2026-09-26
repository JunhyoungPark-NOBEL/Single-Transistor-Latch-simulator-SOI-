"""Mean V_LU and sigma versus V_G (dark and 3.51 mW) for this device's setup: C2C from the -1.8 V dark calibration,
channel body coupling gamma, ramp 1200 V/s. Frozen states: fine Gaussian fold mixture + first-passage noise at GH5 nodes."""
import json,sys,numpy as np
from scipy.interpolate import PchipInterpolator
import photo_fpt as F,setup_photo as S
cal=json.load(open('c2c_calibration_m18_dark.json'));conv=json.load(open('photo_conversion_fit.json'));R=conv['R_A_per_mW']
dg0,sig=cal['delta_phi_G0_V'],cal['sigma_phi_V'];G=float(open('gamma_probe.txt').read());RATE=1200.
x5,w5=np.polynomial.hermite.hermgauss(5);w5=w5/np.sqrt(np.pi)
t=np.linspace(-4,4,81);wt=np.exp(-t*t/2);wt/=wt.sum()
def fold_of(vg,iph,dg):
    z=S.MODEL.classify(S.params(vg,iph,dg,gamma=G),S.m.state_grid(601));return (np.nan,np.nan) if z is None else (float(z[3][0]),float(z[3][1]))
VGS=np.round(np.arange(-3.6,-0.85,0.1),2)
import os
rows=json.load(open('vg_sweep_1200Vps.json')) if os.path.exists('vg_sweep_1200Vps.json') else []
done={(r['P_mW'],r['VG']) for r in rows}
for P in (0.,3.51):
    iph=R*P
    for vg in VGS:
        if (P,float(vg)) in done:continue
        fl=np.array([fold_of(vg,iph,dg0+sig*tj)[0] for tj in t]);ok=np.isfinite(fl)
        if ok.sum()<10:rows.append(dict(P_mW=P,VG=float(vg),latch=False));json.dump(rows,open('vg_sweep_1200Vps.json','w'),indent=1);print('P=%.2f VG=%.2f no latch'%(P,vg),flush=True);continue
        wk=wt[ok]/wt[ok].sum();fmu=float(np.sum(wk*fl[ok]));fsd=float(np.sqrt(np.sum(wk*(fl[ok]-fmu)**2)))
        vld=fold_of(vg,iph,dg0)[1]
        noise=[];shift=[];ws=[]
        for xi,wi in zip(x5,w5):
            rec=F.hazard_curve(vg,iph,dg0+np.sqrt(2)*sig*xi,gamma=G);q=F.quantiles(rec,RATE)
            if q is None:continue
            noise.append(q.std());shift.append(q.mean()-rec['fold_V']);ws.append(wi)
        ws=np.array(ws)/sum(ws);nsd=float(np.sqrt(np.sum(ws*np.array(noise)**2)));msh=float(np.sum(ws*np.array(shift)))
        rows.append(dict(P_mW=P,VG=float(vg),latch=True,I_PH_pA=iph*1e12,mean_VLU_V=fmu+msh,sigma_VLU_mV=1e3*np.sqrt(fsd**2+nsd**2),fold_mean_V=fmu,state_sd_mV=1e3*fsd,noise_sd_mV=1e3*nsd,VLD_fold_V=vld,weight_without_latch=float(1-wt[ok].sum())))
        print('P=%.2f VG=%.2f: mean %.3f sigma %5.1f (state %5.1f, noise %4.1f) VLD %.3f'%(P,vg,fmu+msh,rows[-1]['sigma_VLU_mV'],1e3*fsd,1e3*nsd,vld),flush=True)
        json.dump(rows,open('vg_sweep_1200Vps.json','w'),indent=1)
print('DONE',flush=True)
