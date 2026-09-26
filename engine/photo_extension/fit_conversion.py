from pathlib import Path
"""Light power -> photogeneration current: one linear conversion I_PH = R * P, with R fitted so that the model's
steady-state drain current (photo supply amplified by the floating-body bipolar action) matches the measured
pre-latch plateau of the light-dependent I-V sweeps."""
import json,numpy as np
from scipy.optimize import brentq,minimize_scalar
import setup_photo as S
m=S.m;M=S.MODEL
D=np.load(str(Path(__file__).resolve().parents[1]/'data'/'idvd_light.npy'));P=np.array([0,0.799,1.42,2.55,2.80,3.31])
sel=(D[:,0]>=0.5)&(D[:,0]<=2.0);meas=D[sel,1:].mean(0);dark=meas[0];Iph_meas=meas[1:]-dark;VDs=1.25
def comp(u,r,p):return m.components(u,r,p,M.na,M.vbi,M.rg,M.fg,M.table)
def drain_at(vg,iph,vd=VDs):
    p=S.params(vg,iph)
    def r_of_u(u):
        f=lambda r:comp(u,r,p)[0]-vd
        return brentq(f,0.,vd,xtol=1e-15) if f(0.)<0 else None
    us=np.r_[np.geomspace(1e-14,1e-3,40),np.linspace(1.1e-3,vd,80)];prev=None
    for u in us:
        r=r_of_u(u)
        if r is None:break
        z=comp(u,r,p)
        if prev is not None and prev[1][2]>0 and z[2]<=0:
            uu=brentq(lambda x:comp(x,r_of_u(x),p)[2],prev[0],u,xtol=1e-16);zz=comp(uu,r_of_u(uu),p);return zz[1],uu,zz
        prev=(u,z)
    return prev[1][1],prev[0],prev[1]
for vg in (-1.8,-1.1):
    print('VG=%.1f V, VD=%.2f V: model drain current vs I_PH'%(vg,VDs))
    for iph in (0.5e-12,1e-12,2e-12,3e-12,5e-12):
        I,u,z=drain_at(vg,iph);print('   I_PH=%.1f pA -> I_D=%.2f pA (gain %.2f), u=%.3f V, seed=%.2f pA'%(iph*1e12,I*1e12,I/iph,u,z[3]*1e12))
def cost(R,vg=-1.8):
    return sum((np.log(drain_at(vg,R*p)[0])-np.log(i))**2 for p,i in zip(P[1:],Iph_meas))
res=minimize_scalar(cost,bounds=(1e-13,5e-12),method='bounded');R=res.x
print('\nfitted R = %.3f pA/mW (log-LSQ, VG=-1.8 V, VD=%.2f V)'%(R*1e12,VDs))
for p,i in zip(P[1:],Iph_meas):print('   P=%.2f mW: measured %.2f pA, model %.2f pA (I_PH=%.2f pA)'%(p,i*1e12,drain_at(-1.8,R*p)[0]*1e12,R*p*1e12))
resB=minimize_scalar(lambda R:cost(R,-1.1),bounds=(1e-13,5e-12),method='bounded');print('if the I-V sweeps were at VG=-1.1 V: R = %.3f pA/mW'%(resB.x*1e12))
json.dump(dict(R_A_per_mW=float(R),VD_plateau_V=VDs,VG_assumed=-1.8,P_mW=P.tolist(),measured_plateau_minus_dark_A=Iph_meas.tolist(),
               I_PH_pA={str(p):float(R*p*1e12) for p in (1.15,2.55,3.51)}),open('photo_conversion_fit.json','w'),indent=1)
print('I_PH at 1.15 / 2.55 / 3.51 mW: %.2f / %.2f / %.2f pA'%(R*1.15e12,R*2.55e12,R*3.51e12))
