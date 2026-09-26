"""Export lookup tables for a JavaScript/WASM port of the deterministic core.
For each (VG, I_PH): on a (u, r) grid -> V_D(u,r), I_D(u,r), F(u,r) = net hole current into the body (A), Q_B(u,r) (C),
plus the unit-event current (GIDL + junction BTBT + photo) and the II-cluster current, so that dQ_B/dt = F can be
integrated in any circuit simulator and the stochastic increments (Eq. 2) generated from the same rows.
Also exports M(r) (multiplication), the II cluster pmf vs reverse voltage, and the calibrated parameters."""
import sys,json,numpy as np
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import stl_api as A
S=A.S;m=A.m;M=A.MODEL;ct=S.ct
def table(vg,iph,us=np.linspace(0.,1.0,101),rs=np.linspace(0.,5.0,101),**kw):
    p=S.params(vg,iph,**kw);VD=np.full((len(us),len(rs)),np.nan);ID=VD.copy();F=VD.copy();QB=VD.copy();UNIT=VD.copy();II=VD.copy();CH=VD.copy()
    for i,u in enumerate(us):
        for j,r in enumerate(rs):
            z=m.components(u,r,p,M.na,M.vbi,M.rg,M.fg,M.table)
            if not np.isfinite(z[0]):continue
            psi=u-m.VT*np.log1p(z[10]);ratio=(p[5]*1e-7/(m.TSI_M*100))*(p[7]*1e-7/z[11]);qb=(z[13]-m.COX_F*u)/(1+ratio);qa=qb*ratio
            VD[i,j]=z[0];ID[i,j]=z[1];F[i,j]=z[2];QB[i,j]=qb+qa;UNIT[i,j]=z[8]+z[9]+z[18];II[i,j]=(z[1]-z[3]-z[16])-(z[8]+z[9]+z[18]);CH[i,j]=z[16]
    return dict(VG=vg,I_PH=iph,u=us,r=rs,V_D=VD,I_D=ID,F_net_hole_A=F,Q_B_C=QB,unit_event_A=UNIT,II_hole_A=II,channel_A=CH)
if __name__=='__main__':
    out=H/'data'/'tables';out.mkdir(exist_ok=True)
    conds=[(-2.0,0.),(-1.8,0.),(-1.8,2.63e-12),(-1.1,0.)]
    for vg,iph in conds:
        t=table(vg,iph);name='table_VG%+.1f_IPH%.2fpA'%(vg,iph*1e12);np.savez_compressed(out/(name+'.npz'),**t)
        ok=np.isfinite(t['V_D']);print(name,'valid cells %d/%d, V_D range %.2f..%.2f, I_D range %.1e..%.1e'%(ok.sum(),ok.size,np.nanmin(t['V_D']),np.nanmax(t['V_D']),np.nanmin(t['I_D']),np.nanmax(t['I_D'])),flush=True)
    rg=M.rg;np.savez_compressed(out/'multiplication_and_clusters.npz',reverse_V=rg,M=M.fg[0],junction_BTBT=M.fg[1],cluster_reverse_V=ct.cf.rv,cluster_pmf=ct.cf.pmf)
    json.dump(dict(parameter_vector=S.BASE.tolist(),indices={'0':'beta (diffusion ratio)','1':'tau_bulk_s','2':'tau_junction_s','3':'Rcontact_ohm','4':'lGIDL_nm','5':'t_access_nm','6':'NA_access_cm3','7':'L_access_nm','8':'tau_p/tau_n','9':'phi_gidl mean offset (V)','10':'phi_emitter mean offset (V)','11':'VG (V)','12':'channel II scale','13':'I_PH (A)','14':'DIBL eta (V/V)','15':'body-to-channel gamma','16':'slope kappa (1/V)','17':'high-VD channel seed I_p at VG=-1.8 V (A)','18':'its slope S (V/dec)','19':'junction potential offset (V)','20':'log scale of (M-1)','21':'local avalanche strength','22':'local path saturation (A)','23':'local path log fluctuation','24':'local path carrier definition (0 edge incl. channel, 1 bulk, 2 edge excl. channel)','25':'kappaF (1/V)'},
              NA_cm3=S.D['NA_cm3'],sigma_phi_G_V=S.SIGG,sigma_phi_E_V=S.SIGE,kinetics=S.C['kinetic_fit'] if 'kinetic_fit' in S.C else None),open(out/'parameters.json','w'),indent=1)
    print('written to',out)
