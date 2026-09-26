"""Geometry-updated reduced BJT/II/BTBT/loss model; no synthetic jitter.

All currents A, potentials V, transport quantities in cm units internally.
The channel is frozen from low-Vd IDVG and has no invented body coupling.
State u is a quasi-Fermi/injection proxy, not a solved electrostatic potential.
"""
from pathlib import Path
import sys,json
from dataclasses import dataclass,asdict
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq,minimize_scalar

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'stl_stochastic_research'))
from process_randomness.standard_mean import van_overstraeten_300k,Q,KB

T=300.;VT=KB*T/Q;EPS0=8.8541878128e-12
WIDTH_M=200e-9;LENGTH_M=500e-9;TSI_M=50e-9;EOT_M=14.1e-9
AREA_CM2=WIDTH_M*TSI_M*1e4
L_BODY_CM=.8*LENGTH_M*100
L_JUNCTION_CM=.2*LENGTH_M*100
MU_BASE_CM2_VS=450.;DIFFUSIVITY_CM2_S=MU_BASE_CM2_VS*VT
COX_F=3.9*EPS0/EOT_M*WIDTH_M*LENGTH_M
NI_CM3=1e10;ND_CM3=1e20
BB_A=4e14;BB_B=19e6
CHANNEL=json.loads((HERE.parent/'model_review/channel_fit/fit_results.json').read_text())['selected_candidate']['params']

def channel_current(vd):
    vg=-2.;ov=vg-CHANNEL['Vth0_V'];n=CHANNEL['n']
    pp=ov/n;f=np.logaddexp(0,pp/(2*VT));r=np.logaddexp(0,(pp-np.asarray(vd))/(2*VT))
    vv=n*VT*np.logaddexp(0,ov/(n*VT))
    beta=CHANNEL['beta0_A_V2']/(1+CHANNEL['theta_per_V']*vv)
    return 2*n*beta*VT**2*(f-r)*(f+r)

class Field:
    def __init__(self,na_cm3,space_points=501,reverse_points=1501):
        self.na=float(na_cm3)
        self.vbi=VT*np.log(ND_CM3*self.na/NI_CM3**2)
        eps_cm=11.7*EPS0/100
        rr=np.linspace(0,5.,reverse_points)
        width=np.sqrt(2*eps_cm*(rr+self.vbi)/(Q*self.na))
        peak=2*(rr+self.vbi)/width
        zz=np.linspace(0,1,space_points)
        field=peak[:,None]*zz
        a,b=van_overstraeten_300k(field)
        expint=cumulative_trapezoid(a-b,zz,axis=1,initial=0)*width[:,None]
        den=1-width*np.trapezoid(a*np.exp(-expint),zz,axis=1)
        valid=(den>1e-3)&(peak<=1.2e6)
        stop=np.flatnonzero(~valid)
        end=int(stop[0]) if len(stop) else len(rr)
        if end<5: raise ValueError('Insufficient finite subcritical local-II domain')
        rr=rr[:end];width=width[:end];peak=peak[:end];field=field[:end]
        generation=BB_A*field**2.5*np.exp(-BB_B/np.maximum(field,1.))
        bb=Q*AREA_CM2*width*np.trapezoid(generation,zz,axis=1)
        self.max_reverse=float(rr[-1]);self.rr=rr
        self.gain=PchipInterpolator(rr,1/den[:end],extrapolate=False)
        self.btbt=PchipInterpolator(rr,bb,extrapolate=False)
        self.peak=PchipInterpolator(rr,peak,extrapolate=False)
        self.width=PchipInterpolator(rr,width,extrapolate=False)
        self.max_gain=float(1/den[end-1])
        self.limits=dict(max_reverse_V=self.max_reverse,max_peak_V_cm=float(peak[-1]),max_gain=self.max_gain,
          margin_min=float(den[end-1]),space_points=space_points,reverse_points=reverse_points)

@dataclass
class Parameters:
    beta_diff:float=10.
    tau_bulk_s:float=6.64e-10
    tau_junction_s:float=3.16e-8
    Rseries_ohm:float=30000.
    gidl_field_length_nm:float=42.3
    gidl_active_length_nm:float=5.

class Model:
    def __init__(self,field,parameters):
        self.field=field;self.p=parameters;self.na=field.na
        self.n0=2*NI_CM3**2/(self.na+np.sqrt(self.na**2+4*NI_CM3**2))
        self.p0=self.na+self.n0
        # Assumed highly-doped drain-side tunnelling region; not body NA.
        self.gidl_depth_cm=min(np.sqrt(2*(11.7*EPS0/100)*1.12/(Q*7e19)),TSI_M*100)
        self.gidl_volume_cm3=WIDTH_M*100*parameters.gidl_active_length_nm*1e-7*self.gidl_depth_cm
        self.k=L_BODY_CM/np.sqrt(DIFFUSIVITY_CM2_S*parameters.tau_bulk_s)
    def excess(self,u):
        prod=NI_CM3**2*np.expm1(np.asarray(u)/VT);s=self.n0+self.p0
        return 2*prod/(s+np.sqrt(s*s+4*prod))
    def components(self,u,r):
        u,r=np.broadcast_arrays(np.asarray(u,float),np.asarray(r,float));p=self.p
        delta=self.excess(u)
        ref=Q*AREA_CM2*DIFFUSIVITY_CM2_S/L_BODY_CM*delta
        seed=ref*self.k/np.sinh(self.k)
        emitter=ref*self.k/np.tanh(self.k)
        bulk=ref*self.k*np.tanh(self.k/2)
        pair=Q*AREA_CM2*delta*np.sqrt(DIFFUSIVITY_CM2_S*p.tau_bulk_s)*np.tanh(self.k/2)
        # Mid-intrinsic-level symmetric SRH in separate junction volume.
        junction=Q*AREA_CM2*L_JUNCTION_CM*NI_CM3/(2*p.tau_junction_s)*np.expm1(u/(2*VT))
        diffusion=ref/p.beta_diff
        gain=self.field.gain(r);bjt_ii=(gain-1)*seed
        eg=np.maximum((u+r+2.-.3-1.12)/(p.gidl_field_length_nm*1e-7),0)
        gidl=Q*self.gidl_volume_cm3*BB_A*eg**2.5*np.exp(-BB_B/np.maximum(eg,1.))
        bbj=self.field.btbt(r);btbt=bbj+gidl
        ch=channel_current(u+r)
        current=seed+bjt_ii+btbt+ch
        loss=diffusion+bulk+junction
        return dict(u_V=u,reverse_V=r,internal_drain_V=u+r,drain_V=u+r+p.Rseries_ohm*current,
            drain_A=current,reference_A=ref,seed_A=seed,emitter_A=emitter,II_BJT_A=bjt_ii,II_channel_A=np.zeros_like(u),
            channel_A=ch,BTBT_A=btbt,BTBT_junction_A=bbj,GIDL_A=gidl,diffusion_A=diffusion,bulk_A=bulk,
            junction_A=junction,total_loss_A=loss,net_A=bjt_ii+btbt-loss,pair_charge_C=pair,
            excess_cm3=delta,injection_ratio=delta/self.na,mean_gain=gain,peak_field_V_cm=self.field.peak(r),
            gidl_field_V_cm=eg,depletion_width_cm=self.field.width(r))
    def branch(self,points=701):
        u=np.unique(np.r_[np.geomspace(1e-12,.02,110),np.linspace(.02,1.1,points)])
        lo=np.zeros_like(u);hi=np.full_like(u,self.field.max_reverse)
        fl=self.components(u,lo)['net_A'];fh=self.components(u,hi)['net_A']
        valid=(fl<=0)&(fh>=0)
        u=u[valid];lo=lo[valid];hi=hi[valid]
        for _ in range(39):
            mid=(lo+hi)/2;f=self.components(u,mid)['net_A'];yes=f>0
            hi=np.where(yes,mid,hi);lo=np.where(yes,lo,mid)
        if len(u)<8:return None
        out=self.components(u,(lo+hi)/2)
        v=out['drain_V'];dv=np.diff(v)
        maxima=np.flatnonzero((dv[:-1]>0)&(dv[1:]<0))+1
        minima=np.flatnonzero((dv[:-1]<0)&(dv[1:]>0))+1
        if not len(maxima) or not len(minima):return dict(data=out,folds=None)
        i=int(maxima[0]);after=minima[minima>i]
        if not len(after):return dict(data=out,folds=None)
        j=int(after[0]);folds=[]
        for index,kind in [(i,'VLU'),(j,'VLD')]:
            coef=np.polyfit(u[index-1:index+2]-u[index],v[index-1:index+2],2)
            du=-coef[1]/(2*coef[0]);uu=u[index]+du
            fold={k:float(np.interp(uu,u,val)) for k,val in out.items()}
            fold.update(kind=kind,drain_V=float(np.polyval(coef,du)),u_V=float(uu),index=index)
            folds.append(fold)
        return dict(data=out,folds=folds,hrs_slice=slice(0,i+1),lrs_slice=slice(j,None))
    def at_u(self,u):
        r=brentq(lambda x: float(self.components(u,x)['net_A']),0,self.field.max_reverse,xtol=2e-13)
        return {k:float(v) for k,v in self.components(u,r).items()}
    def refine_folds(self,b):
        out=[];u=b['data']['u_V']
        for f in b['folds']:
            i=f['index'];sign=-1 if f['kind']=='VLU' else 1
            fit=minimize_scalar(lambda x: sign*self.at_u(float(x))['drain_V'],bounds=(u[max(0,i-3)],u[min(len(u)-1,i+3)]),method='bounded',options={'xatol':1e-12})
            point=self.at_u(float(fit.x));point['kind']=f['kind'];out.append(point)
        return out

def geometry_audit(na):
    f=Field(na)
    return dict(NA_cm3=na,L_m=LENGTH_M,W_m=WIDTH_M,Tsi_m=TSI_M,EOT_m=EOT_M,
       emitter_and_collector_area_cm2=AREA_CM2,body_length_cm=L_BODY_CM,junction_length_cm=L_JUNCTION_CM,
       body_volume_cm3=AREA_CM2*L_BODY_CM,junction_volume_cm3=AREA_CM2*L_JUNCTION_CM,
       Cgeom_F=COX_F,base_mobility_cm2_Vs=MU_BASE_CM2_VS,D_base_cm2_s=DIFFUSIVITY_CM2_S,
       Vbi_V=f.vbi,ND_assumed_cm3=ND_CM3,ni_cm3=NI_CM3,
       inherited_R_scaling_ohm=5e-3*(1e17/na)**.5*LENGTH_M/(TSI_M*WIDTH_M),
       field_limits=f.limits)
