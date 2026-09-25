"""Reference-normalized geometry extension of the frozen photo_mean kernel.

The original 26 calibration entries are unchanged. Geometry is appended in nm/cm^-3;
per-device field lookup data follows. This module never changes engine globals, allowing
heterogeneous devices in one circuit. Exact reference geometry dispatches to frozen code.
"""
from functools import lru_cache
import numpy as np
from numba import njit
import photo_mean as original
from photo_mean import Q, VT, EPS0, NI_CM3, DN, BB_A, BB_B, srh_solve
from server import params

FIELD_SIZE = 1501
PACK_SIZE = 36 + 2 * FIELD_SIZE

@lru_cache(maxsize=32)
def _field(na):
    f=original.Field(na)
    return float(f.vbi), f.rr, np.array([f.gain(f.rr), f.btbt(f.rr)])

def pack_p(p, force=False):
    p=np.asarray(p,dtype=float)
    if len(p)>33:
        return p
    if len(p)==26 and not force:
        return p
    g=np.array([params.GEOMETRY[k] for k in params.GEOMETRY_KEYS]) if len(p)==26 else p[26:32]
    out=np.zeros(PACK_SIZE)
    out[:26]=p[:26];out[26:32]=g
    out[32]=p[32] if len(p)==33 else 0.0
    # domain checks; server.payloads.check_geometry_domain runs the L/Nbody and VBG ones before a job is queued
    vbi=VT*np.log(1e20*float(g[5])/NI_CM3**2)
    wd0=np.sqrt(2*(11.7*EPS0/100)*vbi/(Q*float(g[5])))
    if g[0]*1e-7 <= 2*wd0+1e-7:
        raise ValueError("geometry-domain-unavailable: this L/Nbody combination fully depletes the lateral "
                         "neutral base assumed by this compact model; increase L or Nbody")
    if abs(g[3]/(g[4]+g[2]/3.0)*out[32]) > params.BACKGATE_SHIFT_MAX_V:
        raise ValueError("geometry-domain-unavailable: back-gate coupling beyond the linear (depleted back-interface) "
                         "range; reduce |V_BG| or EOT/Tbox")
    try:
        vb, rg, fg=_field(float(g[5]))
    except ValueError as exc:
        raise ValueError("geometry-domain-unavailable: the specified Nbody is outside the finite avalanche-field "
                         "domain of this compact model") from exc
    if rg[-1] < params.FIELD_MIN_REVERSE_V:
        # the tabulated (finite, subcritical) avalanche field ends before the latch-up bias range: a truncated
        # table would be reported as a physical "no latch"
        raise ValueError("geometry-domain-unavailable: the avalanche-field table for this Nbody "
                         f"({float(g[5]):.3g} cm^-3) ends at a reverse bias of {rg[-1]:.2f} V "
                         f"(< {params.FIELD_MIN_REVERSE_V:g} V); lower Nbody")
    out[33]=vb;out[34]=len(rg);out[35]=rg[1]-rg[0]
    out[36:36+len(rg)]=fg[0]
    # Field.btbt was integrated with the frozen junction area; scale its area here.
    area_ratio=g[1]*g[2]/(200.*50.)
    out[36+FIELD_SIZE:36+FIELD_SIZE+len(rg)]=fg[1]*area_ratio
    return out

@njit(cache=True)
def constants_from_p(p):
    if len(p)<32:
        return (original.LENGTH_M*100., original.WIDTH_M*100., original.TSI_M*100.,
                original.AREA_CM2, original.COX_F, params_NA, params_VBI)
    length=p[26]*1e-7;width=p[27]*1e-7;tsi=p[28]*1e-7
    # Reference-anchored split-gate charge: Cf*(psi-VG) + (Cb-Cb0)*psi - Cb*VBG.
    # Cb0 is the reference-calibration counterterm, not a negative physical capacitor.
    area_m2=(p[27]*1e-9)*(p[26]*1e-9)
    cf=3.9*EPS0*area_m2/(p[29]*1e-9)
    cb=3.9*EPS0*area_m2/((p[28]/3.0+p[30])*1e-9)
    cb0=3.9*EPS0*area_m2/((50.0/3.0+140.0)*1e-9)
    cbody=cf+cb-cb0
    return length,width,tsi,width*tsi,cbody,p[31],VT*np.log(1e20*p[31]/NI_CM3**2)

params_NA=float(params.NA_CM3)
params_VBI=float(VT*np.log(1e20*params_NA/NI_CM3**2))

@njit(cache=True)
def backgate_charge(p):
    """Incremental capacitive backgate charge relative to calibrated VBG=0."""
    if len(p)<33:
        return 0.0
    area_m2=(p[27]*1e-9)*(p[26]*1e-9)
    cback=3.9*EPS0*area_m2/((p[30]+p[28]/3.0)*1e-9)
    return -cback*p[32]

@njit(cache=True)
def gate_charge_offset(p):
    """Applied-gate terms -Cf*VG-Cb*VBG in the reference-anchored charge law."""
    if len(p)<32:
        return -original.COX_F*p[11]
    area_m2=(p[27]*1e-9)*(p[26]*1e-9)
    cf=3.9*EPS0*area_m2/(p[29]*1e-9)
    return -cf*p[11]+backgate_charge(p)

@njit(cache=True)
def packed_field(r,p,kind):
    n=int(p[34]);pos=r/p[35];i=max(0,min(int(pos),n-2));a=pos-i
    off=36+kind*FIELD_SIZE
    return p[off+i]*(1-a)+p[off+i+1]*a

@njit(cache=True)
def channel_current(u,r,p):
    vg=p[11]
    if len(p)>=33:
        vg += p[29]/(p[30]+p[28]/3.0)*p[32]
    n=1.7786684648788609*(1+p[16]*(u+r))
    ov=vg-(-0.49032524444873615)+p[14]*(u+r)+p[15]*u
    pp=ov/n
    # softplus without exp overflow for strongly-on channels (large VG + VBG coupling); unchanged below x = 700
    xf=pp/(2*VT);xr=(pp-u-r)/(2*VT);xo=ov/(n*VT)
    sf=xf if xf>700.0 else np.log1p(np.exp(xf));sr=xr if xr>700.0 else np.log1p(np.exp(xr))
    so=xo if xo>700.0 else np.log1p(np.exp(xo))
    cur=2*n*7.52135238967614e-5*VT*VT*(sf-sr)*(sf+sr)/(1+0.6335606399651017*n*VT*so)
    if len(p)>=32:
        cur *= (p[27]/200.)*(500./p[26])*(14.1/p[29])
    return cur

@njit(cache=True)
def geometry_components(u,r,p,na,vbi,rg,fg,table):
    length_cm, width_cm, tsi_cm, area_cm2, cox_f, na, vbi = constants_from_p(p)
    # p: beta,tau_bulk,tau_junc,Rcontact,lGIDL_nm,taccess_nm,NAaccess_cm3,Laccess_nm
    beta,tau,tj,rc,lg,ta,naa,la,tau_ratio=p[:9]
    width_ratio = width_cm / (200e-7)
    rc /= width_ratio
    lg *= p[29] / 14.1
    # Surface-dominated junction-SRH closure: tau_eff / tau_ref = Tsi / Tsi_ref.
    # This is an uncalibrated scaling assumption, not extracted surface recombination.
    tj *= p[28] / 50.0
    phi_gidl=p[9];phi_emitter=p[10];vg=p[11];channel_II_scale=p[12]
    iph=p[13] if p.shape[0]>13 else 0.   # photogeneration current (A), uniform hole supply into the body
    eps=11.7*EPS0/100
    prod=NI_CM3**2*np.expm1(u/VT)
    delta=2*prod/(na+np.sqrt(na*na+4*prod))
    wd=np.sqrt(2*eps*(vbi+r)/(Q*na))
    # u is local quasi-Fermi splitting. The junction electrostatic reduction
    # is u-VT*log(p_source/NA), not u itself, in this neutrality approximation.
    source_barrier=vbi-u+VT*np.log1p(delta/na)
    if source_barrier<=0:return np.full(19,np.nan)
    ws=np.sqrt(2*eps*source_barrier/(Q*na))
    length=length_cm-wd-ws
    if length<=1e-7:return np.full(19,np.nan)
    prod=NI_CM3**2*np.expm1(u/VT)
    delta=2*prod/(na+np.sqrt(na*na+4*prod))
    scale=Q*area_cm2*DN*na/length
    mult=1.+(packed_field(r+(p[19] if p.shape[0]>19 else 0.), p, 0)-1.)*np.exp(p[20] if p.shape[0]>20 else 0.)   # p[19]: local junction potential offset (V), p[20]: log-scale of (M-1) [state hypotheses]
    balance=-np.expm1(-r/VT)
    bbj=packed_field(r+(p[19] if p.shape[0]>19 else 0.), p, 1)*balance
    eg=max((u+r-vg-.3-1.12+phi_gidl)/(lg*1e-7),0.)
    depth=min(np.sqrt(2*eps*1.12/(Q*7e19)),tsi_cm)
    volume=width_cm*5e-7*depth
    gidl=Q*volume*BB_A*eg**2.5*np.exp(-BB_B/max(eg,1.))*balance
    # Low-VD IDVG fit frozen; DIBL/body coupling not extracted and not invented.
    ch=channel_current(u,r,p)
    # Holes from channel-electron avalanches enter the neutral-body boundary.
    # Direct channel electrons do not flow through the neutral-base BJT solver.
    # High-VD channel seed of this device (floating-body DIBL): I_p*10^((VG+1.8)/S) added to the channel electrons at the drain.
    ch=ch+((p[17]*width_ratio*10.**((vg+1.8)/p[18])) if (p.shape[0]>18 and p[17]>0.) else 0.)
    ii_ch=max(mult-1.,0.)*ch*channel_II_scale
    ii_ph=max(mult-1.,0.)*iph   # photo-electrons collected at the drain also multiply
    # Local avalanche path (microplasma-type) near the drain junction, saturating (negligible in the LRS).
    # p[21] strength, p[22] saturation (A), p[23] log-fluctuation, p[24] carriers it multiplies:
    #   0 = surface/edge definition (GIDL + junction BTBT + channel + photo electrons)
    #   1 = bulk definition (BJT-injected electrons crossing the junction + photo electrons + junction BTBT), channel and surface GIDL excluded
    loc_on=(p.shape[0]>23 and p[21]>0.);bulk=(p.shape[0]>24 and p[24]>0.5)
    iloc=0.
    # optional gate-drain-field dependence of the local path: strength x exp(kappaF*(V_GD - 5.6 V)), p[25] = kappaF (1/V)
    fdep=np.exp(p[25]*((u+r-vg)-5.6)) if p.shape[0]>25 else 1.
    edge_only=(p.shape[0]>24 and 1.5<p[24]<2.5)   # p[24]=2: gate-edge carriers only (GIDL, junction BTBT, photo); channel electrons excluded
    if loc_on and not bulk:iloc=min(p[21]*np.exp(p[23])*fdep*max(mult-1.,0.)*(gidl+bbj+iph+(0. if edge_only else ch)),p[22]*width_ratio)
    js,lb,qa,hole_drop_over_VT,iterations=srh_solve(delta/na,mult,(bbj+gidl+ii_ch+iph+ii_ph+iloc)/scale,length*length/(DN*tau),1/3,64,tau_ratio,(1+tau_ratio)*NI_CM3/na)
    if loc_on and bulk and np.isfinite(js):
        for _ in range(3):   # fixed-point iteration: the local path multiplies the self-consistent BJT seed
            iloc=min(p[21]*np.exp(p[23])*max(mult-1.,0.)*(scale*js+iph+bbj),p[22]*width_ratio)
            js,lb,qa,hole_drop_over_VT,iterations=srh_solve(delta/na,mult,(bbj+gidl+ii_ch+iph+ii_ph+iloc)/scale,length*length/(DN*tau),1/3,64,tau_ratio,(1+tau_ratio)*NI_CM3/na)
            if not np.isfinite(js):break
    if not np.isfinite(js):return np.full(19,np.nan)
    seed=scale*js;bulk=scale*lb;emitter=seed+bulk
    # Nonlinear SRH integral returned by kernel; bulk is NOT Qpair/tau.
    # Low-injection minority-hole diffusion into the n+ source (the BJT emitter). The source doping and its
    # hole-diffusion length are not changed by L or Nbody, so its saturation current scales with the junction
    # area W*Tsi only: beta and the reference base length/doping stay those of the calibrated device.
    # High-injection base current no longer shares its exp(u/2VT) scaling.
    lref=original.LENGTH_M*100-2*np.sqrt(2*eps*params_VBI/(Q*params_NA))
    isp=Q*area_cm2*DN*NI_CM3**2/(params_NA*lref*beta)
    diff=isp*np.exp(-phi_emitter/VT)*np.expm1(u/VT)
    junction=Q*area_cm2*ws*NI_CM3/(2*tj)*np.expm1(u/(2*VT))
    avg=na*qa
    # Explicit independent access slab. This density-sharing assumption is not Poisson-derived.
    sigma=Q*(450*avg+150*(naa+avg))
    racc=(la*1e-7)/(width_cm*ta*1e-7*sigma)
    mult=1.+(packed_field(r+(p[19] if p.shape[0]>19 else 0.), p, 0)-1.)*np.exp(p[20] if p.shape[0]>20 else 0.)   # p[19]: local junction potential offset (V), p[20]: log-scale of (M-1) [state hypotheses]
    ii=(mult-1)*seed
    # Net reverse BTBT detailed-balance ansatz: zero at equilibrium r=0.
    balance=-np.expm1(-r/VT)
    bbj=packed_field(r+(p[19] if p.shape[0]>19 else 0.), p, 1)*balance
    eg=max((u+r-vg-.3-1.12+phi_gidl)/(lg*1e-7),0.)
    depth=min(np.sqrt(2*eps*1.12/(Q*7e19)),tsi_cm)
    volume=width_cm*5e-7*depth
    gidl=Q*volume*BB_A*eg**2.5*np.exp(-BB_B/max(eg,1.))*balance
    drain=seed+ii+bbj+gidl+ch+ii_ch+iph+ii_ph+iloc
    net=ii+ii_ch+bbj+gidl+iph+ii_ph+iloc-diff-bulk-junction
    vd=u+r+VT*hole_drop_over_VT+(rc+racc)*drain
    qpair=Q*area_cm2*length*avg
    qaccess=Q*(width_cm*ta*1e-7)*(la*1e-7)*avg
    charge=cox_f*u+qpair+qaccess
    return np.array([vd,drain,net,seed,emitter,bulk,diff,junction,bbj,gidl,delta/na,length,racc,charge,ws,wd,ch,VT*hole_drop_over_VT,iph])

@njit(cache=True)
def geometry_curve_grid(p,na,vbi,rg,fg,table,ug):
    length_cm, width_cm, tsi_cm, area_cm2, cox_f, na, vbi = constants_from_p(p)
    out=np.empty((len(ug),21));count=0
    eps=11.7*EPS0/100
    for u in ug:
        if u==0:
            z=geometry_components(0.,0.,p,na,vbi,rg,fg,table)
            out[count,:17]=z[:17];out[count,17]=0.;out[count,18]=0.;out[count,20]=z[17]
            out[count,19]=np.sqrt(2*Q*na*vbi/eps)
            count+=1;continue
        prod=NI_CM3**2*np.expm1(u/VT)
        delta=2*prod/(na+np.sqrt(na*na+4*prod))
        source_barrier=vbi-u+VT*np.log1p(delta/na)
        if source_barrier<=0:continue
        ws=np.sqrt(2*eps*source_barrier/(Q*na))
        lavail=length_cm-ws-1.01e-7
        if lavail<=0:continue
        rgeo=Q*na*lavail*lavail/(2*eps)-vbi
        hi=min((p[34]-1)*p[35],rgeo);lo=0.
        if hi<=0:continue
        a=geometry_components(u,lo,p,na,vbi,rg,fg,table)
        b=geometry_components(u,hi,p,na,vbi,rg,fg,table)
        if not np.isfinite(a[2]) or a[2]>0 or (np.isfinite(b[2]) and b[2]<0):continue
        loglo=np.log(1e-100);loghi=np.log(hi)
        for _ in range(29):
            mid=np.exp((loglo+loghi)/2)
            z=geometry_components(u,mid,p,na,vbi,rg,fg,table)
            if not np.isfinite(z[2]) or z[2]>0:loghi=np.log(mid)
            else:loglo=np.log(mid)
        r=np.exp((loglo+loghi)/2)
        z=geometry_components(u,r,p,na,vbi,rg,fg,table)
        if not np.isfinite(z[2]) or abs(z[2])>1e-5*max(z[1],1e-25):continue
        out[count,:17]=z[:17];out[count,17]=u;out[count,18]=r;out[count,20]=z[17]
        out[count,19]=np.sqrt(2*Q*na*(vbi+r)/eps)
        count+=1
    return out[:count]

@njit(cache=True)
def components(u,r,p,na,vbi,rg,fg,table):
    if len(p)<32:
        return original.components(u,r,p,na,vbi,rg,fg,table)
    return geometry_components(u,r,p,na,vbi,rg,fg,table)

class GeometryModel(original.FastModel):
    def __init__(self, ref):
        self.__dict__.update(ref.__dict__)
        self.reference=ref
    def branch(self,p,ug=original.UG):
        p=np.asarray(p,float)
        if len(p)<32:
            return self.reference.branch(p,ug)
        p=pack_p(p)
        vbi=p[33]
        if len(ug)>3:
            gap=np.geomspace(1e-7,.02,17)
            ug=np.unique(np.r_[ug,vbi-gap,vbi,vbi+gap])
            ug=ug[(ug>=0)&(ug<=1.12)]
        return geometry_curve_grid(p,self.na,self.vbi,self.rg,self.fg,self.table,ug)
    def classify(self,p,ug=original.UG):
        if len(p)<32:
            return self.reference.classify(p,ug)
        return super().classify(p,ug)

class ModuleProxy:
    components=staticmethod(components)
    def __getattr__(self,name):
        return getattr(original,name)
