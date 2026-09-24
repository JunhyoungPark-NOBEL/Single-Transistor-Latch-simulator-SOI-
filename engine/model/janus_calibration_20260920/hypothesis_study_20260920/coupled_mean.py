# Generated from frozen v3 mean; see derivative_manifest.json.
# Two explicit conditional couplings:
# phi_gidl: additional local voltage drop across the drain-edge tunneling field.
# phi_emitter: modulation of the emitter minority-hole diffusion prefactor.
# Neither is random output-voltage jitter. These local couplings are hypotheses,
# not a self-consistent 2D Poisson solution or identified defect positions.
"""Current-carrying quasi-neutral candidate with asymmetric density-dependent SRH.
Local emitter/collector quasi-Fermi voltage partition includes the neutral-base hole drop.
Auger is explicitly absent; effective lifetime cannot separate bulk/interface.

The carrier-partition drift term is included using a shooting solution.
This is not a full Poisson/degenerate/field-mobility model. Access excess is assumed
equal to the mean base excess, a separate unverified quasi-static closure.
Source/drain depletion widths retain abrupt-junction low-injection electrostatics.
No stochastic jitter or fitted readout current floor is present.
"""
from pathlib import Path
import sys,time,json
import numpy as np
from numba import njit
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'idvd_model'))
from mean_model import Field,Q,VT,AREA_CM2,LENGTH_M,WIDTH_M,TSI_M,COX_F,NI_CM3,CHANNEL,BB_A,BB_B,EPS0

sys.path.insert(0,str(HERE.parent/'high_injection_review'))
from current_carrying_transport import integrate as transport_integrate
sys.path.insert(0,str(HERE.parent/'idvd_model_v3'))
from srh_transport import solve_voltage as srh_solve
DN=450*VT;DP=150*VT
LOG_Y=np.linspace(-14,4,241)
LOG_K=np.linspace(-3,1,161)

@njit(cache=True)
def transport_solve_full(h_end,M,b,kappa,r=1/3,steps=64):
    # Same current-carrying sensitivity solver, with relative tolerance that
    # remains meaningful in the equilibrium, very-low-injection limit.
    if h_end==0 and b==0:return 0.,0.,0.,0
    j0=max(2*h_end-np.log1p(h_end),1e-100)
    for it in range(24):
        h,je,deriv,qavg=transport_integrate(j0,M,b,kappa,r,steps)
        if not np.isfinite(h) or deriv<=0:return np.nan,np.nan,np.nan,it
        residual=h-h_end
        if abs(residual)<1e-9*max(h_end,1e-100):return j0,je-j0,qavg,it+1
        j0=max(j0*.1,j0-residual/deriv)
    return np.nan,np.nan,np.nan,24

@njit(cache=True)
def transport_solve(h_end,M,b,kappa,r=1/3,steps=64):
    if h_end<1e-8:
        k=np.sqrt(kappa)
        if k<1e-6:
            j0=h_end*(1-kappa/6);qa=h_end*(.5-kappa/24)
        else:
            j0=h_end*k/np.sinh(k)
            qa=h_end*2*np.sinh(k/2)**2/(k*np.sinh(k))
        if M*j0+b<1e-8:return j0,kappa*qa,qa,0
    return transport_solve_full(h_end,M,b,kappa,r,steps)

@njit(cache=True)
def hn(y,t):
    # H(y*t)/(DN*y*y), H(z)=integral_0^z Da(s)*s ds.
    dc=DN+DP;ginf=2*DP/dc;z=dc/DP*y*t
    if z<1e-3:
        ratio=(dc/DP*t)**2*(.5-z/3+z*z/4-z*z*z/5+z**4/6)
    else:ratio=(z-np.log1p(z))/(y*y)
    coeff=DP*DP*(DN-DP)/(dc**3)
    return ginf*t*t/2+coeff*ratio

@njit(cache=True)
def make_table(ly,lk,t,w):
    out=np.empty((len(ly),len(lk),3))
    for i in range(len(ly)):
        y=10**ly[i]
        for j in range(len(lk)):
            k=10**lk[j];lo=-35.;hi=0.
            for _ in range(48):
                z=(lo+hi)/2;a=np.exp(z);val=0.
                for h in range(len(t)):
                    g=(1+2*y*t[h])/(1+(1+DN/DP)*y*t[h])
                    val+=w[h]*g/np.sqrt(a*a+2*k*k*hn(y,t[h]))
                if val>1:lo=z
                else:hi=z
            a=np.exp((lo+hi)/2);charge=0.
            for h in range(len(t)):
                g=(1+2*y*t[h])/(1+(1+DN/DP)*y*t[h])
                charge+=w[h]*t[h]*g/np.sqrt(a*a+2*k*k*hn(y,t[h]))
            out[i,j,0]=a
            out[i,j,1]=np.sqrt(a*a+2*k*k*hn(y,1.))
            out[i,j,2]=charge
    return out

@njit(cache=True)
def lookup(y,k,table):
    ay=(np.log10(max(y,1e-14))+14)/18*(table.shape[0]-1)
    ak=(np.log10(max(k,1e-3))+3)/4*(table.shape[1]-1)
    iy=max(0,min(int(ay),table.shape[0]-2));ik=max(0,min(int(ak),table.shape[1]-2))
    fy=max(0.,min(1.,ay-iy));fk=max(0.,min(1.,ak-ik))
    return (1-fy)*((1-fk)*table[iy,ik]+fk*table[iy,ik+1])+fy*((1-fk)*table[iy+1,ik]+fk*table[iy+1,ik+1])

@njit(cache=True)
def f_interp(r,grid,values):
    pos=r/(grid[1]-grid[0]);i=max(0,min(int(pos),len(grid)-2));a=pos-i
    return values[i]*(1-a)+values[i+1]*a

@njit(cache=True)
def components(u,r,p,na,vbi,rg,fg,table):
    # p: beta,tau_bulk,tau_junc,Rcontact,lGIDL_nm,taccess_nm,NAaccess_cm3,Laccess_nm
    beta,tau,tj,rc,lg,ta,naa,la,tau_ratio=p[:9]
    phi_gidl=p[9];phi_emitter=p[10]
    eps=11.7*EPS0/100
    prod=NI_CM3**2*np.expm1(u/VT)
    delta=2*prod/(na+np.sqrt(na*na+4*prod))
    wd=np.sqrt(2*eps*(vbi+r)/(Q*na))
    # u is local quasi-Fermi splitting. The junction electrostatic reduction
    # is u-VT*log(p_source/NA), not u itself, in this neutrality approximation.
    source_barrier=vbi-u+VT*np.log1p(delta/na)
    if source_barrier<=0:return np.full(18,np.nan)
    ws=np.sqrt(2*eps*source_barrier/(Q*na))
    length=LENGTH_M*100-wd-ws
    if length<=1e-7:return np.full(18,np.nan)
    prod=NI_CM3**2*np.expm1(u/VT)
    delta=2*prod/(na+np.sqrt(na*na+4*prod))
    scale=Q*AREA_CM2*DN*na/length
    mult=f_interp(r,rg,fg[0])
    balance=-np.expm1(-r/VT)
    bbj=f_interp(r,rg,fg[1])*balance
    eg=max((u+r+2.-.3-1.12+phi_gidl)/(lg*1e-7),0.)
    depth=min(np.sqrt(2*eps*1.12/(Q*7e19)),TSI_M*100)
    volume=WIDTH_M*100*5e-7*depth
    gidl=Q*volume*BB_A*eg**2.5*np.exp(-BB_B/max(eg,1.))*balance
    js,lb,qa,hole_drop_over_VT,iterations=srh_solve(delta/na,mult,(bbj+gidl)/scale,length*length/(DN*tau),1/3,64,tau_ratio,(1+tau_ratio)*NI_CM3/na)
    if not np.isfinite(js):return np.full(18,np.nan)
    seed=scale*js;bulk=scale*lb;emitter=seed+bulk
    # Nonlinear SRH integral returned by kernel; bulk is NOT Qpair/tau.
    # Standard low-injection minority-hole diffusion in an n+ emitter.
    # beta is the low-injection reference ratio at fixed zero-bias base length.
    # High-injection base current no longer shares its exp(u/2VT) scaling.
    lref=LENGTH_M*100-2*np.sqrt(2*eps*vbi/(Q*na))
    isp=Q*AREA_CM2*DN*NI_CM3**2/(na*lref*beta)
    diff=isp*np.exp(-phi_emitter/VT)*np.expm1(u/VT)
    junction=Q*AREA_CM2*ws*NI_CM3/(2*tj)*np.expm1(u/(2*VT))
    avg=na*qa
    # Explicit independent access slab. This density-sharing assumption is not Poisson-derived.
    sigma=Q*(450*avg+150*(naa+avg))
    racc=(la*1e-7)/(WIDTH_M*100*ta*1e-7*sigma)
    mult=f_interp(r,rg,fg[0])
    ii=(mult-1)*seed
    # Net reverse BTBT detailed-balance ansatz: zero at equilibrium r=0.
    balance=-np.expm1(-r/VT)
    bbj=f_interp(r,rg,fg[1])*balance
    eg=max((u+r+2.-.3-1.12+phi_gidl)/(lg*1e-7),0.)
    depth=min(np.sqrt(2*eps*1.12/(Q*7e19)),TSI_M*100)
    volume=WIDTH_M*100*5e-7*depth
    gidl=Q*volume*BB_A*eg**2.5*np.exp(-BB_B/max(eg,1.))*balance
    # Frozen IDVG channel, no added body effect or fitted floor.
    n=1.7786684648788609;ov=-2.-(-.49032524444873615);pp=ov/n
    sf=np.log1p(np.exp(pp/(2*VT)));sr=np.log1p(np.exp((pp-u-r)/(2*VT)))
    ch=2*n*7.52135238967614e-5*VT*VT*(sf-sr)*(sf+sr)/(1+.6335606399651017*n*VT*np.log1p(np.exp(ov/(n*VT))))
    drain=seed+ii+bbj+gidl+ch
    net=ii+bbj+gidl-diff-bulk-junction
    vd=u+r+VT*hole_drop_over_VT+(rc+racc)*drain
    qpair=Q*AREA_CM2*length*avg
    qaccess=Q*(WIDTH_M*100*ta*1e-7)*(la*1e-7)*avg
    charge=COX_F*u+qpair+qaccess
    return np.array([vd,drain,net,seed,emitter,bulk,diff,junction,bbj,gidl,delta/na,length,racc,charge,ws,wd,ch,VT*hole_drop_over_VT])

@njit(cache=True)
def curve_grid(p,na,vbi,rg,fg,table,ug):
    out=np.empty((len(ug),21));count=0
    eps=11.7*EPS0/100
    for u in ug:
        if u==0:
            z=components(0.,0.,p,na,vbi,rg,fg,table)
            out[count,:17]=z[:17];out[count,17]=0.;out[count,18]=0.;out[count,20]=z[17]
            out[count,19]=np.sqrt(2*Q*na*vbi/eps)
            count+=1;continue
        prod=NI_CM3**2*np.expm1(u/VT)
        delta=2*prod/(na+np.sqrt(na*na+4*prod))
        source_barrier=vbi-u+VT*np.log1p(delta/na)
        if source_barrier<=0:continue
        ws=np.sqrt(2*eps*source_barrier/(Q*na))
        lavail=LENGTH_M*100-ws-1.01e-7
        if lavail<=0:continue
        rgeo=Q*na*lavail*lavail/(2*eps)-vbi
        hi=min(rg[-1],rgeo);lo=0.
        if hi<=0:continue
        a=components(u,lo,p,na,vbi,rg,fg,table)
        b=components(u,hi,p,na,vbi,rg,fg,table)
        if not np.isfinite(a[2]) or a[2]>0 or (np.isfinite(b[2]) and b[2]<0):continue
        loglo=np.log(1e-100);loghi=np.log(hi)
        for _ in range(29):
            mid=np.exp((loglo+loghi)/2)
            z=components(u,mid,p,na,vbi,rg,fg,table)
            if not np.isfinite(z[2]) or z[2]>0:loghi=np.log(mid)
            else:loglo=np.log(mid)
        r=np.exp((loglo+loghi)/2)
        z=components(u,r,p,na,vbi,rg,fg,table)
        if not np.isfinite(z[2]) or abs(z[2])>1e-5*max(z[1],1e-25):continue
        out[count,:17]=z[:17];out[count,17]=u;out[count,18]=r;out[count,20]=z[17]
        out[count,19]=np.sqrt(2*Q*na*(vbi+r)/eps)
        count+=1
    return out[:count]

def state_grid(points=201):
    # Deep equilibrium branch must be resolved in injection potential, not
    # filled by a line from zero to the first measurable-bias state.
    return np.unique(np.r_[0.,np.geomspace(1e-80,.02,121),np.linspace(.02,1.12,points)])
UG=state_grid()

def load_transport_table(force=False):
    # The current-carrying kernel solves transport directly; pair table is unused.
    return np.empty((1,1,3)),dict(action='no_transport_table_required',seconds=0.,shape=[1,1,3])

class FastModel:
    def __init__(self,na,table):
        start=time.perf_counter();f=Field(na)
        self.na=na;self.vbi=float(f.vbi);self.rg=f.rr
        self.fg=np.array([f.gain(f.rr),f.btbt(f.rr)])
        self.table=table;self.field_initialization_s=time.perf_counter()-start
    def branch(self,p,ug=UG):
        if len(ug)>3:
            gap=np.geomspace(1e-7,.02,17)
            ug=np.unique(np.r_[ug,self.vbi-gap,self.vbi,self.vbi+gap])
            ug=ug[(ug>=0)&(ug<=1.12)]
        return curve_grid(np.asarray(p,float),self.na,self.vbi,self.rg,self.fg,self.table,ug)
    def classify(self,p,ug=UG):
        b=self.branch(p,ug)
        if len(b)<8:return None
        dv=np.diff(b[:,0]);mx=np.flatnonzero((dv[:-1]>0)&(dv[1:]<0))+1
        mn=np.flatnonzero((dv[:-1]<0)&(dv[1:]>0))+1
        if not len(mx) or not len(mn):return None
        i=int(mx[0]);after=mn[mn>i]
        if not len(after):return None
        # During optimization the topology may have intermediate folds. Use the
        # terminal stable branch; the validation report counts all extrema.
        j=int(after[-1]);folds=[]
        for ind in [i,j]:
            xx=b[ind-1:ind+2,17]-b[ind,17];co=np.polyfit(xx,b[ind-1:ind+2,0],2)
            v=float(np.polyval(co,-co[1]/(2*co[0])))
            folds.append(v)
        return b,i,j,np.array(folds)
    def double_curve(self,p,points=131):
        # New physical branch and folds on every call; interpolation only maps its
        # monotonic equilibrium segments to the requested terminal voltage grid.
        cb=self.classify(p,state_grid(points))
        if cb is None:raise ValueError('No two-fold branch')
        b,i,j,fold=cb
        v=np.r_[np.linspace(0,4,401),np.linspace(3.99,0,400)]
        output=np.full((801,4),np.nan);output[:,0]=v
        low=b[:i+1];high=b[j:]
        for ind,on in [(i,False),(j,True)]:
            xx=b[ind-1:ind+2,17]-b[ind,17]
            co=np.polyfit(xx,b[ind-1:ind+2,0],2)
            uf=b[ind,17]-co[1]/(2*co[0])
            rr=self.branch(p,np.array([uf]))
            if len(rr):
                if not on and rr[0,0]>low[-1,0]:low=np.vstack([low,rr])
                if on and rr[0,0]<high[0,0]:high=np.vstack([rr,high])
        for on,part in [(False,low),(True,high)]:
            wanted=np.where(np.arange(801)<=400,v>fold[0],v>=fold[1])==on
            wanted&=(v>=part[0,0])&(v<=part[-1,0])
            output[wanted,1]=np.maximum(np.exp(np.interp(v[wanted],part[:,0],np.log(np.maximum(part[:,1],1e-300))))-1e-300,0.)
            output[wanted,2]=np.interp(v[wanted],part[:,0],part[:,17])
            output[wanted,3]=np.interp(v[wanted],part[:,0],part[:,18])
        # Equilibrium is an analytic boundary of the same equations, all net sources zero.
        output[v==0]=[0.,0.,0.,0.]
        return output,fold

    def validity_flags(self,output):
        prod=NI_CM3**2*np.expm1(output[:,2]/VT)
        delta=2*prod/(self.na+np.sqrt(self.na*self.na+4*prod))
        barrier=self.vbi-output[:,2]+VT*np.log1p(delta/self.na)
        return dict(numerically_missing=int(np.sum(~np.isfinite(output[:,1]))),
                    source_flatband_or_beyond=int(np.sum(barrier<=0)),
                    minimum_source_electrostatic_barrier_V=float(np.nanmin(barrier)),
                    max_source_injection_potential_V=float(np.nanmax(output[:,2])),
                    source_built_in_V=self.vbi)

@njit(cache=True)
def sample_root(vd,ulo,uhi,p,na,vbi,rg,fg,table):
    # Nested u/r solve using branch brackets; avoids implicit state interpolation.
    last=np.zeros(20)
    for _ in range(13):
        u=(ulo+uhi)/2;grid=curve_grid(p,na,vbi,rg,fg,table,np.array([u]))
        if len(grid)==0:return np.full(20,np.nan)
        last=grid[0]
        if last[0]>vd:uhi=u
        else:ulo=u
    return last

if __name__=='__main__':
    tab,meta=load_transport_table();m=FastModel(3e17,tab)
    p=[10.,1e-9,3e-8,1000.,34.,10.,1e18,70.]
    t=time.perf_counter();a=m.classify(p)
    print(json.dumps(dict(table=meta,field_s=m.field_initialization_s,first_branch_s=time.perf_counter()-t,
        folds=None if a is None else a[3].tolist(),points=None if a is None else len(a[0]))),flush=True)
