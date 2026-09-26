"""Independent current-carrying quasi-neutral transport check, not a device fit.

Coordinate z=(collector-x)/L. h=excess/NA, j=i_e/(q A Dn NA/L).
Constant total terminal current J=M*j0+b enters the carrier partition explicitly.
R=excess/tau; constant mobilities and neutrality are declared approximations.
"""
from pathlib import Path
import json,time
import numpy as np
from numba import njit

@njit(cache=True)
def rhs(h,j,hs,js,j0,M,b,kappa,r):
    den=r+(1+r)*h
    fn=h/den
    g=den/(r*(1+2*h))
    total=M*j0+b
    dh=(j-fn*total)*g
    dj=kappa*h
    fp=r/(den*den)
    gp=(1-r)/(r*(1+2*h)**2)
    dhs=(-fp*total*g+(j-fn*total)*gp)*hs+g*js-fn*M*g
    return dh,dj,dhs,kappa*hs,h

@njit(cache=True)
def integrate(j0,M,b,kappa,r,steps):
    h=0.;j=j0;hs=0.;js=1.;charge=0.;dz=1./steps
    for _ in range(steps):
        a=rhs(h,j,hs,js,j0,M,b,kappa,r)
        c=rhs(h+dz*a[0]/2,j+dz*a[1]/2,hs+dz*a[2]/2,js+dz*a[3]/2,j0,M,b,kappa,r)
        d=rhs(h+dz*c[0]/2,j+dz*c[1]/2,hs+dz*c[2]/2,js+dz*c[3]/2,j0,M,b,kappa,r)
        e=rhs(h+dz*d[0],j+dz*d[1],hs+dz*d[2],js+dz*d[3],j0,M,b,kappa,r)
        h+=dz*(a[0]+2*c[0]+2*d[0]+e[0])/6
        j+=dz*(a[1]+2*c[1]+2*d[1]+e[1])/6
        hs+=dz*(a[2]+2*c[2]+2*d[2]+e[2])/6
        js+=dz*(a[3]+2*c[3]+2*d[3]+e[3])/6
        charge+=dz*(a[4]+2*c[4]+2*d[4]+e[4])/6
        if h<0 or not np.isfinite(h):return np.nan,np.nan,np.nan,np.nan
    return h,j,hs,charge

@njit(cache=True)
def solve(h_end,M,b,kappa,r=1/3,steps=64):
    # Exact no-recombination/no-BTBT M=1 solution supplies a good initial scale.
    j0=max(2*h_end-np.log1p(h_end),1e-30)
    for it in range(20):
        h,je,deriv,q=integrate(j0,M,b,kappa,r,steps)
        if not np.isfinite(h) or deriv<=0:return np.nan,np.nan,np.nan,it
        residual=h-h_end
        if abs(residual)<1e-9*max(h_end,1e-12):return j0,je-j0,q,it+1
        change=residual/deriv
        j0=max(j0*.1,j0-change)
    return np.nan,np.nan,np.nan,20

@njit(cache=True)
def batch(h,M,b,kappa,steps=64):
    n=len(h);out=np.empty((n,4))
    for i in range(n):
        s,l,q,k=solve(h[i],M[i],b[i],kappa[i],1/3,steps)
        out[i,0]=s;out[i,1]=l;out[i,2]=q;out[i,3]=k
    return out

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    hh=np.geomspace(1e-8,100,201);ones=np.ones_like(hh);zero=ones*0
    t=time.perf_counter();test=batch(hh,ones,zero,zero);jit=time.perf_counter()-t
    expected=2*hh-np.log1p(hh)
    rel=float(np.max(abs(test[:,0]-expected)/expected))
    assert rel<.004 # RK boundary resolution at very high injection is recorded, not hidden.
    controls=[]
    for h in [.001,1.,10.,100.]:
        for M in [1.,1.1,1.25]:
            vals=[]
            for ns in [32,64,128,256,1024]:
                vals.append([ns,*[float(x) for x in solve(h,M,0.,.02,1/3,ns)]])
            controls.append(dict(h=h,M=M,steps_seed_loss_charge_iterations=vals))
    # Shape and input vary on each call; this is transport-kernel timing only.
    times=[]
    for i in range(51):
        t=time.perf_counter();b=batch(hh*(1+.001*np.sin(i)),ones*1.1,zero,ones*.02,64)
        if i:times.append(time.perf_counter()-t)
    ans=dict(status='Independent transport kernel, NOT calibrated STL model or full-sweep benchmark',
        exact_M1_zeroR_test_max_relative_error=rel,compile_or_cache_load_s=jit,
        transport_solves_per_batch=len(hh),warm_median_s=float(np.median(times)),warm_p95_s=float(np.quantile(times,.95)),
        controls=controls,limitations=['Quasi-neutral region only, constant mobilities, linear bulk lifetime, collector minority boundary zero.',
        'Current partition included; degeneracy, field mobility, depletion dynamics and real device access region still require closure.'])
    (root/'current_carrying_transport_audit.json').write_text(json.dumps(ans,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:v for k,v in ans.items() if k!='controls'},indent=2))
