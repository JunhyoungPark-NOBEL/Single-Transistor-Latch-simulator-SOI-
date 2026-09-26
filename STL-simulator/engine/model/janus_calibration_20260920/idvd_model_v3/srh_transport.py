"""Current-carrying quasi-neutral transport with density-dependent SRH.

Dimensionless carrier excess h=delta/NA; x runs collector to emitter.
No Auger. Constant mobilities, Einstein relation, and quasi-neutral interior.
The SRH lifetime is effective; a flat transverse profile cannot distinguish
bulk SRH from interface SRH through one IDVD curve.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def reaction(h,kappa,tau_ratio,trap_offset):
    # tau_ratio=tau_p/tau_n; trap_offset=(tau_p*n1+tau_n*p1)/(tau_n*NA).
    den=1.+trap_offset+(1.+tau_ratio)*h
    num=h*(1.+h)
    value=kappa*num/den
    derivative=kappa*((1.+2.*h)*den-(1.+tau_ratio)*num)/(den*den)
    return value,derivative


@njit(cache=True)
def rhs(h,j,hs,js,j0,M,b,kappa,r,tau_ratio,trap_offset):
    den=r+(1.+r)*h
    fn=h/den
    g=den/(r*(1.+2.*h))
    total=M*j0+b
    dh=(j-fn*total)*g
    rr,dr=reaction(h,kappa,tau_ratio,trap_offset)
    fp=r/(den*den)
    gp=(1.-r)/(r*(1.+2.*h)**2)
    dhs=(-fp*total*g+(j-fn*total)*gp)*hs+g*js-fn*M*g
    return dh,rr,dhs,dr*hs,h,(total-j)/(r*(1.+h))


@njit(cache=True)
def integrate_voltage(j0,M,b,kappa,r,steps,tau_ratio=1.,trap_offset=0.):
    h=0.;j=j0;hs=0.;js=1.;charge=0.;drop=0.;dz=1./steps
    for _ in range(steps):
        a=rhs(h,j,hs,js,j0,M,b,kappa,r,tau_ratio,trap_offset)
        c=rhs(h+dz*a[0]/2,j+dz*a[1]/2,hs+dz*a[2]/2,js+dz*a[3]/2,j0,M,b,kappa,r,tau_ratio,trap_offset)
        d=rhs(h+dz*c[0]/2,j+dz*c[1]/2,hs+dz*c[2]/2,js+dz*c[3]/2,j0,M,b,kappa,r,tau_ratio,trap_offset)
        e=rhs(h+dz*d[0],j+dz*d[1],hs+dz*d[2],js+dz*d[3],j0,M,b,kappa,r,tau_ratio,trap_offset)
        h+=dz*(a[0]+2*c[0]+2*d[0]+e[0])/6
        j+=dz*(a[1]+2*c[1]+2*d[1]+e[1])/6
        hs+=dz*(a[2]+2*c[2]+2*d[2]+e[2])/6
        js+=dz*(a[3]+2*c[3]+2*d[3]+e[3])/6
        charge+=dz*(a[4]+2*c[4]+2*d[4]+e[4])/6
        drop+=dz*(a[5]+2*c[5]+2*d[5]+e[5])/6
        if h<0 or not np.isfinite(h) or h>1e80:
            return np.nan,np.nan,np.nan,np.nan,np.nan
    return h,j,hs,charge,drop


@njit(cache=True)
def integrate(j0,M,b,kappa,r,steps,tau_ratio=1.,trap_offset=0.):
    h,j,hs,q,drop=integrate_voltage(j0,M,b,kappa,r,steps,tau_ratio,trap_offset)
    return h,j,hs,q


@njit(cache=True)
def solve_full_voltage(h_end,M,b,kappa,r=1/3,steps=64,tau_ratio=1.,trap_offset=0.):
    if h_end==0 and b==0:return 0.,0.,0.,0.,0
    # Non-recombining M=1 limit sets the trial scale; solve is still nonlinear.
    j0=max(2*h_end-np.log1p(h_end),1e-100)
    for it in range(32):
        h,je,deriv,qavg,drop=integrate_voltage(j0,M,b,kappa,r,steps,tau_ratio,trap_offset)
        if not np.isfinite(h) or deriv<=0:
            return np.nan,np.nan,np.nan,np.nan,it
        residual=h-h_end
        if abs(residual)<1e-9*max(h_end,1e-100):
            return j0,je-j0,qavg,drop,it+1
        j0=max(j0*.1,j0-residual/deriv)
    return np.nan,np.nan,np.nan,np.nan,32


@njit(cache=True)
def solve_full(h_end,M,b,kappa,r=1/3,steps=64,tau_ratio=1.,trap_offset=0.):
    seed,loss,charge,drop,it=solve_full_voltage(h_end,M,b,kappa,r,steps,tau_ratio,trap_offset)
    return seed,loss,charge,it


@njit(cache=True)
def solve_voltage(h_end,M,b,kappa,r=1/3,steps=64,tau_ratio=1.,trap_offset=0.):
    """Return (seed, integrated recombination, mean h, QF drop / VT, iterations).

    Multiplying the first two outputs by q*A*Dn*NA/L gives ampere.
    Qpair=q*A*L*NA*mean_h. For nonlinear SRH, loss is NOT Qpair/tau_n.
    Local junction u,r require VD=u+r+VT*drop+I*Routside. This drop is
    the hole quasi-Fermi difference, NOT the electrostatic field integral.
    The low-injection analytic guard only drops small O(h^2,h*J) terms.
    """
    if h_end<1e-8 and (1+tau_ratio)*h_end<1e-8:
        kk=kappa/(1.+trap_offset)
        k=np.sqrt(kk)
        if k<1e-6:
            j0=h_end*(1-kk/6);qa=h_end*(.5-kk/24)
        else:
            j0=h_end*k/np.sinh(k)
            qa=h_end*np.tanh(k/2)/k
        if M*j0+b<1e-8:
            return j0,kk*qa,qa,(M*j0+b-h_end)/r,0
    return solve_full_voltage(h_end,M,b,kappa,r,steps,tau_ratio,trap_offset)


@njit(cache=True)
def solve(h_end,M,b,kappa,r=1/3,steps=64,tau_ratio=1.,trap_offset=0.):
    seed,loss,charge,drop,it=solve_voltage(h_end,M,b,kappa,r,steps,tau_ratio,trap_offset)
    return seed,loss,charge,it


@njit(cache=True)
def batch(h,M,b,kappa,r=1/3,steps=64,tau_ratio=1.,trap_offset=0.):
    out=np.empty((len(h),4))
    for i in range(len(h)):
        a,b0,c,d=solve(h[i],M[i],b[i],kappa[i],r,steps,tau_ratio,trap_offset)
        out[i,0]=a;out[i,1]=b0;out[i,2]=c;out[i,3]=d
    return out
