"""Source-resolved mean components, separate from the immutable legacy model.

The voltage argument is quasi-Fermi splitting / q, not automatically the
electrostatic floating-body potential. No self-consistent VLU is claimed.
Junction and neutral-body SRH integrals use disjoint illustrative volumes.
beta_diff is explicitly I_e,injected / I_h,out, excluding recombination.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

Q=1.602176634e-19
KB=1.380649e-23


def van_overstraeten_300k(field_v_cm):
    e=np.asarray(field_v_cm,dtype=float)
    if np.any(~np.isfinite(e)) or np.any(e<0):raise ValueError('Finite nonnegative field magnitude required')
    effective=np.maximum(e,1.)
    alpha=7.03e5*np.exp(-1.231e6/effective)
    beta=np.where(e<4e5,1.582e6*np.exp(-2.036e6/effective),6.71e5*np.exp(-1.693e6/effective))
    return np.where(e>0,alpha,0),np.where(e>0,beta,0)


def spatial_mean_gain(x_cm,field_v_cm):
    """Electron injected at x[0], electron +x, hole -x; local-field theory.

    Me = [1-integral alpha(x) exp(-integral_0^x(alpha-beta)) dx]^-1.
    Negative/zero denominator means no finite subcritical local mean.
    """
    x=np.asarray(x_cm,dtype=float);e=np.asarray(field_v_cm,dtype=float)
    if x.ndim!=1 or len(x)<3 or e.shape!=x.shape or np.any(~np.isfinite(x)) or np.any(np.diff(x)<=0):
        raise ValueError('Matching increasing 1D spatial grid and field required')
    a,b=van_overstraeten_300k(e)
    exponent=cumulative_trapezoid(a-b,x,initial=0)
    integral=float(trapezoid(a*np.exp(-exponent),x))
    denominator=1-integral
    return dict(gain=(1/denominator if denominator>0 else None),breakdown_margin=denominator,
                finite_subcritical=bool(denominator>0),method='local ionization integral; numerical spatial quadrature')


def srh_density_rate(n,p,ni,tau_n,tau_p,n1=None,p1=None):
    n,p=np.broadcast_arrays(np.asarray(n,dtype=float),np.asarray(p,dtype=float))
    if np.any(~np.isfinite(n)) or np.any(~np.isfinite(p)) or np.any(n<0) or np.any(p<0):raise ValueError('Finite nonnegative densities required')
    for x in [ni,tau_n,tau_p]:
        if not np.isfinite(x) or x<=0:raise ValueError('Positive finite ni and lifetimes required')
    n1=ni if n1 is None else n1;p1=ni if p1 is None else p1
    if not np.isfinite(n1) or not np.isfinite(p1) or n1<=0 or p1<=0:raise ValueError('Positive finite n1,p1 required')
    return (n*p-ni**2)/(tau_p*(n+n1)+tau_n*(p+p1))


@dataclass(frozen=True)
class ResolvedMean:
    temperature_K:float=300.
    ni_cm3:float=1e10
    acceptors_cm3:float=3e17
    tau_n_s:float=3e-7
    tau_p_s:float=3e-7
    # .5 x .65 x .05 um^3 total; .1 um aggregate SCR and .4 um neutral length.
    junction_volume_cm3:float=3.25e-15
    body_volume_cm3:float=1.30e-14
    beta_diff:float=50.

    def __post_init__(self):
        if any(not np.isfinite(x) or x<=0 for x in self.__dict__.values()):raise ValueError('All parameters must be positive finite')

    @property
    def VT(self):return KB*self.temperature_K/Q

    def recombination(self,Vqf_V):
        v=np.asarray(Vqf_V,dtype=float)
        if np.any(~np.isfinite(v)) or np.any(abs(v/self.VT)>600):raise ValueError('Finite supported splitting required')
        product=self.ni_cm3**2*np.exp(v/self.VT)
        equal=self.ni_cm3*np.exp(v/(2*self.VT))
        p=(self.acceptors_cm3+np.sqrt(self.acceptors_cm3**2+4*product))/2
        n=2*product/(self.acceptors_cm3+np.sqrt(self.acceptors_cm3**2+4*product))
        uj=srh_density_rate(equal,equal,self.ni_cm3,self.tau_n_s,self.tau_p_s)
        ub=srh_density_rate(n,p,self.ni_cm3,self.tau_n_s,self.tau_p_s)
        # These limits retain zero net rate at equilibrium, but are not generic all-bias laws.
        junction_limit=self.ni_cm3/(self.tau_n_s+self.tau_p_s)*np.expm1(v/(2*self.VT))
        body_low_limit=self.ni_cm3**2/(self.acceptors_cm3*self.tau_n_s)*np.expm1(v/self.VT)
        return dict(junction_A=Q*self.junction_volume_cm3*uj,body_A=Q*self.body_volume_cm3*ub,
                    junction_limit_A=Q*self.junction_volume_cm3*junction_limit,
                    body_low_injection_A=Q*self.body_volume_cm3*body_low_limit,
                    neutral_n_cm3=n,neutral_p_cm3=p,junction_n_cm3=equal,
                    low_injection_ratio=n/self.acceptors_cm3)

    def hole_budget(self,Vqf_V,electron_injection_A,ii_holes_per_s,btbt_holes_per_s):
        """Conditional expected body-hole budget; no hidden recombination in beta."""
        for val in [electron_injection_A,ii_holes_per_s,btbt_holes_per_s]:
            if not np.isfinite(val) or val<0:raise ValueError('Finite nonnegative injection/generation required')
        rec=self.recombination(Vqf_V)
        loss_diff=electron_injection_A/self.beta_diff
        return dict(**rec,diffusion_out_A=loss_diff,
                    net_holes_per_s=ii_holes_per_s+btbt_holes_per_s-(loss_diff+rec['junction_A']+rec['body_A'])/Q)
