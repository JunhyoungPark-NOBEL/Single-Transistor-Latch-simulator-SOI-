"""Conditional LU first passage from the accepted mean; no histogram fitting.

Unclustered single-hole Poisson control, quasi-static ramp hazard, exact
birth/death MFPT recurrence, and independent backward-equation checks.
Frozen mean-model files are imported without modification.
"""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
from scipy.special import logsumexp
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve_banded

H=Path(__file__).resolve().parent
ROOT=H.parent
sys.path.insert(0,str(ROOT/'idvd_model_v3'))
import mean_model_voltage_v3 as m
import cache_provenance as cache
CANDIDATE=ROOT/'idvd_model_v3/final_candidate.json'
d=json.loads(CANDIDATE.read_text());p=np.array(d['parameters']);table,_=m.load_transport_table();model=m.FastModel(d['NA_cm3'],table)
Q=m.Q;COX=m.COX_F;NA=model.na;AREA=m.AREA_CM2;RAMP=.4
OUT=H/'lu_fpt_outputs';OUT.mkdir(exist_ok=True)

def state(u,vd):
    def at(r):return m.components(u,r,p,NA,model.vbi,model.rg,model.fg,table)
    r=brentq(lambda r:at(r)[0]-vd,0.,vd-u,xtol=1e-11)
    z=at(r);h=z[10];psi=u-m.VT*np.log1p(h)
    ratio=(p[5]*1e-7/(m.TSI_M*100))*(p[7]*1e-7/z[11])
    qb=(z[13]-COX*u)/(1+ratio);qa=qb*ratio;qbg=Q*NA*AREA*z[11]
    birth=(z[1]-z[3]-z[16])/Q
    death=(z[5]+z[6]+z[7])/Q
    # Return inventory basis terms so charge controls share identical mean rates.
    return np.array([u,r,z[1],birth,death,COX*psi,qb+qa,qbg,z[2],z[3],z[8]+z[9]])

def mfpt_log(birth,death,start):
    # States0..B-1 transient, B absorbing. Lower boundary is reflecting.
    logpi=np.r_[0.,np.cumsum(np.log(birth[:-1])-np.log(death[1:]))]
    logs=np.logaddexp.accumulate(logpi)
    terms=logs-logpi-np.log(birth)
    return float(logsumexp(terms[start:])),terms

def backward_solve(birth,death):
    n=len(birth);death=death.copy();death[0]=0.
    total=birth+death
    band=np.zeros((3,n));band[1]=1.;band[0,1:]=-birth[:-1]/total[:-1];band[2,:-1]=-death[1:]/total[1:]
    return solve_banded((1,1),band,1./total)

def generator_for(rows,gate_scale=1.,background=True,lower_u=.1,lattice_offset=0.):
    u=rows[:,0];F=rows[:,3]-rows[:,4]
    stable=np.flatnonzero((F[:-1]>0)&(F[1:]<0))
    unstable=np.flatnonzero((F[:-1]<0)&(F[1:]>0))
    if not len(stable) or not len(unstable):raise ValueError('Missing well/saddle in off-equilibrium domain')
    k=stable[0];ks=unstable[unstable>k][0]
    uw=brentq(PchipInterpolator(u,F),u[k],u[k+1]);us=brentq(PchipInterpolator(u,F),u[ks],u[ks+1])
    ia=np.flatnonzero(rows[:,2]>=1e-8)[0]
    ua=brentq(PchipInterpolator(u,rows[:,2]-1e-8),u[ia-1],u[ia])
    if ua<=us:raise ValueError('10nA boundary does not lie beyond saddle')
    inventory=(gate_scale*rows[:,5]+rows[:,6]+(rows[:,7] if background else 0.))/Q
    ds=np.diff(inventory)/np.diff(u)
    path=(u[:-1]>=lower_u)&(u[:-1]<=ua)
    if np.any(ds[path]<=0):raise ValueError('Non-monotone hole inventory')
    qfun=PchipInterpolator(u,inventory);qlow=float(qfun(lower_u));qabs=float(qfun(ua));qw=float(qfun(uw))
    ngrid=np.arange(np.ceil(qlow-lattice_offset)+lattice_offset,np.ceil(qabs-lattice_offset)+lattice_offset,1.)
    if len(ngrid)<10:raise ValueError('Insufficient charge domain')
    nstart=int(np.clip(np.round(qw-ngrid[0]),0,len(ngrid)-1))
    plus=np.exp(PchipInterpolator(inventory,np.log(rows[:,3]))(ngrid))
    minus=np.exp(PchipInterpolator(inventory,np.log(rows[:,4]))(ngrid))
    logT,terms=mfpt_log(plus,minus,nstart)
    slope=float(PchipInterpolator(u,F).derivative()(uw))
    cwell=float(qfun.derivative()(uw))*Q
    relaxation=-cwell/(Q*slope)
    logpi=np.r_[0.,np.cumsum(np.log(plus[:-1])-np.log(minus[1:]))]
    # Tail probability proxy relative to well; strict stationary distribution is
    # only used on the left basin, before the unstable state.
    tail=float(np.exp(min(logpi[0]-logpi[nstart],0.)))
    return dict(logT_s=logT,well_u_V=uw,saddle_u_V=us,absorbing_u_V=ua,n_states=len(ngrid),start_index=nstart,min_Ceff_fF=float(ds[path].min()*Q*1e15),well_Ceff_fF=cwell*1e15,relaxation_s=relaxation,reflecting_tail_ratio=tail,charge_boundary_rounding_holes=float(np.ceil(qabs-lattice_offset)+lattice_offset-qabs)),plus,minus

def load_or_build():
    fn=OUT/'offequilibrium_grid.npz';meta=OUT/'offequilibrium_grid_metadata.json'
    sha=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    fold=json.loads((ROOT/'idvd_model_v3/validation_summary.json').read_text())['prediction']['folds_V'][0]
    # Very near the fold an unresolved sub-grid well/saddle is replaced by the
    # deterministic terminal atom; no negative/invalid generator is forced.
    voltage=np.r_[np.arange(3.10,fold-.001,.002),fold-.0005]
    u=np.linspace(.1,.85,301)
    provenance=cache.expected('LU',voltage,u,fold)
    cached=cache.load_compatible(fn,meta,provenance)
    if cached is not None:return (*cached,fold)
    start=time.perf_counter();allrows=[]
    for k,v in enumerate(voltage):
        allrows.append(np.array([state(x,v) for x in u]))
        if k%60==0:print(f'Off-equilibrium table {k}/{len(voltage)} VD={v:.3f}',flush=True)
    allrows=np.array(allrows);np.savez_compressed(fn,VD=voltage,u=u,rows=allrows)
    metadata=dict(candidate_sha256=sha,grid_voltages=len(voltage),grid_u=len(u),seconds=time.perf_counter()-start,columns=cache.COLUMNS,mean_refit=False)
    meta.write_text(json.dumps(cache.attach(metadata,fn,provenance),indent=2))
    return voltage,u,allrows,fold

def curve(voltage,rows,gate_scale=1.,background=True,lower_u=.1,subsample=False,lattice_offset=0.):
    result=[]
    for vv,rr in zip(voltage,rows):
        r,_,_=generator_for(rr[::2] if subsample else rr,gate_scale,background,lower_u,lattice_offset)
        r['VD_V']=float(vv);result.append(r)
    hazard=np.exp(np.clip(-np.array([r['logT_s'] for r in result]),-740,700))
    cumulative=cumulative_trapezoid(hazard/RAMP,voltage,initial=0.)
    for r,h,c in zip(result,hazard,cumulative):r.update(hazard_per_s=float(h),cumulative_hazard=float(c),CDF=float(-np.expm1(-c)))
    return result

def samples(result,fold,n,seed):
    rng=np.random.default_rng(seed);haz=-np.log(rng.random(n));V=np.array([r['VD_V'] for r in result]);cum=np.array([r['cumulative_hazard'] for r in result])
    v=np.interp(haz,cum,V,left=V[0],right=fold)
    midpoint=np.floor(v/.01)*.01+.005
    return np.c_[np.arange(n),v,midpoint,haz>cum[-1]]

def summary(s):
    return dict(N=len(s),continuous_mean_V=float(np.mean(s[:,1])),continuous_SD_V=float(np.std(s[:,1],ddof=1)),quantized_midpoint_mean_V=float(np.mean(s[:,2])),quantized_midpoint_SD_V=float(np.std(s[:,2],ddof=1)),quantiles_V=np.quantile(s[:,1],[.01,.1,.5,.9,.99]).tolist(),deterministic_fold_atom_fraction=float(np.mean(s[:,3])))

def main():
    started=time.perf_counter();V,u,rows,fold=load_or_build()
    results=curve(V,rows);s100=samples(results,fold,100,20260920);large=samples(results,fold,100000,20260921)
    np.savetxt(OUT/'LU_MC100.csv',s100,delimiter=',',header='sample,VLU_continuous_V,VLU_10mV_midpoint_V,at_fold_atom',comments='')
    np.savetxt(OUT/'LU_MC100000.csv',large,delimiter=',',header='sample,VLU_continuous_V,VLU_10mV_midpoint_V,at_fold_atom',comments='')
    import csv
    with (OUT/'hazard_curve.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(results[0]));wr.writeheader();wr.writerows(results)
    # Exact birth/death analytical control: constant upward/downward rates.
    a=np.full(20,3.);b=np.full(20,2.);lt,_=mfpt_log(a,b,0);known=20/(3-2)-2/(3-2)**2*(1-(2/3)**20)
    analytic=dict(MFPT_recurrence_s=float(np.exp(lt)),closed_form_s=known,backward_s=float(backward_solve(a,b)[0]))
    # Independent tridiagonal backward solve at distribution10/50/90% points.
    checks=[]
    for probability in [.1,.5,.9]:
        idx=int(np.argmin(abs(np.array([r['CDF'] for r in results])-probability)))
        rr,lp,lm=generator_for(rows[idx]);t=backward_solve(lp,lm)[rr['start_index']];ref=np.exp(rr['logT_s'])
        checks.append(dict(CDF_target=probability,VD_V=float(V[idx]),recurrence_MFPT_s=float(ref),backward_MFPT_s=float(t),relative_difference=float(t/ref-1)))
    # Native one-hole grid retained; change state-interpolation density or
    # reflecting truncation without changing event size or mean parameters.
    controls={}
    for name,kw in [('u_grid151',dict(subsample=True)),('left_boundary_u0p15',dict(lower_u=.15)),('half_hole_lattice_offset',dict(lattice_offset=.5)),('without_junction_inventory',dict(background=False)),('gate_capacitance_half',dict(gate_scale=.5))]:
        rr=curve(V,rows,**kw);ss=samples(rr,fold,100000,20260921);controls[name]=summary(ss)
    coarse=results[::2]
    hh=np.array([r['hazard_per_s'] for r in coarse]);vv=np.array([r['VD_V'] for r in coarse]);cc=cumulative_trapezoid(hh/RAMP,vv,initial=0)
    coarse=[dict(r,cumulative_hazard=float(c)) for r,c in zip(coarse,cc)]
    controls['voltage_grid4mV']=summary(samples(coarse,fold,100000,20260921))
    window=[r for r in results if .01<r['CDF']<.99]
    deriv=np.gradient(np.log(np.maximum([r['hazard_per_s'] for r in results],1e-300)),V)
    eps=[r['relaxation_s']*RAMP*abs(dd) for r,dd in zip(results,deriv) if .01<r['CDF']<.99]
    report=dict(status='CONDITIONAL_LU_FPT_PROTOTYPE_UNCLUSTERED_POISSON_NOT_HISTOGRAM_FIT',accepted_mean_candidate_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),mean_parameters_unchanged=True,Auger_enabled=False,ramp_V_per_s=RAMP,absorbing_current_A=1e-8,charge_closure='Cox*psiE+base_excess+access_excess+qNA*A*(Lneutral-Lref)',II_noise='unclustered independent single-hole Poisson control; not complete avalanche branching noise',deterministic_fold_V=fold,MC100=summary(s100),MC100000=summary(large),mean_shift_from_fold_V=summary(large)['continuous_mean_V']-fold,analytic_control=analytic,backward_checks=checks,controls=controls,minimum_Ceff_fF=float(min(r['min_Ceff_fF'] for r in results)),max_reflecting_tail_ratio=float(max(r['reflecting_tail_ratio'] for r in results)),max_quasistatic_relaxation_times_loghazard_ramp_in_1to99percent=float(max(eps)) if eps else None,unresolved_last_interval_V=float(fold-V[-1]),survival_at_last_node=float(np.exp(-results[-1]['cumulative_hazard'])),finite_rate_grid=True,independent_cycles_only=True,trap_memory_included=False,source_noise_attribution_identified=False,seconds_total=time.perf_counter()-started)
    report['max_relaxation_over_MFPT_in_1to99percent']=float(max(r['relaxation_s']*r['hazard_per_s'] for r in window))
    (OUT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
