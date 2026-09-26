"""Conditional LD first passage and paired plotting ensembles.

Down ramp0.4V/s is an explicit assumption. LU/LD events are independent white
controls; paths between switches use the frozen quasi-static branch means.
"""
from pathlib import Path
import json,time,hashlib,csv
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid
import lu_fpt as lu
H=Path(__file__).resolve().parent;OUT=H/'ld_fpt_outputs';OUT.mkdir(exist_ok=True)

def generator_for(rows,gate_scale=1.,background=True,upper_u=1.03,lattice_offset=0.,commitment_depth=.05):
    u=rows[:,0];F=rows[:,3]-rows[:,4]
    stable=np.flatnonzero((F[:-1]>0)&(F[1:]<0))
    unstable=np.flatnonzero((F[:-1]<0)&(F[1:]>0))
    if not len(stable) or not len(unstable):raise ValueError('Missing LD saddle/well')
    k=stable[-1];ks=unstable[unstable<k][-1]
    ff=PchipInterpolator(u,F);uw=brentq(ff,u[k],u[k+1]);us=brentq(ff,u[ks],u[ks+1])
    ia=np.flatnonzero(rows[:,2]>=1e-8)[0];uth=brentq(PchipInterpolator(u,rows[:,2]-1e-8),u[ia-1],u[ia]);ua=min(uth,us-commitment_depth)
    if ua<=u[0]:raise ValueError('Committed HRS boundary outside calculated domain')
    inventory=(gate_scale*rows[:,5]+rows[:,6]+(rows[:,7] if background else 0.))/lu.Q
    ds=np.diff(inventory)/np.diff(u);path=(u[:-1]>=ua)&(u[:-1]<=upper_u)
    if np.any(ds[path]<=0):raise ValueError('Non-monotone LD hole inventory')
    qfun=PchipInterpolator(u,inventory)
    # x=-Q/q: physical hole loss is the upward jump toward absorption.
    xlo=-float(qfun(upper_u));xabs=-float(qfun(ua));xw=-float(qfun(uw))
    xgrid=np.arange(np.ceil(xlo-lattice_offset)+lattice_offset,np.ceil(xabs-lattice_offset)+lattice_offset,1.);start=int(np.clip(np.round(xw-xgrid[0]),0,len(xgrid)-1))
    xx=-inventory[::-1]
    plus=np.exp(PchipInterpolator(xx,np.log(rows[::-1,4]))(xgrid))
    minus=np.exp(PchipInterpolator(xx,np.log(rows[::-1,3]))(xgrid))
    logT,_=lu.mfpt_log(plus,minus,start)
    cwell=float(qfun.derivative()(uw))*lu.Q;relax=-cwell/(lu.Q*float(ff.derivative()(uw)))
    lp=np.r_[0.,np.cumsum(np.log(plus[:-1])-np.log(minus[1:]))]
    return dict(logT_s=logT,well_u_V=uw,saddle_u_V=us,absorbing_u_V=ua,threshold_u_V=uth,absorbing_ID_A=float(PchipInterpolator(u,rows[:,2])(ua)),commitment_depth_V=commitment_depth,n_states=len(xgrid),start_index=start,min_Ceff_fF=float(ds[path].min()*lu.Q*1e15),well_Ceff_fF=cwell*1e15,relaxation_s=relax,reflecting_tail_ratio=float(np.exp(min(lp[0]-lp[start],0.))),charge_boundary_rounding_holes=float(np.ceil(xabs-lattice_offset)+lattice_offset-xabs)),plus,minus

def load_or_build():
    fn=OUT/'offequilibrium_grid.npz';meta=OUT/'offequilibrium_grid_metadata.json';sha=hashlib.sha256(lu.CANDIDATE.read_bytes()).hexdigest()
    fold=json.loads((lu.ROOT/'idvd_model_v3/validation_summary.json').read_text())['prediction']['folds_V'][1]
    voltage=np.r_[np.arange(3.2,fold+.001,-.002),fold+.0005];u=np.linspace(.55,1.05,301)
    provenance=lu.cache.expected('LD',voltage,u,fold)
    cached=lu.cache.load_compatible(fn,meta,provenance)
    if cached is not None:return (*cached,fold)
    start=time.perf_counter();rows=[]
    for k,v in enumerate(voltage):
        rows.append(np.array([lu.state(x,v) for x in u]))
        if k%60==0:print(f'LD off-equilibrium table {k}/{len(voltage)} VD={v:.3f}',flush=True)
    rows=np.array(rows);np.savez_compressed(fn,VD=voltage,u=u,rows=rows)
    metadata=dict(candidate_sha256=sha,grid_voltages=len(voltage),grid_u=len(u),seconds=time.perf_counter()-start,columns=lu.cache.COLUMNS,mean_refit=False)
    meta.write_text(json.dumps(lu.cache.attach(metadata,fn,provenance),indent=2))
    return voltage,u,rows,fold

def curve(voltage,rows,subsample=False,**kwargs):
    result=[]
    for vv,rr in zip(voltage,rows):
        r,_,_=generator_for(rr[::2] if subsample else rr,**kwargs);r['VD_V']=float(vv);result.append(r)
    hazard=np.exp(np.clip(-np.array([r['logT_s'] for r in result]),-740,700));cum=cumulative_trapezoid(hazard/lu.RAMP,-voltage,initial=0.)
    for r,h,c in zip(result,hazard,cum):r.update(hazard_per_s=float(h),cumulative_hazard=float(c),CDF=float(-np.expm1(-c)))
    return result

def samples(result,fold,n,seed):
    rng=np.random.default_rng(seed);haz=-np.log(rng.random(n));V=np.array([r['VD_V'] for r in result]);cum=np.array([r['cumulative_hazard'] for r in result])
    v=np.interp(haz,cum,V,left=V[0],right=fold);mid=np.floor(v/.01)*.01+.005
    return np.c_[np.arange(n),v,mid,haz>cum[-1]]

def double_sweeps(ld100):
    lu100=np.loadtxt(lu.OUT/'LU_MC100.csv',delimiter=',',skiprows=1)
    branch,i,j,fold=lu.model.classify(lu.p,lu.m.state_grid(801));low=branch[:i+1];high=branch[j:]
    V=np.r_[np.linspace(0,4,401),np.linspace(3.99,0,400)];current=np.full((801,100),np.nan)
    for k in range(100):
        on=np.where(np.arange(801)<=400,V>lu100[k,1],V>ld100[k,1])
        for state,part in [(False,low),(True,high)]:
            take=on==state
            if np.any((V[take]<part[0,0])|(V[take]>part[-1,0])):raise ValueError('FPT switch requests nonexistent stable branch; no filling allowed')
            current[take,k]=np.exp(np.interp(V[take],part[:,0],np.log(np.maximum(part[:,1],1e-300))))-1e-300
        current[V==0,k]=0.
    np.savez_compressed(H/'conditional_MC100_double_sweeps.npz',VD_V=V,ID_A=current,VLU_continuous_V=lu100[:,1],VLD_continuous_V=ld100[:,1])
    np.savetxt(H/'conditional_MC100_double_sweeps.csv',np.c_[V,current],delimiter=',',header='VD_V,'+','.join(f'I_cycle{k+1:03d}_A' for k in range(100)),comments='')
    extraction=[]
    for k in range(100):
        iu=np.flatnonzero(current[:401,k]>=1e-8)[0];part=np.r_[current[400,k],current[401:,k]];dv=np.linspace(4,0,401);idd=np.flatnonzero(part<1e-8)[0]
        extraction.append([(V[iu-1]+V[iu])/2,(dv[idd-1]+dv[idd])/2])
    extraction=np.array(extraction)
    return dict(points=801,trials=100,finite=bool(np.all(np.isfinite(current))),max_LU_midpoint_difference_V=float(np.max(abs(extraction[:,0]-lu100[:,2]))),max_LD_midpoint_difference_V=float(np.max(abs(extraction[:,1]-ld100[:,2]))),intrabranchevent_current_fluctuations_included=False,LU_LD_independent=True,stable_branch_only_between_FPT_events=True)

def main():
    started=time.perf_counter();V,u,rows,fold=load_or_build();results=curve(V,rows)
    s100=samples(results,fold,100,20260922);large=samples(results,fold,100000,20260923)
    for name,s in [('LD_MC100.csv',s100),('LD_MC100000.csv',large)]:np.savetxt(OUT/name,s,delimiter=',',header='sample,VLD_continuous_V,VLD_10mV_midpoint_V,at_fold_atom',comments='')
    with (OUT/'hazard_curve.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(results[0]));wr.writeheader();wr.writerows(results)
    checks=[]
    for prob in [.1,.5,.9]:
        idx=int(np.argmin(abs(np.array([r['CDF'] for r in results])-prob)));rr,lp,lm=generator_for(rows[idx]);ref=np.exp(rr['logT_s']);value=lu.backward_solve(lp,lm)[rr['start_index']]
        checks.append(dict(CDF_target=prob,VD_V=float(V[idx]),recurrence_MFPT_s=float(ref),backward_MFPT_s=float(value),relative_difference=float(value/ref-1)))
    controls={}
    for name,kw in [('u_grid151',dict(subsample=True)),('upper_boundary_u1p00',dict(upper_u=1.00)),('half_hole_lattice_offset',dict(lattice_offset=.5)),('commitment_depth25mV',dict(commitment_depth=.025)),('commitment_depth75mV',dict(commitment_depth=.075)),('without_junction_inventory',dict(background=False)),('gate_capacitance_half',dict(gate_scale=.5))]:controls[name]=lu.summary(samples(curve(V,rows,**kw),fold,100000,20260923))
    coarse=results[::2];hh=np.array([r['hazard_per_s'] for r in coarse]);vv=np.array([r['VD_V'] for r in coarse]);cc=cumulative_trapezoid(hh/lu.RAMP,-vv,initial=0.)
    coarse=[dict(r,cumulative_hazard=float(c)) for r,c in zip(coarse,cc)];controls['voltage_grid4mV']=lu.summary(samples(coarse,fold,100000,20260923))
    derivative=np.gradient(np.log(np.maximum([r['hazard_per_s'] for r in results],1e-300)),V);eps=[r['relaxation_s']*lu.RAMP*abs(g) for r,g in zip(results,derivative) if .01<r['CDF']<.99]
    report=dict(status='CONDITIONAL_LD_FPT_PROTOTYPE_UNCLUSTERED_POISSON_NOT_HISTOGRAM_FIT',accepted_mean_candidate_sha256=hashlib.sha256(lu.CANDIDATE.read_bytes()).hexdigest(),mean_parameters_unchanged=True,Auger_enabled=False,down_ramp_V_per_s=-lu.RAMP,down_ramp_is_assumed_not_measured=True,absorbing_current_below_A=1e-8,charge_axis='x=-Qh/q',deterministic_fold_V=fold,MC100=lu.summary(s100),MC100000=lu.summary(large),mean_shift_from_fold_V=lu.summary(large)['continuous_mean_V']-fold,backward_checks=checks,controls=controls,minimum_Ceff_fF=float(min(r['min_Ceff_fF'] for r in results)),max_reflecting_tail_ratio=float(max(r['reflecting_tail_ratio'] for r in results)),max_quasistatic_relaxation_times_loghazard_ramp_in_1to99percent=float(max(eps)) if eps else None,unresolved_last_interval_V=float(V[-1]-fold),survival_at_last_node=float(np.exp(-results[-1]['cumulative_hazard'])),independent_cycles_only=True,trap_memory_included=False,source_noise_attribution_identified=False,seconds_total=time.perf_counter()-started)
    report['max_relaxation_over_MFPT_in_1to99percent']=float(max(r['relaxation_s']*r['hazard_per_s'] for r in results if .01<r['CDF']<.99))
    report['absorption_commitment_rule']='ID<10nA and u<=saddle_u−50mV; depth25/75mV controls included'
    report['MC100_double_sweep_overlay']=double_sweeps(s100)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
