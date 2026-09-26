"""V3: transition-adjacent currents, aligned windows, full branches; Auger off."""
from pathlib import Path
import json,time,sys
import numpy as np
from scipy.optimize import least_squares
import mean_model_v3 as physics
HERE=Path(__file__).resolve().parent
RAW=np.loadtxt(HERE.parent/'inputs/IDVD_100cycle.txt')
UP=RAW[:,1:101];DOWN=RAW[:,102:]
THRESHOLD=1e-8
BASELINE=.5329e-12;BASELINE_SD=.3937e-12
OFFSETS=np.array([0,1,3,5,10])
PRE_UP=np.array([np.flatnonzero(UP[:,k]>=THRESHOLD)[0]-1 for k in range(100)])
PRE_DN=np.array([np.flatnonzero(DOWN[:,k]<THRESHOLD)[0]-1 for k in range(100)])
TARGET_UP=np.array([np.median(UP[PRE_UP-o,np.arange(100)]) for o in OFFSETS])
TARGET_DN=np.array([np.median(DOWN[PRE_DN-o,np.arange(100)]) for o in OFFSETS])
VLU=float(np.mean((RAW[PRE_UP,0]+RAW[PRE_UP+1,0])/2))
VLD=float(np.mean((RAW[PRE_DN,101]+RAW[PRE_DN+1,101])/2))
LRS_V=np.round(np.arange(2.8,4.0001,.05),2)
LRS_I=np.array([np.median(DOWN[np.argmin(abs(RAW[:,101]-v))]) for v in LRS_V])
HRS_V=np.array([2.8,3.,3.1,3.2])
HRS_I=np.array([np.median(UP[np.argmin(abs(RAW[:,0]-v))]) for v in HRS_V])
# beta is the reference emitter diffusion ratio only. No Auger or noise parameter.
LO=np.array([np.log10(2),-12,-12,0,14.1,3,17.])
HI=np.array([5.,-6,-4,4.3,42.3,50,20.])
FIT_GRID=np.unique(np.r_[0.,np.geomspace(1e-16,.02,16),np.linspace(.02,1.12,121)])
NRES=2+2+len(OFFSETS)*2+len(LRS_V)+len(HRS_V)+4+4

def decode(x):return np.array([10**x[0],10**x[1],10**x[2],10**x[3],x[4],x[5],10**x[6],70.])

def predict(model,x,grid=FIT_GRID):
    p=decode(x);cb=model.classify(p,grid)
    if cb is None:return None
    b,i,j,fold=cb;lo=b[:i+1];hi=b[j:]
    if np.any(np.diff(lo[:,0])<=0) or np.any(np.diff(hi[:,0])<=0):return None
    if np.any(hi[:,1]<=0):return None
    def cur(v,part):return np.exp(np.interp(v,part[:,0],np.log(np.maximum(part[:,1],1e-100))))
    vu=fold[0]-(OFFSETS+.5)*.01;vd=fold[1]+(OFFSETS+.5)*.01
    up=cur(vu,lo);dn=cur(vd,hi)
    lrs=cur(LRS_V,hi);hrs=cur(HRS_V,lo)
    # Same current threshold as measured extraction; geometric folds alone do not suffice.
    thup=float(np.interp(np.log(THRESHOLD),np.log(np.maximum(lo[:,1],1e-100)),lo[:,0])) if lo[-1,1]>=THRESHOLD else float(fold[0])
    thdn=float(fold[1]) if hi[0,1]>=THRESHOLD else float(np.interp(np.log(THRESHOLD),np.log(hi[:,1]),hi[:,0]))
    rows=np.array([[np.interp(v,hi[:,0],hi[:,k]) for k in range(20)] for v in [3.,3.5,3.8,4.]])
    na=model.na;scale=physics.Q*physics.AREA_CM2*physics.DN*na/rows[:,11]
    mult=np.interp(rows[:,18],model.rg,model.fg[0]);js=rows[:,3]/scale;bb=(rows[:,8]+rows[:,9])/scale
    ec=physics.VT/rows[:,11]*abs((mult-1+1/3)*js+bb)/(1/3)
    velocity=450*ec/1e7
    # Infer the electrostatic barrier actually used by the model from Ws;
    # u may be quasi-Fermi splitting rather than electrostatic junction bias.
    source_barrier=physics.Q*na*rows[:,14]**2/(2*(11.7*physics.EPS0/100))
    dv=np.diff(b[:,0]);extrema=int(np.sum(dv[1:]*dv[:-1]<0))
    return dict(folds_V=fold.tolist(),thresholds_V=[thup,thdn],aligned_up_V=vu.tolist(),aligned_down_V=vd.tolist(),aligned_up_A=up.tolist(),aligned_down_A=dn.tolist(),lrs_A=lrs.tolist(),hrs_A=hrs.tolist(),LRS4_A=float(lrs[-1]),preLU_A=float(up[0]),preLD_A=float(dn[0]),grid_fold_currents_A=[float(lo[-1,1]),float(hi[0,1])],domain_shortfall_V=[float(max(vu.max()-lo[-1,0],0)+max(lo[0,0]-vu.min(),0)),float(max(vd.max()-hi[-1,0],0)),float(max(LRS_V.max()-hi[-1,0],0)),float(max(hi[0,0]-LRS_V.min(),0))],source_barrier_V=source_barrier.tolist(),collector_muE_over_vsat=velocity.tolist(),injection_ratios=rows[:,10].tolist(),extrema_count=extrema)

def residual(model,x,physical_weight=.3):
    a=predict(model,x)
    if a is None:return np.full(NRES,100.)
    # Observation baseline enters only this uncertainty-aware residual, never
    # generation rates, terminal intrinsic current, or state equations.
    up=(np.log1p(np.array(a['aligned_up_A'])/BASELINE_SD)-np.log1p(np.maximum(TARGET_UP-BASELINE,0)/BASELINE_SD))/(np.log(10)*.35)
    dn=np.log10(np.array(a['aligned_down_A'])/TARGET_DN)/.12
    up[0]*=2.;dn[0]*=2.
    lrs=np.log10(np.array(a['lrs_A'])/LRS_I)/(.10*np.sqrt(len(LRS_I)/5))
    hrs=(np.arcsinh(np.array(a['hrs_A'])/BASELINE_SD)-np.arcsinh((HRS_I-BASELINE)/BASELINE_SD))/(2*np.sqrt(len(HRS_I)))
    # Soft field screening supports selection; it is not a field-mobility model.
    phys=physical_weight*np.maximum(np.array(a['collector_muE_over_vsat'])-1,0)
    phys+=np.maximum(-np.array(a['source_barrier_V']),0)/.01
    r=np.r_[(np.array(a['folds_V'])-[VLU,VLD])/[.02,.012],(np.array(a['thresholds_V'])-[VLU,VLD])/[.02,.012],up,dn,lrs,hrs,np.array(a['domain_shortfall_V'])/.01,phys]
    if len(r)!=NRES:raise RuntimeError((len(r),NRES))
    return r

def main():
    global physics
    is_srh='srh' in sys.argv[1:]
    if is_srh:
        import mean_model_srh_v3
        physics=mean_model_srh_v3
    prefix='srh' if is_srh else 'linear'
    started=time.perf_counter();rng=np.random.default_rng(20260923+(1 if is_srh else 0));table,_=physics.load_transport_table();all_rows=[]
    prior=json.loads((HERE.parent/'idvd_model_v2/emitter_control_results.json').read_text())['records']
    if is_srh and (HERE/'linear_profile_results.json').exists():
        prior=json.loads((HERE/'linear_profile_results.json').read_text())['records']+prior
    profiles=np.geomspace(1e17,1e18,9)
    for na in profiles:
        model=physics.FastModel(na,table)
        nearest=sorted(prior,key=lambda d:abs(np.log(d['NA_cm3']/na)))[:2]
        seeds=[np.clip(np.asarray(d['x']),LO+1e-8,HI-1e-8) for d in nearest]
        # Deliberately paired stronger recombination and increased beta controls.
        base=seeds[0]
        for db,dt in [(1,-.5),(2,-1),(3,-1.5),(.5,-1.)]:
            x=base.copy();x[0]+=db;x[1]+=dt;x[2]+=dt;seeds.append(np.clip(x,LO+1e-8,HI-1e-8))
        for _ in range(24):
            seeds.append(np.array([rng.uniform(.4,4.8),rng.uniform(-11.3,-8.5),rng.uniform(-10.5,-6.5),rng.uniform(0,4.2),rng.uniform(24,40),rng.uniform(3,45),rng.uniform(17,19.5)]))
        scores=[float(np.sum(residual(model,x)**2)) for x in seeds]
        print('PROFILE',na,'seed_scores',sorted(scores)[:4],flush=True)
        profile=[]
        for ind in np.argsort(scores)[:3]:
            if scores[ind]>=1e4:continue
            start=time.perf_counter();fit=least_squares(lambda x:residual(model,x),seeds[ind],bounds=(LO,HI),diff_step=1e-3,max_nfev=100,xtol=2e-6,ftol=2e-6,gtol=2e-6,x_scale='jac')
            row=dict(NA_cm3=float(na),x=fit.x.tolist(),parameters=decode(fit.x).tolist(),objective=float(fit.fun@fit.fun),prediction=predict(model,fit.x),nfev=int(fit.nfev),optimizer_success=bool(fit.success),fit_seconds=time.perf_counter()-start,closure=prefix+'_effective_bulk_lifetime_standard_emitter_diffusion_Auger_OFF',status='DIAGNOSTIC_PENDING_FULL_VALIDATION')
            profile.append(row);all_rows.append(row);print(json.dumps(row),flush=True)
            (HERE/(prefix+'_profile_results.json')).write_text(json.dumps(dict(records=all_rows,elapsed_s=time.perf_counter()-started,targets=dict(VLU_V=VLU,VLD_V=VLD,offset_steps=OFFSETS.tolist(),aligned_up_A=TARGET_UP.tolist(),aligned_down_A=TARGET_DN.tolist(),LRS_V=LRS_V.tolist(),LRS_A=LRS_I.tolist())),indent=2))
            best=min(all_rows,key=lambda d:d['objective']);(HERE/(prefix+'_best_attempt.json')).write_text(json.dumps(best,indent=2))
        if profile:prior=sorted(profile,key=lambda d:d['objective'])[:1]+prior
    print('DONE',time.perf_counter()-started,flush=True)
if __name__=='__main__':main()
