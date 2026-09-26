"""State-dependent compound-II first passage with slowly modulated junction SRH.

log_j changes the physically positive junction lifetime. phi_e changes only
the minority-hole injection prefactor. Rates and branch currents are recomputed.
"""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
import coupled_mean as m
import compound_fpt as cf
H=Path(__file__).resolve().parent;R=H.parent
d=json.loads((R/'idvd_model_v3/final_candidate.json').read_text());p0=np.r_[d['parameters'],0.,0.]
model=m.FastModel(d['NA_cm3'],m.load_transport_table()[0]);table=model.table
prob=np.r_[0.,np.geomspace(1e-6,.001,10),np.linspace(.002,.998,499),1-np.geomspace(.001,1e-6,10),1.]

def parameters(log_j,phi_e):
    p=p0.copy();p[2]*=np.exp(log_j);p[10]=phi_e;return p

def state(u,vd,p):
    def comp(r):return m.components(u,r,p,model.na,model.vbi,model.rg,model.fg,table)
    r=brentq(lambda r:comp(r)[0]-vd,0.,vd-u,xtol=1e-11);z=comp(r)
    psi=u-m.VT*np.log1p(z[10]);ratio=(p[5]*1e-7/(m.TSI_M*100))*(p[7]*1e-7/z[11]);qb=(z[13]-m.COX_F*u)/(1+ratio);qa=qb*ratio
    return np.array([u,r,z[1],(z[1]-z[3]-z[16])/m.Q,(z[5]+z[6]+z[7])/m.Q,m.COX_F*psi,qb+qa,m.Q*model.na*m.AREA_CM2*z[11],z[2],z[3],z[8]+z[9]])

def calculate(log_j,phi_e,fine=False):
    p=parameters(log_j,phi_e);b,i,j,fold=model.classify(p,m.state_grid(401))
    result=[];metadata=[]
    for direction,foldV,uf in [('LU',fold[0],b[i,17]),('LD',fold[1],b[j,17])]:
        step=.002 if fine else .004
        volts=np.arange(foldV-.28,foldV-.001,step) if direction=='LU' else np.arange(foldV+.30,foldV+.001,-step)
        ug=np.unique(np.round(np.r_[np.linspace(.10,.90,181) if direction=='LU' else np.linspace(.55,1.04,181),np.linspace(uf-.065,uf+.065,61)],12))
        hazards=[];accepted=[];bad=[]
        for vd in volts:
            rr=np.array([state(u,vd,p) for u in ug])
            try:
                xx,ix,r,bt,ii,death=cf.make_lattice(rr,direction,.06);tm,A,check=cf.backward(r,bt,ii,death,direction)
            except ValueError:
                bad.append(float(vd));continue
            h=1/tm[ix] if np.isfinite(tm[ix]) and tm[ix]>0 else np.nan
            hazards.append(h);accepted.append(vd)
        V=np.array(accepted);h=np.array(hazards);valid=(np.isfinite(h))&(h>=1e-4)
        badidx=np.flatnonzero(~valid);begin=int(badidx[-1]+1) if len(badidx) else 0
        if begin>=len(h):raise ValueError('No resolved hazard')
        h[:begin]=0;progress=V if direction=='LU' else -V;cum=cumulative_trapezoid(h/.4,progress,initial=0)
        # Any remaining survival is an explicitly reported deterministic fold
        # atom, not silently discarded or renormalized. Extreme tail states can
        # reach the fold; accuracy is checked near the calibrated state range.
        assert cum[-1]>0,(log_j,phi_e,direction,cum[-1])
        targets=-np.log1p(-np.minimum(prob,1-1e-15));qv=np.interp(targets,cum,V,right=foldV);result.append(qv)
        metadata.append({'direction':direction,'fold_V':float(foldV),'start_hazard_per_s':float(h[begin]),'early_mass_bound_monotone':float(abs(V[begin]-V[0])*h[begin]/.4),'end_survival':float(np.exp(-cum[-1])),'skipped_near_fold_nodes':bad,'voltage_V':V.tolist(),'hazard_per_s':h.tolist()})
    # Full branch currents are retained to reconstruct loops at arbitrary states.
    Vout=np.linspace(0,4,401);curves=[]
    for part in [b[:i+1],b[j:]]:
        good=(Vout>=part[0,0])&(Vout<=part[-1,0]);vals=np.full_like(Vout,np.nan);vals[good]=np.exp(np.interp(Vout[good],part[:,0],np.log(np.maximum(part[:,1],1e-300))))-1e-300;curves.append(vals)
    return np.array(result),fold,np.array(curves),metadata

def main():
    out=H/'conditional_table';out.mkdir(exist_ok=True);J=np.linspace(-1.6,1.6,9);E=np.array([.0005,.002,.0035,.005,.0065]);start=time.perf_counter();records=[]
    hashes={name:hashlib.sha256((H/name).read_bytes()).hexdigest() for name in ['coupled_mean.py','compound_fpt.py','conditional_table.py','avalanche/cluster_pmf.npz']}
    sig=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    qq=np.empty((len(J),len(E),2,len(prob)));folds=np.empty((len(J),len(E),2));curr=np.empty((len(J),len(E),2,401))
    for j,x in enumerate(J):
        for e,y in enumerate(E):
            path=out/f'node_{j}_{e}.npz';meta=out/f'node_{j}_{e}.json'
            if path.exists() and meta.exists() and json.loads(meta.read_text()).get('signature')==sig:
                z=np.load(path);qv,f,ci=z['quantiles'],z['folds'],z['current'];rec=json.loads(meta.read_text())
            else:
                qv,f,ci,checks=calculate(x,y);rec={'signature':sig,'log_junction_tau_ratio':float(x),'emitter_phi_V':float(y),'checks':checks};np.savez_compressed(path,quantiles=qv,folds=f,current=ci);meta.write_text(json.dumps(rec,indent=2))
            qq[j,e]=qv;folds[j,e]=f;curr[j,e]=ci;records.append(rec);print(f'Conditional nodes {j*len(E)+e+1}/{len(J)*len(E)}: logTau={x:.2f}, phiE={y*1000:.2f}mV',flush=True)
    np.savez_compressed(out/'lookup.npz',log_j=J,phi_e=E,probability=prob,quantiles=qq,folds=folds,current=curr,VD=np.linspace(0,4,401))
    (out/'summary.json').write_text(json.dumps({'nodes':len(records),'runtime_s':time.perf_counter()-start,'source_hashes':hashes,'probability_points':len(prob),'records':records,'fast_SRH_count_model':'Poisson count comparator retained; separate exact four-channel count calculation shows anti-correlation and no cycle-scale retention. Slow modulation changes mean SRH coefficient explicitly.'},indent=2))
if __name__=='__main__':main()
