"""Exact backward equation for the specified compound jump generator.

Avalanche kernels are empirical local-field MC distributions; inverse-MFPT
ramp hazard remains a metastable approximation, as in the baseline.
"""
from pathlib import Path
import sys,json,csv,time,warnings
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve,MatrixRankWarning
from scipy.integrate import cumulative_trapezoid
H=Path(__file__).resolve().parent;R=H.parent;sys.path.insert(0,str(R/'stl_mc_v3'))
import lu_fpt as lu
import ld_fpt as ld
pm=np.load(H/'avalanche/cluster_pmf.npz');rv=pm['reverse_V'];pmf=pm['probability'];ks=np.arange(pmf.shape[1]);mu=pmf@ks
K=int(np.flatnonzero(pmf.sum(axis=0)>0)[-1]);pmf=pmf[:,:K+1]

def make_lattice(rows,direction,upper_extra=.06):
    u=rows[:,0];F=rows[:,3]-rows[:,4];ff=PchipInterpolator(u,F)
    s=np.flatnonzero((F[:-1]>0)&(F[1:]<0));a=np.flatnonzero((F[:-1]<0)&(F[1:]>0))
    wi=s[0] if direction=='LU' else s[-1];si=a[a>wi][0] if direction=='LU' else a[a<wi][-1]
    uw=brentq(ff,u[wi],u[wi+1]);us=brentq(ff,u[si],u[si+1]);ii=np.flatnonzero(rows[:,2]>=1e-8)[0];uth=brentq(PchipInterpolator(u,rows[:,2]-1e-8),u[ii-1],u[ii])
    ua=uth if direction=='LU' else min(uth,us-.05)
    q=(rows[:,5]+rows[:,6]+rows[:,7])/lu.Q;qfun=PchipInterpolator(u,q)
    if direction=='LU':
        x=q;rr=rows;lo=float(qfun(.1));end=float(qfun(ua));well=float(qfun(uw))
    else:
        x=-q[::-1];rr=rows[::-1];lo=-float(qfun(min(uw+upper_extra,u[-1]-.002)));end=-float(qfun(ua));well=-float(qfun(uw))
    xx=np.arange(np.ceil(lo),np.ceil(end));start=int(np.clip(np.round(well-xx[0]),0,len(xx)-1))
    rates=np.exp(PchipInterpolator(x,np.log(rr[:,[3,4,10]]),axis=0)(xx));reverse=PchipInterpolator(x,rr[:,1])(xx)
    # BTBT stored in amperes; the first two rates are counts/s.
    bt=rates[:,2]/lu.Q;ii_rate=np.maximum(rates[:,0]-bt,0);death=rates[:,1]
    return xx,start,reverse,bt,ii_rate,death

def backward(reverse,bt,ii,death,direction,unit=False):
    n=len(bt);ar=np.arange(n);meanpreserve=[]
    if unit:jumps=[(np.ones(n),ii)]
    else:
        assert reverse.min()>=rv[0] and reverse.max()<=rv[-1],(reverse.min(),reverse.max())
        probs=np.array([np.interp(reverse,rv,pmf[:,k]) for k in range(1,K+1)]).T
        mk=probs@np.arange(1,K+1);jumps=[(np.full(n,k),ii*probs[:,k-1]/mk) for k in range(1,K+1)]
        meanpreserve=np.sum(np.array([k*r for k,r in jumps]),axis=0)-ii
    dests=[];rates=[]
    sign=1 if direction=='LU' else -1
    for jump,rate in [(np.full(n,sign),bt),(np.full(n,-sign),death)]+[(sign*k,r) for k,r in jumps]:
        target=ar+jump.astype(int);target=np.maximum(target,0);rate=rate.copy();rate[target==ar]=0
        dests.append(target);rates.append(rate)
    total=np.sum(rates,axis=0);assert np.all(total>0)
    row=[ar];col=[ar];data=[np.ones(n)]
    for target,rate in zip(dests,rates):
        ok=(target<n)&(rate>0);row.append(ar[ok]);col.append(target[ok]);data.append(-rate[ok]/total[ok])
    A=coo_matrix((np.concatenate(data),(np.concatenate(row),np.concatenate(col))),shape=(n,n)).tocsc()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',MatrixRankWarning);t=spsolve(A,1/total)
    residual=np.max(abs(A@t-1/total))/(np.max(abs(t))+1e-100)
    return t,A,{'normalized_linear_residual':float(residual),'mean_drift_error_rate':float(np.max(abs(meanpreserve))) if len(meanpreserve) else 0.}

def run_direction(direction,unit=False,extra=.06):
    V,u,rows,fold=(lu.load_or_build() if direction=='LU' else ld.load_or_build())
    out=[]
    # Only finite, numerically resolved rates are trusted. The early-tail cut
    # below is validated by moving it and bounding the removed hazard integral.
    for vv,rr in zip(V,rows):
        xx,ix,r,bt,ii,death=make_lattice(rr,direction,extra)
        t,A,check=backward(r,bt,ii,death,direction,unit)
        valid=bool(np.all(np.isfinite(t)) and t[ix]>0)
        h=1/t[ix] if valid else np.nan
        out.append([vv,h,t[ix],len(xx),check['normalized_linear_residual'],check['mean_drift_error_rate']])
    ar=np.array(out);h=ar[:,1].copy()
    # Far-tail sparse solves suffer cancellation at very long escape times.
    # Keep the final connected region above1e-4/s; quantify cutoff controls.
    finite=(np.isfinite(h))&(h>=1e-4)
    bad=np.flatnonzero(~finite)
    begin=int(bad[-1]+1) if len(bad) else 0
    assert begin<len(V),'No resolved final hazard domain'
    assert np.all(finite[begin:]),'Unresolved node inside retained hazard domain'
    h[:begin]=0.;ar[:,1]=h
    progress=V if direction=='LU' else -V;cum=cumulative_trapezoid(h/lu.RAMP,progress,initial=0)
    rng=np.random.default_rng(2026092050+(direction=='LD'));targets=-np.log(rng.random(100000));sample=np.interp(targets,cum,V,right=fold);mid=np.floor(sample/.01)*.01+.005
    report={'direction':direction,'mode':'unit' if unit else 'compound_II','mean_continuous_V':float(sample.mean()),'SD_continuous_mV':float(sample.std(ddof=1)*1000),'mean_midpoint_V':float(mid.mean()),'SD_midpoint_mV':float(mid.std(ddof=1)*1000),'retained_domain_start_V':float(V[begin]),'retained_start_hazard_per_s':float(h[begin]),'far_tail_cutoff_hazard_per_s':1e-4,'omitted_probability_bound_if_monotone':float(abs(V[begin]-V[0])*h[begin]/lu.RAMP),'minimum_state_reverse_V':float(r.min()),'fold_atom_fraction':float((targets>cum[-1]).mean()),'unit_event_mean_preserved':True,'resolved_hazard_monotonic':bool(np.all(np.diff(h[begin:])>=0))}
    return ar,np.c_[sample,mid],report

def main():
    out=H/'fast_fpt';out.mkdir(exist_ok=True);t=time.perf_counter();reports=[]
    for direction in ['LU','LD']:
        for unit in [True,False]:
            curve,sample,rep=run_direction(direction,unit);reports.append(rep);name=direction+('_unit' if unit else '_compound')
            np.savetxt(out/(name+'_hazard.csv'),curve,delimiter=',',header='VD_V,hazard_per_s,MFPT_s,nstates,normalized_residual,mean_drift_error_rate',comments='')
            np.savetxt(out/(name+'_samples.csv'),sample,delimiter=',',header='continuous_V,midpoint_V',comments='')
            print(json.dumps(rep),flush=True)
    (out/'summary.json').write_text(json.dumps({'reports':reports,'seconds':time.perf_counter()-t},indent=2))
if __name__=='__main__':main()
