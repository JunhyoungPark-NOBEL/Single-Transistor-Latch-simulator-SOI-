"""Local nonuniform-field two-carrier avalanche genealogy, vanOst 300K.

No dead-space assumption or fitted excess-noise multiplier. Conditional field
is the same abrupt-junction triangular profile used by the accepted DC model.
"""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.integrate import cumulative_trapezoid
from numba import njit
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'stl_mc_v3'))
import lu_fpt as lu
from process_randomness.standard_mean import van_overstraeten_300k

@njit(cache=True)
def simulate(n,seed,z,ae,ah,cap=10000):
    np.random.seed(seed);counts=np.zeros(n,np.int64);censored=0
    xs=np.empty(2*cap+2);types=np.empty(2*cap+2,np.int64)
    for i in range(n):
        xs[0]=0.;types[0]=1;top=1;k=0
        while top:
            top-=1;x=xs[top];typ=types[top];a=ae if typ==1 else ah
            while True:
                old=np.interp(x,z,a);need=-np.log(max(np.random.random(),1e-300))
                target=old+need if typ==1 else old-need
                if target>=a[-1] or target<=0:break
                x=np.interp(target,a,z);k+=1
                if k>=cap:top=0;censored+=1;break
                xs[top]=x;types[top]=1;top+=1;xs[top]=x;types[top]=-1;top+=1
        counts[i]=k
    return counts,censored

def main():
    start=time.perf_counter();out=H/'avalanche';out.mkdir(exist_ok=True)
    rr=np.linspace(.7,4.5,77);z=np.linspace(0,1,1201);pmf=[];stats=[];n=400000
    for j,r in enumerate(rr):
        eps=11.7*lu.m.EPS0/100;width=np.sqrt(2*eps*(r+lu.model.vbi)/(lu.Q*lu.NA));peak=2*(r+lu.model.vbi)/width
        a,b=van_overstraeten_300k(peak*z);ae=cumulative_trapezoid(a,z,initial=0)*width;ah=cumulative_trapezoid(b,z,initial=0)*width
        k,censored=simulate(n,2026092010+j,z,ae,ah);assert censored==0
        assert k.max()<100,'Increase stored support; no silent truncation'
        h=np.bincount(k,minlength=100)/n;pmf.append(h)
        mu=k.mean();var=k.var(ddof=1);expected=float(np.interp(r,lu.model.rg,lu.model.fg[0]))-1
        stats.append([r,peak,width,n,mu,var,(k*k).mean()/mu,expected,(mu-expected)/np.sqrt(var/n),k.max(),censored])
    np.savez_compressed(out/'cluster_pmf.npz',reverse_V=rr,probability=np.array(pmf),pairs=np.arange(100))
    np.savetxt(out/'cluster_moments.csv',stats,delimiter=',',header='reverse_V,peak_field_Vcm,width_cm,N,mean_pairs,variance_pairs,compound_count_Fano,DC_expected_pairs,mean_zscore,max_pairs,censored',comments='')
    report={'N_per_bias':n,'biases':len(rr),'total_cascades':n*len(rr),'maximum_abs_mean_zscore':float(np.max(abs(np.array(stats)[:,8]))),'censored':0,'runtime_s':time.perf_counter()-start,'mean_closure':'Same triangular field as frozen DC; jump rates use II/q / empirical E[K] so conditional mean drift is preserved exactly.','limitations':'Local coefficients; no energy history or dead space; BTBT seeds not additionally multiplied because mean model does not do so.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
