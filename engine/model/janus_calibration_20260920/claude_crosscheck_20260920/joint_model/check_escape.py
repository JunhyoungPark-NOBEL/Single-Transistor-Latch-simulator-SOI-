"""Compute actual compound-generator escape for each joint refit iteration."""
from pathlib import Path
import sys,json,time
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1]
sys.path.insert(0,str(R/'hypothesis_study_20260920'))
import conditional_table as ct
import avalanche_clusters as ac
from scipy.integrate import cumulative_trapezoid

def install_kernel(na):
 path=H/f'avalanche_{na:.8e}.npz'
 if not path.exists():
  rr=np.linspace(.7,4.5,77);z=np.linspace(0,1,1201);prob=[];stats=[];n=200000
  for j,r in enumerate(rr):
   eps=11.7*ct.m.EPS0/100;width=np.sqrt(2*eps*(r+ct.model.vbi)/(ct.m.Q*na));peak=2*(r+ct.model.vbi)/width
   a,b=ac.van_overstraeten_300k(peak*z);ae=cumulative_trapezoid(a,z,initial=0)*width;ah=cumulative_trapezoid(b,z,initial=0)*width
   k,cens=ac.simulate(n,2026092800+j,z,ae,ah);assert cens==0 and k.max()<300
   prob.append(np.bincount(k,minlength=300)/n)
   expected=np.interp(r,ct.model.rg,ct.model.fg[0])-1
   stats.append([r,k.mean(),k.var(ddof=1),expected,(k.mean()-expected)/np.sqrt(k.var(ddof=1)/n),int(k.max())])
  np.savez_compressed(path,reverse_V=rr,probability=prob,stats=stats,N=n)
  print('New-NA avalanche kernel',na,flush=True)
 z=np.load(path);cf=ct.cf;cf.rv=z['reverse_V'];cf.pmf=z['probability'];cf.K=int(np.flatnonzero(cf.pmf.sum(axis=0)>0)[-1]);cf.pmf=cf.pmf[:,:cf.K+1]
 return dict(kernel=str(path.name),N_per_bias=int(z['N']),max_abs_mean_zscore=float(np.max(abs(z['stats'][:,4]))),max_pairs=int(np.max(z['stats'][:,5])))

def calculate(iteration):
 rec=json.loads((H/f'refit_{iteration}.json').read_text());p=np.r_[rec['prediction']['parameters'],0.,0.]
 ct.model=ct.m.FastModel(rec['NA_cm3'],ct.m.load_transport_table()[0]);ct.table=ct.model.table
 kernel=install_kernel(rec['NA_cm3'])
 ct.parameters=lambda j,e:p.copy()
 start=time.perf_counter();qv,fold,cur,meta=ct.calculate(0.,0.,fine=True)
 probs=(np.arange(200000)+.5)/200000
 samp=np.array([np.interp(probs,ct.prob,q) for q in qv])
 result=dict(iteration=iteration,means_V=samp.mean(axis=1).tolist(),SD_mV=(samp.std(axis=1)*1000).tolist(),folds_V=fold.tolist(),seconds=time.perf_counter()-start,checks=meta,kernel=kernel)
 np.savez_compressed(H/f'fpt_{iteration}.npz',quantiles=qv,probability=ct.prob,folds=fold,current=cur,samples=samp)
 (H/f'fpt_{iteration}.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
if __name__=='__main__':calculate(int(sys.argv[1]) if len(sys.argv)>1 else 0)
