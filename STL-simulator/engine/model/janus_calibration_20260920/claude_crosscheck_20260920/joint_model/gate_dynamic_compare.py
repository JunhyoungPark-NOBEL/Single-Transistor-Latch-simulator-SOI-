"""Joint-current-calibrated internal-state Monte Carlo.

Explicitly separate the observed acquisition trend from stationary fluctuations.
State evolution modulates rates and branch currents, never output-voltage jitter.
Up/down simulations are separate records. Reduced kinetics remain conditional.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import least_squares
from scipy.stats import qmc,norm
from scipy.signal import lfilter
H=Path(__file__).resolve().parent;R=H.parents[1]

def ou(n,dt,tau,rng):
 rho=np.exp(-dt/tau);a,_=lfilter([np.sqrt(1-rho*rho)],[1,-rho],rng.standard_normal(n),zi=[rho*rng.standard_normal()]);return a

def setup():
 t=np.load(H/'gate_state_lookup.npz');J,E,P=t['J'],t['E'],t['probability'];fd=RGI((J,E),t['fold'],bounds_error=False,fill_value=None)
 hz=[RGI((J,E,t['dist']),t['hazard'][:,:,d],bounds_error=False,fill_value=0.) for d in range(2)]
 ci=[RGI((J,E,t['du']),t['log_current_up'],bounds_error=False,fill_value=None),RGI((J,E,t['dd']),t['log_current_down'],bounds_error=False,fill_value=None)]
 quant=[RGI((J,E,P),t['quantiles'][:,:,d],bounds_error=False,fill_value=None) for d in range(2)]
 return t,fd,hz,ci,quant

def calibrate():
 t,fd,hz,ci,quant=setup();raw=np.load(R/'outputs/measured_idvd_parsed.npz');vm=np.array([(raw['VLU_low']+raw['VLU_high'])/2,(raw['VLD_low']+raw['VLD_high'])/2])
 sob=qmc.Sobol(4,scramble=True,seed=2026092910).random_base2(15);z=norm.ppf(sob[:,:2]);U=sob[:,2:]
 target=np.r_[vm.mean(1),vm.std(1,ddof=1)]
 def predict(x):
  # e mean/SD units are millivolts for optimization conditioning.
  st=np.c_[x[0]+x[2]*z[:,0],.001*(x[1]+x[3]*z[:,1])]
  v=np.array([quant[d](np.c_[st,U[:,d]]) for d in range(2)])
  return np.r_[v.mean(1),np.sqrt(v.var(1)+.01**2/12)],v,st
 fit=least_squares(lambda x:(predict(x)[0]-target)/[.005,.003,.004,.002],[0,0,.19,.45],bounds=([-.25,-.5,.02,.02],[.25,.5,.30,1.0]),diff_step=1e-3,xtol=1e-9,ftol=1e-9,gtol=1e-8)
 x=fit.x;met,v,st=predict(x);J,E=t['J'],t['E']
 outside=(st[:,0]<J[0])|(st[:,0]>J[-1])|(st[:,1]<E[0])|(st[:,1]>E[-1])
 # Fit time response to training voltages only, keeping amplitude fixed by sigmaLD.
 vd=np.round(np.arange(3,4.00001,.02),8);Y=np.log(raw['Idown'][[np.argmin(abs(raw['Vdown']-vv)) for vv in vd]])
 cov=(Y-Y.mean(1,keepdims=True))@(vm[1]-vm[1].mean())/(len(vm[1])-1);sd=Y.std(1,ddof=1)*vm[1].std(ddof=1)
 def mean_event(j,e,d):return np.mean(quant[d](np.c_[np.full(2000,j),np.full(2000,e), (np.arange(2000)+.5)/2000]))
 j0,e0=x[0],x[1]*.001;dj=.001;de=.00001
 dlJ=(mean_event(j0+dj,e0,1)-mean_event(j0-dj,e0,1))/(2*dj)
 dlE=(mean_event(j0,e0+de,1)-mean_event(j0,e0-de,1))/(2*de)
 def logcurrent(j,e):
  states=np.c_[np.full(len(vd),j),np.full(len(vd),e)];offset=vd-fd(states)[:,1];return ci[1](np.c_[states,offset])
 sJ=(logcurrent(j0+dj,e0)-logcurrent(j0-dj,e0))/(2*dj);sE=(logcurrent(j0,e0+de)-logcurrent(j0,e0-de))/(2*de)
 train=vd<=3.500001;test=vd>=3.599999;delta=(vd-vm[1].mean())/.4
 def pred_cov(p):
  fraction,tau=p;return dlE*sE*(x[3]*.001)**2*(fraction+(1-fraction)*np.exp(-delta/tau))+dlJ*sJ*x[2]**2*np.exp(-delta/5.)
 scale=np.max(abs(cov[train]));tf=least_squares(lambda p:(pred_cov(p)[train]-cov[train])/scale,[.05,1.0],bounds=([0,.05],[.8,20.]))
 pc=pred_cov(tf.x);static=pred_cov([1,1]);report=dict(parameter_names=['mean_GIDL_phi_V','mean_emitter_phi_mV','SD_GIDL_phi_V','SD_emitter_phi_mV'],parameters=x.tolist(),fit_targets=['mean_VLU','mean_VLD','SD_VLU','SD_VLD'],target=target.tolist(),predicted=met.tolist(),outside_static_fraction=float(outside.mean()),kinetic_fit=dict(fraction_slow=float(tf.x[0]),tau_fast_s=float(tf.x[1]),slow_tau_s=1000.,up_residual_tau_s=5.,training_voltage_V=[3,3.5],heldout_voltage_V=[3.6,4],heldout_covariance_RMSE=float(np.sqrt(np.mean((pc[test]-cov[test])**2))),heldout_r_RMSE=float(np.sqrt(np.mean(((pc-cov)/sd)[test]**2))),static_heldout_r_RMSE=float(np.sqrt(np.mean(((static-cov)/sd)[test]**2))),LD_event_derivatives=[dlJ,dlE]),scope='Internal-state amplitudes calibrated to widths; kinetics to same-record training covariance. No microscopic defect identification. Up residual tau5s and slow1000s are conditional choices, not extracted lifetimes.')
 (H/'gate_dynamic_calibration.json').write_text(json.dumps(report,indent=2));np.savetxt(H/'gate_model_covariance.csv',np.c_[vd,cov,pc,static,cov/sd,pc/sd,static/sd,sJ,sE],delimiter=',',header='VD_V,measured_covariance,new_dynamic_covariance,static_covariance,measured_r,new_dynamic_r,static_r,dlnI_dGIDLphi,dlnI_dphiE',comments='')
 print(json.dumps(report),flush=True)
 return report

def simulate(n=100,seed=2026092920,mode='dynamic',dv=.002,rate=.4,amplitude_scale=1.,tau_up_s=None):
 t,fd,hz,ci,quant=setup();fit=json.loads((H/'gate_dynamic_calibration.json').read_text());j0,e0,sj,se=fit['parameters'];se*=.001;e0*=.001;kin=fit['kinetic_fit'];sf=kin['fraction_slow'];te=kin['tau_fast_s']
 raw=np.load(R/'outputs/measured_idvd_parsed.npz');lu=(raw['VLU_low']+raw['VLU_high'])/2;k=np.linspace(-1,1,100);trend=np.polyval(np.polyfit(k,lu,2),k);frac=np.var(trend,ddof=1)/np.var(lu,ddof=1)
 # Acquisition trend is a calibrated nuisance trajectory. Because increasing GIDL field
 # lowers VLU, map the centered voltage trend with the opposite sign into the GIDL potential offset.
 trend=-(trend-trend.mean())/trend.std(ddof=1)
 sj*=amplitude_scale;se*=amplitude_scale
 dt=dv/rate;steps=int(round(4/dv));rng=np.random.default_rng(seed);draws=[];currents=[];outside=[];records=[]
 for d in range(2):
  count=n*(steps+1);tt=np.arange(count)*dt
  x=ou(count,dt,5. if tau_up_s is None else tau_up_s,rng);y=np.sqrt(sf)*ou(count,dt,1000.,rng)+np.sqrt(1-sf)*ou(count,dt,te,rng)
  if d==0 and mode!='stationary':
   # Repeat the calibrated 100-acquisition window for ensemble checks; each window
   # is simulated separately by callers, so no artificial window boundary exists.
   trendn=np.interp(np.linspace(0,99,n),np.arange(100),trend)
   x=np.sqrt(frac)*np.repeat(trendn,steps+1)+np.sqrt(1-frac)*x
  if mode=='frozen':x=np.repeat(rng.normal(size=n),steps+1);y=np.repeat(rng.normal(size=n),steps+1)
  if mode=='fast_only':x[:]=0;y[:]=0
  state=np.c_[j0+sj*x,e0+se*y]
  outside.append(float(np.mean((state[:,0]<t['J'][0])|(state[:,0]>t['J'][-1])|(state[:,1]<t['E'][0])|(state[:,1]>t['E'][-1]))))
  V=np.linspace(0,4,steps+1) if d==0 else np.linspace(4,0,steps+1);volts=np.tile(V,n);fold=fd(state)[:,d];distance=fold-volts if d==0 else volts-fold
  h=hz[d](np.c_[state,np.clip(distance,0,.32)]).reshape(n,-1);h[distance.reshape(n,-1)>.32]=0
  integrated=np.cumsum(h*dt,axis=1)-h[:,0,None]*dt
  event=(integrated>=rng.exponential(size=n)[:,None])|(distance.reshape(n,-1)<=0)
  hit=event.any(1);idx=event.argmax(1);v=np.where(hit,V[idx],np.nan);mid=np.floor(v/.01)*.01+.005
  selected=np.arange(0,steps+1,int(round(.01/dv)));ss=state.reshape(n,-1,2)[:,selected];vv=V[selected];fs=fd(ss.reshape(-1,2)).reshape(n,-1,2);on=vv[None,:]>=v[:,None] if d==0 else vv[None,:]>v[:,None]
  on[~hit]=bool(d==1) # Unswitched up remains HRS; unswitched down remains LRS.
  cur=np.empty_like(on,dtype=float)
  for active in [0,1]:
   off=fs[:,:,0]-vv if active==0 else vv-fs[:,:,1];values=np.exp(ci[active](np.c_[ss.reshape(-1,2),np.maximum(off.ravel(),0.)])).reshape(cur.shape);cur[on==bool(active)]=values[on==bool(active)]
  cur[:,vv==0]=0;draws.append(np.c_[v,mid]);currents.append(cur)
  valid=np.isfinite(v);pre=np.array([np.flatnonzero(vv<x)[-1] if d==0 else np.flatnonzero(vv>x)[-1] for x in v[valid]])
  prei=cur[np.flatnonzero(valid),pre]
  rec=dict(direction=['LU','LD'][d],mean_V=float(np.nanmean(mid)),SD_mV=float(np.nanstd(mid,ddof=1)*1000),pre_current_median_A=float(np.median(prei)),censored=int((~hit).sum()),current_mean_I4_A=float(cur[:,np.argmax(vv)].mean()),current_CV_I4=float(cur[:,np.argmax(vv)].std(ddof=1)/cur[:,np.argmax(vv)].mean()),r_I4=float(np.corrcoef(mid[valid],np.log(cur[valid,np.argmax(vv)]))[0,1]),r_I3=float(np.corrcoef(mid[valid],np.log(cur[valid,np.argmin(abs(vv-3))]))[0,1]))
  records.append(rec)
 return dict(transition=np.array(draws),current=np.array(currents),Vup=np.linspace(0,4,401),Vdown=np.linspace(4,0,401)),dict(mode=mode,seed=seed,internal_voltage_step=dv,rate_V_per_s=rate,amplitude_scale=amplitude_scale,outside_fraction=outside,records=records)

def main():
 start=time.perf_counter();calibrate();reps=[]
 for mode in ['fast_only','frozen','dynamic']:
  z,r=simulate(mode=mode);np.savez_compressed(H/f'gate_{mode}_mc100.npz',**z);reps.append(r);print(json.dumps(r),flush=True)
 # Ten fixed seeds, each a separate 100-acquisition record, no seed selection.
 allz=[]
 for k in range(10):
  z,r=simulate(seed=2026093000+k);allz.append(z);reps.append(r)
 np.savez_compressed(H/'gate_dynamic_mc1000.npz',transition=np.concatenate([z['transition'] for z in allz],axis=1),current=np.concatenate([z['current'] for z in allz],axis=1),Vup=allz[0]['Vup'],Vdown=allz[0]['Vdown'])
 (H/'gate_dynamic_results.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,records=reps,protocol='Up/down unpaired; .4V/s assumed for both; current readout is quasistatic branch, no empirically added current floor or gain noise. Trend included only in up-state and fitted to this observed record.'),indent=2))
if __name__=='__main__':main()
