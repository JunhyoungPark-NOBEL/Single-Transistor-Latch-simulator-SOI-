"""Low-Vd IDVG-only, constant-slope EKV fits; no latch or stochastic fit.

Run with the existing stl_stochastic_research/.venv Python.
Two columns remain series_1 and series_2 because their direction is unspecified.
No instrument floor is added to the physical channel current.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT = ROOT / "inputs" / "idvg_vd005_user.tsv"
T = 300.0
UT = 1.380649e-23*T/1.602176634e-19
COX = 3.9*8.8541878128e-12/14.1e-9
W, L, VD = 200e-9, 500e-9, 0.05
MODELS = ("constant_mobility", "mobility_degradation", "symmetric_access_resistance", "mobility_and_access_diagnostic")

def decode(p, model):
    ans = dict(Vth0_V=float(p[0]), n=float(p[1]), mu0_cm2_Vs=float(np.exp(p[2])))
    ans["theta_per_V"] = float(p[3]) if model in (MODELS[1], MODELS[3]) else 0.0
    ans["Rsd_ohm"] = float(p[-1]*1e4) if model in (MODELS[2], MODELS[3]) else 0.0
    ans["beta0_A_V2"] = ans["mu0_cm2_Vs"]*1e-4*COX*W/L
    ans["Is_A"] = 2*ans["n"]*ans["beta0_A_V2"]*UT**2
    return ans

def intrinsic(vg, vd, par):
    ov = vg-par["Vth0_V"]
    p = ov/par["n"]
    f = np.logaddexp(0.0, p/(2*UT))
    r = np.logaddexp(0.0, (p-vd)/(2*UT))
    # Smooth positive overdrive scale is fixed by n*UT, not an extra fitting parameter.
    ov_eff = par["n"]*UT*np.logaddexp(0.0, ov/(par["n"]*UT))
    beta = par["beta0_A_V2"]/(1+par["theta_per_V"]*ov_eff)
    return 2*par["n"]*beta*UT**2*(f-r)*(f+r)

def predict(vg, p, model, vd=VD):
    par = decode(p,model)
    if par["Rsd_ohm"] == 0:
        return intrinsic(vg, vd, par)
    # Symmetric Rs=Rd assumption is declared; one low-Vd curve cannot infer the split.
    rsum = par["Rsd_ohm"]
    lo = np.zeros_like(np.asarray(vg,dtype=float))
    hi = np.minimum(intrinsic(vg,vd,par), vd/rsum)
    for _ in range(46):
        mid = .5*(lo+hi)
        target = intrinsic(vg-mid*rsum/2, vd-mid*rsum,par)
        below = mid < target
        lo = np.where(below,mid,lo)
        hi = np.where(below,hi,mid)
    return .5*(lo+hi)

def fit(vg, current, model, floor, initial=None):
    used = current >= floor
    strong = current >= 1e-7
    ymax = np.max(current)
    names = ["Vth0_V", "n", "log_mu0_cm2_Vs"]
    p0 = [-.40,1.7,np.log(650.)]
    lower = [-1.5,1.,np.log(5.)]
    upper = [.5,4.,np.log(1400.)]
    if model in (MODELS[1],MODELS[3]):
        names += ["theta_per_V"]; p0 += [.6]; lower += [0.]; upper += [10.]
    if model in (MODELS[2],MODELS[3]):
        names += ["Rsd_10kohm"]; p0 += [.8]; lower += [0.]; upper += [2.]
    def residual(p):
        out = predict(vg,p,model)
        # Equal total weights for log and strong-current blocks. Scales are declared
        # fitting tolerances, not measured independent instrument standard errors.
        rlog = (np.log10(np.maximum(out[used],1e-300))-np.log10(current[used]))/.05/np.sqrt(used.sum())
        rlin = (out[strong]-current[strong])/(.02*ymax)/np.sqrt(strong.sum())
        return np.r_[rlog,rlin]
    starts=[p0]
    if initial is not None: starts.insert(0,initial)
    opts=[least_squares(residual,s,bounds=(lower,upper),xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=900,x_scale="jac") for s in starts]
    opt=min(opts,key=lambda x: np.dot(x.fun,x.fun))
    out = predict(vg,opt.x,model)
    logr=np.log10(np.maximum(out[used],1e-300)/current[used])
    sj=np.linalg.svd(opt.jac,compute_uv=False)
    cov=np.linalg.pinv(opt.jac.T@opt.jac)
    scale=np.sqrt(np.maximum(np.diag(cov),0))
    corr=cov/np.maximum(scale[:,None]*scale[None,:],1e-300)
    transition=used & (current<=1e-7)
    return dict(model=model, floor_A=floor, params=decode(opt.x,model), optimizer_parameters=opt.x.tolist(),
        optimizer_parameter_names=names, success=bool(opt.success), nfev=int(opt.nfev),
        objective=float(np.dot(opt.fun,opt.fun)), log_rms_decades=float(np.sqrt(np.mean(logr**2))),
        strong_rms_A=float(np.sqrt(np.mean((out[strong]-current[strong])**2))),
        strong_rms_fraction_of_max=float(np.sqrt(np.mean((out[strong]-current[strong])**2))/ymax),
        transition_log_rms_decades=float(np.sqrt(np.mean(np.log10(out[transition]/current[transition])**2))),
        n_above_floor=int(used.sum()), n_strong=int(strong.sum()),
        scaled_jacobian_condition=float(sj[0]/sj[-1]), local_parameter_correlation=corr.tolist(),
        bound_distance_fraction=(np.minimum(opt.x-lower,np.array(upper)-opt.x)/(np.array(upper)-lower)).tolist(),
        predicted_channel_at_VG_minus2_A=float(predict(np.array([-2.]),opt.x,model)[0]))

def main():
    raw=np.loadtxt(INPUT)
    assert raw.shape==(101,3) and np.all(np.diff(raw[:,0])>0)
    vg=np.tile(raw[:,0],2)
    current=np.r_[raw[:,1],raw[:,2]]
    rows=[]
    for floor in (5e-12,1e-11):
        for model in MODELS:
            row=fit(vg,current,model,floor)
            row["series"]="joint"
            rows.append(row)
            print(row["series"],floor,model,row["params"],row["log_rms_decades"],row["strong_rms_fraction_of_max"],flush=True)
    for j in (1,2):
        for floor in (5e-12,1e-11):
            for model in MODELS[:3]:
                row=fit(raw[:,0],raw[:,j],model,floor)
                row["series"]=f"series_{j}"
                rows.append(row)
    selected=next(r for r in rows if r["series"]=="joint" and r["model"]==MODELS[1] and r["floor_A"]==5e-12)
    meta=dict(input=str(INPUT),data_shape=list(raw.shape),temperature_K=T,drain_V=VD,width_m=W,length_m=L,EOT_m=14.1e-9,
        Cox_F_m2=COX, UT_V=UT, column_labels=["VG_V","series_1_A","series_2_A"],
        objective="RMS log residual /0.05 decade and RMS strong-current residual /(0.02*Imax), equal block weights; strong>=100nA",
        censoring="Current below 5 or 10 pA excluded from fit, no fitted physical leakage floor added",
        parameter_bounds=dict(Vth0_V=[-1.5,.5],n=[1.,4.],mu0_cm2_Vs=[5.,1400.],theta_per_V=[0.,10.],Rsd_ohm=[0.,20000.]),
        limitations=["One drain bias only: no DIBL/saturation/II or body coupling extraction", "No NA extraction", "300K assumed pending temperature metadata", "Series labels are not sweep direction", "Local covariance is an identifiability diagnostic, not a measurement confidence interval", "Mobility is conditional on nominal W/L and EOT, not a unique transport extraction"],
        selected_candidate=selected, fits=rows)
    (HERE/"fit_results.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    prediction=[raw[:,0],raw[:,1],raw[:,2]]
    names=["VG_V","series_1_A","series_2_A"]
    for row in rows:
        if row["series"]=="joint":
            prediction.append(predict(raw[:,0],row["optimizer_parameters"],row["model"]))
            names.append(f"{row['model']}_floor_{row['floor_A']:.0e}_A")
    np.savetxt(HERE/"predictions.csv",np.column_stack(prediction),delimiter=",",header=",".join(names),comments="")

if __name__=="__main__": main()
