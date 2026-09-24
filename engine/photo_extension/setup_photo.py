"""Photo-extended model wiring (copies only; the original package is untouched)."""
import sys,json,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1]/'model'/'janus_calibration_20260920'
sys.path.insert(0,str(R/'hypothesis_study_20260920'));sys.path.insert(0,str(R/'claude_crosscheck_20260920/joint_model'));sys.path.insert(0,str(R/'reader_fig3_20260921/gate_model'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import photo_mean as m
import conditional_table as ct
import check_escape as ce
D=json.loads((R/'claude_crosscheck_20260920/joint_model/refit_3.json').read_text());C=json.loads((R/'claude_crosscheck_20260920/joint_model/gate_dynamic_calibration.json').read_text())
BASE=np.r_[D['prediction']['parameters'],C['parameters'][0],.001*C['parameters'][1],-2.,1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,2e-11,0.,0.,0.]   # ... p[21] local strength, p[22] saturation, p[23] log-fluctuation, p[24] bulk switch, p[25] kappaF (1/V)
SIGG=C['parameters'][2];SIGE=.001*C['parameters'][3]
MODEL=m.FastModel(D['NA_cm3'],m.load_transport_table()[0]);ct.m=m;ct.model=MODEL;ct.table=MODEL.table
ce.install_kernel(D['NA_cm3'])
GM=R/'reader_fig3_20260921/gate_model'
if (GM/'gate_avalanche_extended.npz').exists():
    zz=np.load(GM/'gate_avalanche_extended.npz');ct.cf.rv=zz['reverse_V'];ct.cf.pmf=zz['probability'];ct.cf.K=int(np.flatnonzero(ct.cf.pmf.sum(axis=0)>0)[-1]);ct.cf.pmf=ct.cf.pmf[:,:ct.cf.K+1]
def state(u,vd,p):
    """ct.state re-implemented for the photo model: photogeneration counted among the unit events (column 10)."""
    def comp(r):return m.components(u,r,p,MODEL.na,MODEL.vbi,MODEL.rg,MODEL.fg,MODEL.table)
    r=ct.brentq(lambda r:comp(r)[0]-vd,0.,vd-u,xtol=1e-11);z=comp(r)
    psi=u-m.VT*np.log1p(z[10]);ratio=(p[5]*1e-7/(m.TSI_M*100))*(p[7]*1e-7/z[11]);qb=(z[13]-m.COX_F*u)/(1+ratio);qa=qb*ratio
    return np.array([u,r,z[1],(z[1]-z[3]-z[16])/m.Q,(z[5]+z[6]+z[7])/m.Q,m.COX_F*psi,qb+qa,m.Q*MODEL.na*m.AREA_CM2*z[11],z[2],z[3],z[8]+z[9]+z[18]])
def params(vg,iph,dg=0.,de=0.,dibl=0.,gamma=0.,kappa=0.,ip=0.,S=1.,dj=0.,dm=0.,aloc=0.,isat=2e-11,dloc=0.,bulk=0.,kappaF=0.):
    p=BASE.copy();p[11]=vg;p[13]=iph;p[9]+=dg;p[10]+=de;p[14]=dibl;p[15]=gamma;p[16]=kappa;p[17]=ip;p[18]=S;p[19]=dj;p[20]=dm;p[21]=aloc;p[22]=isat;p[23]=dloc;p[24]=bulk;p[25]=kappaF;return p
