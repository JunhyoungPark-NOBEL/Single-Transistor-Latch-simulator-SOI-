"""Real Simple/Detailed circuit timing; equal external drives, no result cache."""
import argparse, copy, json, os, platform, statistics, sys, time
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
parser=argparse.ArgumentParser()
parser.add_argument('--project', default=str(Path.cwd()))
parser.add_argument('--out', default=str(Path(__file__).resolve().parent/'circuit-benchmark'))
parser.add_argument('--cases', default='csvm_1pF,csvm_1fF,four_oscillators,body_rc,five_terminal_csvm')
args=parser.parse_args()
ROOT=Path(args.project).resolve()
OUT=Path(args.out).resolve()
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
from server.compute.circuit import run_circuit
from server.compute.circuit import custom
from server import jsonutil
import numpy as np
from scipy.signal import find_peaks

def csvm(cap=1e-12,duration=.015,n=1):
    els=[dict(type='V',name='VG',nodes=['gate','0'],wave=dict(kind='dc',value=-3))]
    probes=[]
    for k in range(1,n+1):
        d='drain' if n==1 else f'd{k}'
        els.extend([
            dict(type='I',name=f'I{k}',nodes=['0',d],wave=dict(kind='dc',value=1e-9)),
            dict(type='C',name=f'C{k}',nodes=[d,'0'],value=cap),
            dict(type='STL',name=f'X{k}',nodes=dict(d=d,g='gate',s='0'),device=dict(vg=-3))])
        probes.extend([f'V({d})',f'I(X{k}.d)',f'X{k}.vb'])
    return dict(bench='custom',mode='deterministic',netlist=dict(elements=els),
        tran=dict(t_stop_s=duration,t_start_save_s=0,dt_max_s=duration/3000,
        dt_min_s=1e-15,method='BE',reltol=1e-3),
        detect=dict(i_threshold_A=1e-8,hysteresis=10),probes=probes)

cases={'csvm_1pF':csvm(),'csvm_1fF':csvm(1e-15,4e-5),'four_oscillators':csvm(n=4),
       'body_rc':json.loads((ROOT/'web/review/body-terminals/live-request.json').read_text()),
       'five_terminal_csvm':csvm(1e-15,4e-5)}
els=cases['five_terminal_csvm']['netlist']['elements']
els[-1]['nodes'].update(bg='bg',b='body')
els.extend([dict(type='V',name='VBG',nodes=['bg','0'],wave=dict(kind='sine',vo=0,va=.02,freq=5e4)),
            dict(type='R',name='RB',nodes=['body','0'],value=1e12),
            dict(type='C',name='CB',nodes=['body','0'],value=1e-16)])
cases['five_terminal_csvm']['probes'].extend(['V(body)','V(bg)','I(X1.b)'])
for el in cases['body_rc']['netlist']['elements']:
    if el['type']=='STL': el['device']['vg']=-3
    if el['type']=='V' and el['name']=='VG':
        w=el['wave']
        for key in ('v1','v2','vo','value'):
            if key in w: w[key]-=1


def summarize(result,wall):
    signals={s['key']:np.asarray(s['values']) for s in result['runs'][0]['signals']}
    t=np.asarray(result['runs'][0]['t'])
    key=next((key for key in signals if key in ('V(drain)','V(d1)','V(d)')),None)
    if key:
        y=signals[key]
        peaks,_=find_peaks(y,prominence=.05)
        kept=peaks[t[peaks]>=t[-1]*.2]
        freq=1/np.mean(np.diff(t[kept])) if len(kept)>=2 else None
    else: peaks=[];freq=None
    return dict(wall_s=wall,reported_s=result['runtime_s'],solver=result['solver_stats'],
        t_reached=float(t[-1]),saved_points=len(t),event_count=len(result['events']),
        drain_peaks=len(peaks),waveform_frequency_Hz=freq,warnings=result['warnings'])

results=dict(python=sys.version,platform=platform.platform(),notes=[
 'Equal input, VG=-3V, deterministic BE, identical tolerances. Different models yield different waveforms and step counts.',
 'Approximate timings on a shared worker. Three paired warm runs; other project checks may share CPU time.' ,
 'Warm wall time includes circuit preparation and result assembly, but no API/result cache/network/rendering.',
 'First run includes profile setup and lazy JIT compile/cache load. It is not a warm speed comparison.',
 'Frequency is a qualitative stored-waveform check via prominence=0.05V after first 20%; not an accuracy benchmark.'
],cases={})
for name in args.cases.split(','):
    base=cases[name]
    per={}
    for model in ('detailed','simple'):
        p=copy.deepcopy(base)
        for el in p['netlist']['elements']:
            if el['type']=='STL': el['device']['model']=model
        (OUT/f'{name}.{model}.input.json').write_text(json.dumps(p,indent=2))
        custom._PROFILE_CACHE.clear()
        tic=time.perf_counter();res=run_circuit(copy.deepcopy(p));first=summarize(res,time.perf_counter()-tic)
        (OUT/f'{name}.{model}.output.json').write_bytes(jsonutil.dumps(res))
        per[model]=dict(first=first,warm=[])
        print(json.dumps(dict(case=name,model=model,first=first)),flush=True)
    for repeat in range(3):
        for model in (('simple','detailed') if repeat%2 else ('detailed','simple')):
            p=json.loads((OUT/f'{name}.{model}.input.json').read_text())
            tic=time.perf_counter();res=run_circuit(p);per[model]['warm'].append(summarize(res,time.perf_counter()-tic))
    for model in per:
        per[model]['median_wall_s']=statistics.median(r['wall_s'] for r in per[model]['warm'])
        per[model]['median_solver_s']=statistics.median(r['solver']['runtime_s'] for r in per[model]['warm'])
    per['speedup_wall']=per['detailed']['median_wall_s']/per['simple']['median_wall_s']
    per['speedup_solver']=per['detailed']['median_solver_s']/per['simple']['median_solver_s']
    results['cases'][name]=per
    (OUT/'measurements.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(dict(case=name,speedup_wall=per['speedup_wall'],speedup_solver=per['speedup_solver'],
                         detailed_s=per['detailed']['median_wall_s'],simple_s=per['simple']['median_wall_s'])),flush=True)
