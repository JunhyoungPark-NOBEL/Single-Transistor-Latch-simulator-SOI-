"""Measure shipped real compute paths without API result caching.

python scripts/benchmark_performance.py --repeats 5
Uses a fresh stochastic *table* cache per invocation. Existing Numba disk cache is
retained: first_call_s is not a fresh-install/uncached-JIT measurement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def environment():
    import numpy, scipy, numba
    info = dict(python=platform.python_version(),platform=platform.platform(),machine=platform.machine(),
                numpy=numpy.__version__,scipy=scipy.__version__,numba=numba.__version__,
                cpu_model=platform.processor() or "unknown",cpu_logical_count=os.cpu_count(),shared_host=True,
                thread_limits={k:os.environ[k] for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS")})
    try:
        cpu = Path('/proc/cpuinfo').read_text()
        info['cpu_model'] = next(s.split(':',1)[1].strip() for s in cpu.splitlines() if s.startswith('model name'))
    except (OSError,StopIteration):
        pass
    try:
        quota,period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        info['cpu_quota_cores'] = int(quota)/int(period) if quota != 'max' else None
    except (OSError,ValueError):
        info['cpu_quota_cores'] = None
    return info


def fingerprint():
    digest = hashlib.sha256()
    files = sorted((ROOT/'engine').rglob('*.py')) + sorted((ROOT/'server'/'compute').rglob('*.py'))
    files += [ROOT/'server'/name for name in ('geometry_model.py','simple_model.py','simple_config.py','params.py')]
    for file in files:
        digest.update(str(file.relative_to(ROOT)).encode());digest.update(file.read_bytes())
    return digest.hexdigest()


def _json_default(obj):
    if hasattr(obj,'tolist'):return obj.tolist()
    if hasattr(obj,'item'):return obj.item()
    raise TypeError(type(obj).__name__)


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False,default=_json_default)+'\n')


def run_case(case):
    from server.compute import resolve
    fn = resolve(case['kind'])
    payload = copy.deepcopy(case['payload'])
    tic = time.perf_counter();cpu = time.process_time()
    result = fn(payload)
    wall = time.perf_counter()-tic;cpu_s = time.process_time()-cpu
    detail = dict(wall_s=wall,cpu_s=cpu_s,reported_s=result.get('runtime_s'),solver=result.get('solver_stats'),
                  warnings=result.get('warnings',[]),engine=result.get('engine'))
    if case['kind']=='circuit':
        truncated = next((item.get('value',0) for item in result.get('summary',[]) if item.get('key')=='truncated_runs'),0)
        if truncated:
            raise RuntimeError(f"Incomplete transient {case['id']}: {truncated} truncated runs")
        stops = [float(r['t'][-1]) if len(r['t']) else 0 for r in result['runs']]
        expected = case['payload']['tran']['t_stop_s']
        if any(abs(stop-expected)>max(1e-15,expected*1e-8) for stop in stops):
            raise RuntimeError(f"Incomplete transient {case['id']}: reached {stops}, expected {expected}")
        detail.update(t_reached_s=stops,saved_points=[len(r['t']) for r in result['runs']],event_count=len(result.get('events',[])))
    if 'folds' in result:detail['folds']=result['folds']
    if case['kind']=='simple_calibrate':detail['rmse_log10']=result['rmse_log10']
    return detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--cases',default='',help='Comma-separated comparison ids; empty = all')
    parser.add_argument('--out',type=Path,default=ROOT/'docs'/'benchmarks'/'performance')
    parser.add_argument('--catalog',type=Path,default=ROOT/'server'/'data'/'performance_catalog.json')
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be >=1')
    # Dedicated on-disk table cache documents preparation separately and cannot
    # invalidate normal user tables. Preserve within run for realistic warm timings.
    cache_dir=tempfile.mkdtemp(prefix='stl-benchmark-stochastic-')
    os.environ['STL_STOCH_CACHE_DIR']=cache_dir
    from server.performance_cases import benchmark_cases,workload_features
    from server.payloads import normalize,normalize_device
    cases=benchmark_cases()
    selected=set(args.cases.split(',')) if args.cases else None
    if selected:cases=[c for c in cases if c['comparison_id'] in selected]
    for case in cases:
        if case['supported']:
            p=case['payload']
            if case['kind']=='circuit':
                for el in p['netlist']['elements']:
                    if el['type']=='STL':el['device']=normalize_device(el.get('device'),[])
            case['payload'],_=normalize(case['kind'],p)
            case['features']=workload_features(case['kind'],case['payload'])
            case['payload_sha256']=hashlib.sha256(json.dumps(case['payload'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        case.setdefault('notes',[])
        if case['kind'] in ('sweep_mc','hazard','vg_curve_stochastic'):
            case['notes'].append('Stochastic fold/hazard table cache retained between repeats; first-call follows earlier cases in this process.')
        if case['model']=='simple':case['notes'].append('Simple initial parameters are not fitted to the Detailed model. Timings do not compare accuracy.')
        save(args.out/(case['id']+'.input.json'),case['payload'])
    result=dict(schema_version=1,generated_at=datetime.now(timezone.utc).isoformat(),environment=environment(),
                engine_fingerprint=fingerprint(),methodology=[
                'Real backend compute wall time; excludes HTTP, result-cache hits, queue waiting, serialization, network and chart rendering.',
                'Five paired warm repetitions by default, alternating model order. Individual rows report actual repeat count.',
                'Shared CPU host; medians and dispersion are observations, not hardware guarantees.',
                'First-call uses existing Numba disk cache and a process potentially warmed by earlier cases. Circuit profile cache is cleared once before each model pair, then retained for both models and all warm repeats.',
                'A fresh stochastic table-cache directory is created per benchmark invocation and retained between cases/repeats.',
                'Detailed and Simple use equal external stimuli and solver tolerances but have different equations, waveforms and accepted-step counts.',
                'Synthetic HRS fitting workload tests runtime, not independent model accuracy.',
                'Half-duration CSVM rows are held out of runtime reference fitting to inspect interpolation uncertainty.'
                ],cases=cases)
    groups=[]
    for case in cases:
        if case['comparison_id'] not in groups:groups.append(case['comparison_id'])
    raw={}
    for group in groups:
        paired=[c for c in cases if c['comparison_id']==group and c['supported']]
        from server.compute.circuit import custom
        custom._PROFILE_CACHE.clear()
        for case in paired:
            try:
                first=run_case(case)
                raw[case['id']]=dict(first=first,warm=[])
                print(json.dumps(dict(case=case['id'],phase='first',wall_s=first['wall_s'])),flush=True)
            except Exception as exc:
                case['measurement_error']=f'{type(exc).__name__}: {exc}'
                case['measured']=False
                print(json.dumps(dict(case=case['id'],error=case['measurement_error'])),flush=True)
        paired=[c for c in paired if c['id'] in raw]
        repeats=min(args.repeats,2) if group.endswith('holdout') else args.repeats
        if paired:repeats=min([repeats]+[c.get('repeats',args.repeats) for c in paired])
        if paired and max(raw[c['id']]['first']['wall_s'] for c in paired)>10:repeats=min(repeats,3)
        for repeat in range(repeats):
            for case in paired if repeat%2==0 else list(reversed(paired)):
                raw[case['id']]['warm'].append(run_case(case))
        for case in paired:
            import numpy as np
            rows=raw[case['id']]['warm'];times=[r['wall_s'] for r in rows]
            case['measured']=True
            case['timings']=dict(median_s=statistics.median(times),min_s=min(times),max_s=max(times),
                                 p25_s=float(np.quantile(times,.25)),p75_s=float(np.quantile(times,.75)),
                                 repeats=len(times),samples_s=times,first_call_s=raw[case['id']]['first']['wall_s'],
                                 median_cpu_s=statistics.median(r['cpu_s'] for r in rows))
            case['execution_summary']=rows[len(rows)//2]
            save(args.out/(case['id']+'.timings.json'),raw[case['id']])
            print(json.dumps(dict(case=case['id'],phase='warm',median_s=case['timings']['median_s'],n=len(times))),flush=True)
        save(args.catalog,result);save(args.out/'measurements.json',result)
    print(json.dumps(dict(catalog=str(args.catalog),measured=sum(c.get('measured',False) for c in cases),total=len(cases))),flush=True)


if __name__=='__main__':main()
