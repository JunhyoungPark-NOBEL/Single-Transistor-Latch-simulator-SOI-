"""Explicit short calibration on the same worker and production kernels as jobs."""
from __future__ import annotations

import statistics
import time

from server.payloads import normalize
from server.performance_cases import calibration_cases


def run_performance_calibrate(payload: dict, progress) -> dict:
    from server.compute import resolve
    cases = calibration_cases()
    samples = []
    for index, case in enumerate(cases):
        normal, _ = normalize(case["kind"], case["payload"])
        function = resolve(case["kind"])
        timings = []
        for repeat in range(3):
            progress((index + repeat / 3) / len(cases),
                     f'{case["model"]} · {case["family"]} · {repeat + 1}/3')
            start = time.perf_counter()
            result = function(normal, lambda fraction=0, message="":
                              progress((index + (repeat + float(fraction)) / 3) / len(cases),
                                       f'{case["model"]} · {case["family"]}'))
            elapsed = time.perf_counter() - start
            if any(row.get("key") == "truncated_runs" and float(row.get("value") or 0) > 0
                   for row in result.get("summary", []) if isinstance(row, dict)):
                raise ValueError("Calibration workload stopped before its requested final time")
            timings.append(elapsed)
        warm = statistics.median(timings[1:])
        samples.append(dict(case_id=case["id"], kind=case["kind"], model=case["model"],
                            family=case["family"], features=case["features"],
                            first_s=timings[0], warm_s=warm, warm_samples_s=timings[1:],
                            observed_setup_s=max(0.0, timings[0] - warm)))
    progress(1, "calibration complete")
    return dict(samples=samples, measured_at=time.time(),
                warnings=["동일한 계산 worker에서 재측정했습니다. 최초 준비 차이는 컴파일·캐시 읽기·실행 변동을 포함합니다.",
                          "측정한 모델·해석 종류에만 서버 보정을 적용합니다. 다른 종류는 기준 환경 추정입니다."])
