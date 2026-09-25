# Update verification — 2026-09-25

## Latest: geometry, surface recombination and back-gate coupling

- Frontend: **33 files / 514 tests passed**. TypeScript and production build passed.
- Backend: **163 passed, 7 deselected** across geometry, geometry-noise guards, deterministic and circuit regression suites. HTTP-worker tests and slow tests remain excluded for the existing environment limitation below.
- **14 targeted browser tests passed** (`geometry.spec.ts` + `device-csvm.spec.ts`). The six geometry controls are topmost. Save/load/import, circuit snapshots, fixed-geometry LTspice/Verilog-A provenance, legacy reference migration and refusal of offline resized results are covered. Desktop 1440px and mobile 390px previews show actual resized-model curves after Plotly trace readiness; the Geometry guide opens the responsive model document.
- Exact original reference folds are preserved: VLU=3.703688985 V / VLD=2.597866939 V. Across L=200/300/400/500/700nm, both folds increase with L. Across Tsi=5/10/20/30/50/70nm, both decrease with Tsi; no explicit fold correction is applied. Shorter lengths can lose the latch or the assumed lateral neutral base, so this is not a global monotonic guarantee.
- W doubled at dark bias gives the same branch voltages and twice the currents/charges. Explicit Iph and external Iin/Cdrain are not silently scaled.
- Final reference-anchored body capacitance at Tbox=70/140/280nm: 2.627056 / 2.449031 / 2.345016 ×10^-16 F. At VBG=1V, VG=-0.8V, the channel coupling contribution changes with Tbox; measured channel currents at u=.6V,r=2.7V are 6.0597/1.3775/0.56482nA. VBG=0 yields no invented DC fold shift from Tbox.
- Actual geometry CSVM run: L400nm, VG=-2V, VBG=0, Iin=1nA, Cdrain=1pF, 8ms completed with 5976 accepted steps / 3062 saved samples, four LU and four LD events. LU spacing gives about 899.230Hz; event voltages are about 3.6510/2.54527V. All sampled terminal currents and body charge are finite. Saved-output KCL residual is 30pA (about 2.47ppm of peak current), limited by seven-significant-digit output rounding; this is not the unrounded Newton residual.
- The surface-dominated interpretation of the existing effective junction lifetime is an explicit assumption, not an extracted surface recombination fraction. Tbox140nm is nominal. No arbitrary quantum knee, BJT barrier shift, or fold-voltage offset was introduced. Domain errors and nonreference noise requests are rejected explicitly.
- These are dimensional/numerical consistency checks, **not validation against multi-geometry measurements or TCAD**. See `./GEOMETRY_MODEL_KO.md`, `web/e2e/fixtures/geometry/numerical-report.json`, and `csvm-check.json` (the latter two relative to the repository root).
- All **66 original engine files** match the input source ZIP byte for byte. Numerical extensions live in server-owned code. No deployment was performed.

Final server command:

```bash
python3 -m pytest server/tests/test_geometry_model.py server/tests/test_geometry_stochastic.py server/tests/test_circuit_geometry.py server/tests/test_deterministic.py server/tests/test_circuit.py server/tests/test_circuit_custom.py server/tests/test_circuit_basic.py -q -m 'not slow' -k 'not api_custom_circuit'
```

## Follow-up: supplied symbol, CSVM, export selector

- Final frontend suite: **31 files / 494 tests passed**; TypeScript and production build passed.
- **19 targeted browser tests passed**: 9 workspace/library/export cases, 5 migrated
  device/statistics/schematic cases, and 5 new CSVM cases. Desktop/mobile previews were
  visually inspected. Modal opacity and stacking were checked after entry animations.
- The CSVM frontend request matches the request accepted by the actual numerical solver.
  Tests use authentic direct-solver recordings through the HTTP job contract because the
  worker-pool limitation below still applies. Production does not use these fixtures.
- Device 1 at VG = −2 V, Iin = 1 nA, Cdrain = 1 pF, 15 ms produced
  Vtop ≈ 3.7422 V, Vbottom ≈ 2.5857 V, f ≈ 860.025 Hz after startup exclusion.
  Iin = 2 nA gave ≈ 1685.994 Hz; Cdrain = 2 pF gave ≈ 437.152 Hz.
  A 4 ms run and the tested nonoscillating bias do not display a fabricated frequency.
  Insufficient waveform resolution also suppresses unsupported metrics.
- Verilog-A source interpolation is parsed independently and compared with actual engine
  currents. The selector, offline protection, canceled generation and input validation are
  tested. Commercial simulator executables were not run. Sentaurus export is explicitly
  unavailable; see `./SIMULATOR_EXPORTS_KO.md`.
- All **66 original engine files** still match the supplied source archive byte for byte.

## Passed

The following table records the initial redesign verification; the frontend suite above
supersedes its earlier test count. Numerical engine/backend code was unchanged by the follow-up.

| Check | Result |
|---|---|
| `npm run typecheck` | Pass |
| `npm test` | 28 files, 464 tests passed |
| `npm run build` | Pass (the delivery included the compiled frontend; the repository does not commit `web/dist`) |
| `python3 engine/stl_api.py` | Reference folds 3.7037/2.5979 V; FPT 3.6442 V / 8.03 mV; dynamic MC 3.6344 V / 119.0 mV |
| Relevant server numerical tests | 127 passed, 7 deselected |
| `ux-device.spec.ts` + `reference-sweep.spec.ts` | 12 passed |
| `ux-shell.spec.ts` + `ux-guide.spec.ts` | 14 passed |
| `workspace-redesign.spec.ts` | 6 passed |
| Original `engine/` comparison | All original files byte-identical |

Server test command:

```bash
python3 -m pytest server/tests/test_deterministic.py server/tests/test_circuit.py server/tests/test_circuit_custom.py server/tests/test_circuit_basic.py -q -m 'not slow' -k 'not api_custom_circuit'
```

Front-end-generated example netlists were evaluated by the actual backend numerical code:

- CMOS inverter: output approximately 1.08 nV–1.8 V, low output for high input.
- Diode clipper: output −1.5–0.65057 V.
- BJT switch: output 0.09923–3.3 V.
- Output-node KCL residuals ≤1e−11 A for these checks.

LTspice export tests parse generated branch tables independently and compare roughly 4,000
real-engine voltage-sweep points; relative errors <1e−9 away from switching discontinuities.
This checks table fidelity, metadata, threshold mapping and state selection, not LTspice execution.

## Important limits

- This environment did not run the production HTTP worker pool: SyncManager's Unix socket failed
  with `PermissionError: Operation not permitted`. Direct numerical calculations and browser
  tests succeeded. Tests of the deployed HTTP pool, full API suite and slow tests were not run.
- LTspice and commercial Verilog-A executable integration were not run. Export is a fixed-bias quasi-static model and
  does not preserve physical body-charge dynamics or stochastic behavior.
- New generic MOS/diode/BJT models are educational 300 K memoryless models; see
  `BASIC_CIRCUIT_DEVICES.md` for unsupported effects.
- The full historical end-to-end suite was not claimed as passing. Updated suites above cover
  the changed interfaces. Five preset/library tests were migrated in the follow-up; some
  remaining legacy assertions still target removed bench UI.
- No deployment or git push was performed. Cloudflare prevented inspection of the existing
  Claude artifact in the cloud browser. Full source and original handoff were available locally.
- Locked-artifact recording/rebuild was not run. Existing snapshots need regeneration for the
  new example circuits and reference workflow. The ordinary `web/dist` build (not committed) is for the live server.

## Reference benchmark interpretation

The fixed reference uses authentic measured forward/reverse median currents and actual
calibrated engine sweeps at VG=−2 V, dark, 0–4 V. The two sets of measured sweeps are not
paired cycles. The current comparison includes the measurement floor: full-range log RMSE
is about 5.49 decades because the model predicts much smaller low currents, while linear
range-normalized RMSE is approximately 5.083% up and 3.451% down. These metrics are
reported as errors, not an artificial overall accuracy percentage or a self-test pass rate.
The expandable method explains the formulas and valid points.

## Preview images

The preview screenshots delivered with this update (`PREVIEW/`) are not kept in the repository.

## 2026-09-25 변경 요약

전달본의 `START_HERE_KO.md`에 있던 화면 변경 요약입니다. 실행·설치 방법은 `RUNNING.md`, `LOCAL_INSTALL.md`, `DEPLOY_LAB.md`를 따릅니다.

- **Geometry:** 파라미터 맨 위에서 L·W·Tsi·Tox(EOT)·Tbox·Nbody를 바꿉니다. VG 아래 VBG로 백게이트 바이어스를 설정합니다. 저장 소자·회로·내보내기에 함께 보존합니다. 기준 치수·VBG=0은 기존 결과를 유지하고, 변경 조건은 결정론 VSCM·CSVM으로 계산합니다. 표면 재결합과 백게이트 결합의 구체적인 가정·수치 검사·한계는 `GEOMETRY_MODEL_KO.md`와 Geometry ⓘ의 상세 문서에 있습니다. Tbox 140 nm는 원본에서 추출한 값이 아닌 명시적 기준 가정입니다.
- **소자:** VSCM(전압 구동)과 CSVM(전류 구동 + Cdrain)을 화면 위에서 전환합니다. VSCM은 ID–VD 상향·하향 스윕, CSVM은 VD(t)와 Vtop·Vbottom·주파수를 보여줍니다. CSVM의 Iin·Cdrain·관측 시간을 직접 바꿀 수 있습니다. 초기 구간을 제외한 실제 주기에서 지표를 계산하며, 주기가 부족하면 주파수를 표시하지 않습니다. 상세 설명은 ⓘ에 접었습니다.
- **기본 라이브러리:** `Device 1` 하나만 표시합니다. 이전 보정 프리셋의 수치 데이터와 저장된 사용자 소자·회로는 유지합니다.
- **사용자 소자:** 결과 오른쪽에 5개 슬롯이 있습니다. 현재 보정값을 저장하고, 불러오거나 같은 이름으로 업데이트합니다. 소자 관리에서 이름 변경·삭제·JSON 가져오기/내보내기가 가능합니다. 브라우저 로컬 저장이며 이전에 5개보다 많이 저장된 데이터는 삭제하지 않습니다.
- **회로:** 자유 배치·배선·파라미터 편집이 기본입니다. 예제는 보조 메뉴입니다. 빠른 벤치 화면은 제거했습니다. 새 MOSFET/다이오드/BJT도 회로의 비선형 MNA 해석에 참여합니다.
- **기본 소자:** NMOS/PMOS의 L·W·Vth·SS, 다이오드 Is·n, NPN/PNP의 Is·βF·βR을 설정합니다. 세부 식과 적용 범위는 `BASIC_CIRCUIT_DEVICES.md`에 있습니다. BSIM/Gummel–Poon 같은 제조사 모델은 아니며 기본 동작용 모델입니다.
- **상용 시뮬레이터로 내보내기:** LTspice(`.cir`·`.lib`)와 Verilog-A(`.va`) 중 선택합니다. 고정 게이트·광조건의 ID–VD 및 래치 문턱을 재현하는 준정적 모델이며, CSVM의 동적 Vtop·Vbottom·주파수와 확률 잡음은 포함하지 않습니다. 실제 서버 계산에서만 내보냅니다. 상용 프로그램에서의 실행 검증은 수행하지 않았습니다. Sentaurus TCAD는 구조·메시·도핑·접촉 정보와 별도 연동 검증이 필요하므로, 선택창에서 이유를 안내하고 다운로드를 비활성화했습니다.
- **레퍼런스:** 실제 측정 ID–VD 중앙값과 기준 보정 모델을 겹쳐 보여 줍니다. 상향/하향의 오차 지표와 계산 방법을 확인할 수 있습니다. 항상 고정 기준 조건이며 현재 사용자 보정값과 혼동하지 않도록 표시합니다. `scripts/build_reference.py`로 실제 소스 데이터에서 재생성할 수 있습니다.
- **표시 오류:** 로그축 확대 범위가 Linear 전환 후 남는 문제를 수정했습니다.
- **브랜딩:** 사용자가 제공한 biristor 심볼(원 안의 기울어진 평행사변형과 좌우 단자선)을 로고·아이콘에 적용했습니다. 회로 심볼은 같은 내부 형상을 쓰며 실제 SOI 모델의 D/G/S 연결을 유지합니다. 제작자 정보는 KAIST · NOBEL 연구실만 접어서 보여줍니다.

검증 화면·회로 예제·기본 UI가 바뀌었으므로 기존 정적 스냅샷을 다시 배포할 때는 새로 녹화해야 합니다. Geometry·VBG를 바꾼 조건은 해당 조건의 실제 계산 기록이 없으면 오프라인 데모 결과로 대체하지 않습니다.
