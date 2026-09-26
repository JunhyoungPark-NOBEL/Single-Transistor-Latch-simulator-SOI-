# STL simulator 물리 모델 감사

2026-09-25 · 소스 코드와 제공된 보정 자료 기준

## 결론

기본 Device 1은 제공받은 **300 K 기준 소자의 보정 모델**을 사용한다. 래치 전압을
원하는 모양으로 이동시키는 별도 식이나 S자 출력 곡선을 새로 넣지 않았다. 평형 바디
전류의 합이 0이 되는 해와 그 곡선의 fold에서 VLU·VLD를 계산한다.

그러나 코드 검토만으로 모든 내부 가정이 실제 소자에서 성립한다고 증명할 수는 없다.
원본도 2D Poisson·완전한 반도체 수송 해석이 아닌 축약 모델이다. 특히 아래 치수·백게이트
확장은 **데이터로 식별되지 않은 가정**이다. 새로운 치수에서의 실험적 정확도로
표시해서는 안 된다.

1. `tau_junction ∝ Tsi`: 원본의 유효 접합 재결합을 표면 성분 100%로 해석한다.
2. `Cbody = Cf + Cb − Cb0`: 원본 보정을 보존하려고 선택한 증분 전하식이다.
3. `Δψ = Cb/(Cf+Cb)·VBG`: v2에서 BJT source injection에 적용하는 미보정 바디 바이어스 근사다.

기준 모델을 기본값으로 유지하고, 치수 또는 VBG 변경은 미보정 외삽으로 표시한다.
새로운 양자 효과, 임의의 얇은 바디 knee 또는 별도 BJT β 배율은 추가하지
않았다. VBG의 주입 장벽 결합은 위에 명시한 가정이며 측정에서 새로 추출한 관계가 아니다. 공정 전반의 설계 인증 또는 TCAD 대체라는 주장은 지원하지 않는다.

## 증거와 기준값

| 소스 | 확인한 내용 |
|---|---|
| `engine/photo_extension/photo_mean.py` | 실제 활성 수송·재결합·GIDL·II 식, 가설적 확장 항 |
| `engine/model/janus_calibration_20260920/idvd_model/mean_model.py` | 치수 상수, 300 K 전계·BTBT 생성, 단위 |
| `engine/model/janus_calibration_20260920/claude_crosscheck_20260920/joint_model/refit_3.json` | 실제 Nbody와 기준 유효 파라미터 |
| `server/geometry_model.py` | 치수·VBG 외삽, 소자별 전계표 |
| `server/compute/deterministic.py` | 평형 가지, fold, 바디 전하 좌표 |
| `server/compute/circuit/element.py`, `mna.py` | 바디 전하 ODE와 회로 KCL, 역바이어스 밖 연장 |
| `server/compute/circuit/basic.py` | MOS·diode·BJT의 명시적 교육용 모델 |
| `server/compute/data.py` | 제공된 측정 배열과 기준 IDVD 읽기 |

기준값은 L=500 nm, W=200 nm, Tsi=50 nm, EOT=14.1 nm,
Nbody=2.295773162796593×10¹⁷ cm⁻³이다. Tbox=140 nm는 원본에 없던 **명시적 기준 가정**이다.
`tau_bulk=9.266283807625294e−7 s`, `tau_junction=5.358373401760838e−9 s`,
`beta=7.166501201841884`, GIDL 유효 길이는 28.754392875051614 nm이다.
이 값들을 다른 연구 메모의 전형값으로 교체하지 않았다.

## 원본 모델: 물리 항과 유효 가정의 구분

| 항목 | 실제 계산 | 적용 한계 |
|---|---|---|
| 바디 생성·소멸 | II + 접합 BTBT + GIDL + 광생성 − SRH − 확산 | 모든 항이 동시에 실험에서 독립 추출된 것은 아님 |
| 바디 수송 | 중성 베이스에서 전류를 운반하는 1D 수송과 비선형 SRH 풀이 | 횡방향 중성 베이스가 필요; 2D 전위 분포는 없음 |
| 접합 공핍폭 | `sqrt(2 εSi Vjunction/(q Nbody))` | abrupt-junction, 비축퇴 근사; 고주입에서의 정전기 재분포를 완전히 풀지 않음 |
| 충격 이온화 | 300 K local-field 계수의 전계 적분으로 M 생성 | 비국소 dead-space·가열 효과 검증 아님; 채널과 BJT에 같은 접합 M을 사용하는 근사 |
| GIDL | `q·volume·A·E^2.5 exp(−B/E)` | 유효 전계 길이와 gate-edge 부피를 사용; 실제 산화막/드레인 2D 형상 해석 아님 |
| 재결합 | 기준 유효 bulk·junction 수명 | bulk/계면 성분 분리가 안 됨; Auger는 없음 |
| access 저항 | 별도 slab의 전도도에서 R 계산 | access excess density를 평균 바디 excess와 같게 둔 미검증 closure |
| 채널 전류 | 낮은 VD의 IDVG에 보정한 매끄러운 채널 식 | 고전계·임의 도핑·온도 범위의 채널 모델 아님 |
| 국소 상태 | φG·φE, OU 상태와 선택적 seed/avalanche 항 | 결함 위치·새 미시적 경로의 실험적 식별과 동일하지 않음 |

`u`는 내부 quasi-Fermi 분리/주입 좌표다. 이를 별도로 해결된 정전기 바디 전위라고
부르면 안 된다. 정전기 근사에는 `ψ = u − VT ln(1+δ/Nbody)`를 사용한다.
바디 접촉 단자 B는 u에 연결하고, 표시하는 정전기 전위 ψB와 구분한다. VBG가 있는 v2에서는 아래의 유효 injection density를 사용한다.

## 치수별 감사

| 입력 | 단위 및 반영 경로 | 확인된 한계 |
|---|---|---|
| L | nm→cm, 중성 베이스 길이와 수송; 채널 W/L | 영바이어스 공핍폭 합보다 짧은 경우 명시적으로 오류 처리 |
| W | 접합 면적·GIDL 부피·채널·저장 전하에 선형; 저항은 역비례 | dark에서 threshold 불변은 이 축약 모델의 불변성; 좁은 폭/edge 효과 없음 |
| Tsi | 접합 면적과 저장 부피; 아래 표면 우세 수명; Csi | 5 nm까지 입력 가능하다는 사실은 양자/계면 정확도 보장 아님 |
| EOT | `Cf=εox LW/EOT`; 채널 계수; GIDL 길이의 기준 비율 | EOT가 같아도 high-k 실제 두께·장벽·edge 전계가 같다는 보장은 없음 |
| Tbox | Csi·Cbox 직렬 용량, VBG에 의한 유효 BJT source injection·바디 전위 | 결합은 미보정 가정; 독립 back-channel/2D Poisson·collector field correction 없음 |
| Nbody | 중성 조건, Vbi, 공핍폭, SRH·주입과 전계표 재계산 | 이동도·수명·밴드갭 축소·축퇴통계·채널 Vth 도핑 의존성은 재추출하지 않음 |

cm 기반 전류식에서 `εSi=11.7 ε0/100`은 F/cm, 면적은 cm², 농도는 cm⁻³,
이동도는 cm²/(V·s), 확산계수는 cm²/s로 일관된다. 게이트 정전용량은 SI 길이와 F/m로
계산한다. 조사한 활성 경로에서 추가적인 nm/cm/m 변환 오류는 발견하지 못했다.

사용자가 입력한 Iph는 **절대 A**이며 W 변경 시 자동 확대하지 않는다. 따라서 고정
Iph 또는 고정 외부 Iin·Cdrain에서 폭을 바꾸면 모든 응답이 불변일 필요가 없다.
기준 폭에서 추출한 채널 seed 및 local-path saturation 전류만 W에 비례한다.
`t_access`, `L_access`, `NA_access`는 독립 유효 slab 값으로 남긴다.

### Tsi와 표면 재결합: 어느 부분이 가정인가

두 표면과 부피 내 excess가 거의 균일하다고 두면 다음 관계를 유도할 수 있다.

```text
1/τeff ≈ 1/τbulk + (Stop + Sbottom)/Tsi
```

현재 구현은 이 전체 관계를 독립적으로 추출하지 않았다. 원본의 접합 SRH 수명에
`τj,eff = τj,ref·Tsi/Tsi,ref`를 적용한다. 이에 따라

```text
Ij = q·W·Tsi·ws·ni/(2τj,eff) · expm1(u/(2VT))
   = q·W·ws·ni·Tsi,ref/(2τj,ref) · expm1(u/(2VT))
```

가 되어 이 항의 유효 부피 의존성을 접합 부근 표면적 의존성으로 바꾼다. 표면 재결합
자체는 실제 물리 현상이지만 **이 소자의 유효 τj를 100% 표면 기여로 해석하는 선택은
측정에서 확인되지 않았다**. 등가 `Stop+Sbottom≈933.119 cm/s`는 이 선택의 역산값이고
추출된 계면 속도가 아니다. 원본 `tau_bulk`에도 계면 기여가 포함됐을 수 있어 성분의
물리적 분리 역시 유일하지 않다. 중성 바디 전체의 계면 SRH를 따로 푼 것이 아니다.

Tsi 감소 시 VLU·VLD 상승과 작은 Tsi에서의 기울기 증가가 코드 테스트에서 나와도,
이는 이 가정의 수치 결과다. 다중 두께 데이터와의 검증을 대신하지 않는다. 이를
양자 구속 또는 새로운 얇은 층 전이의 증거로 해석하지 않는다.

### Tbox와 백게이트: 유효 BJT 바디 바이어스와 보정 보상항

v2는 이전의 channel-only VBG 항을 제거하고, 사용자가 지정한 BJT 바디 바이어스 가정을 명시적으로 구현한다.

```text
Cf = εox LW/EOT; Csi = εSi LW/Tsi; Cbox = εox LW/Tbox
Cb = (1/Csi + 1/Cbox)^−1
αBG = Cb/(Cf+Cb); Δψ = αBG VBG
P = ni² exp(Δψ/VT) expm1(u/VT)
δinj = 2P/(Nbody + sqrt(Nbody²+4P))
ψB = u − VT ln(1+δinj/Nbody) + Δψ
```

정전용량 divider는 무차원이며 source barrier를 조절하는 지수 인자는 실제 gate-controlled injection의 축약 가정이다. **계수의 크기와 이 소자에 대한 적용성은 VBG 데이터로 보정되지 않았다.** δinj를 source minority boundary에 사용하되 hole diffusion과 junction-SRH의 전압 인자는 실제 u를 유지한다. source depletion은 vbi−ψB를 사용한다. u=0에서 excess injection은 VBG와 무관하게 0이고, VBG=0에서 기존 식을 유지한다.

전면 GIDL·채널은 실제 VG, u+r를 사용한다. 기존 r 기반 collector field 근사는 보존하며 독립적인 back-channel 또는 2D Poisson 해로 주장하지 않는다. 이 제약 때문에 전체 전위 분포/고주파 terminal-charge 정확도를 보장하지 않는다. source barrier가 없어지는 범위는 오류 처리하며 인위적으로 clamp하지 않는다.

```text
Qcap = (Cf+Cb−Cb0) ψB − Cf VG − Cb VBG
Cb0 = εox LW/(140 nm + 50 nm/3)
```

Cb0는 기존 보정 전하를 유지하는 counterterm이며 음의 물리 커패시터가 아니다. 동일한 ψB와 수송 carrier charge를 한 번만 사용한다. ψB 고정의 미분은 −Cb이고, 실제 contact u 고정의 미분은 `(Cf+Cb−Cb0)∂ψB/∂VBG−Cb`다. 두 편미분을 구분하며 analytic/finite-difference 일치를 검사한다. 상세 유도와 한계는 `GEOMETRY_MODEL_KO.md`에 기록한다.

## 회로 모델과 발진 지표

STL 회로는 `dQ/dt=G−L`과 외부 노드의 KCL을 함께 풀며, CSVM은 전류원과 Cdrain을
실제로 연결한다. `oscillator.py`의 fold 사이 준정적 walk는 작업량/시간 스텝을 위한
추정이다. 그것을 실계산의 transient 또는 측정 발진 주파수로 제시하면 안 된다.
Vtop·Vbottom·주파수는 실제 시간 파형의 충분히 해상된 정상 주기에서만 읽어야 한다.

회로용 STL은 원본 평형 도메인을 벗어나는 `u<0`, `r<0`에 diode-law 연장을 갖는다.
이는 C¹ 연결/대칭 접합 가정을 사용한 제한적 연장이며 해당 역방향 영역의 측정 보정은
없다. 급격한 gate/source 파형이나 역방향 구동에 대해 SPICE PDK 수준의 검증을 주장하지
않는다. BG를 연결하면 VBG는 회로의 시간 가변 소스 기준 전압이고, B를 연결하면 hole quasi-Fermi contact 전압 u에 외부 RC가 연결된다. 정전기 ψB는 별도로 표시한다.

| 기본 소자 | 구현 | 미포함 |
|---|---|---|
| MOS | 대칭 softplus square-law, W/L, Vth, SS, λ | 바디 다이오드, 기생 전하, 공정 PDK, 고속/온도 모델 |
| Diode | Shockley `Is·expm1(V/(nVT))` | 역항복, 접합 전하, 직렬저항/고주입 추출 |
| BJT | reciprocal Ebers–Moll, βF·βR | Early 효과, transit charge, high-injection/온도 모델 |

모든 기본 소자는 300 K 교육용 compact model이다. MOS SS 하한은 60 mV/dec로
제한했다. 이 식에는 subthermal 동작을 설명하는 메커니즘이 없으므로 그보다 작은 값을
입력해 일반 상온 MOSFET처럼 계산하지 않는다. 별도 외부 C를 추가할 수 있다. 기본 소자의 terminal-current
합과 analytic Jacobian의 구조는 KCL에 일관된다.

수치적 장치도 물리 효과와 구분한다. `mna.py`의 GMIN=10⁻¹⁸ S는 수렴 정규화용이고,
diode/BJT 지수 인자가 40을 넘을 때 선형으로 이어지는 것은 overflow 방지다. 이 영역의
전류를 물리적 high-injection 예측으로 사용하면 안 된다.

## 발견한 오류와 조치

| 발견 사항 | 조치 |
|---|---|
| 기준 치수라는 이유만으로 임의 VG·광·수명에도 `validated=true` | `server/params.py`에서 기준 보정 구성에만 true. 그 경우도 독립 예측 검증이 아닌 보정 비교임을 payload에 명시 |
| `loc_carriers=2`가 원본 `bulk=(p[24]>.5)` 분기에 먼저 걸림 | 원본은 보존. `aloc>0 && loc_carriers=2` 요청을 명시적으로 거부해 잘못된 경로 계산 방지 |
| 일반 300 K MOS에 SS=10 mV/dec 등 미설명 subthermal 입력 허용 | API·UI의 SS 하한을 60 mV/dec로 제한 |
| geometry·VBG의 stochastic kernel은 미보정 | 기준 밖 stochastic 요청의 기존 거부 정책 유지 |
| demo/mock 및 근접 snapshot은 물리 계산의 증거가 아님 | 연구용 모드에서 실계산 결과와 혼합 금지; 서버 연결 실패를 합성 파형으로 대체하지 않음 |

기준 `aloc=0`에는 발견한 carrier-mode 분기 오류의 영향이 없다. 활성 0/1도 가설적
경로이므로 특정 미시적 avalanche mechanism이 확인됐다는 의미는 아니다.

`server/jobs.py`의 엔진 버전 hash에는 `geometry_model.py`가 들어가며, 회로 profile
cache는 전체 p 벡터를 key로 사용한다. 다른 치수의 기준 결과를 재사용하는 cache-key
누락은 현재 감사 시점에 확인되지 않았다.

## 데이터와 검증 표현

제공된 기준 IDVD는 VG=−2 V, dark, 약 0.4 V/s, 별도 100개 up·100개 down 기록이다.
이들을 100개의 짝지어진 cycle로 바꿔 부르면 안 된다. 광 데이터는 별도 조건이며,
일부 IDVD의 VG는 변환 피팅에서 가정된 값이다. 같은 데이터를 사용한 파라미터 보정과
오차 표시는 유용하지만 독립 테스트셋 정확도와 구분해야 한다.

이 감사에서는 `server/tests/test_geometry_model.py`의 26개 검사가 통과했다. 기준
엔진과의 정확한 가지 일치, L/Tsi 방향성, W 전류 스케일, 단위 관련 불변량, VBG·전하
계수, 도메인 거부, 메타데이터와 미지원 경로 차단을 확인한다. 테스트 통과는 구현의
내적 일관성이지 새 치수의 실험적 검증이 아니다. 원본 `engine/`는 수정하지 않았다.

LTspice·Verilog-A 내보내기는 현재 선택한 고정 치수·바이어스에서의 **준정적 IDVD
behavioral model**이다. 전체 바디 전하 ODE, CSVM 주파수, 가변 gate/geometry/temperature
모델을 옮긴 것이 아니다. Sentaurus 구조/mesh/물리 deck을 임의 생성하지 않는다.
상용 simulator 실행 검증 전에는 호환성 시험 완료로 표시하지 않는다.

## 외부 근거

- UC Berkeley, [BSIMSOI v4.4 Users’ Manual](https://ngspice.sourceforge.io/external-documents/models/BSIMSOIv4.4_UsersManual.pdf), §3.3 GIDL, §4 terminal charge, §6.1 front/back coupling. 본 앱은 BSIMSOI 구현이 아니며, 이 문헌이 앱의 보상항이나 계수를 검증하지 않는다.
- UC Berkeley, [BSIM-IMG 모델 소개](https://bsim.berkeley.edu/models/bsimimg/). 독립 전·후면 gate에는 전압·일함수·유전체 정보가 필요함을 확인하는 자료다.
- J. Brody 등의 [effective/bulk lifetime와 표면 재결합 관계 검토](https://www.sciencedirect.com/science/article/pii/S0927024802003501), 2003. 이번 감사에서는 공개 초록의 근사식 유효성 범위 설명을 확인했으며 유료 본문을 확보하지 않았다. 이 소자의 표면 기여율을 제공하는 자료가 아니다.

표면/부피 비, 직렬 정전용량, 평형 전류 보존이라는 물리적 근거와 이 소자에 대한
파라미터 식별은 서로 다른 문제다. 후자는 새 치수·백게이트·표면 처리 조건의 측정
또는 TCAD 자료가 추가될 때 갱신해야 한다.
