# Geometry 모델 — 차원 감사와 적용 범위

2026-09-25 · `fdsoi-bjt-body-bias-v2`

기존 보정 소자를 기준으로 차원 관계를 확장했습니다. 새로운 치수의 측정·TCAD 데이터에
맞춘 모델은 아닙니다. 원본 `engine/`는 보존하고 `server/geometry_model.py`와 회로 연결부에
확장을 구현했습니다. 기준 치수·VBG=0은 기존 수치 커널을 그대로 사용합니다.

## 기준점과 입력

Geometry는 왼쪽 파라미터의 첫 항목입니다. Tox는 산화막의 물리 두께가 아닌 **EOT**로 입력합니다.
VBG는 VG 바로 아래에서 조절합니다. 소자 모드에서는 소스 기준 바이어스이고, 회로에서는 BG 단자를 전압원에 연결해 시간에 따라 바꿀 수 있습니다.

| 입력 | 기준 | 입력 범위 | 출처 |
|---|---:|---:|---|
| L | 500 nm | 100–2,000 nm | 원본 구조 |
| W | 200 nm | 20–10,000 nm | 원본 구조 |
| Tsi | 50 nm | 5–200 nm | 원본 구조 |
| Tox / EOT | 14.1 nm | 1–100 nm | 원본 보정 |
| Tbox | 140 nm | 10–1,000 nm | **원본에 없어 설정한 명시적 기준 가정** |
| Nbody | 2.295773162796593×10¹⁷ cm⁻³ | 10¹⁵–10¹⁹ cm⁻³ | 원본 보정 JSON의 실제 값 |
| VBG | 0 V | −10–10 V | 기준 바이어스 |

범위는 입력 허용 범위이며, 전체 조합의 모델 유효성을 보장하지 않습니다. 특히 이 수송
모델은 횡방향 중성 베이스를 가정합니다. `L > 2 wd0 + 1 nm`를 만족하지 않거나 도핑에 대한
avalanche 전계 계산이 성립하지 않으면 범위 밖 오류를 표시합니다. 여기서
`wd0 = sqrt(2 εSi vbi / (q Nbody))`, `vbi = VT ln(Nsource Nbody / ni²)`입니다.
횡방향 중성 베이스와 수직 방향 fully-depleted 근사는 별개이며, 이 확장은 FD 상태를 자동
판정하는 2D Poisson 해석기가 아닙니다.

## 기존 식에서 빠져 있던 차원 관계

| 항목 | 연결한 관계 | 유지한 보정값·한계 |
|---|---|---|
| L | 접합 공핍폭을 뺀 수송 길이, 확산·재결합, 영바이어스 베이스 길이, 채널 W/L | 임의의 VLU·VLD 이동식 없음 |
| W | 접합 면적·GIDL 생성 부피·채널·전하 ∝ W, 접촉·access 저항 ∝ 1/W | 어두운 조건에서 래치 전압 불변; 가장자리·좁은 폭 효과는 미보정 |
| Tsi | 접합 면적 W·Tsi, 저장 부피, 접합 표면 SRH 수명, Si/BOX 직렬 정전용량 | bulk 유효 수명은 유지; 얇은 바디의 양자 보정·임의 knee 없음 |
| EOT | Cf=εox LW/EOT, 채널 계수 ∝1/EOT, GIDL 유효 전계 길이 ∝EOT | 기존 GIDL 추출 길이를 기준으로 비율 적용 |
| Tbox | Si/BOX 직렬 정전용량, BJT 유효 바디 바이어스, 증분 저장 전하 | 미보정 electrostatic divider; β를 별도로 변경하지 않음 |
| Nbody | 중성 조건·내장전위·공핍폭·SRH/주입·avalanche/BTBT 전계표 | 채널 Vth·이동도·수명의 도핑 의존성은 별도 추출하지 않음 |

`t_access`, `L_access`, `NA_access`는 독립적으로 보정한 access slab입니다. L 또는 Tsi와
자동으로 동일시하지 않습니다. W에 대한 저항 스케일은 적용합니다. `beta`는 영바이어스
기준 확산비로 유지하고, 실제 주입 식에는 변경한 베이스 길이를 사용합니다.

사용자가 지정하는 광전류 `Iph`는 **절대 전류**입니다. W를 바꿔도 자동으로 늘리지 않습니다.
반면 기준 폭의 전류로 추출된 채널 seed와 local saturation 전류에는 폭 비율을 적용합니다.
따라서 고정 Iph 또는 고정 외부 Iin·Cdrain에서 W를 바꿀 때까지 모든 전압·주파수가
불변이라는 뜻은 아닙니다.

## Tsi: 접합 표면 SRH 우세 가정

얇은 층에서 표면 재결합의 유효 손실률은 표면/부피 비에 의존합니다. 이번 구현은 원본의
유효 접합 SRH 항에 표면 기여가 우세하다는 가정을 적용합니다.

```text
τj,eff = τj,ref · Tsi / Tsi,ref
Ij = q · W · Tsi · ws · ni / (2 τj,eff) · expm1(u / 2VT)
   = q · W · ws · ni · (Tsi,ref / τj,ref) / 2 · expm1(u / 2VT)
```

즉 이 손실 항은 접합 부근의 표면적 W·ws에 비례합니다. 새로운 재결합 전류를 더하지 않고
기존 항의 두께 관계를 바꿉니다. 기준 보정에서 유도되는 `Stop+Sbottom ≈ 933 cm/s`는
**이 가정의 등가값**이며 측정으로 추출한 표면 재결합 속도가 아닙니다.

원본 τj는 유효 접합 수명으로, 표면/벌크 성분이 독립적으로 식별되지 않았습니다.
여기서 표면 기여를 100%로 보는 선택과 기존 `tau_bulk`의 물리적 분해에는 불확실성이
남아 있습니다. 중성 바디 전체의 계면 재결합을 새로 푼 것도 아닙니다. 따라서 Tsi가 매우
얇을 때 변화의 **정량 크기**까지 검증했다고 해석하면 안 됩니다. 여러 Tsi에서의 측정과
표면 처리 조건이 있어야 이를 재추출할 수 있습니다.

## Tbox와 VBG: BJT 바디 바이어스 근사

사용자가 요청한 BJT 바디 바이어스 경로를 다음과 같이 명시적인 **미보정 compact-model 가정**으로 구현합니다. Si/BOX 직렬 정전용량과 전면 용량의 divider를 사용합니다. 각 C는 F, 모든 전위는 V, αBG는 무차원입니다.

```text
Cf = εox · L · W / EOT
Csi = εSi · L · W / Tsi
Cbox = εox · L · W / Tbox
Cb = (1/Csi + 1/Cbox)⁻¹
αBG = Cb / (Cf + Cb)
Δψ = αBG · VBG
P = ni² · exp(Δψ/VT) · expm1(u/VT)
δinj = 2P / (Nbody + sqrt(Nbody² + 4P))
ψB = u − VT ln(1 + δinj/Nbody) + Δψ
```

`u`는 소스에 대한 hole quasi-Fermi/contact 전압입니다. 실제 B 단자는 이 전압에 연결하고, `ψB`는 별도의 근사 정전기 바디 전위로 표시합니다. 둘을 같은 변수로 취급하지 않습니다. `P`는 cm⁻⁶, `δinj`는 cm⁻³이며 기존 BJT 수송 solver의 source minority-injection boundary에 사용합니다. 낮은 주입에서는 gate-controlled barrier의 지수 효과를, 높은 주입에서는 기존 제곱근 관계를 유지합니다. 실제 carrier quasi-Fermi 분리를 새로 풀었다는 뜻은 아닙니다.

`expm1(u/VT)`를 유지하므로 u=0에서는 VBG와 무관하게 excess injection이 0입니다. source depletion은 `vbi−ψB`를 사용하고, hole out-diffusion 및 junction-SRH의 전압 인자는 실제 u를 유지합니다. junction-SRH의 depletion-width 인자는 바뀔 수 있습니다. β를 별도로 곱하거나 기존 손실 식의 모든 u를 이동시키지 않습니다.

이전 버전의 채널 overdrive에 VBG를 직접 더하던 항은 제거했습니다. 전면 채널과 GIDL은 실제 VG와 기존 u+r를 사용합니다. collector 전계는 기존 r 기반 근사를 유지하며, 별도 후면 drain-field correction이나 독립 back-channel 전류를 새로 가정하지 않습니다. **이 방식은 전위·전하의 2D Poisson 해도, 임의 VBG에서 검증된 BJT 모델도 아닙니다.** back-gate sweep 데이터가 있어야 결합 크기와 고주입 유효성을 검증할 수 있습니다.

전하에는 동일한 ψB를 정확히 한 번 사용합니다.

```text
Cbody = Cf + Cb − Cb0
Qcap = Cbody ψB − Cf VG − Cb VBG
Cb0 = εox · L · W / (Tbox,ref + Tsi,ref/3)
Qtotal = Qcap + 동일 수송 계산의 excess 전하 + 기존 ionized-dopant 전하
```

`Cb0`는 현재 L·W 면적에서 기준 보정을 유지하는 보상항이며 물리적 음의 커패시터가 아닙니다. 기준 치수·VBG=0의 원본 계산은 그대로 유지합니다. ψB를 고정했을 때 `∂Qcap/∂VBG=−Cb`이고, contact u를 고정하면 `∂Qcap/∂VBG=Cbody·∂ψB/∂VBG−Cb`입니다. 이 두 미분을 혼동하거나 Δψ를 다시 더하지 않습니다. 전압·전하 편미분은 유한차분으로 점검합니다.

VBG=0이면 새 injection factor는 정확히 1입니다. Tbox만 바꿨다는 이유로 DC VLU·VLD를 강제로 이동시키지 않습니다. source barrier가 0 이하이거나 중성 베이스가 사라지는 조건은 잘라서 계산하지 않고 모델 범위 밖으로 처리합니다. 실제 영바이어스 Tbox 의존성에는 back-gate 일함수·flat-band·계면 전하 정보가 더 필요합니다.

## 실제 수치 검사

아래는 VG=−2 V, VBG=0, dark에서 현재 solver를 직접 호출한 결과입니다. 독립 측정과의
정확도 비교가 아닙니다. 나머지 파라미터는 기준값입니다.

| 변경 | VLU (V) | VLD (V) |
|---|---:|---:|
| 기준 L=500, Tsi=50 nm | 3.703689 | 2.597867 |
| L=400 nm | 3.609538 | 2.555622 |
| L=200 nm | 2.650380 | 2.149015 |
| Tsi=30 nm | 3.814662 | 2.635083 |
| Tsi=10 nm | 4.068936 | 2.733356 |
| Tsi=5 nm | 4.240943 | 2.818399 |

L=200–700 nm의 검사점에서 L 감소에 따른 두 전압의 단조 감소를 확인했습니다.
이보다 짧아 래치가 사라지거나 수송 가정이 무효가 되는 영역까지 이를 보장하지 않습니다.
Tsi=5–70 nm 검사점에서는 얇아질수록 두 전압이 증가하고 얇은 쪽의 기울기가 더 큽니다.
4 V를 넘는 VLU를 스윕에서 보려면 VD 최대값도 높여야 합니다.

W를 200→400 nm로 바꾸면 dark branch 전압은 같고 전류·전하가 2배가 됩니다.
새 VBG 검사는 같은 u/r/VG에서 BJT injection이 변하고, 전면 GIDL·채널 전류에 VBG를 직접 더하지 않는지 확인합니다. 기존 `web/e2e/fixtures/geometry/`의 VBG 채널 수치는 v1의 역사적 기록이며 v2의 근거로 사용하지 않습니다. VBG=0 결과는 유지됩니다.

L=400 nm, VG=−2 V, VBG=0, Iin=1 nA, Cdrain=1 pF의 실제 CSVM 과도 해석도
8 ms까지 완료했습니다. 래치업·래치다운이 각각 4회 발생했고 주기는 약 1.11206 ms,
주파수는 899.23 Hz였습니다. 동적 이벤트 전압은 약 3.65100/2.54527 V로 위 DC fold와
구분합니다. 실제 요청과 solver 요약은 `web/e2e/fixtures/geometry/csvm-check.json`에 있습니다.

## 저장·회로·내보내기

- 여섯 치수와 VBG는 저장 소자·회로의 소자별 snapshot·JSON에서 유지됩니다. 예전 파일은 기준 치수·VBG=0으로 복원합니다.
- 서로 다른 치수/도핑의 소자를 회로에 함께 배치할 수 있습니다. 각 소자는 자체 전계표를 사용하며 전역 도핑 상수를 바꾸지 않습니다.
- 회로는 D·S·G·BG·B의 5단자를 사용합니다. 연결하지 않은 BG는 저장된 고정 VBG, 연결하지 않은 B는 floating body입니다. B에 외부 R·C를 연결하면 실제 body contact 전류가 동역학에 들어갑니다.
- Geometry/VBG 변경은 **결정론 VSCM·CSVM**에 적용합니다. 기준에서만 보정한 확률 잡음 모델은 변경 조건에서 거부합니다.
- 오프라인 데모는 변경 치수의 결과를 만들어 내거나 기준 결과로 대체하지 않습니다. 실제 계산 서버가 필요합니다.
- LTspice·Verilog-A는 선택한 **고정 치수·VG·VBG·광조건**의 준정적 결과를 내보냅니다. 치수를 바꿀 때는 다시 계산하고 내보내야 합니다. 메타데이터가 다른 이전 결과는 거부합니다.
- Sentaurus는 여전히 미지원입니다. 이 compact geometry 확장만으로 메시·도핑 프로파일·접촉·경계조건을 갖춘 TCAD deck이 만들어지지 않습니다.

## 참고 문헌과 근거의 구분

- [Bawedin 외, Solid-State Electronics 54, 104–114 (2010), 저자 기관 기록](https://research.dial.uclouvain.be/entities/publication/cc47b194-936a-442c-9f40-062d812586df), DOI 10.1016/j.sse.2009.12.004: FD-SOI에서도 동적 바디 전위가 gate/drain 과도 전류에 영향을 준다는 연구 배경입니다. 공개 초록과 서지 정보를 확인했으며, 본 업데이트의 축약 injection 식을 그 논문에서 추출했다는 의미가 아닙니다.
- [UC Berkeley BSIM-SOI 공식 모델 안내](https://bsim.berkeley.edu/models/bsimsoi/): SOI 회로 모델과 관련 기술 문서를 제공하는 공식 자료입니다. 이 업데이트는 BSIM-SOI 구현이나 해당 모델의 검증을 이어받는 것이 아닙니다.
- [UC Berkeley BSIMSOI v4.4 매뉴얼](https://ngspice.sourceforge.io/external-documents/models/BSIMSOIv4.4_UsersManual.pdf), §3.3, §4.2, §6.1: SOI의 GIDL, 전하, 전면/후면 정전용량 경로를 확인하는 근거입니다. 본 STL 식이나 새 계수를 검증한 문헌은 아닙니다.
- [UC Berkeley BSIM-IMG](https://bsim.berkeley.edu/models/bsimimg/): 독립 게이트가 있는 얇은 SOI를 위한 별도 모델 체계입니다. 이번 수정은 BSIM-IMG 구현이 아닙니다.
- [Brody 외, effective/bulk lifetime 관계 검토](https://www.sciencedirect.com/science/article/abs/pii/S0927024802003501): 표면·벌크 수명의 분리에는 가정이 필요함을 다룬 문헌입니다. 이번 τj의 표면 기여율을 제공하지 않습니다.

이 문서의 기준 보정 증분식, 기존 τj를 사용하는 표면 우세 해석, 기존 GIDL 길이의 EOT
비율 확장은 본 업데이트에서 선택한 compact-model 근사입니다. 다중 치수 데이터가
없는 상태에서 실험적으로 확정된 법칙이나 소자 정확도로 제시하지 않습니다.
