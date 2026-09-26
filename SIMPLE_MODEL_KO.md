# Simple Model

Simple Model은 게재 확정된 논문 **J.-H. Park, H.-B. Noh, S.-W. Lee, S.-Y. Yun, and Y.-K. Choi, "Analytical Model for Single Transistor Latch in MOSFETs," *IEEE Electron Device Letters*, 2026, doi: 10.1109/LED.2026.3737574**의 해석 모델(식 (1)–(4), (7), Table I)을 그대로 구현한 빠른 결정론적 모델이다. 공간적으로 분포된 SRH 수송 문제를 풀지 않고, 바디 전하와 전류를 직접 계산한다. 이 문서는 `server/simple_model.py`와 `server/simple_config.py`의 구현을 설명한다.

시뮬레이터 전체의 틀(바디 전하 균형, 증배, BTBT, 재결합·확산, 래치 조건)은 이 논문을 따른다. 두 모델의 관계는 다음과 같다.

| 모델 | 내용 | 표기 |
|---|---|---|
| **Simple Model** | 논문의 수식 (1)–(4), (7)과 Table I. 확산 β만 상수(식 (8) 미사용). | 논문 모델 |
| **Detailed Model** | 같은 틀을 분포 SRH 수송·전계 테이블·측정 보정으로 확장한 모델. | Updated accuracy 모델 |

기존 Detailed Model과 선택하여 사용한다. Simple Model의 기본값은 논문 Table I의 값을 앱의 기준 Geometry에 맞춰 환산한 값이며, Device 1의 측정 보정을 그대로 옮긴 값이 아니다. 새로운 모델의 측정 정확도를 독립적으로 검증했다는 의미도 아니다.

## 사용 방법

1. 소자 모델에서 **Simple Model**을 선택한다. 기본 시작 바이어스는 $V_{\mathrm G}=-3\ \mathrm V$, $V_{\mathrm{BG}}=0\ \mathrm V$이다. Detailed Model의 기본값 $-2\ \mathrm V$와 구분한다.
2. Geometry와 측정 바이어스를 맞춘다. 단일 HRS 전압·전류로 $I_{\mathrm S}$ 또는 $\tau_{\mathrm B}$ 중 하나를 보정한다.
3. ID–VD에서 보정 결과를 확인하고 CSVM 또는 회로 해석으로 진행한다. 정적 적합과 과도 동작의 정확도는 따로 비교한다.
4. 외부 Body RC를 사용할 때 접점 전압과 내부 저장 영역 전위를 구분한다.

현재 지원 범위는 **FDSOI, 결정론적 VSCM/CSVM 및 BE 회로 과도 해석**이다. Simple Model에서는 확률적 해석과 상용 시뮬레이터 내보내기(LTspice, Verilog-A, Sentaurus)를 지원하지 않는다. 기존 Detailed Model의 지원 범위와 보정 데이터는 별도로 유지된다.

## 1. 상태와 전류

모든 단자 전압은 소스 S를 기준으로 한다. $V_{\mathrm T}=kT/q$이며 현재 온도는 300 K이다.

| 기호 | 의미 | 단위 |
|---|---|---|
| $u$ | 소스 쪽 BJT 유효 이미터/바디 접점 전압 | V |
| $r$ | 내부 컬렉터 전압 $V_{\mathrm D}-I_{\mathrm D}R_{\mathrm{LRS}}$ | V |
| $w$ | 집중된 전하 저장 영역의 전위 $u+I_{\mathrm D}R_{\mathrm{LRS}}$ | V |
| $Q$ | 1차 완화 대상인 바디 전하 좌표 | C |
| $V_{\mathrm{bias}}$ | 게이트와 백게이트가 만드는 유효 전위 이동 | V |
| $I_{\mathrm{seed}}$ | 증배 전 BJT 컬렉터 주입 전류 | A |

구현식은 다음과 같다.

$$
M(r)=\left[1-\left(\frac{r}{V_{\mathrm{BR}}}\right)^\eta\right]^{-1},\qquad
I_{\mathrm{seed}}=I_{\mathrm S}\exp\!\left(\frac{u}{V_{\mathrm T}}\right).
$$

$$
I_{\mathrm D}=M I_{\mathrm{seed}}+I_{\mathrm{BTBT}}+I_{\mathrm{ph}},\qquad
V_{\mathrm D}=r+I_{\mathrm D}R_{\mathrm{LRS}}.
$$

$$
I_{\mathrm{II}}=(M-1)I_{\mathrm{seed}},\qquad
I_{\mathrm{DIFF}}=\frac{I_{\mathrm{seed}}}{\beta}.
$$

$I_{\mathrm{BTBT}}$는 측방향 BTBT와 GIDL의 합이다. 광생성 $I_{\mathrm{ph}}$는 입력된 유효 전류를 드레인과 바디 생성 항에 각각 한 번 더하는 선택적 확장이다. Simple Model에서는 이 광생성 전류를 다시 avalanche 증배하지 않는다.

기존 Detailed Model의 채널 전류, 분포 SRH 손실, 접합 SRH 손실, ambipolar 수송 전하, 추가 access 저항을 이 식에 중복해서 더하지 않는다. 기존 분포 모델의 호출 뒤 손실 항만 바꾸는 구현이 아니며, Simple kernel 자체가 SRH 수송 풀이를 생략한다.

## 2. 재결합과 바디 전하 좌표

$$
w=u+I_{\mathrm D}R_{\mathrm{LRS}},\qquad
Q=C_{\mathrm B}(w-V_{\mathrm{bias}}).
$$

$$
\frac{\mathrm dQ}{\mathrm dt}
=I_{\mathrm{II}}+I_{\mathrm{BTBT}}+I_{\mathrm{ph}}
-I_{\mathrm{DIFF}}-\frac{Q}{\tau_{\mathrm B}}+I_{\mathrm{B,ext}}.
$$

$I_{\mathrm{B,ext}}$는 외부 B 접점에서 소자 안으로 들어오는 홀 전류이다. 부유 바디에서는 0이다. 전하 좌표가 음수인 경우에도 구현은 부호를 유지한 선형 완화 $Q/\tau_{\mathrm B}$를 사용하며, 이를 별도의 양의 미시적 재결합률로 해석하지 않는다.

논문의 식 (1)–(3)을 동적으로 일관되게 사용하는 핵심은 감쇠하는 좌표를 $Q=C_{\mathrm B}(w-V_{\mathrm{bias}})$로 구분하는 것이다. 단순히 $Q=C_{\mathrm B}u$로 대체하면 바이어스 이동과 $I_{\mathrm D}R_{\mathrm{LRS}}$가 재결합 및 전하식에서 서로 다르게 적용된다.

부유 바디의 정상 상태에서는

$$
u=R_{\mathrm{REC}}\left(I_{\mathrm{II}}+I_{\mathrm{BTBT}}+I_{\mathrm{ph}}-I_{\mathrm{DIFF}}\right)
+V_{\mathrm{bias}}-I_{\mathrm D}R_{\mathrm{LRS}},\qquad
R_{\mathrm{REC}}=\frac{\tau_{\mathrm B}}{C_{\mathrm B}},
$$

가 되어 논문의 바디 전위 피드백 구조를 얻는다. CSVM은 여기에 회로 방정식 $C_{\mathrm D}\,\mathrm dV_{\mathrm D}/\mathrm dt=I_{\mathrm{IN}}-I_{\mathrm D}$를 함께 적분한다. 별도의 가상 상태 전환 규칙으로 발진 파형을 만들지 않는다.

## 3. Geometry와 확산 β

### 바이어스에 대해 일정하고 Geometry에 따라 변하는 β

논문의 바이어스 의존 β 식 (8)은 사용하지 않는다. 고정된 에미터 도핑, 확산 계수와 에미터 홀 확산 길이를 가정하면

$$
I_{\mathrm{C0}}\propto \frac{A}{N_{\mathrm{body}}L_{\mathrm{base}}},\qquad
I_{\mathrm{B,diff0}}\propto A,
$$

이므로 저주입 확산 근사에서 $\beta\propto 1/(N_{\mathrm{body}}L_{\mathrm{base}})$이다. 구현은 논문의 치수 관계와 같이 $L_{\mathrm{base}}\simeq L$을 사용한다.

$$
\beta=\beta_{\mathrm{ref}}\frac{L_{\mathrm{ref}}}{L}\frac{N_{\mathrm{ref}}}{N_{\mathrm{body}}},
$$

$$
I_{\mathrm S}=I_{\mathrm{S,ref}}
\frac{W}{W_{\mathrm{ref}}}\frac{T_{\mathrm{Si}}}{T_{\mathrm{Si,ref}}}
\frac{L_{\mathrm{ref}}}{L}\frac{N_{\mathrm{ref}}}{N_{\mathrm{body}}},
$$

$$
R_{\mathrm{LRS}}=R_{\mathrm{LRS,ref}}
\frac{L}{L_{\mathrm{ref}}}\frac{W_{\mathrm{ref}}}{W}
\frac{T_{\mathrm{Si,ref}}}{T_{\mathrm{Si}}}.
$$

두 확산 전류의 접합 면적이 같다는 가정에서 $W T_{\mathrm{Si}}$는 β의 비율에서 상쇄된다. 따라서 β에 임의의 폭·두께 거듭제곱을 추가하지 않는다. $I_{\mathrm S}/\beta$에는 접합 면적에 비례하는 관계가 남는다. 고주입 LRS에 같은 β를 사용하는 것은 Simple Model의 근사이며, 고주입 수송을 정확하게 푼 결과가 아니다.

### 두께와 수명

$$
\tau_{\mathrm B}=\frac{\tau_{\mathrm{ref}}}
{(1-f_{\mathrm{surf}})+f_{\mathrm{surf}}T_{\mathrm{Si,ref}}/T_{\mathrm{Si}}}.
$$

$f_{\mathrm{surf}}$는 기준 두께에서 수명 역수에 대한 표면 기여율이다. 기본값 1은 표면 우세 가정이며 측정으로 추출하지 않았다. 이 값이 1이면 수명이 두께에 비례하고, 0이면 두께와 무관한 기준 수명을 사용한다. 별도의 초박막 급상승 함수나 양자 효과를 넣지 않는다. 두께 변화에 따른 정량 예측에는 추가 데이터가 필요하다.

## 4. 게이트 결합과 정전용량

$a=(L/L_{\mathrm{ref}})(W/W_{\mathrm{ref}})$라고 하면 유효 정전용량은

$$
C_{\mathrm G}=C_{\mathrm{B,ref}}\gamma_{\mathrm G}a\frac{\mathrm{EOT}_{\mathrm{ref}}}{\mathrm{EOT}},
$$

$$
C_{\mathrm{BG}}=C_{\mathrm{B,ref}}\gamma_{\mathrm{BG}}a
\frac{T_{\mathrm{box,ref}}+T_{\mathrm{Si,ref}}/3}{T_{\mathrm{box}}+T_{\mathrm{Si}}/3},
$$

$$
C_{\mathrm S}=C_{\mathrm{B,ref}}(1-\gamma_{\mathrm G}-\gamma_{\mathrm{BG}})a,\qquad
C_{\mathrm B}=C_{\mathrm G}+C_{\mathrm{BG}}+C_{\mathrm S}.
$$

여기서 $T_{\mathrm{Si}}/3$은 Si와 SiO₂ 유전율 비를 사용한 직렬 정전용량 표현이다. $\gamma_{\mathrm G}+\gamma_{\mathrm{BG}}\le1$을 유지하여 $C_{\mathrm S}\ge0$이 되도록 한다. 이 분할은 논문의 결합 계수에 맞춰 구성한 유효 집중소자 근사로, 새로운 Poisson 해나 별도 백채널 모델이 아니다.

논문 식 (2)의 $V_{\mathrm{BS,bias}}=\gamma_1(V_{\mathrm{FG}}-\varphi_{\mathrm{FB}})+\gamma_2(V_{\mathrm{BG}}-\varphi_{\mathrm{FB}})$를 정전용량 비로 쓰면 다음과 같다. 논문은 **축적(accumulation) 상태에서는 축적된 정공층이 게이트–바디 결합을 차폐하므로 결합 계수를 0으로 둔다**. 구현에서는 각 게이트 전압을 평탄대 전압에서 잘라 이를 연속적으로 적용한다.

$$
V_{\mathrm G}^{\mathrm{eff}}=\max(V_{\mathrm G},V_{\mathrm{FB}}),\qquad
V_{\mathrm{BG}}^{\mathrm{eff}}=\max(V_{\mathrm{BG}},V_{\mathrm{FB}}),
$$

$$
V_{\mathrm{bias}}=
\frac{C_{\mathrm G}(V_{\mathrm G}^{\mathrm{eff}}-V_{\mathrm{FB}})+C_{\mathrm{BG}}(V_{\mathrm{BG}}^{\mathrm{eff}}-V_{\mathrm{FB}})}{C_{\mathrm B}}.
$$

즉 $V_{\mathrm G}$가 $V_{\mathrm{FB}}$ 아래로 내려가도 $V_{\mathrm{bias}}$는 더 이상 내려가지 않는다(전면 게이트 항이 0). 반면 GIDL 전계는 계속 커지므로 $I_{\mathrm{BTBT}}$가 급증하고, 그 결과 $V_{\mathrm{LU}}(V_{\mathrm G})$는 $V_{\mathrm{FB}}$ 부근에서 최대가 된 뒤 다시 내려가는 종 모양(bell shape, 논문 Fig. 5(a))이 된다. 이전 구현은 이 차폐가 없어 $V_{\mathrm G}\le V_{\mathrm{FB}}$에서 $V_{\mathrm{LU}}$가 $V_{\mathrm{BR}}$에 붙어 버렸다. 백게이트 항은 논문 스크립트처럼 유지하되, 백게이트가 자체 평탄대 아래로 내려가면 같은 방식으로 잘라 연속성을 유지한다(논문 본문은 "두 계수 모두 0"이라 적었으나 전면 게이트 축적에서 백게이트 항까지 한꺼번에 0으로 두면 $V_{\mathrm{FB}}$에서 $V_{\mathrm{bias}}$가 불연속으로 뛰고 Fig. 5(a)의 매끄러운 곡선이 나오지 않는다).

단자 변위 전류는 같은 저장 영역 전위와 같은 유효 게이트 전압을 사용하는 전하로 계산한다.

$$
Q_{\mathrm G}=C_{\mathrm G}(V_{\mathrm G}^{\mathrm{eff}}-w),\qquad
Q_{\mathrm{BG}}=C_{\mathrm{BG}}(V_{\mathrm{BG}}^{\mathrm{eff}}-w).
$$

G/BG 전압 변화와 $I_{\mathrm D}R_{\mathrm{LRS}}$에 따른 $w$의 변화가 전하 도함수에 함께 들어간다. 축적 상태에서 게이트 전압 변화로 생기는 추가 변위 전하는 차폐 정공층에 놓이며, 집중 저장 영역 밖이므로 $Q$, $Q_{\mathrm G}$, $Q_{\mathrm{BG}}$, $C_{\mathrm S}w$의 증분 보존은 그대로 성립한다. 큰 게이트 바이어스 범위에 대한 일반적 정확도를 의미하지 않는다.

## 5. BTBT 및 유효 범위

논문 Table I의 $I_{\mathrm{BTBT}}=q\int_V A\,E^{2.5}\exp(-B/E)\,dv$ ($A=4\times10^{14}\ \mathrm{cm^{-0.5}V^{-2.5}s^{-1}}$, $B=19\ \mathrm{MV/cm}$)를 논문 참고 스크립트와 같이 "한 전계에서의 Kane 생성률 × 생성 체적"으로 평가한다. 두 항 모두 $1-\exp(-r/V_{\mathrm T})$를 곱하고 $r\le0$에서는 0이며, 전체에 배율 $s_{\mathrm{BTBT}}$가 곱해진다.

**접합(측방향) BTBT** — 급격 n⁺ 드레인 접합의 최대 전계와 공핍 체적을 사용한다.

$$
V_{\mathrm{bi}}=V_{\mathrm T}\ln\frac{10^{20}N_{\mathrm{body}}}{n_i^2}-\gamma_{\mathrm G}\,(V_{\mathrm G}^{\mathrm{eff}}-V_{\mathrm{FB}}),\qquad
W_{\mathrm d}=\sqrt{\frac{2\varepsilon_{\mathrm{Si}}(V_{\mathrm{bi}}+r)}{qN_{\mathrm{body}}}},\qquad
E_{\mathrm j}=\frac{2(V_{\mathrm{bi}}+r)}{W_{\mathrm d}},
$$

$$
I_{\mathrm{BTBT,j}}=q\,A\,E_{\mathrm j}^{2.5}\exp(-B/E_{\mathrm j})\;W\,T_{\mathrm{Si}}\,W_{\mathrm d}.
$$

게이트가 평탄대 위에 있을 때 접합 내장 전위를 $\gamma_{\mathrm G}(V_{\mathrm G}-V_{\mathrm{FB}})$만큼 낮추는 것은 논문 스크립트의 정의이며, 축적 상태에서는 변하지 않는다.

**GIDL** — 게이트–드레인 가장자리의 수직 전계와 고정 체적을 사용한다.

$$
E_{\mathrm g}=\frac{r-V_{\mathrm G}-V_{\mathrm{FB0}}-E_{\mathrm g}^{\mathrm{Si}}}{3\,\mathrm{EOT}}
=\frac{r-V_{\mathrm G}+1.2-1.12}{3\,\mathrm{EOT}},
$$

$$
I_{\mathrm{GIDL}}=q\,A\,E_{\mathrm g}^{2.5}\exp(-B/E_{\mathrm g})\;V_{\mathrm{GIDL}}\frac{W}{W_{\mathrm{ref}}},\qquad W_{\mathrm{ref}}=200\ \mathrm{nm}.
$$

- $V_{\mathrm{FB0}}=-1.2\ \mathrm V$는 게이트–n⁺ 드레인 사이의 평탄대 전압으로 논문 스크립트의 값이며, 바디의 $V_{\mathrm{FB}}=-3.35\ \mathrm V$와 다르다. $E_{\mathrm g}\le0$이면 GIDL은 0이다.
- $V_{\mathrm{GIDL}}$(GIDL 생성 체적, `gidl_volume_ref_cm3`)은 기준 폭 200 nm에서의 체적을 부피 단위로 직접 입력하며, 모델은 폭에 비례해 환산한다(화면 단위 nm³, 1 cm³ = $10^{21}$ nm³). 기본값 $4.55\times10^{-16}\ \mathrm{cm^3}=4.55\times10^{5}\ \mathrm{nm^3}$는 논문 참고 스크립트의 유효 체적 $W\,L_{\mathrm{ov}}\,W_{\mathrm t}\times100$이다. 여기서 $L_{\mathrm{ov}}=5$ nm는 게이트–드레인 중첩 길이, $W_{\mathrm t}=\sqrt{2\varepsilon_{\mathrm{Si}}\cdot1.12/(q\cdot7\times10^{19})}\approx4.55$ nm는 n⁺ 드레인의 터널링 깊이로, 둘 다 기본값을 유도하는 데만 쓰이며 모델은 입력한 체적만 사용한다(따라서 $T_{\mathrm{Si}}$에는 의존하지 않는다). 스크립트는 $W\,L_{\mathrm{ov}}$의 m²→cm² 환산에 $10^4$ 대신 $10^6$을 곱해 물리 체적의 100배를 사용했고, Fig. 5의 결과는 이 체적으로 얻어진 것이다. 물리 체적 $W\cdot5\ \mathrm{nm}\cdot4.55\ \mathrm{nm}=4.55\times10^{3}\ \mathrm{nm^3}$을 입력하면 GIDL이 너무 작아 $V_{\mathrm{LU}}$가 $V_{\mathrm{BR}}$에 붙는다(이전 구현의 증상). GIDL은 $\exp(-B/E_{\mathrm g})$에 지배되므로 체적 100배는 전계 길이를 $3\,\mathrm{EOT}$ 대신 약 $2.1$–$2.3\,\mathrm{EOT}$로 잡는 것과 같은 크기의 보정이다.
- GIDL 전압에는 외부 $V_{\mathrm D}$ 대신 내부 $r$을 사용한다. HRS에서는 $I_{\mathrm D}R_{\mathrm{LRS}}$가 무시할 만해 차이가 없고, LRS에서는 전류에 의존하는 암시적 BTBT 풀이를 피하는 명시적 근사이다.
- Miller 식은 $0\le r<V_{\mathrm{BR}}$에서 사용한다. $r\ge V_{\mathrm{BR}}$에서는 유효하지 않은 평가로 처리하며, 증배 계수를 임의의 최대값으로 대체하지 않는다.
- 시작점과 작은 링잉의 수치 연장을 위해 $-50\ \mathrm{mV}\le r\le0$에서는 $M=1$, BTBT = 0을 사용한다. 그보다 작은 $r$은 지원하지 않는다.
- 논문의 $\exp(u/V_{\mathrm T})$를 그대로 유지한다. 정확한 영바이어스 평형을 만족하는 완전한 Ebers–Moll 소자나 역방향 트랜지스터 모델이 아니다. 역방향 구동에 활용하면 안 된다.

**확인한 결과** (논문 소자 $W=650$ nm, $T_{\mathrm{ox}}=13$ nm, $N_{\mathrm{body}}=3\times10^{17}$, $C_{\mathrm B}$·$R_{\mathrm{LRS}}$·$I_{\mathrm S}$를 Table I 값으로 맞춤, 상수 $\beta=2.3$):

| $V_{\mathrm G}$ (V) | −2.2 | −2.6 | −3.0 | −3.2 | −3.4 | −3.8 | −4.0 |
|---|---|---|---|---|---|---|---|
| $V_{\mathrm{LU}}$ (V) | 1.76 | 1.95 | 2.21 | 2.23 | 2.19 | 1.92 | 1.80 |
| $V_{\mathrm{LD}}$ (V) | 1.75 | 1.76 | 1.76 | 1.76 | 1.76 | 1.76 | 1.76 |
| $I_{\mathrm{BTBT}}/I_{\mathrm{gen}}$ at $V_{\mathrm{LU}}$ | 0.00 | 0.07 | 0.58 | 0.73 | 0.80 | 0.75 | 0.57 |

논문 Fig. 5(a)(최대 ≈2.22 V at −3.2 V, 1.75 V at −2.2 V, ≈1.85 V at −4.0 V)와 Fig. 5(b)의 경향을 재현한다. $V_{\mathrm G}=-3$ V에서 $V_{\mathrm{BG}}$를 0→2 V로 올리면 $V_{\mathrm{LU}}$는 2.21→1.84 V로 내려간다(Fig. 5(c)). 상수 β를 쓰므로 $V_{\mathrm{LD}}$는 논문과 달리 $V_{\mathrm G}$·$V_{\mathrm{BG}}$에 거의 무관하고, 그 결과 $V_{\mathrm{BG}}\gtrsim3.3$ V에서는 래치 창이 닫힌다(논문은 식 (8)의 β 증가로 $V_{\mathrm{LD}}$가 함께 내려가 4 V까지 창이 유지된다).

## 6. HRS 보정

### 보정 대상과 데이터 수

UI는 $I_{\mathrm S}$ 또는 $\tau_{\mathrm B}$ 한 변수를 HRS 한 점 이상에서 보정할 수 있다. 두 변수를 동시에 보정하려면 전압이 구분되는 HRS 데이터가 최소 3점 필요하며, 추가로 두 변수의 식별 가능성을 확인한다. 데이터 수 조건만 만족한다고 유일한 파라미터가 정해지는 것은 아니다.

정적 바디 방정식에는 $R_{\mathrm{REC}}=\tau_{\mathrm B}/C_{\mathrm B}$가 들어간다. 따라서 **HRS 보정 중 $C_{\mathrm B}$를 고정**한다. 정적 데이터만으로 수명과 정전용량을 각각 독립적으로 추출했다고 해석하지 않는다. $I_{\mathrm S}$와 $\tau_{\mathrm B}$ 역시 제한된 전압 범위에서는 강한 공변성을 보일 수 있다. 보정이 불안정하면 한 파라미터를 고정하거나 더 넓은 HRS 구간의 데이터를 추가한다.

### 한 점 관계식

Geometry와 바이어스가 지정된 측정점 $(V_{\mathrm h},I_{\mathrm h})$에서 다른 파라미터를 고정하면

$$
r_{\mathrm h}=V_{\mathrm h}-I_{\mathrm h}R_{\mathrm{LRS}},\qquad
s_{\mathrm h}=\frac{I_{\mathrm h}-I_{\mathrm{BTBT,h}}-I_{\mathrm{ph}}}{M(r_{\mathrm h})},
$$

$$
J_{\mathrm h}=\left[M(r_{\mathrm h})-1-\frac1\beta\right]s_{\mathrm h}
+I_{\mathrm{BTBT,h}}+I_{\mathrm{ph}}.
$$

$I_{\mathrm S}$를 고정한 경우

$$
\tau_{\mathrm B}=C_{\mathrm B}
\frac{V_{\mathrm T}\ln(s_{\mathrm h}/I_{\mathrm S})-V_{\mathrm{bias}}+I_{\mathrm h}R_{\mathrm{LRS}}}{J_{\mathrm h}}.
$$

$\tau_{\mathrm B}$를 고정한 경우

$$
I_{\mathrm S}=s_{\mathrm h}\exp\!\left[
-\frac{R_{\mathrm{REC}}J_{\mathrm h}+V_{\mathrm{bias}}-I_{\mathrm h}R_{\mathrm{LRS}}}{V_{\mathrm T}}
\right].
$$

이 식의 $I_{\mathrm S},\tau_{\mathrm B}$는 해당 Geometry에서의 유효값이다. 저장하는 기준 파라미터는 앞서 설명한 Geometry 관계를 통해 환산한다. 양의 유한한 파라미터, $s_{\mathrm h}>0$, Miller 유효 범위와 HRS 적합성을 확인해야 한다. $J_{\mathrm h}$가 0에 가까우면 수명 추정이 민감해진다.

### 구현의 선형 보정과 가지 재확인

각 점에 대해 기준 전류 $I_*=1\ \mathrm A$를 사용하여

$$
y_{\mathrm h}=V_{\mathrm T}\ln(s_{\mathrm h}/I_*)+I_{\mathrm h}R_{\mathrm{LRS}}-V_{\mathrm{bias}}
=R_{\mathrm{REC}}J_{\mathrm h}+V_{\mathrm T}\ln(I_{\mathrm S}/I_*)
$$

로 변환한다. 로그 안의 전류 비는 무차원이다. 한 변수를 보정할 때는 기울기 또는 절편을 고정하고, 두 변수를 보정할 때는 이 선형 관계를 적합한다. $J_{\mathrm h}$의 차이가 충분하지 않으면 기울기와 절편을 안정적으로 구분할 수 없다.

파라미터를 얻은 뒤에는 실제로 가장 낮은 전류의 안정 HRS 가지를 다시 계산·정밀화하고, 측정 전압에서의 로그 전류 RMSE를 보고한다. HRS 밖의 데이터, 식별 불가능한 조합, 로그 전류 잔차가 지나치게 큰 결과는 적용을 거부한다. 현재 큰 오차 거부 기준은 1 decade이며, 이 수치가 정확도 보증 기준을 뜻하지는 않는다. 회귀 식이 맞더라도 올바른 안정 가지에서 재현되는지 별도로 확인하는 과정이다.

일반적인 HRS 한 점은 한 개의 정상 상태 제약을 제공한다. 그 점을 자동으로 fold라고 가정하지 않는다. 실제 pre-latch fold라고 별도 확인된 경우에만 접선 조건을 추가 정보로 사용할 수 있다. 현재 HRS 보정 자체가 fold 조건을 독립적으로 검증했다는 의미는 아니다.

입력한 HRS에 대한 잔차는 **그 데이터에 대한 보정 결과**이다. 미사용 데이터의 ID–VD, fold 위치, 첫 오버슈트, 반복 주기의 극값·주파수와 Body RC 응답을 비교해야 과도 모델의 적용 범위를 평가할 수 있다.

## 7. 5단자와 Body RC

D·S·G·BG·B를 지원한다. G/BG는 회로 전압원으로 변조할 수 있다. B는 소스 쪽의 유효 바디/이미터 접점에 연결하며, $V(\mathrm B)-V(\mathrm S)=u$를 사용한다. 바디 저장 영역 전위는 $w=u+I_{\mathrm D}R_{\mathrm{LRS}}$이므로 두 값이 특히 LRS에서 다를 수 있다.

| 회로 출력 | Simple Model에서의 의미 |
|---|---|
| `X1.vbody` | B 접점의 S 기준 전압 $u$ |
| `X1.vb` | 전하 저장 영역의 S 기준 전위 $w$; 바디 플롯 값 |
| `I(X1.b)` | 소자 안으로 들어오는 B 접점 전류 |

외부 RC 전류는 바디 전하 방정식과 단자 KCL에 실제로 결합된다. 다만, 이 외부 접점 위치와 저항성 전압 강하의 배치는 논문의 3단자 모델을 5단자로 확장한 **현상론적 집중소자 가정**이다. 실제 바디 접점의 분포 저항이나 전류 경로를 독립적으로 추출한 결과가 아니다. Detailed Model의 `vb`와 내부 전위 정의가 완전히 같은 것도 아니다.

## 8. 초기값의 출처

앱 기준 Geometry는 다음과 같다. 논문 Table I의 폭 650 nm에서 앱 기준 폭 200 nm로 $I_{\mathrm S},C_{\mathrm B}$를 비례 환산하고 $R_{\mathrm{LRS}}$를 반비례 환산한다. 아래 값은 보정 전 시작점이다.

| 항목 | 기준값 | 출처 또는 가정 |
|---|---:|---|
| $L,W,T_{\mathrm{Si}}$ | 500, 200, 50 nm | 앱 기준 치수 |
| EOT, $T_{\mathrm{box}}$ | 14.1, 140 nm | 앱 기준; BOX는 명시적 기준 가정 |
| $N_{\mathrm{body}}$ | $2.295773162796593\times10^{17}\ \mathrm{cm}^{-3}$ | 기존 앱 기준 도핑; 논문의 $3\times10^{17}$과 구분 |
| $\beta_{\mathrm{ref}}$ | 2.3 | 바이어스 일정 확산 β의 시작 가정; 논문 식 (8) 대체(사용자 결정) |
| $\tau_{\mathrm{ref}}$ | 200 ns | 논문 Table I |
| $C_{\mathrm{B,ref}}$ | $0.86\times200/650=0.264615$ fF | 논문 Table I의 폭 환산 |
| $R_{\mathrm{LRS,ref}}$ | $44\times650/200=143$ kΩ | 논문 Table I의 폭 환산 |
| $I_{\mathrm{S,ref}}$ | $2\times10^{-16}\times200/650=6.153846\times10^{-17}$ A | 논문 Table I의 폭 환산 |
| $V_{\mathrm{BR}},\eta$ | 2.35 V, 4 | 논문 Table I |
| $\gamma_{\mathrm G},\gamma_{\mathrm{BG}}$ | 0.2, 0.0525 | 논문 Table I에서 시작한 유효 결합 |
| $V_{\mathrm{FB}}$ | −3.35 V | 논문 Table I |
| BTBT 배율 $s_{\mathrm{BTBT}}$ | 1 | 기본값 |
| $f_{\mathrm{surf}}$ | 1 | 표면 우세 가정; 미추출 |
| GIDL 생성 체적 $V_{\mathrm{GIDL}}$ (기준 폭 200 nm) | $4.55\times10^{5}$ nm³ ($4.55\times10^{-16}$ cm³) | 논문 참고 스크립트의 유효 체적 $W\cdot5\ \mathrm{nm}\cdot4.55\ \mathrm{nm}\times100$(§5) |
| $V_{\mathrm G},V_{\mathrm{BG}}$ | −3 V, 0 V | UI의 초기 사용 조건 |

논문 수치의 폭 환산이 앱의 다른 EOT·도핑까지 동일한 측정 소자로 만들어 주는 것은 아니다. 모든 Geometry·바이어스에서 논문의 정확도가 자동으로 유지되지 않는다.

## 9. 이번 변경 내용

2026-09-26 (논문 정합):

- 축적 상태의 결합 차폐를 구현했다: 각 게이트 전압을 $V_{\mathrm{FB}}$에서 잘라 $V_{\mathrm{bias}}$에 넣는다(§4). 이로써 $V_{\mathrm{LU}}(V_{\mathrm G})$의 종 모양이 재현된다.
- 접합 BTBT와 GIDL을 논문 참고 스크립트의 정의(최대 전계 × 체적, 게이트 변조 내장 전위, $V_{\mathrm{FB0}}=-1.2$ V)로 바꿨다(§5). 이전의 16점 Gauss 적분과 GIDL 상수(−0.3 V)는 제거했다.
- GIDL 생성 체적을 부피 단위 파라미터 `gidl_volume_ref_cm3`(기본 $4.55\times10^{-16}$ cm³ = $4.55\times10^{5}$ nm³, 폭에 비례 환산)로 직접 입력하도록 했고 모델 버전을 `simple-edl2026-paper-v3`로 올렸다. `/api/meta`의 `simple_model.reference`에 논문 서지 정보를 넣었다.
- 논문 소자에 대한 회귀 테스트(`server/tests/test_simple_model.py`)로 종 모양·BTBT 비율·백게이트 경향을 고정했다.

이전 변경:

- Detailed / Simple Model 선택과 모델별 파라미터를 추가했다.
- Simple Model은 SRH 수송 풀이와 그 전계 테이블 준비를 거치지 않는 직접 평가 경로를 사용한다.
- 바이어스 일정 β에 길이·도핑의 확산 관계를 적용하고, 전류·저항·수명·게이트 결합의 Geometry 관계를 명시했다.
- HRS 보정에서 $I_{\mathrm S}$ 또는 수명을 선택하도록 하고, 동시 보정은 여러 점과 식별 가능성 검사를 요구한다.
- 같은 전하 정의로 CSVM, G/BG 변조, 5단자 Body RC를 연결했다.
- Simple Model의 확률적 해석과 상용 내보내기는 미지원으로 표시한다.

성능 수치는 실제 측정한 검증 기록을 참조한다. 실행 시간은 회로 크기, 스텝 수, JIT 최초 컴파일과 캐시 상태에 따라 달라지므로 이 문서에서 고정된 가속 배율을 약속하지 않는다.

## 근거

J.-H. Park, H.-B. Noh, S.-W. Lee, S.-Y. Yun, and Y.-K. Choi, "Analytical Model for Single Transistor Latch in MOSFETs," *IEEE Electron Device Letters*, 2026, doi: 10.1109/LED.2026.3737574의 식 (1)–(4), (7), Table I, 축적 상태의 결합 차폐 설명, Fig. 4–5의 수치와 저자가 제공한 참고 스크립트(snap-back 계산)를 사용했다. β 식 (8)은 저자의 지시에 따라 적용하지 않고 상수 β를 확산 손실에만 사용한다. Geometry의 확산 스케일링, 수명의 표면 기여율, 유효 정전용량 분할, 내부 전압을 사용하는 GIDL, 외부 B 접점은 위에서 각각 구분한 구현 가정 또는 확장이다.
