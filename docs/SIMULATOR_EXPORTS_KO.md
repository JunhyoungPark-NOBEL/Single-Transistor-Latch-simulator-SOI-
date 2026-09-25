# 상용 시뮬레이터 내보내기

소자 화면의 **상용 시뮬레이터로 내보내기**에서 형식을 선택합니다. 현재 보정값을 계산 서버로 다시 계산한 뒤 모델을 만듭니다. 데모·기록된 결과는 내보내기에 사용하지 않습니다.

| 형식 | 제공 파일 | 범위 |
|---|---|---|
| LTspice | `.cir`, 선택적 `.lib`, 사용 안내 | 고정 조건의 ID–VD와 정적 히스테리시스 |
| Verilog-A | `.va`, 사용 안내 | 같은 ID–VD를 표준 Verilog-A 동작 모델로 표현 |
| Sentaurus TCAD | 현재 다운로드 미지원 | 구조·격자·도핑·접촉 및 연동 구현이 추가로 필요 |

## LTspice

`.cir`을 열어 실행한 뒤 `-I(Vdrive)`를 세로축, `V(d)`를 가로축으로 선택합니다. 다른 회로에서는 `.lib` 파일을 포함하고 `X1 drain source STL_소자이름`으로 연결합니다. 정확한 모델명과 등록 예제는 다운로드한 파일에 들어 있습니다.

## Verilog-A

`.va`를 Verilog-A 지원 시뮬레이터의 모델 소스로 등록합니다. `disciplines.vams`는 해당 제품의 표준 include 경로에 있어야 합니다. Spectre에서는 다음 형태로 등록하고 연결합니다.

```spectre
ahdl_include "STL_Device_1.va"
X1 (d 0) STL_Device_1
```

파일명과 모듈명은 실제 내보낸 값에 맞춥니다. 0 V에서 시작하는 삼각파 전압원의 **과도해석**으로 ID–VD를 확인합니다. 상승 방향 VLU와 하강 방향 VLD 통과 시 branch를 바꾸며, 초기 전압이 VLU 미만이면 HRS에서 시작합니다. DC sweep의 히스테리시스는 검증하지 않았습니다.

## 정확한 지원 범위

두 형식 모두 L·W·Tsi·EOT·Tbox·Nbody, VG·VBG, 광전류·보정값을 고정한 **준정적 동작 모델**입니다. 웹 엔진에서 계산한 안정 branch 전류를 log(I) 선형 보간하고, 같은 VLU/VLD를 사용합니다. 단자는 D, S입니다. 치수와 바이어스는 파일 메타데이터에 기록되며, 보정이나 고정 조건을 바꿨다면 다시 계산하고 내보내야 합니다. 메타데이터에는 기본적으로 보정 파라미터와 엔진 파라미터 벡터를 넣지 않습니다(`calibration_included: false`). 다만 전류 표 자체가 보정된 ID–VD 곡선을 그대로 담으므로 파일을 공유할 범위를 확인해 주세요. 상용 프로그램 안에서 치수만 바꿔 재계산하는 물리 모델은 아닙니다.

바디 전하 시간 적분과 잡음은 포함하지 않습니다. CSVM의 Cdrain, Vtop, Vbottom, 발진 주파수를 재현하는 물리 과도 모델이 아닙니다. 허용 VDS 범위는 각 파일에 기록되어 있으며, 범위 밖에서는 끝값 고정 또는 0 A로 제한합니다. 이 제한은 물리적 외삽이 아닙니다.

현재 검증은 실제 엔진의 ID–VD 데이터와 **생성된 파일을 독립적으로 읽어 얻은 전류값의 비교**, 임계값·메타데이터·오류 처리 및 UI 동작 확인입니다. LTspice와 상용 Verilog-A 시뮬레이터 바이너리 실행은 수행하지 않았습니다.

## Sentaurus 검토 결과

현재 compact calibration은 물리 소자 구조를 유일하게 결정하지 않습니다. 따라서 실행 가능한 물리 TCAD deck을 만들려면 구조·격자·도핑 분포·접촉·물리 모델 등의 원본 설정이 더 필요합니다.

Sentaurus의 compact model 연동은 물리 TCAD 소자를 만드는 작업과 별개입니다. 확인한 Synopsys CMI 자료는 C++ 모델과 인터페이스 정의를 해당 Sentaurus 환경에서 컴파일하는 절차를 설명합니다. 이 앱의 `.va`를 Sentaurus Device에 직접 불러오는 경로는 확인·검증하지 못했습니다. 실행되지 않는 템플릿을 모델로 제공하지 않도록 해당 형식의 다운로드를 비활성화했습니다.

## 근거 자료

- [Accellera Verilog-AMS 2023 언어 표준](https://www.accellera.org/images/downloads/standards/v-ams/VAMS-LRM-2023.pdf): analog function, contribution, initial_step, cross. cross는 DC 초기화를 대신하지 않습니다.
- [Cadence의 Verilog-A 소스 포함 설명](https://community.cadence.com/cadence_technology_forums/f/custom-ic-design/52254/how-to-include-the-veriloga-content-in-the-netlist/1384857): Spectre의 ahdl_include 사용.
- [Analog Devices의 LTspice 히스테리시스 스위치 설명](https://www.analog.com/en/resources/analog-dialogue/articles/how-to-add-a-voltage-controlled-switch.html): Vt/Vh의 의미.
- [Synopsys Sentaurus Device](https://www.synopsys.com/manufacturing/tcad/device-simulation/sentaurus-device.html): 물리 소자 및 사용자 모델링 제품 범위.
- [Synopsys Compact Models User Guide N-2017.09 공개 사본](https://manuals.plus/m/c8a17f66d262a105725e2db0306af3f2c5e7fe3bf34aecef3463c0d9f31026ff.pdf): CMI의 C++ 모델·인터페이스·컴파일 흐름. 설치 버전의 공식 매뉴얼과 라이브러리를 기준으로 별도 구현이 필요합니다.
