# 소자 구동 · VSCM / CSVM

## VSCM — 전압 구동
드레인 전압을 올렸다 내리며 ID–VD, VLU, VLD, 히스테리시스 폭을 확인합니다.

## CSVM — 전류 구동 + Cdrain
DC 전류원 Iin이 드레인에 전류를 공급하고, Cdrain은 드레인–접지 사이에 연결됩니다. 게이트는 현재 VG를 사용합니다. 보정한 소자 파라미터와 광조사 조건을 그대로 적용해 회로와 소자 내부 상태를 함께 과도 해석합니다.

- **Iin (nA)**: 드레인에 공급하는 전류. 발진 가능한 전류 범위는 소자와 바이어스에 따라 달라집니다.
- **Cdrain (pF)**: 외부 드레인 커패시턴스.
- **Time (ms)**: 해석 시간. 관측 주기가 부족하면 늘려주세요.

시작값은 1 nA, 1 pF, 15 ms입니다. 모든 보정 소자나 게이트 조건에서 발진을 보장하는 값은 아닙니다.

### 결과 읽기
**Vtop / Vbottom**은 저장된 VD(t)의 완전한 주기마다 최댓값 / 최솟값을 찾은 뒤 평균한 값입니다. 시작 구간과 첫 주기는 제외합니다. **주파수**는 그 이후 완전한 주기가 2개 이상 있을 때만 `1 / 평균 주기`로 표시합니다. 발진이 없거나 시간이 부족하면 주파수는 `—`로 남습니다.

첫 피크가 높게 올라가는 오버슈트는 파형에 그대로 남습니다. 전류 문턱 이벤트가 첫 피크만 검출하거나 일부 반복을 놓쳐도 실제 전압 파형에서 완전한 반복 주기를 찾아 지표를 계산합니다. 원래 파형을 평활화하거나 피크를 잘라내지 않습니다. 주기 검출용 prominence만 초기 극값에 과도하게 좌우되지 않도록 정합니다.

아래 그래프의 **VB(t)**는 소스 기준 내부 정전기적 바디 전위입니다. 위의 VD(t)와 시간축을 공유합니다. 회로에서 외부 R·C를 연결하는 **B 접점 전압**은 hole quasi-Fermi/contact 전압 u로, 내부 정전기적 전위와 구분합니다. 회로에서는 `X1.vb`가 내부 정전기 전위, `X1.vbody`가 소스 기준 접점 전압입니다.

극값은 저장된 파형의 표본에 기반한 값입니다. 시간 간격보다 빠른 진동이나 불충분한 표본에서는 극값을 신뢰할 수 없습니다. 조건이 바뀌면 기존 값은 흐리게 표시되며, 다시 실행한 뒤 갱신됩니다.

확률적 모드에서는 소자 탭의 난수 시드, 캐리어 잡음, 국소 상태 설정으로 단일 과도 파형을 계산합니다. 표시는 이 파형 안의 주기 통계이며 여러 소자의 독립 실행 평균은 아닙니다. 전압 스윕 횟수·속도·간격은 CSVM에 적용되지 않습니다.

CSVM은 실제 과도 해석 서버 연결이 필요합니다. 오프라인 데모에는 물리적 CSVM 계산이 없습니다. 파형의 CSV / PNG는 그래프 메뉴에서 내보냅니다.

## English
VSCM applies a drain-voltage sweep. CSVM drives the current calibrated device with Iin and Cdrain while retaining VG and illumination. The existing circuit/body-state transient solver computes VD(t) and the source-relative electrostatic body potential VB(t). The latter is distinct from the ohmic body-contact voltage u. Startup overshoot remains in the unmodified plot; actual voltage cycles can be measured when current-threshold events miss later repetitions. Vtop and Vbottom are sampled waveform cycle extrema averaged after startup; frequency requires at least two complete retained cycles. Stochastic mode uses one trace and the current seed/noise/local-state settings. A live computation server is required.
