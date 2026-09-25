# Basic circuit devices

MOSFET, diode and BJT elements participate in the same nonlinear MNA operating-point and
transient solve as STL, resistor, capacitor and source elements. Circuits do not need an STL.
These are educational compact models at 300 K. They omit parasitic capacitance, MOS body
diodes, junction breakdown, self-heating and process-specific effects. Add external capacitors
for circuit dynamics; these devices have no internal dynamic charge state.

## JSON elements

```json
{"type":"MOS","name":"M1","nodes":{"d":"out","g":"in","s":"0"},
 "model":{"polarity":"nmos","L_um":1,"W_um":10,"Vth_V":0.5,
          "SS_mV_dec":80,"k_uA_V2":100,"lambda_per_V":0.02}}
```

```json
{"type":"D","name":"D1","nodes":{"a":"in","k":"0"},
 "model":{"Is_A":1e-14,"n":1}}
```

```json
{"type":"BJT","name":"Q1","nodes":{"c":"out","b":"in","e":"0"},
 "model":{"polarity":"npn","Is_A":1e-15,"beta_F":100,"beta_R":1}}
```

Values shown are defaults. MOS polarity accepts `nmos` or `pmos`; BJT accepts `npn` or `pnp`.
`Vth_V` is a threshold **magnitude**; negative inputs are converted to their absolute value.
All pins are required. Ground is `0`; ordered node lists are also accepted in the displayed pin order.

| Parameter | Accepted range |
|---|---|
| L | 0.001–10,000 µm |
| W | 0.001–100,000 µm |
| Vth magnitude | 0–100 V |
| SS | 10–1,000 mV/dec |
| k | 0.001–1,000,000 µA/V² |
| λ | 0–10 V⁻¹ |
| Diode/BJT Is | 10⁻³⁰–1 A |
| Diode n | 0.1–10 |
| BJT βF, βR | 0.01–10⁶ |

These validation bounds avoid malformed input; they do not establish physical validity across the range.

## Models

For MOS, let σ = +1 for NMOS and −1 for PMOS and transform each terminal voltage as
v′ = σv. Set s = SS/ln(10), where SS is in V/dec, and

\[
f(x)=2s\ln[1+\exp(x/(2s))],\qquad K=kW/L.
\]

The drain current, positive into the drain, is

\[
I_D=\frac{\sigma K}{2}
\{f(v'_g-v'_s-|V_{th}|)^2-f(v'_g-v'_d-|V_{th}|)^2\}
(1+\lambda|v'_d-v'_s|).
\]

This smooth symmetric square-law model approaches square-law strong inversion and has the
specified subthreshold swing in weak inversion. Gate current is zero and source current is −ID.
For large positive/negative softplus arguments the code uses their numerically stable limits.

The diode uses I = Is[exp((Va−Vk)/(nVT))−1], VT = 0.025852 V.

The BJT uses reciprocal Ebers–Moll with **transport** saturation current Is. Let σ = +1 for NPN,
−1 for PNP, F = exp[σ(Vb−Ve)/VT]−1, R = exp[σ(Vb−Vc)/VT]−1,
αF = βF/(βF+1), αR = βR/(βR+1). Currents into the terminals are

\[
I_C=\sigma I_s(F-R/\alpha_R),\quad
I_E=\sigma I_s(R-F/\alpha_F),\quad I_B=-I_C-I_E.
\]

The diode/BJT exponential has a C¹ linear continuation above an argument of 40 to keep Newton
trial values finite. Use the models in their ordinary junction-bias range; this continuation is
a numerical safeguard, not a high-injection model. Analytic Jacobians are stamped into MNA;
large junction-voltage updates are damped during iteration.

## Probes and validation

`I(M1.d)`, `I(M1.g)`, `I(M1.s)` and `I(Q1.c)`, `I(Q1.b)`, `I(Q1.e)` are currents **into**
each terminal. `I(D1)` flows from anode to cathode. Node probes retain `V(node)` syntax.
The result echoes resolved parameters under each element's `model` key.

`server/tests/test_circuit_basic.py` checks L/W scaling, threshold influence, subthreshold
swing, BJT forward/reverse gain, and full nonlinear ramp transients with finite outputs and
KCL for both transistor polarities and a diode.
