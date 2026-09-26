# Five-terminal circuit fixture

`circuit.json` is a fully wired schematic imported through **파일 → 가져오기**.
The test temporarily removes one capacitor wire, draws it through the ordinary UI,
and edits Rb before a real API calculation. No responses are intercepted and no
physical data are synthesized.

| Element | Setting |
| --- | --- |
| VD | DC 1 V |
| VG | −2 → −1.95 V pulse; period 20 µs |
| VBG | 0 ± 0.1 V sine; 50 kHz |
| Rb | 10 GΩ in fixture; edited to 20 GΩ in the live test |
| Cb | 10 fF between B and ground |
| Transient | 40 µs, maximum step 0.2 µs, deterministic BE |

The current Device 1 snapshot is preserved. Saved local-state settings are retained
for provenance; this fixture is simulated deterministically. `payload.json` is the
initial direct custom-circuit request. `X1.vbody` is the source-relative body-contact
voltage u; `X1.vb` is the distinct electrostatic body potential ψ_B.

Actual submitted request/response and reviewed screenshots are in
`web/review/body-terminals/live-{request,response}.json` and `circuit-*.png`.
The fully wired final browser document, including the 20 GΩ edit, is exported to
`examples/body-rc.stl-circuit.json`: open **파일 → 가져오기** and select that file.

To repeat, serve the production frontend with the actual API, then run from `web/`:

```sh
STUDIO_LIVE=1 E2E_PORT=8015 npx playwright test e2e/five-terminal-live.spec.ts
```

This verifies editor connectivity and the supported calculation path, not an
independent experimental calibration of back-gate coupling or body contacts.
