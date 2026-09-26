# Biristor Studio verification

This release separates numerical checks from browser and connection checks. The current Studio gates are in `web/e2e/studio.spec.ts`. On 2026-09-25, all **9 targeted Chromium gates passed in 21.5 seconds** against the production build. This file does not treat fixture-driven tests as proof of live computation.

## Test paths

- **Live HTTP:** browser → FastAPI job API → actual numerical worker. Release verification used the built frontend served by FastAPI; development reproduction can use the Vite proxy. VSCM reference and L = 400 nm; repeated log/linear switches after zoom; CSVM current source and drain capacitor; freely placed and wired 2 V / 1 kΩ circuit; STL load-line example.
- **Live remote authentication:** a second, password-protected FastAPI process on a different port, an empty result cache, and an exact allowed browser origin. The browser entered the URL and ephemeral password, authenticated, submitted a model job with its memory-only bearer token, and displayed the actual result. No route interception was used in this gate.
- **Browser data:** five saved devices, geometry persistence, reload/load, fixed measured versus calibrated IDVD reference.
- **Connection contract:** local/server selection, remote token authentication, endpoint switching, stale response rejection, and unavailable-server refusal. Routed responses in these tests check transport behavior only.
- **Visuals:** light desktop at 1440 px and phone at 390 px. Every result screenshot waits for populated Plotly data and rendered trace paths.

## Reproduction

Run the Python API from the repository root, then run the browser gates from `web/`:

```sh
STL_WORKERS=2 python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

```sh
STUDIO_LIVE=1 npx playwright test e2e/studio.spec.ts
```

Vite starts on the fixed port configured by Playwright and proxies `/api` to port 8000. A missing live API fails the opt-in live checks; without `STUDIO_LIVE=1`, those checks are explicitly skipped.

## Observed results

| Actual solver path | Observed result |
| --- | --- |
| Reference VSCM, L = 500 nm | VLU = 3.703688985 V; VLD = 2.597866939 V |
| Geometry VSCM, L = 400 nm | VLU = 3.609537559 V; VLD = 2.555622003 V |
| CSVM, Iin = 1 nA, Cdrain = 1 pF, 15 ms | UI: Vtop 3.742 V, Vbottom 2.586 V, f 860 Hz; startup excluded |
| Freely placed V source + resistor | V = 2 V; I(R) = 2 mA for R = 1 kΩ |
| STL load-line example | One latch-up; event VD = 3.703917241 V |
| Cross-origin authenticated reference computation | Reference folds reproduced; actual remote job, empty cache at startup |

Local gates allow the API's normal numerical-result cache after a prior real solve. The remote gate explicitly started with an empty disk cache and required a noncached completion. `web/review/studio/live-*.json` records job responses without credentials. The separately named connection-contract test uses an existing authentic recorded curve only to test response retirement and authentication headers; it is not part of the numerical evidence above.

The final screenshots in `web/review/studio/` contain populated real traces, correctly rendered bundled Korean fonts, closed transient notifications, and no horizontal page overflow at 390 px. The mobile navigation overlap found during review was corrected before the final capture.

The ninth gate is opt-in when a second real server is available. Its `STUDIO_REMOTE_URL` must match that server; `STUDIO_REMOTE_PASSWORD` is supplied through the environment. Set the second server's `STL_CORS_ORIGINS` to the exact frontend origin (for example, `http://127.0.0.1:5174` for the default development test runner). The release run used frontend/API port 8000 and protected API port 8001; all processes were stopped afterward.

## Scope

Passing software checks verifies the tested calculations and interface contracts. It does not establish geometry-model accuracy outside the calibrated reference data, full TCAD equivalence, commercial simulator compatibility, or operation on an unspecified laboratory server.
