# deploy/local — password-protected local installer kit

Sources of the kit that lab members run on their own computers (Windows, macOS, Linux; Docker based). The kit is
built by `scripts/build_local_bundle.py`, which encrypts the app and copies these files into
`dist-local/STL-Simulator-Installer-<version>/` (+ `.zip`). User guide, troubleshooting and the maintainer steps:
**docs/LOCAL_INSTALL.md**.

```bash
python3 scripts/build_local_bundle.py --generate-password    # suggestion for a strong password
read -rs STL_BUNDLE_PASSWORD && export STL_BUNDLE_PASSWORD
python3 scripts/build_local_bundle.py --version 2026.10.01   # weak passwords need --allow-weak-password
unset STL_BUNDLE_PASSWORD
```

Before building a kit, make sure the GitHub repository is **private**: the encryption protects nothing while the source
is public. Share the ZIP by a restricted Drive link, NAS or USB (not e-mail: Gmail blocks ZIPs with .bat/.cmd/.ps1).

| File | Role |
|---|---|
| `install-windows.bat` | Windows entry point (ASCII; runs the PowerShell installer) |
| `install-mac.command`, `install-linux.sh` | macOS / Linux entry points (run `stl-local.sh install`) |
| `installer-files/install-windows.ps1` | Windows installer + control script (PowerShell 5.1, UTF-8 with BOM) |
| `installer-files/stl-pipe.cmd` | byte-exact cmd.exe pipe: decrypting container → `docker build -` |
| `installer-files/stl-local.sh` | macOS / Linux installer + control script (bash 3.2 compatible) |
| `installer-files/stl_payload.py` | generic STLLOC2 encrypt/decrypt helper (no application code, AES in-process) |
| `설치-안내.txt` | quick guide copied into the kit (`{VERSION}`, `{DATE}` are filled in) |

Never put a real password in these files, in commits or on a command line; the builder reads it only from
`STL_BUNDLE_PASSWORD`, the installers only from a hidden prompt (or `STL_INSTALL_PASSWORD` / stdin for tests) and
hand it to the decrypting container on stdin (macOS/Linux) or as `-e STL_PW` by name (Windows).
