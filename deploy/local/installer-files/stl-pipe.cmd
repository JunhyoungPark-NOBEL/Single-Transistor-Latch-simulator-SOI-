@echo off
rem STL Simulator local installer (Windows): byte-exact pipes for install-windows.ps1.
rem
rem Windows PowerShell 5.1 re-encodes data piped between native programs as text, which would corrupt the
rem encrypted payload and the decrypted tar stream. cmd.exe pipes and "<" redirection pass bytes unchanged,
rem so the two binary steps run here. The plaintext tar goes from the decrypting container's stdout straight
rem into "docker build -" and is never written to disk.
rem
rem All inputs are environment variables set by install-windows.ps1. The password is in STL_PW and is handed
rem to Docker by name ("-e STL_PW"); its value never appears on a command line. (It is not sent on stdin as on
rem macOS/Linux: "echo" would write it in the console code page, not UTF-8, and would parse & | < > ^ in it.)
rem   STL_PAYLOAD        payload file name (this folder)     STL_HELPER_IMAGE  python:3.11-slim
rem   STL_HELPER_B64     stl_payload.py, base64              STL_EXPECT_SHA256 SHA-256 of the decrypted tar
rem   STL_IMAGE          image tag to build                   STL_VERSION / STL_BUILD_ID  image labels
rem   STL_DECRYPT_LOG    file for the decrypting container's messages
rem   STL_PROGRESS_ARG   --progress=plain when buildx is available (else empty)
rem   STL_BUILD_EXTRA_ARGS  optional extra "docker build" arguments
rem
rem Usage: stl-pipe.cmd verify   exit 0 = password ok, 3 = wrong password, 4 = damaged payload
rem        stl-pipe.cmd build    exit code of docker build
setlocal
if /i "%~1"=="verify" goto verify
if /i "%~1"=="build" goto build
echo usage: stl-pipe.cmd verify^|build 1>&2
exit /b 2

:verify
docker run --rm -i --network none -e STL_PW -e STL_HELPER_B64 -e STL_EXPECT_SHA256 %STL_HELPER_IMAGE% python3 -c "import os,base64;exec(base64.b64decode(os.environ['STL_HELPER_B64']))" verify < "%STL_PAYLOAD%"
exit /b %errorlevel%

:build
docker run --rm -i --network none -e STL_PW -e STL_HELPER_B64 -e STL_EXPECT_SHA256 %STL_HELPER_IMAGE% python3 -c "import os,base64;exec(base64.b64decode(os.environ['STL_HELPER_B64']))" decrypt < "%STL_PAYLOAD%" 2> "%STL_DECRYPT_LOG%" | docker build %STL_PROGRESS_ARG% --label org.stl-simulator=1 --label "org.stl-simulator.version=%STL_VERSION%" --label "org.stl-simulator.build=%STL_BUILD_ID%" %STL_BUILD_EXTRA_ARGS% -t "%STL_IMAGE%" - 2>&1
exit /b %errorlevel%
