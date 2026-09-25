#!/usr/bin/env python3
"""Build the password-protected local installer kit (docs/LOCAL_INSTALL.md, maintainer section).

    read -rs STL_BUNDLE_PASSWORD && export STL_BUNDLE_PASSWORD      # typed, not shown, not in the shell history
    python3 scripts/build_local_bundle.py [--version 2026.09.25] [--ref HEAD] [--allow-weak-password]
    unset STL_BUNDLE_PASSWORD
    python3 scripts/build_local_bundle.py --generate-password     # print a strong passphrase suggestion and exit

Output: dist-local/STL-Simulator-Installer-<version>/ and dist-local/STL-Simulator-Installer-<version>.zip:

    install-windows.bat  install-mac.command  install-linux.sh  설치-안내.txt
    installer-files/     install-windows.ps1  stl-pipe.cmd  stl-local.sh  stl_payload.py
                         stl-simulator.stlenc (the encrypted app)  kit-info.txt

The app payload is `git archive` of <ref> restricted to what the Docker build needs (PAYLOAD_PATHS; tests, e2e
screenshots and docs are left out), gzip-compressed without timestamps, then encrypted with
deploy/local/installer-files/stl_payload.py (format STLLOC2: PBKDF2-HMAC-SHA256 + AES-256-CTR + HMAC-SHA256).  The
installers decrypt it inside a python:3.11-slim container and stream it straight into `docker build -`.

Requirements: git, Python >= 3.9 (stdlib only).  AES runs in this process (libcrypto through ctypes on Linux, else
a built-in AES, about 1 s per MB); no key or password is ever passed to another program on its command line.  The
password is read ONLY from the environment variable STL_BUNDLE_PASSWORD; it is never printed or written.  Weak
passwords (shorter than 16 characters, containing a name from the kit such as the lab or the simulator, or one word
plus digits) are refused unless --allow-weak-password: anyone who has the kit can test guesses offline.  Checks
before the kit is written: the Dockerfile's COPY sources are in the payload, the payload decrypts back to exactly the
archive (SHA-256), no plaintext kit file contains model strings, the commit hash or the password, the shell scripts
put no $VARIABLE directly before non-ASCII text (bash 3.2 on macOS would misread it), and (when Docker is
available, unless --no-docker-check) the payload decrypts inside python:3.11-slim exactly as the installers do it
(password on stdin) and a wrong password is refused.

The kit names the build by an opaque BUILD_ID (not the commit hash); the builder appends "BUILD_ID commit version"
to <out>/kit-builds.txt (next to the kit, not inside it) so the maintainer can map it back.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
KIT_SRC = ROOT / "deploy" / "local"
HELPER_IMAGE = "python:3.11-slim"
PAYLOAD_NAME = "stl-simulator.stlenc"
KIT_PREFIX = "STL-Simulator-Installer-"
GUIDE_NAME = "설치-안내.txt"

# What the Docker build needs (keep in sync with the Dockerfile's COPY lines; checked below).
PAYLOAD_PATHS = ["Dockerfile", ".dockerignore", "engine", "server", "scripts", "web"]
PAYLOAD_EXCLUDE = ["server/tests", "web/e2e", "scripts/build_local_bundle.py"]

# Kit files: (source relative to deploy/local, destination relative to the kit folder, kind)
#   kind: "unix" = LF + executable, "win" = CRLF, "ps1" = CRLF + UTF-8 BOM, "py" = LF
KIT_FILES = [
    ("install-windows.bat", "install-windows.bat", "win"),
    ("install-mac.command", "install-mac.command", "unix"),
    ("install-linux.sh", "install-linux.sh", "unix"),
    ("installer-files/install-windows.ps1", "installer-files/install-windows.ps1", "ps1"),
    ("installer-files/stl-pipe.cmd", "installer-files/stl-pipe.cmd", "win"),
    ("installer-files/stl-local.sh", "installer-files/stl-local.sh", "unix"),
    ("installer-files/stl_payload.py", "installer-files/stl_payload.py", "py"),
]
ASCII_ONLY = {"install-windows.bat", "installer-files/stl-pipe.cmd"}

# Strings that must never appear in a plaintext kit file (in addition to parameter names/values read from the
# model files of the payload itself).  Matched case-insensitively.
FORBIDDEN_FIXED = ["GIDL", "calib", "impact", "V_LU", "V_LD", "3.7037", "2.5979", "0.2794", "BTBT", "avalanche",
                   "MODEL_PARAMETERS", "parameter_vector", "photo_extension", "janus"]

# Words an attacker who holds the kit tries first (lab, institute, program names).  Matched case-insensitively.
GUESSABLE_WORDS = ["nobel", "kaist", "stl", "simulator", "latch", "biristor", "transistor", "lab", "soi",
                   "password", "passwd", "qwerty", "admin", "1234", "abcd"]
MIN_STRONG_LENGTH = 16
# Passphrase generator: consonant-vowel syllables (no "l": no generated word can spell a GUESSABLE_WORDS entry).
_CONSONANTS = "bdfghjkmnprstvz"
_VOWELS = "aeiou"


def die(msg: str, code: int = 1) -> NoReturn:
    print(f"build_local_bundle: error: {msg}", file=sys.stderr)
    sys.exit(code)


def info(msg: str) -> None:
    print(f"[kit] {msg}", flush=True)


def load_helper():
    sys.dont_write_bytecode = True  # keep deploy/local free of __pycache__
    spec = importlib.util.spec_from_file_location("stl_payload", KIT_SRC / "installer-files" / "stl_payload.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def git(repo: Path, *args: str, binary: bool = False):
    res = subprocess.run(["git", "-C", str(repo), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        die(f"git {' '.join(args)} failed: {res.stderr.decode(errors='replace').strip()}")
    return res.stdout if binary else res.stdout.decode().strip()


def generate_password(words: int = 3, syllables: int = 3) -> str:
    """Three made-up 6-letter words joined by '-' (9 random syllables of 75 = 56 bits); easy to type, IME-free ASCII."""
    while True:
        pw = "-".join("".join(secrets.choice(_CONSONANTS) + secrets.choice(_VOWELS) for _ in range(syllables))
                      for _ in range(words))
        if not weak_reasons(pw):
            return pw


def weak_reasons(pw: str) -> list[str]:
    reasons = []
    if len(pw) < MIN_STRONG_LENGTH:
        reasons.append(f"shorter than {MIN_STRONG_LENGTH} characters")
    low = pw.lower()
    hits = [w for w in GUESSABLE_WORDS if w in low]
    if hits:
        reasons.append("contains a name or word an attacker tries first: " + ", ".join(hits))
    if re.fullmatch(r"[^\W\d_]+[\W_]?\d*[\W_]{0,2}", pw) or re.fullmatch(r"[\d\W_]+", pw):
        reasons.append("one word plus digits/symbols (the first pattern a password guesser tries)")
    return reasons


def guess_seconds(iterations: int, helper) -> float:
    """Time of one password check on this computer's CPU (one core), measured."""
    t = time.perf_counter()
    helper.derive_keys("x" * 12, b"\0" * 16, iterations)
    return time.perf_counter() - t


def read_password(allow_weak: bool, iterations: int, helper) -> str:
    pw = os.environ.get("STL_BUNDLE_PASSWORD")
    if pw is None or pw == "":
        die("set the password in the environment variable STL_BUNDLE_PASSWORD (it is never taken from the "
            "command line):  read -rs STL_BUNDLE_PASSWORD && export STL_BUNDLE_PASSWORD", 2)
    if pw != pw.strip():
        die("the password in STL_BUNDLE_PASSWORD has leading or trailing whitespace; remove it", 2)
    if len(pw) < 8:
        die("the password must have at least 8 characters", 2)
    reasons = weak_reasons(pw)
    if reasons:
        per_guess = guess_seconds(iterations, helper)
        pattern = any(not r.startswith("shorter") for r in reasons)
        lines = [f"weak password: {'; '.join(reasons)}.",
                 "Anyone who has the kit ZIP (forwarded mail, a copied USB stick) can test passwords offline, without "
                 "limit and without anyone noticing.",
                 f"One guess costs {per_guess:.2f} s on one CPU core here; a PC tries ~{8 / per_guess:.0f} per second, "
                 "a gaming GPU thousands per second."]
        if pattern:
            lines.append("A lab/program name plus a few digits is among the first ~100,000 guesses: found in minutes "
                         "on one PC, well under a second on a GPU. It stops casual access only.")
        lines.append("약한 비밀번호입니다: 키트(ZIP)를 가진 사람은 비밀번호를 오프라인으로 무제한 추측할 수 있습니다."
                     + (" 연구실·프로그램 이름 + 숫자 형태는 PC 한 대로 몇 분 안에 풀립니다." if pattern else ""))
        lines.append("Stronger: python3 scripts/build_local_bundle.py --generate-password  (e.g. three made-up words "
                     "of 6 letters; share it by messenger, not together with the ZIP).")
        if not allow_weak:
            die("\n  ".join(lines + ["Refused. To build with this password anyway (you accept the risk above), add "
                                      "--allow-weak-password."]), 2)
        for line in lines:
            info("WARNING: " + line)
        info("WARNING: building with a weak password because --allow-weak-password was given.")
    if not pw.isascii():
        info("warning: the password has non-ASCII characters; typing them in a hidden password prompt can be "
             "awkward (input method editors). ASCII is recommended.")
    return pw


def make_archive(repo: Path, commit: str) -> bytes:
    specs = PAYLOAD_PATHS + [f":(exclude){p}" for p in PAYLOAD_EXCLUDE]
    tar = git(repo, "archive", "--format=tar", commit, "--", *specs, binary=True)
    return gzip.compress(tar, compresslevel=9, mtime=0)


def archive_members(archive: bytes) -> dict[str, bytes | None]:
    out: dict[str, bytes | None] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
        for m in tf.getmembers():
            name = m.name.rstrip("/")
            out[name] = tf.extractfile(m).read() if m.isfile() else None  # type: ignore[union-attr]
    return out


def check_dockerfile_sources(members: dict[str, bytes | None]) -> None:
    df = members.get("Dockerfile")
    if not df:
        die("the payload has no Dockerfile")
    names = set(members)
    missing = []
    for line in df.decode().splitlines():
        parts = line.split()
        if not parts or parts[0].upper() not in ("COPY", "ADD"):
            continue
        args = [p for p in parts[1:] if not p.startswith("--")]
        if any(p.startswith("--from") for p in parts[1:]) or len(args) < 2:
            continue
        for src in args[:-1]:
            s = src.rstrip("/")
            if s not in names and not any(n.startswith(s + "/") for n in names):
                missing.append(src)
    if missing:
        die(f"Dockerfile COPY sources missing from the payload: {missing} (update PAYLOAD_PATHS)")
    for need in ("server/main.py", "web/package.json", "scripts/warmup.py"):
        if need not in names:
            die(f"{need} is missing from the payload")


def forbidden_strings(members: dict[str, bytes | None], password: str, helper) -> list[bytes]:
    words = set(FORBIDDEN_FIXED)

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if len(k) >= 6 and ("_" in k or len(k) >= 10):
                    words.add(k)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, float):
            s = repr(x)
            if len(s) >= 8:
                words.add(s[:8])
        elif isinstance(x, str) and len(x) >= 12:
            words.add(x[:24])

    for name, data in members.items():
        if data is None:
            continue
        if name.startswith("engine/") and name.endswith(".json") and ("param" in name.lower()):
            try:
                walk(json.loads(data))
            except ValueError:
                pass
        if name.startswith("engine/") and name.endswith(".py"):
            stem = Path(name).stem
            if (len(stem) >= 7 and "_" in stem) or len(stem) >= 9:
                words.add(stem)
    out = [w.lower().encode() for w in words if w.strip()]
    pw = {password.encode(), helper.normalize_password(password)}
    return out + [p for p in pw if p]


def leak_check(files: dict[str, bytes], forbidden: list[bytes], password_bytes: set[bytes],
               commit: str = "") -> None:
    problems = []
    for name, data in files.items():
        low = data.lower()
        for p in password_bytes:
            if p in data:
                problems.append(f"{name}: contains the password")
        if name.endswith(".stlenc"):
            continue
        for w in forbidden:
            if w in low and w not in {p.lower() for p in password_bytes}:
                problems.append(f"{name}: contains a model string ({w.decode(errors='replace')!r})")
        if commit and (commit.encode() in low or commit[:7].encode() in low):
            problems.append(f"{name}: contains the commit hash (it points to the source repository)")
    if problems:
        die("leak check failed:\n  " + "\n  ".join(sorted(set(problems))))


def lint_scripts(files: dict[str, bytes]) -> None:
    """bash 3.2 (macOS) reads bytes >= 0x80 after $NAME as part of the name in UTF-8 locales: "$N개" prints garbage.
    Every variable directly followed by non-ASCII text must be written ${N}."""
    bad = []
    pat = re.compile(rb"\$[A-Za-z_][A-Za-z0-9_]*[\x80-\xff]")
    for name, data in files.items():
        if not name.endswith((".sh", ".command")):
            continue
        for no, line in enumerate(data.split(b"\n"), 1):
            for m in pat.finditer(line):
                var = m.group()[:-1].decode()
                bad.append(f"{name}:{no}: {var} -> ${{{var[1:]}}}   ({line.decode(errors='replace').strip()[:90]})")
    if bad:
        die("write ${NAME} before non-ASCII text (bash 3.2 on macOS):\n  " + "\n  ".join(bad))


def convert(data: bytes, kind: str, name: str) -> bytes:
    text = data.decode("utf-8-sig")
    if name in ASCII_ONLY and not text.isascii():
        die(f"{name} must be ASCII-only")
    text = text.replace("\r\n", "\n")
    if kind in ("win", "ps1"):
        text = text.replace("\n", "\r\n")
    out = text.encode("utf-8")
    return b"\xef\xbb\xbf" + out if kind in ("ps1", "txt") else out


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    res = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    return res.returncode == 0


def docker_check(payload: bytes, helper_src: bytes, password: str, expect_sha: str) -> None:
    """Decrypt in python:3.11-slim exactly as the macOS/Linux installers do: the password is the first line of the
    container's stdin (never argv, never in the container's environment, where `docker inspect` would show it)."""
    import base64
    boot = "import os,base64;exec(base64.b64decode(os.environ['STL_HELPER_B64']))"
    env = dict(os.environ, STL_HELPER_B64=base64.b64encode(helper_src).decode(), STL_EXPECT_SHA256=expect_sha)
    env.pop("STL_BUNDLE_PASSWORD", None)
    base = ["docker", "run", "--rm", "-i", "--network", "none", "-e", "STL_PW_STDIN=1", "-e", "STL_HELPER_B64",
            "-e", "STL_EXPECT_SHA256", HELPER_IMAGE, "python3", "-c", boot]
    good = subprocess.run(base + ["decrypt"], input=password.encode() + b"\n" + payload, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=env)
    if good.returncode != 0 or hashlib.sha256(good.stdout).hexdigest() != expect_sha:
        die(f"docker decrypt check failed (exit {good.returncode}): {good.stderr.decode(errors='replace')[-400:]}")
    bad = subprocess.run(base + ["verify"], input=password.encode() + b"-wrong\n" + payload, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, env=env)
    if bad.returncode != 3 or bad.stdout:
        die(f"docker wrong-password check failed (exit {bad.returncode}, expected 3)")
    info(f"docker check: {HELPER_IMAGE} decrypts the payload to the archive; a wrong password is refused (exit 3)")


def write_zip(kit_dir: Path, zip_path: Path, when: tuple[int, ...]) -> None:
    tmp = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        top = kit_dir.name + "/"
        d = zipfile.ZipInfo(top, when)
        d.create_system, d.external_attr = 3, (0o40755 << 16) | 0x10
        zf.writestr(d, b"")
        for path in sorted(kit_dir.rglob("*")):
            rel = path.relative_to(kit_dir).as_posix()
            if path.is_dir():
                zi = zipfile.ZipInfo(top + rel + "/", when)
                zi.create_system, zi.external_attr = 3, (0o40755 << 16) | 0x10
                zf.writestr(zi, b"")
                continue
            zi = zipfile.ZipInfo(top + rel, when)
            zi.create_system = 3
            mode = 0o755 if rel.endswith((".sh", ".command")) else 0o644
            zi.external_attr = (0o100000 | mode) << 16
            zi.compress_type = zipfile.ZIP_STORED if rel.endswith(".stlenc") else zipfile.ZIP_DEFLATED
            zf.writestr(zi, path.read_bytes())
    tmp.replace(zip_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", help="kit version (default: <commit date YYYY.MM.DD>-<first 7 characters of the build id>)")
    ap.add_argument("--ref", default="HEAD", help="git commit/branch/tag to package (default HEAD)")
    ap.add_argument("--repo", type=Path, default=ROOT, help="git repository to package (default: this one)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <repo>/dist-local)")
    ap.add_argument("--iterations", type=int, default=None, help="PBKDF2 iterations (default 1000000)")
    ap.add_argument("--no-zip", action="store_true", help="write only the folder")
    ap.add_argument("--no-docker-check", action="store_true", help="skip the python:3.11-slim decrypt check")
    ap.add_argument("--allow-weak-password", action="store_true",
                    help="build even though the password is guessable (short, a kit name, or one word + digits)")
    ap.add_argument("--generate-password", action="store_true",
                    help="print a strong passphrase suggestion (share it by messenger) and exit; builds nothing")
    args = ap.parse_args()

    if args.generate_password:
        print(generate_password())
        return 0
    helper = load_helper()
    iterations = args.iterations or helper.DEFAULT_ITERATIONS
    password = read_password(args.allow_weak_password, iterations, helper)
    repo = args.repo.resolve()
    commit = git(repo, "rev-parse", "--verify", f"{args.ref}^{{commit}}")
    commit_date = git(repo, "show", "-s", "--format=%cI", commit)
    cdate = dt.datetime.fromisoformat(commit_date).astimezone(dt.timezone.utc)
    build_id = hashlib.sha256(b"stl-kit-build:" + commit.encode()).hexdigest()[:12]
    version = args.version or f"{cdate:%Y.%m.%d}-{build_id[:7]}"
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._-]{0,48}", version):
        die("--version may use letters, digits, '.', '_' and '-' (max 49 characters, not starting with . or -)")
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        info(f"note: the working tree has uncommitted changes; they are NOT in the kit (it packages {args.ref} = "
             f"{commit[:12]})")

    archive = make_archive(repo, commit)
    members = archive_members(archive)
    check_dockerfile_sources(members)
    archive_sha = hashlib.sha256(archive).hexdigest()
    info(f"payload: git archive of {commit[:12]} ({len(members)} entries, {len(archive) / 1e6:.1f} MB gzip)")

    payload = helper.encrypt(archive, password, iterations)
    if hashlib.sha256(helper.decrypt(payload, password)).hexdigest() != archive_sha:
        die("round trip failed: the payload does not decrypt back to the archive")
    try:
        helper.open_payload(payload, password + "x")
        die("a wrong password was accepted")
    except helper.WrongPassword:
        pass
    info(f"encrypted: {len(payload) / 1e6:.1f} MB, PBKDF2 {iterations} iterations; round trip SHA-256 ok")

    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    files: dict[str, bytes] = {}
    for src, dst, kind in KIT_FILES:
        p = KIT_SRC / src
        if not p.is_file():
            die(f"missing kit source {p}")
        files[dst] = convert(p.read_bytes(), kind, dst)
    guide = (KIT_SRC / GUIDE_NAME).read_text(encoding="utf-8-sig")
    guide = guide.replace("{VERSION}", version).replace("{DATE}", cdate.strftime("%Y-%m-%d"))
    files[GUIDE_NAME] = convert(guide.encode(), "txt", GUIDE_NAME).replace(b"\n", b"\r\n")
    files[f"installer-files/{PAYLOAD_NAME}"] = payload
    files["installer-files/kit-info.txt"] = "".join(f"{k}={v}\n" for k, v in [
        ("FORMAT", helper.MAGIC.rstrip(b"\0").decode()), ("VERSION", version), ("BUILD_ID", build_id),
        ("COMMIT_DATE", commit_date),
        ("BUILT", built), ("PAYLOAD", PAYLOAD_NAME), ("ARCHIVE_SHA256", archive_sha),
        ("ARCHIVE_BYTES", len(archive)), ("PAYLOAD_SHA256", hashlib.sha256(payload).hexdigest()),
        ("HELPER_IMAGE", HELPER_IMAGE)]).encode()

    forbidden = forbidden_strings(members, password, helper)
    pw_bytes = {password.encode(), helper.normalize_password(password)}
    leak_check(files, [w for w in forbidden if w not in pw_bytes], pw_bytes, commit)
    info(f"leak check: {len(files)} kit files, {len(forbidden)} model strings, the commit hash and the password: "
         "none found")
    lint_scripts(files)

    if not args.no_docker_check:
        if docker_available():
            docker_check(payload, files["installer-files/stl_payload.py"], password, archive_sha)
        else:
            info("docker check skipped (Docker is not available here)")

    out = (args.out or (repo / "dist-local")).resolve()
    kit_dir = out / f"{KIT_PREFIX}{version}"
    if kit_dir.exists():
        if not (kit_dir / "installer-files" / "kit-info.txt").is_file():
            die(f"{kit_dir} exists and is not a kit folder; remove it first")
        shutil.rmtree(kit_dir)
    for rel, data in files.items():
        p = kit_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        p.chmod(0o755 if rel.endswith((".sh", ".command")) else 0o644)
    info(f"kit folder: {kit_dir}")

    if not args.no_zip:
        zip_path = out / f"{KIT_PREFIX}{version}.zip"
        write_zip(kit_dir, zip_path, cdate.timetuple()[:6])
        with zipfile.ZipFile(zip_path) as zf:
            zfiles = {n.split("/", 1)[1]: zf.read(n) for n in zf.namelist() if not n.endswith("/")}
        if {k: hashlib.sha256(v).hexdigest() for k, v in zfiles.items()} != \
                {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}:
            die("the zip does not match the kit folder")
        leak_check(dict(zfiles, **{"(zip file)": zip_path.read_bytes()}), [], pw_bytes)
        size = zip_path.stat().st_size / 1e6
        info(f"zip: {zip_path} ({size:.1f} MB){'  WARNING: above 25 MB' if size > 25 else ''}")
    with open(out / "kit-builds.txt", "a", encoding="utf-8") as fh:
        fh.write(f"{build_id} {commit} {version} {built}\n")
    info(f"version {version}  build id {build_id} = commit {commit[:12]} (recorded in {out / 'kit-builds.txt'}, "
         f"not in the kit)  archive sha256 {archive_sha[:16]}…")
    info("Share the ZIP as a Google Drive link restricted to lab members, on a NAS or on USB, and send the password "
         "separately (messenger or in person). Do not e-mail it: Gmail and many university mail servers reject ZIPs "
         "that contain .bat/.cmd/.ps1 files.")
    info("배포: ZIP은 연구실 구성원만 볼 수 있게 제한한 Google Drive 링크·NAS·USB로, 비밀번호는 메신저 등 다른 경로로 "
         "따로 전달하세요. Gmail 등 메일은 .bat/.cmd/.ps1이 든 ZIP을 차단합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
