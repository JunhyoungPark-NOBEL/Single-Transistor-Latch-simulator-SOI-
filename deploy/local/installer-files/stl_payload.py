#!/usr/bin/env python3
"""Password-encrypted payload helper for the STL Simulator local installer kit.

This file is generic: it holds no application code and no key.  The installers run it inside a throwaway
container of the official python:3.11-slim image, so the computer needs no crypto tools of its own:

    { printf '%s\\n' "$PW"; cat payload; } | docker run --rm -i --network none -e STL_PW_STDIN=1 -e STL_HELPER_B64 \\
        python:3.11-slim python3 -c "import os,base64;exec(base64.b64decode(os.environ['STL_HELPER_B64']))" MODE

(STL_HELPER_B64 = this file, base64-encoded.)  The password is the first line of stdin (STL_PW_STDIN=1; macOS,
Linux and the kit builder) or the environment variable STL_PW passed by name only (-e STL_PW; Windows, whose cmd.exe
cannot put a Unicode password on a pipe unchanged).  It is never a command-line argument, and no key derived from it
is ever handed to another program: AES runs in this process (libcrypto through ctypes, else a built-in AES).
scripts/build_local_bundle.py imports this file to encrypt.

Payload format "STLLOC2" (integers big-endian):
    magic       b"STLLOC2\\0"                       8 bytes
    salt                                            16 bytes
    iterations  PBKDF2 iteration count (>= 600000)   4 bytes
    iv          AES-CTR initial counter block        16 bytes
    ciphertext  AES-256-CTR                          n bytes
    tag         HMAC-SHA256 over all bytes above     32 bytes   (encrypt-then-MAC)
Keys: master = PBKDF2-HMAC-SHA256(password, salt, iterations, 32 bytes); AES key = HMAC-SHA256(master, "STL enc"),
MAC key = HMAC-SHA256(master, "STL mac").  (One 32-byte PBKDF2 block: an attacker who tests guesses against the
known gzip header needs the same work per guess as the installer; STLLOC1 derived 64 bytes = two blocks, of which
a guess check needed only the first.)  password = UTF-8 of the Unicode-NFC form, surrounding whitespace removed.

Modes (payload or plaintext on stdin after the password line, or password in STL_PW):
    verify    exit 0 when the password is right, 3 when it is not (nothing is written)
    decrypt   check the tag (and the SHA-256 in STL_EXPECT_SHA256, when set), then write the plaintext to stdout;
              nothing is written unless every check passed
    encrypt   write a new payload for the plaintext (iterations: STL_PBKDF2_ITERATIONS, default 2000000)
    selftest  AES known-answer test (NIST SP 800-38A F.5.5) for libcrypto and the built-in AES (no password needed)
Exit codes: 0 ok, 2 usage or missing password, 3 wrong password, 4 not a payload or damaged, 5 decryption failed.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import struct
import sys
import unicodedata

MAGIC = b"STLLOC2\x00"
SALT_LEN = 16
IV_LEN = 16
TAG_LEN = 32
HEADER_LEN = len(MAGIC) + SALT_LEN + 4 + IV_LEN
MIN_ITERATIONS = 600_000
MAX_ITERATIONS = 50_000_000
DEFAULT_ITERATIONS = 2_000_000
MAX_PASSWORD_LINE = 4096

EXIT_OK, EXIT_USAGE, EXIT_WRONG_PASSWORD, EXIT_BAD_PAYLOAD, EXIT_CIPHER = 0, 2, 3, 4, 5

# NIST SP 800-38A, F.5.5 CTR-AES256.Encrypt (public test vector)
KAT_KEY = bytes.fromhex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")
KAT_IV = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")
KAT_PT = bytes.fromhex("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51"
                       "30c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710")
KAT_CT = bytes.fromhex("601ec313775789a5b7a7f504bbf3d228f443e3ca4d62b59aca84e990cacaf5c5"
                       "2b0930daa23de94ce87017ba2d84988ddfc9c58db67aada613c2dd08457941a6")


class WrongPassword(Exception):
    pass


class PayloadError(Exception):
    pass


class CipherError(Exception):
    pass


# ----------------------------------------------------------------------------------------------- keys
def normalize_password(password: str) -> bytes:
    return unicodedata.normalize("NFC", password.strip()).encode("utf-8", "surrogateescape")


def derive_keys(password: str, salt: bytes, iterations: int) -> tuple[bytes, bytes]:
    master = hashlib.pbkdf2_hmac("sha256", normalize_password(password), salt, iterations, dklen=32)
    return (hmac.new(master, b"STL enc", hashlib.sha256).digest(),
            hmac.new(master, b"STL mac", hashlib.sha256).digest())


# ------------------------------------------------------------------------------------ AES-256-CTR
# In-process only: the key is never passed to another program (an `openssl enc -K <key>` command line would be
# readable by every user of the machine through /proc or `ps`, container processes included).
_LIBCRYPTO = None  # None = not tried yet, False = unavailable


def _ctr_with_libcrypto(lib, key: bytes, iv: bytes, data: bytes) -> bytes:
    import ctypes
    ctx = lib.EVP_CIPHER_CTX_new()
    if not ctx:
        raise CipherError("EVP_CIPHER_CTX_new failed")
    try:
        if lib.EVP_EncryptInit_ex(ctx, lib.EVP_aes_256_ctr(), None, key, iv) != 1:
            raise CipherError("EVP_EncryptInit_ex failed")
        out = bytearray()
        chunk = 1 << 20
        buf = ctypes.create_string_buffer(chunk + 32)
        outl = ctypes.c_int(0)
        for pos in range(0, len(data), chunk):
            piece = data[pos:pos + chunk]
            if lib.EVP_EncryptUpdate(ctx, buf, ctypes.byref(outl), piece, len(piece)) != 1 or outl.value != len(piece):
                raise CipherError("EVP_EncryptUpdate failed")
            out += buf.raw[:outl.value]
        return bytes(out)
    finally:
        lib.EVP_CIPHER_CTX_free(ctx)


def _libcrypto():
    """OpenSSL's libcrypto through ctypes (Linux; python:3.11-slim ships libcrypto.so.3), after a known-answer test.
    Not on macOS (loading the unversioned system libcrypto aborts the process) or Windows: built-in AES there."""
    global _LIBCRYPTO
    if _LIBCRYPTO is not None:
        return _LIBCRYPTO or None
    _LIBCRYPTO = False
    if os.environ.get("STL_PAYLOAD_PURE_PYTHON") == "1" or sys.platform == "darwin" or os.name == "nt":
        return None
    try:
        import ctypes
        import ctypes.util
    except ImportError:
        return None
    names = ["libcrypto.so.3", "libcrypto.so.1.1"]
    try:
        found = ctypes.util.find_library("crypto")
    except Exception:  # noqa: BLE001 - find_library runs helper programs; any failure just means "not found"
        found = None
    if found and found not in names:
        names.append(found)
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            vp, ip = ctypes.c_void_p, ctypes.c_int
            lib.EVP_CIPHER_CTX_new.restype, lib.EVP_CIPHER_CTX_new.argtypes = vp, []
            lib.EVP_aes_256_ctr.restype, lib.EVP_aes_256_ctr.argtypes = vp, []
            lib.EVP_EncryptInit_ex.restype = ip
            lib.EVP_EncryptInit_ex.argtypes = [vp, vp, vp, ctypes.c_char_p, ctypes.c_char_p]
            lib.EVP_EncryptUpdate.restype = ip
            lib.EVP_EncryptUpdate.argtypes = [vp, vp, ctypes.POINTER(ip), ctypes.c_char_p, ip]
            lib.EVP_CIPHER_CTX_free.restype, lib.EVP_CIPHER_CTX_free.argtypes = None, [vp]
            if _ctr_with_libcrypto(lib, KAT_KEY, KAT_IV, KAT_PT) == KAT_CT:
                _LIBCRYPTO = lib
                return lib
        except (OSError, AttributeError, CipherError, ValueError, TypeError):
            continue
    return None


def _build_tables():
    exp, log = [0] * 256, [0] * 256
    x = 1
    for i in range(255):
        exp[i], log[x] = x, i
        x ^= ((x << 1) ^ (0x1B if x & 0x80 else 0)) & 0xFF  # x *= 3 in GF(2^8)
    sbox = [0] * 256
    for i in range(256):
        inv = 0 if i == 0 else exp[(255 - log[i]) % 255]
        s = r = inv
        for _ in range(4):
            r = ((r << 1) | (r >> 7)) & 0xFF
            s ^= r
        sbox[i] = s ^ 0x63
    te0 = []
    for s in sbox:
        s2 = ((s << 1) ^ (0x1B if s & 0x80 else 0)) & 0xFF
        te0.append((s2 << 24) | (s << 16) | (s << 8) | (s2 ^ s))
    ror = lambda w, n: ((w >> n) | (w << (32 - n))) & 0xFFFFFFFF  # noqa: E731
    return sbox, te0, [ror(w, 8) for w in te0], [ror(w, 16) for w in te0], [ror(w, 24) for w in te0]


_TABLES = None


def _expand_key(key: bytes, sbox) -> list[int]:
    w = list(struct.unpack(">8I", key))
    rcon = 1
    for i in range(8, 60):
        t = w[i - 1]
        if i % 8 == 0:
            t = ((t << 8) | (t >> 24)) & 0xFFFFFFFF
            t = (sbox[t >> 24] << 24) | (sbox[(t >> 16) & 255] << 16) | (sbox[(t >> 8) & 255] << 8) | sbox[t & 255]
            t ^= rcon << 24
            rcon = ((rcon << 1) ^ (0x1B if rcon & 0x80 else 0)) & 0xFF
        elif i % 8 == 4:
            t = (sbox[t >> 24] << 24) | (sbox[(t >> 16) & 255] << 16) | (sbox[(t >> 8) & 255] << 8) | sbox[t & 255]
        w.append(w[i - 8] ^ t)
    return w


def _ctr_python(key: bytes, iv: bytes, data: bytes) -> bytes:
    global _TABLES
    if _TABLES is None:
        _TABLES = _build_tables()
    S, T0, T1, T2, T3 = _TABLES
    rk = _expand_key(key, S)
    counter = int.from_bytes(iv, "big")
    out = bytearray()
    chunk = 1 << 16
    for pos in range(0, len(data), chunk):
        piece = data[pos:pos + chunk]
        stream = []
        for _ in range((len(piece) + 15) // 16):
            c = counter
            counter = (counter + 1) & ((1 << 128) - 1)
            s0 = ((c >> 96) & 0xFFFFFFFF) ^ rk[0]
            s1 = ((c >> 64) & 0xFFFFFFFF) ^ rk[1]
            s2 = ((c >> 32) & 0xFFFFFFFF) ^ rk[2]
            s3 = (c & 0xFFFFFFFF) ^ rk[3]
            k = 4
            for _r in range(13):
                t0 = T0[s0 >> 24] ^ T1[(s1 >> 16) & 255] ^ T2[(s2 >> 8) & 255] ^ T3[s3 & 255] ^ rk[k]
                t1 = T0[s1 >> 24] ^ T1[(s2 >> 16) & 255] ^ T2[(s3 >> 8) & 255] ^ T3[s0 & 255] ^ rk[k + 1]
                t2 = T0[s2 >> 24] ^ T1[(s3 >> 16) & 255] ^ T2[(s0 >> 8) & 255] ^ T3[s1 & 255] ^ rk[k + 2]
                t3 = T0[s3 >> 24] ^ T1[(s0 >> 16) & 255] ^ T2[(s1 >> 8) & 255] ^ T3[s2 & 255] ^ rk[k + 3]
                s0, s1, s2, s3 = t0, t1, t2, t3
                k += 4
            o0 = ((S[s0 >> 24] << 24) | (S[(s1 >> 16) & 255] << 16) | (S[(s2 >> 8) & 255] << 8) | S[s3 & 255]) ^ rk[56]
            o1 = ((S[s1 >> 24] << 24) | (S[(s2 >> 16) & 255] << 16) | (S[(s3 >> 8) & 255] << 8) | S[s0 & 255]) ^ rk[57]
            o2 = ((S[s2 >> 24] << 24) | (S[(s3 >> 16) & 255] << 16) | (S[(s0 >> 8) & 255] << 8) | S[s1 & 255]) ^ rk[58]
            o3 = ((S[s3 >> 24] << 24) | (S[(s0 >> 16) & 255] << 16) | (S[(s1 >> 8) & 255] << 8) | S[s2 & 255]) ^ rk[59]
            stream.append(struct.pack(">4I", o0, o1, o2, o3))
        ks = b"".join(stream)[:len(piece)]
        out += (int.from_bytes(piece, "little") ^ int.from_bytes(ks, "little")).to_bytes(len(piece), "little")
    return bytes(out)


def aes256_ctr(key: bytes, iv: bytes, data: bytes) -> bytes:
    """AES-256-CTR (128-bit big-endian counter starting at iv); the same call encrypts and decrypts."""
    if len(key) != 32 or len(iv) != 16:
        raise CipherError("bad key or iv length")
    if not data:
        return b""
    lib = _libcrypto()
    if lib is not None:
        try:
            return _ctr_with_libcrypto(lib, key, iv, data)
        except CipherError:
            pass
    return _ctr_python(key, iv, data)


# ------------------------------------------------------------------------------------ payload
def encrypt(plaintext: bytes, password: str, iterations: int = DEFAULT_ITERATIONS,
            salt: bytes | None = None, iv: bytes | None = None) -> bytes:
    if not (MIN_ITERATIONS <= iterations <= MAX_ITERATIONS):
        raise ValueError(f"iterations must be within {MIN_ITERATIONS}..{MAX_ITERATIONS}")
    if not normalize_password(password):
        raise ValueError("empty password")
    salt = salt if salt is not None else os.urandom(SALT_LEN)
    iv = iv if iv is not None else os.urandom(IV_LEN)
    enc_key, mac_key = derive_keys(password, salt, iterations)
    body = MAGIC + salt + struct.pack(">I", iterations) + iv + aes256_ctr(enc_key, iv, plaintext)
    return body + hmac.new(mac_key, body, hashlib.sha256).digest()


def parse(payload: bytes) -> tuple[bytes, int, bytes]:
    if len(payload) < HEADER_LEN + TAG_LEN or payload[:len(MAGIC)] != MAGIC:
        raise PayloadError("not an STLLOC2 payload (damaged or incomplete download?)")
    p = len(MAGIC)
    salt = payload[p:p + SALT_LEN]
    (iterations,) = struct.unpack(">I", payload[p + SALT_LEN:p + SALT_LEN + 4])
    iv = payload[p + SALT_LEN + 4:HEADER_LEN]
    if not (100_000 <= iterations <= MAX_ITERATIONS):
        raise PayloadError("implausible iteration count (damaged payload?)")
    return salt, iterations, iv


def open_payload(payload: bytes, password: str) -> tuple[bytes, bytes]:
    """Check the tag; returns (aes key, iv).  WrongPassword when the tag does not match."""
    salt, iterations, iv = parse(payload)
    enc_key, mac_key = derive_keys(password, salt, iterations)
    tag = hmac.new(mac_key, payload[:-TAG_LEN], hashlib.sha256).digest()
    if not hmac.compare_digest(tag, payload[-TAG_LEN:]):
        raise WrongPassword("wrong password (or damaged payload)")
    return enc_key, iv


def decrypt(payload: bytes, password: str) -> bytes:
    enc_key, iv = open_payload(payload, password)
    return aes256_ctr(enc_key, iv, payload[HEADER_LEN:-TAG_LEN])


# ------------------------------------------------------------------------------------ command line
def _clean(raw: bytes) -> str | None:
    pw = raw.decode("utf-8", "surrogateescape")
    return pw if pw.strip() else None


def _read_input() -> tuple[str | None, bytes]:
    """(password, data): password from the first line of stdin when STL_PW_STDIN=1, else from STL_PW."""
    stdin = sys.stdin.buffer
    if os.environ.get("STL_PW_STDIN") == "1":
        line = stdin.readline(MAX_PASSWORD_LINE + 2)
        if not line.endswith(b"\n"):
            return None, b""
        return _clean(line.rstrip(b"\r\n")), stdin.read()
    raw = os.environb.get(b"STL_PW") if hasattr(os, "environb") else os.environ.get("STL_PW", "").encode()
    return (_clean(raw) if raw else None), stdin.read()


def _selftest() -> int:
    ok = _ctr_python(KAT_KEY, KAT_IV, KAT_PT) == KAT_CT
    print(f"selftest: built-in AES known-answer test {'ok' if ok else 'FAILED'}", file=sys.stderr)
    key, iv = os.urandom(32), b"\xff" * 12 + os.urandom(4)  # the counter wraps inside the test
    data = os.urandom(70_001)
    py = _ctr_python(key, iv, data)
    ok = ok and _ctr_python(key, iv, py) == data
    lib = _libcrypto()
    if lib is None:
        print("selftest: libcrypto not available (built-in AES is used)", file=sys.stderr)
    else:
        same = _ctr_with_libcrypto(lib, key, iv, data) == py
        print(f"selftest: libcrypto known-answer test ok; libcrypto and built-in AES {'agree' if same else 'DISAGREE'}",
              file=sys.stderr)
        ok = ok and same
    return EXIT_OK if ok else EXIT_CIPHER


def main(argv: list[str]) -> int:
    mode = argv[0] if argv else ""
    if mode == "selftest":
        return _selftest()
    if mode not in ("verify", "decrypt", "encrypt"):
        print("usage: stl_payload.py verify|decrypt|encrypt|selftest  (password: first stdin line with "
              "STL_PW_STDIN=1, or STL_PW; data on stdin)", file=sys.stderr)
        return EXIT_USAGE
    password, data = _read_input()
    if password is None:
        print("error: no password (first line of stdin with STL_PW_STDIN=1, or the environment variable STL_PW)",
              file=sys.stderr)
        return EXIT_USAGE
    try:
        if mode == "encrypt":
            out = encrypt(data, password, int(os.environ.get("STL_PBKDF2_ITERATIONS", DEFAULT_ITERATIONS)))
        elif mode == "verify":
            open_payload(data, password)
            print("password ok", file=sys.stderr)
            return EXIT_OK
        else:
            out = decrypt(data, password)
            expect = os.environ.get("STL_EXPECT_SHA256", "").strip().lower()
            if expect and hashlib.sha256(out).hexdigest() != expect:
                print("error: decrypted data does not match the expected SHA-256", file=sys.stderr)
                return EXIT_CIPHER
    except WrongPassword as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_WRONG_PASSWORD
    except PayloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_PAYLOAD
    except (CipherError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CIPHER
    sys.stdout.buffer.write(out)
    sys.stdout.buffer.flush()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
