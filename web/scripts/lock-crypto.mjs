// File format of the locked artifact build (scripts/build-artifact.mjs --lock), Node side.
//   file = "STLENC1\0" (8 bytes) · IV (12 random bytes) · AES-256-GCM ciphertext · 16-byte tag
//   key  = PBKDF2-HMAC-SHA256(NFC(password), salt (16 random bytes, lock.json), iterations ≥ 600 000) → 32 bytes
//   AAD  = the file's published path relative to the page, UTF-8, without a leading "./" ("snapshot/index.bin")
// The browser side is scripts/lock/lock.js (app bundle) and src/api/snapshot.ts (snapshot files); both read the
// same format with Web Crypto. Nothing here ever logs or stores the password.
import { createCipheriv, createDecipheriv, pbkdf2Sync, randomBytes } from "node:crypto";

export const MAGIC = Uint8Array.from(Buffer.from("STLENC1\0", "latin1"));
export const IV_BYTES = 12;
export const TAG_BYTES = 16;
export const SALT_BYTES = 16;
export const KEY_BYTES = 32;
export const MIN_ITERATIONS = 600_000;

/** Published path → AAD string (the loader and the app compute it the same way). */
export const lockAad = (p) => String(p).replace(/^(\.\/)+/, "");

export const isEncrypted = (bytes) => bytes.length >= MAGIC.length + IV_BYTES + TAG_BYTES && MAGIC.every((x, i) => bytes[i] === x);

export function newSalt() {
  return new Uint8Array(randomBytes(SALT_BYTES));
}

/** Raw 32-byte AES key from the password (NFC-normalised UTF-8). */
export function deriveKey(password, salt, iterations = MIN_ITERATIONS) {
  if (typeof password !== "string" || !password) throw new Error("empty password");
  return new Uint8Array(pbkdf2Sync(Buffer.from(password.normalize("NFC"), "utf8"), salt, iterations, KEY_BYTES, "sha256"));
}

/** Encrypt one file's bytes for the published path `aad`. */
export function encryptFile(plain, key, aad, iv = randomBytes(IV_BYTES)) {
  const c = createCipheriv("aes-256-gcm", key, iv, { authTagLength: TAG_BYTES });
  c.setAAD(Buffer.from(lockAad(aad), "utf8"));
  const body = Buffer.concat([c.update(plain), c.final()]);
  return new Uint8Array(Buffer.concat([MAGIC, iv, body, c.getAuthTag()]));
}

/** Decrypt (throws on a wrong key, a different path or any modified byte). */
export function decryptFile(file, key, aad) {
  if (!isEncrypted(file)) throw new Error("not an encrypted file");
  const iv = file.subarray(MAGIC.length, MAGIC.length + IV_BYTES);
  const tag = file.subarray(file.length - TAG_BYTES);
  const d = createDecipheriv("aes-256-gcm", key, iv, { authTagLength: TAG_BYTES });
  d.setAAD(Buffer.from(lockAad(aad), "utf8"));
  d.setAuthTag(tag);
  return new Uint8Array(Buffer.concat([d.update(file.subarray(MAGIC.length + IV_BYTES, file.length - TAG_BYTES)), d.final()]));
}
