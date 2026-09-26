// Types of the pure helpers exported by scripts/lock/lock.js (the password gate), for src/api/lock.test.ts.
export declare const MAGIC: Uint8Array<ArrayBuffer>;
export declare const STORE_PREFIX: string;
export declare const STATE_PREFIX: string;
export declare const APP_PREFIX: string;
export declare const REMEMBER_MS: number;
export declare const SESSION_MS: number;
export declare const T: Record<string, string>;
export declare class LockError extends Error {
  code: string;
  constructor(code: string);
}
export declare function lockAad(path: string): string;
export declare function isEncrypted(bytes: Uint8Array): boolean;
export declare function isGzip(bytes: Uint8Array): boolean;
export declare function b64(bytes: Uint8Array): string;
export declare function unb64(s: string): Uint8Array<ArrayBuffer>;
export declare function deriveKeyBytes(password: string, salt: Uint8Array<ArrayBuffer>, iterations: number): Promise<Uint8Array<ArrayBuffer>>;
export declare function importAesKey(raw: Uint8Array<ArrayBuffer>): Promise<CryptoKey>;
export declare function decryptFile(bytes: Uint8Array<ArrayBuffer>, key: CryptoKey, aad: string): Promise<Uint8Array<ArrayBuffer>>;
export declare function gunzip(bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>>;
export declare function readSavedKey(id: string, now?: number): Uint8Array<ArrayBuffer> | null;
export declare function forgetKey(id: string): void;
export declare function forgetAll(): void;
export declare function saveKey(id: string, raw: Uint8Array, remember: boolean, now?: number): void;
export declare function stateKey(raw: Uint8Array<ArrayBuffer>): Promise<CryptoKey>;
export declare function sealState(obj: Record<string, string>, key: CryptoKey, id: string): Promise<string>;
export declare function openState(sealed: string, key: CryptoKey, id: string): Promise<Map<string, string>>;
export declare function takePlainAppEntries(): Map<string, string>;
export declare function dropOtherStates(id: string): void;
export declare function protectAppStorage(state: Map<string, string>, persist: () => void): boolean;
export declare function stateWriter(state: Map<string, string>, key: CryptoKey, id: string, write: (k: string, v: string) => void): () => Promise<void>;
export declare function absolutizeCss(css: string, base: string): string;
export declare function applyCss(css: string): Promise<"adopted" | "style" | "blob" | null>;
export declare function runBundle(code: string): Promise<{ method: "blob" | "inline" | "eval" | null; booted: boolean; error: string }>;
