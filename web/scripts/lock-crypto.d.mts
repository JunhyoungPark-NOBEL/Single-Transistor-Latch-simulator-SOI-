// Types of scripts/lock-crypto.mjs (Node side of the locked artifact format), for the unit tests in src/api.
export declare const MAGIC: Uint8Array<ArrayBuffer>;
export declare const IV_BYTES: number;
export declare const TAG_BYTES: number;
export declare const SALT_BYTES: number;
export declare const KEY_BYTES: number;
export declare const MIN_ITERATIONS: number;
export declare function lockAad(path: string): string;
export declare function isEncrypted(bytes: Uint8Array): boolean;
export declare function newSalt(): Uint8Array<ArrayBuffer>;
export declare function deriveKey(password: string, salt: Uint8Array, iterations?: number): Uint8Array<ArrayBuffer>;
export declare function encryptFile(plain: Uint8Array, key: Uint8Array, aad: string, iv?: Uint8Array): Uint8Array<ArrayBuffer>;
export declare function decryptFile(file: Uint8Array, key: Uint8Array, aad: string): Uint8Array<ArrayBuffer>;
