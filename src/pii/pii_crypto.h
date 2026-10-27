/* PII-AGENT :: pii_crypto.h — vault encryption + UUID + purge.
 *
 * Envelope (UTF-8 JSON, written to .vault/[UUID].json):
 *   {"v":1,"alg":"AES-256-CBC-HMAC-SHA256","iv":"b64","ct":"b64","mac":"b64"}
 * where ct = AES-256-CBC(payload JSON) and mac = HMAC-SHA256(iv||ct).
 * Key: 32 bytes from PII_VAULT_KEY (64 hex chars) or .vault/.key
 * (generated once, hex, file mode 0600). Fail-closed: envelope with a bad
 * MAC or a failed AES self-test never decrypts.
 */
#ifndef PII_CRYPTO_H
#define PII_CRYPTO_H

#include <stddef.h>

/* 0 ok, -1 error. Runs NIST AES-256 vector + SHA256("abc") checks first. */
int pii_crypto_selftest(void);

/* UUID v4 string (36+NUL) into out[37]. 0 ok. */
int pii_uuid4(char out[37]);

/* Load (or create) the 32-byte vault key for vault_dir. 0 ok. */
int pii_key_load(const char *vault_dir, unsigned char key[32]);

/* Encrypt payload JSON -> malloc'd envelope JSON. NULL on error. */
char *pii_vault_seal(const char *vault_dir, const char *payload_json);
/* Decrypt envelope JSON -> malloc'd payload JSON. NULL on error/MAC fail. */
char *pii_vault_open(const char *vault_dir, const char *envelope_json);

/* Securely destroy path: overwrite with random+zeros (3 passes) then
 * delete. Retains NOTHING of the file. 0 ok, -1 if missing/unremovable. */
int pii_secure_delete(const char *path);

void pii_memzero(void *p, size_t n);

#endif
