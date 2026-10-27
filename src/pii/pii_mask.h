/* PII-AGENT :: pii_mask.h — deterministic PII detectors + boundary rules.
 *
 * Masked classes (placeholders [NAME_n] [EMAIL_n] [PHONE_n] [FISCAL_n]
 * [ADDRESS_n], consistent per document, per-type counters):
 *   candidate names, third-party names, fiscal IDs (IT codice fiscale,
 *   P.IVA 11-digit, US SSN), phone numbers, personal emails, street addresses.
 *
 * Boundary (NEVER masked): birth dates, educational institutions, employer
 * names, job titles, dates of employment, skill descriptions.
 *
 * Name rule (encodes the boundary without an NER model): a capitalised
 * pair is treated as a person name ONLY when (a) inside the header zone
 * (first 12 lines), or (b) derived from an email local-part, or (c) prefixed
 * by a courtesy/professional title, or (d) on a references/emergency line.
 * Everywhere else capitalised pairs are kept (employer / institution).
 */
#ifndef PII_MASK_H
#define PII_MASK_H

#include <stddef.h>

#define PII_ORIG_MAX   480
#define PII_PH_MAX      32
#define PII_TYPE_MAX    16

typedef struct {
    char placeholder[PII_PH_MAX];   /* e.g. "[NAME_1]"  */
    char original[PII_ORIG_MAX + 1];/* exact surface    */
    char type[PII_TYPE_MAX];        /* NAME/EMAIL/PHONE/FISCAL/ADDRESS */
    /* v003 spatial vault: 2D bounding box (normalized 0..1000, page
     * 1-based). has_box=0 for entries from text-only sources / v1. */
    float bx0, by0, bx1, by1;
    int bpage;
    int has_box;
} PiiEntry;

typedef struct {
    PiiEntry *items;
    int n, cap;
    int c_name, c_email, c_phone, c_fiscal, c_addr;
} PiiVaultMap;

void   pii_map_init(PiiVaultMap *m);
void   pii_map_free(PiiVaultMap *m);
/* find existing entry for (type,original) or append a new numbered one.
 * Returns placeholder string (inside map) or NULL on OOM. */
const char *pii_map_intern(PiiVaultMap *m, const char *type,
                           const char *original, size_t len);

/* Mask full text. Returns malloc'd string (caller frees). Map is filled. */
char *pii_mask_text(const char *in, PiiVaultMap *map);
/* Restore placeholders using map. Returns malloc'd string (caller frees). */
char *pii_unmask_text(const char *in, const PiiVaultMap *map);

/* Serialize map (unencrypted JSON payload) / parse it back.
 * Caller frees returned string. Returns 0 on parse error. */
char *pii_map_to_json(const PiiVaultMap *m, const char *uuid,
                      const char *source_file);
int   pii_map_from_json(const char *json, PiiVaultMap *m,
                        char *uuid_out, size_t uuid_cap);

#endif
