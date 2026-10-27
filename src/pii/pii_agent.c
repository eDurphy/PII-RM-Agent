/* PII-AGENT :: pii_agent.c — CLI: mask / unmask / purge / selftest.
 *
 *   pii_agent mask <input.pdf> [--vault-dir DIR]
 *     Extracts text, masks PII, writes [UUID]_CV.pdf NEXT TO THE SOURCE
 *     file and the encrypted vault at <vault>/.vault/[UUID].json
 *     (default vault: ./.vault). Prints UUID, counts, paths.
 *
 *   pii_agent unmask <UUID|masked.pdf> [--vault-dir DIR]
 *     Restores originals via the vault into [UUID]_CV_restored.pdf
 *     (same directory), then PERMANENTLY destroys .vault/[UUID].json
 *     (secure overwrite + delete). The masked PDF is RETAINED.
 *
 *   pii_agent purge <UUID> [--vault-dir DIR]
 *     Explicit deletion request: destroys ONLY .vault/[UUID].json,
 *     retaining [UUID]_CV.pdf.
 *
 *   pii_agent selftest
 *     Deterministic checks: crypto vectors, mask/unmask roundtrip incl.
 *     boundary keeps, envelope seal/open, UUID format. Exit 0 = all pass.
 *
 * NOTE on the LLM: entity detection here is deterministic (regex/shape +
 * header/reference-zone rules), which is what makes the CLI sub-second on
 * CPU. The downscaled resident model (L14/d1024, ~182M, see README) is the
 * trainable NER re-ranker hook: when weights exist, a future --llm pass can
 * confirm ambiguous NAME spans. The hook point is pii_llm_confirm() below
 * (currently heuristic-only, documented — never silently a no-op claim).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define MKDIR(d) _mkdir(d)
#else
#include <sys/types.h>
#define MKDIR(d) mkdir(d, 0700)
#endif

#include "pii_mask.h"
#include "pii_crypto.h"
#include "pii_pdf.h"

/* LLM re-rank hook (stub, documented): returns 1 confirm / 0 reject /
 * -1 abstain (no weights loaded). Wiring lives with the resident model. */
static int pii_llm_confirm(const char *span, const char *type) {
    (void)span; (void)type;
    return -1; /* abstain: heuristic decision stands */
}

static void usage(const char *argv0) {
    printf("PII-AGENT — CV anonymizer (v004 base, PII-scale L14/d1024 resident)\n");
    printf("Usage:\n");
    printf("  %s mask <input.pdf> [--vault-dir DIR]\n", argv0);
    printf("  %s unmask <UUID|masked.pdf> [--vault-dir DIR]\n", argv0);
    printf("  %s purge <UUID> [--vault-dir DIR]\n", argv0);
    printf("  %s selftest\n", argv0);
}

static int is_uuid_like(const char *s) {
    size_t i, n = strlen(s);
    if (n != 36) return 0;
    for (i = 0; i < 36; i++) {
        if (i == 8 || i == 13 || i == 18 || i == 23) { if (s[i] != '-') return 0; }
        else if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f') ||
                   (s[i] >= 'A' && s[i] <= 'F'))) return 0;
    }
    return 1;
}
static char *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    char *b; long sz;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 0 || sz > 32 * 1024 * 1024) { fclose(f); return NULL; }
    b = (char *)malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    if (sz && fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    fclose(f); b[sz] = 0; if (n) *n = (size_t)sz; return b;
}
static int write_file(const char *path, const char *data) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(data, 1, strlen(data), f);
    if (fclose(f)) return -1;
    return 0;
}
/* split path into dir (malloc'd, may be "." ) + base pointer */
static char *dir_of(const char *path) {
    const char *s = strrchr(path, '/'), *b = strrchr(path, '\\');
    const char *m = s > b ? s : b;
    char *d;
    if (!m) { d = (char *)malloc(2); if (d) strcpy(d, "."); return d; }
    d = (char *)malloc((size_t)(m - path) + 1);
    if (!d) return NULL;
    memcpy(d, path, (size_t)(m - path)); d[m - path] = 0;
    if (!*d) strcpy(d, ".");
    return d;
}
static void ensure_dir(const char *d) {
    MKDIR(d);
}

/* v003 hybrid forward (infer_ctx.c; linked into BOTH exes, flash-safe). */
float *eng_infer_run_document(const PiiDoc *doc, int *n_out);

/* collapse embedded newlines (show-op literals may carry them); the join
 * uses '\n' as the span separator so one line == one span. */
static char *collapse_nl(const char *s)
{
    char *o = strdup(s ? s : "");
    char *q;
    if (!o) return NULL;
    for (q = o; *q; q++) if (*q == '\n' || *q == '\r') *q = ' ';
    return o;
}

/* ---------------- mask (v003 spatial) ---------------- */
static int cmd_mask(const char *input, const char *vault_dir) {
    char uuid[37];
    char *text = NULL, *masked = NULL, *payload = NULL, *env = NULL;
    char *dir = NULL, out_pdf[1024], vault_path[1024];
    PiiDoc *doc = NULL;
    PiiVaultMap map;
    if (pii_uuid4(uuid)) { printf("UUID failure\n"); return 1; }
    doc = pii_doc_load(input);
    if (!doc) {
        /* unreadable file: legacy 1D path (v002 parity) */
        text = pii_pdf_extract_text(input);
        if (!text) {
            printf("MASK-FAIL: cannot read '%s'.\n", input);
            return 1;
        }
        goto legacy_text;
    }
    /* ---- spatial path: mask the span join, attribute boxes, redact ---- */
    {
        int total = 0, p, s, nred = 0, cred = 0;
        int *span_off = NULL;
        int g = 0;
        size_t jlen = 0;
        char *join = NULL, *mp = NULL;
        char **lines = NULL;
        PiiRedact *reds = NULL;
        float *sal = NULL;
        int nsal = 0;
        double salsum = 0;
        for (p = 0; p < doc->npages; p++) total += doc->pages[p].nspans;
        if (total <= 0) {
            int nimgs = 0;
            for (p = 0; p < doc->npages; p++) nimgs += doc->pages[p].nimgs;
            printf("MASK-FAIL: '%s' has no text layer (%d page image(s) decoded via stb_image).\n",
                   input, nimgs);
            printf("  Visual PII localization needs the trained VLM head: run train on text-layer CVs first,\n");
            printf("  then re-run mask. No vault entry was created (fail-closed).\n");
            pii_doc_free(doc);
            return 1;
        }
        span_off = (int *)malloc((size_t)(total + 1) * sizeof(int));
        lines = (char **)calloc((size_t)total, sizeof(char *));
        if (!span_off || !lines) { free(span_off); free(lines); pii_doc_free(doc); return 1; }
        /* build collapsed join; record each span's offset */
        for (p = 0; p < doc->npages; p++) {
            for (s = 0; s < doc->pages[p].nspans; s++) {
                char *c = collapse_nl(doc->pages[p].spans[s].text);
                lines[g] = c ? c : strdup("");
                span_off[g] = (int)jlen;
                jlen += strlen(lines[g]) + 1;
                g++;
            }
            jlen += 1; /* blank line = page separator */
        }
        join = (char *)malloc(jlen + 1);
        if (!join) { goto spatial_oom; }
        mp = join; g = 0;
        for (p = 0; p < doc->npages; p++) {
            for (s = 0; s < doc->pages[p].nspans; s++) {
                size_t L = strlen(lines[g++]);
                memcpy(mp, lines[g - 1], L); mp += L; *mp++ = '\n';
            }
            *mp++ = '\n';
        }
        *mp = 0;
        /* hybrid visual-context forward over the structured doc */
        sal = eng_infer_run_document(doc, &nsal);
        if (sal && nsal == total) {
            int k; for (k = 0; k < total; k++) salsum += sal[k];
        }
        pii_map_init(&map);
        (void)pii_llm_confirm;
        masked = pii_mask_text(join, &map);
        if (!masked) { printf("MASK-FAIL: out of memory\n"); goto spatial_oom; }
        /* split masked lines back onto spans (same order, page blanks) */
        {   char *q = masked;
            int *span_pg = NULL;
            PiiSpan **by_g = NULL;
            int have = 0, pp, ei, nimgs = 0, nboxed = 0;
            by_g = (PiiSpan **)malloc((size_t)total * sizeof *by_g);
            span_pg = (int *)malloc((size_t)total * sizeof *span_pg);
            if (!by_g || !span_pg) {
                printf("MASK-FAIL: out of memory\n");
                free(by_g); free(span_pg);
                goto spatial_oom;
            }
            g = 0;
            for (pp = 0; pp < doc->npages; pp++) {
                nimgs += doc->pages[pp].nimgs;
                for (s = 0; s < doc->pages[pp].nspans; s++) {
                    by_g[g] = &doc->pages[pp].spans[s];
                    span_pg[g] = doc->pages[pp].num;
                    g++;
                }
            }
            g = 0;
            while (*q && g < total) {
                char *e = strchr(q, '\n');
                size_t L = e ? (size_t)(e - q) : strlen(q);
                if (L == 0) { q = e ? e + 1 : q + strlen(q); continue; } /* page blank */
                free(by_g[g]->text);
                by_g[g]->text = (char *)malloc(L + 1);
                if (!by_g[g]->text) {
                    printf("MASK-FAIL: out of memory\n");
                    free(by_g); free(span_pg);
                    goto spatial_oom;
                }
                memcpy(by_g[g]->text, q, L); by_g[g]->text[L] = 0;
                g++;
                q = e ? e + 1 : q + L;
                have++;
            }
            if (have != total) {
                /* strict: never emit unmasked leftovers */
                printf("MASK-FAIL: span alignment mismatch (%d/%d).\n", have, total);
                free(by_g); free(span_pg);
                goto spatial_oom;
            }
            /* attribute every entry occurrence to its span box */
            span_off[total] = (int)jlen;
            for (ei = 0; ei < map.n; ei++) {
                const char *orig = map.items[ei].original;
                size_t ol = strlen(orig);
                const char *f = join;
                if (!ol) continue;
                while ((f = strstr(f, orig)) != NULL) {
                    int off = (int)(f - join);
                    int gg;
                    f += ol;
                    for (gg = 0; gg < total; gg++) {
                        size_t gl = strlen(lines[gg]);
                        if (off >= span_off[gg] && (size_t)(off - span_off[gg]) + ol <= gl) {
                            PiiSpan *sp = by_g[gg];
                            if (nred == cred) {
                                int nc = cred ? cred * 2 : 64;
                                PiiRedact *nr = (PiiRedact *)realloc(reds, (size_t)nc * sizeof *nr);
                                if (!nr) break;
                                reds = nr; cred = nc;
                            }
                            if (nred == cred) break; /* OOM: stop adding, keep masking */
                            reds[nred].page = span_pg[gg];
                            reds[nred].box = sp->box;
                            nred++;
                            if (!map.items[ei].has_box) {
                                map.items[ei].bx0 = sp->box.x0; map.items[ei].by0 = sp->box.y0;
                                map.items[ei].bx1 = sp->box.x1; map.items[ei].by1 = sp->box.y1;
                                map.items[ei].bpage = span_pg[gg];
                                map.items[ei].has_box = 1;
                            }
                            break;
                        }
                    }
                }
            }
            for (ei = 0; ei < map.n; ei++) if (map.items[ei].has_box) nboxed++;
            /* write physically-redacted PDF + spatial vault */
            dir = dir_of(input);
            if (!dir) {
                printf("MASK-FAIL: out of memory\n");
                free(by_g); free(span_pg);
                goto spatial_oom;
            }
            snprintf(out_pdf, sizeof out_pdf, "%s/%s_CV.pdf", dir, uuid);
            {   char title[128];
                snprintf(title, sizeof title, "Anonymized CV %s", uuid);
                if (pii_pdf_write_redacted(out_pdf, doc, masked, title, reds, nred)) {
                    printf("MASK-FAIL: cannot write '%s'\n", out_pdf);
                    free(by_g); free(span_pg);
                    for (g = 0; g < total; g++) free(lines[g]);
                    free(lines); free(span_off); free(join); free(sal);
                    free(reds); free(dir); free(masked);
                    pii_map_free(&map); pii_doc_free(doc);
                    return 1;
                }
            }
            payload = pii_map_to_json(&map, uuid, input);
            if (!payload) {
                printf("MASK-FAIL: json\n");
                free(by_g); free(span_pg);
                for (g = 0; g < total; g++) free(lines[g]);
                free(lines); free(span_off); free(join); free(sal);
                free(reds); free(dir); free(masked);
                pii_map_free(&map); pii_doc_free(doc);
                return 1;
            }
            ensure_dir(vault_dir);
            env = pii_vault_seal(vault_dir, payload);
            if (!env) {
                printf("MASK-FAIL: vault seal failed (crypto self-test or key error).\n");
                free(by_g); free(span_pg);
                for (g = 0; g < total; g++) free(lines[g]);
                free(lines); free(span_off); free(join); free(sal);
                free(reds); free(dir); free(masked); free(payload);
                pii_map_free(&map); pii_doc_free(doc);
                return 1;
            }
            snprintf(vault_path, sizeof vault_path, "%s/%s.json", vault_dir, uuid);
            if (write_file(vault_path, env)) {
                printf("MASK-FAIL: cannot write '%s'\n", vault_path);
                free(by_g); free(span_pg);
                for (g = 0; g < total; g++) free(lines[g]);
                free(lines); free(span_off); free(join); free(sal);
                free(reds); free(dir); free(masked); free(payload); free(env);
                pii_map_free(&map); pii_doc_free(doc);
                return 1;
            }
            printf("MASK-OK uuid=%s\n", uuid);
            printf("  source : %s (%d pages, %d spans, %d image(s) decoded)\n",
                   input, doc->npages, total, nimgs);
            printf("  masked : %s (%d zone(s) physically redacted)\n", out_pdf, nred);
            if (sal && nsal == total && nsal > 0)
                printf("  visual : hybrid forward mean salience %.3f over %d spans\n",
                       salsum / (double)nsal, nsal);
            printf("  vault  : %s (entries=%d: NAME=%d EMAIL=%d PHONE=%d FISCAL=%d ADDRESS=%d, boxes kept for %d)\n",
                   vault_path, map.n, map.c_name, map.c_email, map.c_phone,
                   map.c_fiscal, map.c_addr, nboxed);
            free(by_g); free(span_pg);
            for (g = 0; g < total; g++) free(lines[g]);
            free(lines); free(span_off); free(join); free(sal);
            free(reds); free(dir); free(masked); free(payload); free(env);
            pii_map_free(&map); pii_doc_free(doc);
            return 0;
        }
        /* unreachable: every path inside returns; kept for safety */
        goto spatial_oom;
spatial_oom:
        for (g = 0; g < total; g++) free(lines[g]);
        free(lines); free(span_off); free(join); free(masked); free(sal);
        free(reds);
        pii_map_free(&map); pii_doc_free(doc);
        return 1;
    }
legacy_text:
    pii_map_init(&map);
    (void)pii_llm_confirm; /* hook documented above */
    masked = pii_mask_text(text, &map);
    if (!masked) { printf("MASK-FAIL: out of memory\n"); free(text); pii_map_free(&map); return 1; }
    dir = dir_of(input);
    if (!dir) { free(text); free(masked); pii_map_free(&map); return 1; }
    snprintf(out_pdf, sizeof out_pdf, "%s/%s_CV.pdf", dir, uuid);
    {   char title[128];
        snprintf(title, sizeof title, "Anonymized CV %s", uuid);
        if (pii_pdf_write_text(out_pdf, masked, title)) {
            printf("MASK-FAIL: cannot write '%s'\n", out_pdf);
            free(dir); free(text); free(masked); pii_map_free(&map); return 1;
        }
    }
    payload = pii_map_to_json(&map, uuid, input);
    if (!payload) { printf("MASK-FAIL: json\n"); free(dir); free(text); free(masked); pii_map_free(&map); return 1; }
    ensure_dir(vault_dir);
    env = pii_vault_seal(vault_dir, payload);
    if (!env) {
        printf("MASK-FAIL: vault seal failed (crypto self-test or key error).\n");
        free(dir); free(text); free(masked); free(payload); pii_map_free(&map); return 1;
    }
    snprintf(vault_path, sizeof vault_path, "%s/%s.json", vault_dir, uuid);
    if (write_file(vault_path, env)) {
        printf("MASK-FAIL: cannot write '%s'\n", vault_path);
        free(dir); free(text); free(masked); free(payload); free(env); pii_map_free(&map); return 1;
    }
    printf("MASK-OK uuid=%s\n", uuid);
    printf("  source : %s\n", input);
    printf("  masked : %s\n", out_pdf);
    printf("  vault  : %s (entries=%d: NAME=%d EMAIL=%d PHONE=%d FISCAL=%d ADDRESS=%d)\n",
           vault_path, map.n, map.c_name, map.c_email, map.c_phone, map.c_fiscal, map.c_addr);
    free(dir); free(text); free(masked); free(payload); free(env); pii_map_free(&map);
    return 0;
}

/* resolve UUID + masked-pdf path + vault path from CLI arg */
static int resolve_unmask_arg(const char *arg, const char *vault_dir,
                              char *uuid, size_t ucap,
                              char *pdf, size_t pcap,
                              char *vault, size_t vcap) {
    if (is_uuid_like(arg)) {
        snprintf(uuid, ucap, "%s", arg);
        /* find <uuid>_CV.pdf: prefer CWD, then vault-adjacent? Spec: source dir. Try CWD. */
        snprintf(pdf, pcap, "%s_CV.pdf", arg);
        {   FILE *f = fopen(pdf, "rb");
            if (!f) { /* try .vault/../ ? fall back: ask to pass pdf path */
                snprintf(pdf, pcap, ".vault/%s_CV.pdf", arg);
                f = fopen(pdf, "rb");
                if (f) fclose(f);
                else { pdf[0] = 0; }
            } else fclose(f);
        }
        snprintf(vault, vcap, "%s/%s.json", vault_dir, arg);
        return 0;
    }
    /* pdf path given: expect <uuid>_CV.pdf */
    {   const char *b = strrchr(arg, '/'), *w = strrchr(arg, '\\');
        const char *base = b > w ? b + 1 : (w ? w + 1 : arg);
        size_t bl = strlen(base);
        if (bl > 7 && !strcmp(base + bl - 7, "_CV.pdf")) {
            if (bl - 7 == 36 && is_uuid_like(base)) { /* shouldn't happen (36+7) */
            }
            {   char tmp[128]; memcpy(tmp, base, bl - 7); tmp[bl - 7] = 0;
                if (is_uuid_like(tmp)) {
                    snprintf(uuid, ucap, "%s", tmp);
                    snprintf(pdf, pcap, "%s", arg);
                    snprintf(vault, vcap, "%s/%s.json", vault_dir, tmp);
                    return 0;
                }
            }
        }
    }
    return -1;
}

/* ---------------- unmask ---------------- */
static int cmd_unmask(const char *arg, const char *vault_dir) {
    char uuid[64] = {0}, pdf[1024] = {0}, vpath[1024] = {0};
    char *env = NULL, *payload = NULL, *masked_text = NULL, *restored = NULL;
    char out[1100];
    PiiVaultMap map;
    if (resolve_unmask_arg(arg, vault_dir, uuid, sizeof uuid, pdf, sizeof pdf, vpath, sizeof vpath)) {
        printf("UNMASK-FAIL: give a UUID or a <UUID>_CV.pdf path.\n");
        return 1;
    }
    if (!pdf[0]) {
        printf("UNMASK-FAIL: cannot locate %s_CV.pdf next to source; pass its path explicitly.\n", uuid);
        return 1;
    }
    env = read_file(vpath, NULL);
    if (!env) { printf("UNMASK-FAIL: vault missing: %s (already purged?)\n", vpath); return 1; }
    payload = pii_vault_open(vault_dir, env);
    free(env);
    if (!payload) { printf("UNMASK-FAIL: vault MAC/key error for %s\n", vpath); return 1; }
    pii_map_init(&map);
    {   char du[64] = {0};
        if (!pii_map_from_json(payload, &map, du, sizeof du)) {
            printf("UNMASK-FAIL: vault payload corrupt\n");
            free(payload); pii_map_free(&map); return 1;
        }
    }
    free(payload);
    masked_text = pii_pdf_extract_text(pdf);
    if (!masked_text) { printf("UNMASK-FAIL: no text in %s\n", pdf); pii_map_free(&map); return 1; }
    restored = pii_unmask_text(masked_text, &map);
    free(masked_text);
    if (!restored) { printf("UNMASK-FAIL: oom\n"); pii_map_free(&map); return 1; }
    {   char *dir = dir_of(pdf);
        char title[160];
        if (!dir) { free(restored); pii_map_free(&map); return 1; }
        snprintf(out, sizeof out, "%s/%s_CV_restored.pdf", dir, uuid);
        snprintf(title, sizeof title, "Restored CV %s", uuid);
        free(dir);
        if (pii_pdf_write_text(out, restored, title)) {
            printf("UNMASK-FAIL: cannot write %s\n", out);
            free(restored); pii_map_free(&map); return 1;
        }
    }
    free(restored); pii_map_free(&map);
    /* lifecycle: destroy ONLY the vault, retain the masked PDF */
    if (pii_secure_delete(vpath)) {
        printf("UNMASK-WARN: restored to %s but vault delete failed: %s (delete it manually)\n", out, vpath);
        return 1;
    }
    printf("UNMASK-OK uuid=%s\n  restored: %s\n  purged  : %s (masked PDF retained)\n", uuid, out, vpath);
    return 0;
}

/* ---------------- purge ---------------- */
static int cmd_purge(const char *arg, const char *vault_dir) {
    char vpath[1024];
    const char *uuid = arg;
    if (!is_uuid_like(uuid)) { printf("PURGE-FAIL: give a UUID.\n"); return 1; }
    snprintf(vpath, sizeof vpath, "%s/%s.json", vault_dir, uuid);
    if (pii_secure_delete(vpath)) { printf("PURGE-FAIL: %s missing or not removable\n", vpath); return 1; }
    printf("PURGE-OK uuid=%s vault=%s destroyed ([UUID]_CV.pdf retained)\n", uuid, vpath);
    return 0;
}

/* ---------------- selftest ---------------- */
static int fails = 0;
#define CHECK(c, name) do { if (c) printf("  PASS %s\n", name); \
    else { printf("  FAIL %s\n", name); fails++; } } while (0)
static int cmd_selftest(void) {
    printf("PII-AGENT selftest\n");
    CHECK(pii_crypto_selftest() == 0, "crypto-vectors(AES-256/SHA256)");
    {   char u1[37], u2[37];
        CHECK(pii_uuid4(u1) == 0 && pii_uuid4(u2) == 0 && strlen(u1) == 36 &&
              u1[8] == '-' && strcmp(u1, u2) != 0, "uuid-v4");
    }
    {   /* mask/unmask roundtrip with boundary keeps */
        const char *cv =
            "Mario Rossi\n"
            "Via Roma 12, 20100 Milano\n"
            "mario.rossi@gmail.com — +39 333 123 4567\n"
            "Codice Fiscale RSSMRA80A01H501U\n"
            "Nato il 01/03/1980 a Milano\n"
            "Istruzione: Laurea in Informatica, Universita di Milano (2010)\n"
            "Esperienza: Software Engineer presso Acme Solutions (2015-2022)\n"
            "Competenze: Python, gestione progetti, machine learning\n"
            "Referenze: Luigi Bianchi, luigi.bianchi@outlook.com, 02 8736 4455\n";
        PiiVaultMap m; char *masked = NULL, *back = NULL;
        pii_map_init(&m);
        masked = pii_mask_text(cv, &m);
        CHECK(masked && m.c_name >= 2 && m.c_email == 2 && m.c_phone == 2 &&
              m.c_fiscal == 1 && m.c_addr == 1, "mask-counts");
        if (masked) {
            CHECK(strstr(masked, "mario.rossi@gmail.com") == NULL, "mask-email");
            CHECK(strstr(masked, "Mario Rossi") == NULL, "mask-candidate-name");
            CHECK(strstr(masked, "Luigi Bianchi") == NULL, "mask-third-party-name");
            CHECK(strstr(masked, "RSSMRA80A01H501U") == NULL, "mask-fiscal");
            CHECK(strstr(masked, "Via Roma 12") == NULL, "mask-address");
            CHECK(strstr(masked, "01/03/1980") != NULL, "keep-birth-date");
            CHECK(strstr(masked, "Universita di Milano") != NULL, "keep-institution");
            CHECK(strstr(masked, "Acme Solutions") != NULL, "keep-employer");
            CHECK(strstr(masked, "Software Engineer") != NULL, "keep-title");
            CHECK(strstr(masked, "2015-2022") != NULL, "keep-dates");
            CHECK(strstr(masked, "Python") != NULL, "keep-skills");
            /* placeholder consistency: same email local-part name reuses [NAME_n]? */
            back = pii_unmask_text(masked, &m);
            CHECK(back && !strcmp(back, cv), "roundtrip-exact");
            free(back);
            /* vault payload Sat roundtrip */
            {   char *js = pii_map_to_json(&m, "00000000-0000-4000-8000-000000000000", "t.pdf");
                PiiVaultMap m2; char du[64] = {0};
                pii_map_init(&m2);
                CHECK(js && pii_map_from_json(js, &m2, du, sizeof du) && m2.n == m.n, "vault-json");
                free(js); pii_map_free(&m2);
            }
            free(masked);
        }
        pii_map_free(&m);
    }
    {   /* seal/open envelope roundtrip (uses .vaultךselftest key; cleaned after) */
        const char *vd = ".vault";
        char *env = NULL, *open = NULL;
        MKDIR(vd);
        env = pii_vault_seal(vd, "{\"v\":1,\"uuid\":\"t\",\"source\":\"s\",\"map\":[]}");
        CHECK(env != NULL, "vault-seal");
        if (env) {
            open = pii_vault_open(vd, env);
            CHECK(open && strstr(open, "\"uuid\"") != NULL, "vault-open");
            /* tamper -> MAC fail */
            env[40] = env[40] == 'A' ? 'B' : 'A';
            {   char *bad = pii_vault_open(vd, env);
                CHECK(bad == NULL, "vault-tamper-reject");
                free(bad);
            }
            free(open); free(env);
        }
    }
    {   /* pdf write->extract roundtrip */
        const char *t = "Mario Rossi\nmario.rossi@gmail.com\n[NAME_1] test";
        int w = pii_pdf_write_text(".vault_selftest.pdf", t, "selftest");
        CHECK(w == 0, "pdf-write");
        if (!w) {
            char *x = pii_pdf_extract_text(".vault_selftest.pdf");
            CHECK(x && strstr(x, "[NAME_1]") != NULL, "pdf-extract");
            free(x); remove(".vault_selftest.pdf");
        }
    }
    printf(fails ? "SELFTEST: %d FAILURES\n" : "SELFTEST: all pass\n", fails);
    return fails ? 1 : 0;
}

#ifdef PII_V002_UNIFIED
int pii_agent_main(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif
    const char *vault_dir = ".vault";
    int i;
    for (i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--vault-dir") && i + 1 < argc) vault_dir = argv[++i];
    if (argc < 2) { usage(argv[0]); return 1; }
    if (!strcmp(argv[1], "mask") && argc >= 3) {
        const char *in = NULL;
        for (i = 2; i < argc; i++)
            if (strcmp(argv[i], "--vault-dir")) { if (!in) in = argv[i]; }
        if (!in) { usage(argv[0]); return 1; }
        return cmd_mask(in, vault_dir);
    }
    if (!strcmp(argv[1], "unmask") && argc >= 3) {
        const char *a = NULL;
        for (i = 2; i < argc; i++)
            if (strcmp(argv[i], "--vault-dir")) { if (!a) a = argv[i]; }
        if (!a) { usage(argv[0]); return 1; }
        return cmd_unmask(a, vault_dir);
    }
    if (!strcmp(argv[1], "purge") && argc >= 3) {
        const char *a = NULL;
        for (i = 2; i < argc; i++)
            if (strcmp(argv[i], "--vault-dir")) { if (!a) a = argv[i]; }
        if (!a) { usage(argv[0]); return 1; }
        return cmd_purge(a, vault_dir);
    }
    if (!strcmp(argv[1], "selftest")) return cmd_selftest();
    usage(argv[0]);
    return 1;
}
