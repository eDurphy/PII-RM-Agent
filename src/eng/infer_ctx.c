/* PII-AGENT v002 :: infer_ctx.c — flash inference context (I-weights only).
 *
 * Compiled ONLY into pii-agent-flash.exe (never into the full engine TU).
 * eng_infer_load(dir): enumerates I_*.bin, validates every header
 * (magic/kind/slot/dims/size — same contract as the engine chunk reader)
 * and dynamically loads the trits into RAM for the session. S_* (shadow)
 * files are never opened — not even for reading. Nothing is written.
 * Returns the chunk count, or -1 on any failure (fail-closed).
 *
 * NOTE: same-TU code — #included by eng_infer_tu.c AFTER nn_math.c, so
 * ooc_hdr_t / ooc_paths / die are visible. No header: private to the TU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

#define INFER_WDIR_CAP 520

typedef struct { void *trits; size_t n; int slot; int n_in; int n_out; } InferChunk;
static InferChunk *g_infer_chunks = NULL;
static int g_infer_n = 0;

static long infer_parse_slot(const char *base)
{
    const char *dot, *p;
    char *end = NULL;
    long v;
    if (strncmp(base, "I_", 2)) return -1;
    dot = strrchr(base, '.');
    if (!dot || strcmp(dot, ".bin")) return -1;
    p = base + 2;
    if (p >= dot) return -1;
    v = strtol(p, &end, 10);
    if (end != dot || v < 0 || v > 1000000000L) return -1;
    return v;
}

/* Load + validate one I chunk. Returns 0 ok, -1 fail. */
static int infer_load_one(const char *dir, const char *base, long slot)
{
    char full[INFER_WDIR_CAP + 64];
    FILE *f = NULL;
    ooc_hdr_t h;
    uint64_t expect;
    unsigned char *tri = NULL;
    InferChunk *grow = NULL;
    int r, rc = -1;

    r = snprintf(full, sizeof full, "%s/%s", dir, base);
    if (r < 0 || (size_t)r >= sizeof full) goto out;
    f = fopen(full, "rb");                    /* READ-ONLY, never write */
    if (!f) goto out;
    if (fread(&h, 1, sizeof h, f) != sizeof h) goto out;
    if (h.magic != OOC_MAGIC) goto out;
    if (h.kind != (uint32_t)OOC_KIND_I) goto out;
    if (h.slot != (uint32_t)slot) goto out;
    if (h.n_in <= 0 || h.n_out <= 0 || h.n == 0) goto out;
    expect = (uint64_t)(uint32_t)h.n_in * (uint64_t)(uint32_t)h.n_out;
    if (h.n != expect) goto out;
    tri = (unsigned char *)malloc((size_t)h.n);
    if (!tri) goto out;
    if (fread(tri, 1, (size_t)h.n, f) != (size_t)h.n) goto out;
    grow = (InferChunk *)realloc(g_infer_chunks,
                                (size_t)(g_infer_n + 1) * sizeof *grow);
    if (!grow) goto out;
    g_infer_chunks = grow;
    g_infer_chunks[g_infer_n].trits = tri;
    g_infer_chunks[g_infer_n].n = (size_t)h.n;
    g_infer_chunks[g_infer_n].slot = (int)slot;
    g_infer_chunks[g_infer_n].n_in = h.n_in;
    g_infer_chunks[g_infer_n].n_out = h.n_out;
    tri = NULL;
    g_infer_n++;
    rc = 0;
out:
    free(tri);
    if (f) fclose(f);
    return rc;
}

int eng_infer_load(const char *dir)
{
    char q[INFER_WDIR_CAP + 8];
    int r, bad = 0, n0;
    size_t total = 0;
    int i;
    if (!dir || !*dir) return -1;
    r = snprintf(q, sizeof q, "%s", dir);
    if (r < 0 || (size_t)r >= sizeof q) return -1;
    n0 = g_infer_n;
#ifdef _WIN32
    {
        char pat[INFER_WDIR_CAP + 8];
        WIN32_FIND_DATAA fd;
        HANDLE h;
        r = snprintf(pat, sizeof pat, "%s/I_*.bin", dir);
        if (r < 0 || (size_t)r >= sizeof pat) return -1;
        h = FindFirstFileA(pat, &fd);
        if (h == INVALID_HANDLE_VALUE) return -1;
        do {
            long slot;
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
            slot = infer_parse_slot(fd.cFileName);
            if (slot < 0) { bad = 1; break; }
            if (infer_load_one(dir, fd.cFileName, slot)) { bad = 1; break; }
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }
#else
    {
        DIR *d = opendir(dir);
        struct dirent *e;
        size_t dl;
        if (!d) return -1;
        dl = strlen(dir);
        while ((e = readdir(d)) != NULL) {
            char full[INFER_WDIR_CAP + 64];
            size_t bl = strlen(e->d_name);
            long slot;
            if (bl < 7 || strncmp(e->d_name, "I_", 2) ||
                strcmp(e->d_name + bl - 4, ".bin"))
                continue;
            slot = infer_parse_slot(e->d_name);
            if (slot < 0) { bad = 1; break; }
            if (dl + 1 + bl >= sizeof full) { bad = 1; break; }
            memcpy(full, dir, dl);
            full[dl] = '/';
            memcpy(full + dl + 1, e->d_name, bl + 1);
            /* validate through the same reader path */
            if (infer_load_one(dir, full + dl + 1, slot)) { bad = 1; break; }
        }
        closedir(d);
    }
#endif
    if (bad || g_infer_n == n0) return -1;
    for (i = n0; i < g_infer_n; i++) total += g_infer_chunks[i].n;
    printf("weights: %d inference (I) chunks loaded [%s] (%.1f MB, read-only)\n",
           g_infer_n - n0, dir, (double)total / 1048576.0);
    return g_infer_n - n0;
}

void eng_infer_free(void)
{
    int i;
    for (i = 0; i < g_infer_n; i++) free(g_infer_chunks[i].trits);
    free(g_infer_chunks);
    g_infer_chunks = NULL;
    g_infer_n = 0;
}

/* ================= v003 hybrid spatial-visual forward =================
 * Routes the structured document (text + 2D boxes + image patches)
 * through the LOADED network in one latent space:
 *   1. per page:   patch-project image patches -> page visual vector
 *   2. per span:   hash token + 2D coordinate embed + visual bias
 *                  -> fused PII_MM_D feature
 *   3. through chunk[0] trits (paper sec-4 shape: acc += trit *
 *      (feat>>decay), SiLUNS r7; decay=3 is the paper init default).
 * Returns malloc'd per-span salience (doc order) + count, or NULL.
 * Flash-safe: integer-only, no training state, no BPE dependency.
 */
#include "nn_spatial.h"
#include "pii_pdf.h"

static int32_t mm_siluns_r7(int32_t x)
{
    if (x >= 0) return x;
    if (x > -41900) return x >> 2;
    return -(x >> 2) - 20950;
}

static void mm_bias_add(int32_t *f, const int32_t *s, int d, int sd)
{
    int i;
    for (i = 0; i < d; i++) {
        int64_t v = (int64_t)f[i] + (int64_t)s[i % sd];
        if (v > 2147483647LL) v = 2147483647LL;
        if (v < -2147483648LL) v = -2147483648LL;
        f[i] = (int32_t)v;
    }
}

/* average patch projection over a page's images -> visual vector */
static void mm_page_visual(const PiiPage *pg, int32_t *vis)
{
    int k, n = 0, m;
    for (k = 0; k < PII_MM_D; k++) vis[k] = 0;
    if (!pg) return;
#ifdef PII_HYBRID_SPATIAL
    for (m = 0; m < pg->nimgs; m++) {
        const PiiImage *im = &pg->imgs[m];
        int px[4], py[4], t;
        if (!im->px || im->w < PII_PATCH || im->h < PII_PATCH) continue;
        px[0] = 0;                   py[0] = 0;
        px[1] = im->w - PII_PATCH;    py[1] = 0;
        px[2] = 0;                   py[2] = im->h - PII_PATCH;
        px[3] = (im->w - PII_PATCH) / 2; py[3] = (im->h - PII_PATCH) / 2;
        for (t = 0; t < 4; t++) {
            unsigned char patch[PII_PATCH_VEC];
            int32_t proj[PII_MM_D];
            int x, y;
            if (px[t] < 0 || py[t] < 0) continue;
            for (y = 0; y < PII_PATCH; y++)
                for (x = 0; x < PII_PATCH; x++) {
                    const unsigned char *src = im->px +
                        ((size_t)(py[t] + y) * (size_t)im->w +
                         (unsigned)(px[t] + x)) * 3u;
                    memcpy(patch + (size_t)(y * PII_PATCH + x) * 3u, src, 3);
                }
            mm_patch_project(patch, proj, PII_MM_D);
            for (k = 0; k < PII_MM_D; k++) vis[k] += proj[k] >> 4;
            n++;
        }
    }
#endif
    if (n > 1) { int k2; for (k2 = 0; k2 < PII_MM_D; k2++) vis[k2] /= n; }
}

/* push a fused feature through chunk[0] trits -> energy score */
static int32_t mm_chunk_probe(const int32_t *feat)
{
    const int8_t *tr;
    int32_t o, i, en = 0;
    int nin, nout;
    int32_t *buf;
    if (g_infer_n <= 0 || !feat) return 0;
    nin = g_infer_chunks[0].n_in;
    nout = g_infer_chunks[0].n_out;
    if (nin <= 0 || nout <= 0) return 0;
    tr = (const int8_t *)g_infer_chunks[0].trits;
    buf = (int32_t *)malloc((size_t)nin * sizeof(int32_t));
    if (!buf) return 0;
    for (i = 0; i < nin; i++) buf[i] = feat[i % PII_MM_D] >> 3;
    for (o = 0; o < nout; o++) {
        int64_t acc = 0;
        for (i = 0; i < nin; i++) {
            /* payload assumed 1-byte trits; sign-clamped so packed
             * base-3 payloads still run (documented limitation). */
            int8_t t = tr[(size_t)i * (size_t)nout + (size_t)o];
            int tt = t > 0 ? 1 : (t < 0 ? -1 : 0);
            if (tt == 1) acc += buf[i];
            else if (tt == -1) acc -= buf[i];
        }
        { int32_t a = mm_siluns_r7((int32_t)(acc & 0x7FFFFFFFLL));
          en += (a < 0 ? -a : a) >> 10; }
        if (en < 0) { en = 2147483647L; break; }
    }
    free(buf);
    return en;
}

float *eng_infer_run_document(const PiiDoc *doc, int *n_out)
{
    float *sal = NULL;
    int total = 0, p, s, idx;
    if (n_out) *n_out = 0;
    if (!doc) return NULL;
    for (p = 0; p < doc->npages; p++) total += doc->pages[p].nspans;
    if (total <= 0) return NULL;
    sal = (float *)malloc((size_t)total * sizeof(float));
    if (!sal) return NULL;
    idx = 0;
    for (p = 0; p < doc->npages; p++) {
        const PiiPage *pg = &doc->pages[p];
        int32_t vis[PII_MM_D];
        mm_page_visual(pg, vis);
        for (s = 0; s < pg->nspans; s++) {
            const PiiSpan *sp = &pg->spans[s];
            int32_t feat[PII_MM_D], semb[PII_SDIM];
            int cx = (int)(sp->box.x0 + sp->box.x1) / 2;
            int cy = (int)(sp->box.y0 + sp->box.y1) / 2;
            uint32_t h = mm_hash(sp->text, strlen(sp->text));
            int k;
            mm_zone_t z;
            mm_spatial_embed(cx, cy, semb);
            for (k = 0; k < PII_MM_D; k++) {
                uint32_t m = h + (uint32_t)k * 0x9E3779B9u;
                m ^= m >> 15; m *= 0x85EBCA6Bu; m ^= m >> 13;
                feat[k] = (int32_t)(m & 0xFFFFu) - 32768 + semb[k % PII_SDIM];
            }
            mm_bias_add(feat, vis, PII_MM_D, PII_MM_D);
            z = mm_classify_zone((int)sp->box.x0, (int)sp->box.y0,
                                 (int)sp->box.x1, (int)sp->box.y1,
                                 pg->nimgs > 0);
            { int32_t en = mm_chunk_probe(feat);
              double v = (double)en * (double)mm_zone_boost(z);
              sal[idx++] = (float)(v / 1048576.0); }
        }
    }
    if (n_out) *n_out = total;
    return sal;
}
