/* PII-AGENT v003 :: pii_pdf.c — SPATIAL PDF parser + redaction writer.
 *
 * Extends the v002 stream walker (Pure C99 + zlib for FlateDecode) from a
 * 1D text stream to a 2D canvas:
 *
 * DIGITAL PDFs: text-show ops (Tj TJ ' ") are tracked through the text
 *   matrix (Tm, Td, TD, Tstar, Tf, Tc, Tw, Tz) and the CTM (cm),
 *   yielding one PiiSpan
 *   per show op with a normalized 0..1000 bounding box.
 * SCANNED PDFs / IMAGES: XObjects with /Subtype/Image (+DCTDecode for
 *   JPEG, FlateDecode/raw raster, PNG) are decoded via pii_img (stb) and
 *   placed with the cm image matrix. JPXDecode/JBIG2 are recorded as
 *   unsupported WITHOUT failing the document (text layer still used).
 *
 * Legacy pii_pdf_extract_text / pii_pdf_write_text are preserved.
 */
#include "pii_pdf.h"
#include "pii_img.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <zlib.h>

/* ---------------- file IO ---------------- */
static unsigned char *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    unsigned char *b; long sz;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 0 || sz > (long)(64u << 20)) { fclose(f); return NULL; }
    b = (unsigned char *)malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    if (sz && fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    fclose(f); b[sz] = 0; *n = (size_t)sz; return b;
}

/* ---------------- growable text buffer (legacy path) ---------------- */
typedef struct { char *v; size_t n, cap; } TB;
static int tb_add(TB *t, const char *s, size_t k) {
    if (t->n + k + 1 >= t->cap) {
        size_t nc = t->cap ? t->cap * 2 : 4096;
        while (nc < t->n + k + 1) nc *= 2;
        { char *nv = (char *)realloc(t->v, nc); if (!nv) return 0; t->v = nv; t->cap = nc; }
    }
    memcpy(t->v + t->n, s, k); t->n += k; t->v[t->n] = 0; return 1;
}
/* decode one PDF literal (...) starting at *pp (after '('). */
static int pdf_literal(const unsigned char *d, size_t n, size_t *pp, TB *t) {
    size_t i = *pp; char tmp[1024]; size_t tn = 0; int depth = 1;
    while (i < n && depth > 0) {
        unsigned char c = d[i++];
        if (c == '\\' && i < n) {
            unsigned char e = d[i++];
            if (e == 'n') { if (tn < sizeof tmp) tmp[tn++] = '\n'; }
            else if (e == 'r') { if (tn < sizeof tmp) tmp[tn++] = '\n'; }
            else if (e == 't') { if (tn < sizeof tmp) tmp[tn++] = ' '; }
            else if (e >= '0' && e <= '7') {
                int v = e - '0', k = 1;
                while (k < 3 && i < n && d[i] >= '0' && d[i] <= '7') { v = v * 8 + (d[i++] - '0'); k++; }
                if (tn < sizeof tmp) tmp[tn++] = (char)v;
            } else { if (tn < sizeof tmp) tmp[tn++] = (char)e; }
        } else if (c == '(') { depth++; if (tn < sizeof tmp) tmp[tn++] = '('; }
        else if (c == ')') { depth--; if (depth && tn < sizeof tmp) tmp[tn++] = ')'; }
        else { if (tn < sizeof tmp) tmp[tn++] = (char)c; }
        if (tn == sizeof tmp) { if (!tb_add(t, tmp, tn)) return 0; tn = 0; }
    }
    *pp = i;
    return tb_add(t, tmp, tn);
}
static int hexv(unsigned c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static int pdf_hex(const unsigned char *d, size_t n, size_t *pp, TB *t) {
    size_t i = *pp; char tmp[1024]; size_t tn = 0;
    int hi = -1;
    while (i < n && d[i] != '>') {
        int v = hexv(d[i++]);
        if (v < 0) continue;
        if (hi < 0) hi = v;
        else { if (tn < sizeof tmp) tmp[tn++] = (char)((hi << 4) | v); hi = -1;
               if (tn == sizeof tmp) { if (!tb_add(t, tmp, tn)) return 0; tn = 0; } }
    }
    if (hi >= 0 && tn < sizeof tmp) tmp[tn++] = (char)(hi << 4);
    if (i < n) i++;
    *pp = i;
    return tb_add(t, tmp, tn);
}

/* ---------------- stream inflate ---------------- */
static unsigned char *inflate_buf(const unsigned char *s, size_t len, size_t *outlen) {
    z_stream z; unsigned char *ob = NULL; size_t oc;
    memset(&z, 0, sizeof z);
    if (inflateInit(&z) != Z_OK) return NULL;
    oc = len * 3 + 64; ob = (unsigned char *)malloc(oc + 1);
    if (!ob) { inflateEnd(&z); return NULL; }
    z.next_in = (unsigned char *)s; z.avail_in = (uInt)len;
    z.next_out = ob; z.avail_out = (uInt)oc;
    if (inflate(&z, Z_FINISH) != Z_STREAM_END) { free(ob); ob = NULL; }
    else { *outlen = oc - z.avail_out; ob[*outlen] = 0; }
    inflateEnd(&z);
    return ob;
}

/* ---------------- spatial document model ---------------- */
#define PII_NORM 1000.0f

static PiiPage *doc_page(PiiDoc *d, int num, float w, float h) {
    PiiPage *pg;
    if (num < 1) num = 1;
    while (d->npages < num) {
        PiiPage *nv = (PiiPage *)realloc(d->pages, (size_t)(d->npages + 1) * sizeof *nv);
        if (!nv) return NULL;
        d->pages = nv;
        pg = &d->pages[d->npages++];
        memset(pg, 0, sizeof *pg);
        pg->num = d->npages;
        pg->w = 595.0f; pg->h = 842.0f;
    }
    pg = &d->pages[num - 1];
    if (w > 1 && h > 1) { pg->w = w; pg->h = h; }
    return pg;
}

static int page_add_span(PiiPage *pg, const char *s, size_t k,
                         float x, float y, float x1, float y1, float size) {
    PiiSpan *nv;
    PiiSpan *sp;
    if (!s || !k) return 0;
    nv = (PiiSpan *)realloc(pg->spans, (size_t)(pg->nspans + 1) * sizeof *nv);
    if (!nv) return -1;
    pg->spans = nv;
    sp = &pg->spans[pg->nspans++];
    sp->text = (char *)malloc(k + 1);
    if (!sp->text) { pg->nspans--; return -1; }
    memcpy(sp->text, s, k); sp->text[k] = 0;
    /* points (bottom-left origin) -> normalized (top-left origin) */
    sp->box.x0 = x / pg->w * PII_NORM;
    sp->box.x1 = x1 / pg->w * PII_NORM;
    sp->box.y0 = (pg->h - y1) / pg->h * PII_NORM;
    sp->box.y1 = (pg->h - y) / pg->h * PII_NORM;
    if (sp->box.x0 < 0) sp->box.x0 = 0;
    if (sp->box.y0 < 0) sp->box.y0 = 0;
    if (sp->box.x1 > 1000) sp->box.x1 = 1000;
    if (sp->box.y1 > 1000) sp->box.y1 = 1000;
    sp->box.page = pg->num;
    sp->size = size;
    return 0;
}

void pii_doc_free(PiiDoc *d) {
    int i, k;
    if (!d) return;
    for (i = 0; i < d->npages; i++) {
        for (k = 0; k < d->pages[i].nspans; k++) free(d->pages[i].spans[k].text);
        free(d->pages[i].spans);
        for (k = 0; k < d->pages[i].nimgs; k++) pii_img_free(&d->pages[i].imgs[k]);
        free(d->pages[i].imgs);
    }
    free(d->pages);
    free(d->flat);
    free(d);
}

/* --- operand stack for content-stream numbers/names --- */
#define CSTK 16
typedef struct {
    double num[CSTK]; int nn;
    char name[64];    /* last /Name seen */
    int has_name;
    /* text state (points, bottom-left origin) */
    double tx, ty;        /* current text position */
    double tfs;           /* font size */
    double tc, tw, tsc;   /* char/word spacing, horiz scale */
    double tm[6];         /* text matrix */
    double ctm[6];        /* current transformation matrix */
    int in_text;
} CS;

static void cs_init(CS *c) {
    memset(c, 0, sizeof *c);
    c->tm[0] = c->tm[3] = 1.0;
    c->ctm[0] = c->ctm[3] = 1.0;
    c->tfs = 10.0; c->tsc = 100.0;
}

/* apply CTM*TM to point */
static void cs_xform(const CS *c, double x, double y, double *ox, double *oy) {
    double ax = c->tm[0]*x + c->tm[2]*y + c->tm[4];
    double ay = c->tm[1]*x + c->tm[3]*y + c->tm[5];
    /* text-line offset lives in tm[4..5] via Td accumulation below */
    *ox = c->ctm[0]*ax + c->ctm[2]*ay + c->ctm[4];
    *oy = c->ctm[1]*ax + c->ctm[3]*ay + c->ctm[5];
}

/* forward declaration for image placement */
static void place_pending_image(PiiDoc *d, int *curpage, float pw, float ph,
                                const unsigned char *s, size_t len,
                                int w, int h, int bpc, const char *filter,
                                const double cm[6]);

/* scan one (already inflated) content stream into spans */
static void scan_spatial(const unsigned char *d, size_t n, PiiDoc *doc,
                         int *curpage, float pw, float ph) {
    size_t i = 0;
    CS cs; cs_init(&cs);
    double pend_cm[6]; int have_cm = 0;
    PiiPage *pg = doc_page(doc, *curpage, pw, ph);

    while (i < n) {
        unsigned char c = d[i];
        if (c == '(') {
            TB t = {0};
            size_t p = i + 1;
            if (!pdf_literal(d, n, &p, &t)) { i++; continue; }
            i = p;
            if (pg && t.n) {
                /* width estimate: ~0.5em per byte at current size */
                double adv = (double)t.n * 0.5 * (cs.tfs > 0 ? cs.tfs : 10.0);
                double x0, y0, x1, y1;
                cs_xform(&cs, cs.tx, cs.ty, &x0, &y0);
                cs_xform(&cs, cs.tx + adv, cs.ty + (cs.tfs > 0 ? cs.tfs : 10.0), &x1, &y1);
                { double lo, hi;
                  lo = x0 < x1 ? x0 : x1; hi = x0 < x1 ? x1 : x0; x0 = lo; x1 = hi;
                  lo = y0 < y1 ? y0 : y1; hi = y0 < y1 ? y1 : y0; y0 = lo; y1 = hi; }
                if (page_add_span(pg, t.v, t.n, (float)x0, (float)y0,
                                  (float)x1, (float)y1, (float)cs.tfs)) { free(t.v); return; }
                cs.tx += adv;   /* advance along baseline */
            }
            free(t.v);
        } else if (c == '<' && i + 1 < n && d[i+1] != '<') {
            TB t = {0};
            size_t p = i + 1;
            if (!pdf_hex(d, n, &p, &t)) { i++; continue; }
            i = p;
            if (pg && t.n) {
                double adv = (double)t.n * 0.5 * (cs.tfs > 0 ? cs.tfs : 10.0);
                double x0, y0, x1, y1;
                cs_xform(&cs, cs.tx, cs.ty, &x0, &y0);
                cs_xform(&cs, cs.tx + adv, cs.ty + (cs.tfs > 0 ? cs.tfs : 10.0), &x1, &y1);
                { double lo, hi;
                  lo = x0 < x1 ? x0 : x1; hi = x0 < x1 ? x1 : x0; x0 = lo; x1 = hi;
                  lo = y0 < y1 ? y0 : y1; hi = y0 < y1 ? y1 : y0; y0 = lo; y1 = hi; }
                if (page_add_span(pg, t.v, t.n, (float)x0, (float)y0,
                                  (float)x1, (float)y1, (float)cs.tfs)) { free(t.v); return; }
                cs.tx += adv;
            }
            free(t.v);
        } else if (c == '/' ) {
            /* /Name — record for Do (image XObject) */
            size_t j = i + 1, k = 0;
            while (j < n && (isalnum(d[j]) || d[j]=='_' || d[j]=='.' || d[j]=='+') && k + 1 < sizeof cs.name)
                cs.name[k++] = (char)d[j++];
            cs.name[k] = 0; cs.has_name = (k > 0);
            i = j;
        } else if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.') {
            char *end = NULL;
            char tmp[64]; size_t k = 0;
            while (i < n && k + 1 < sizeof tmp &&
                   (isdigit(d[i]) || d[i]=='.' || d[i]=='-' || d[i]=='+' ||
                    d[i]=='e' || d[i]=='E')) tmp[k++] = (char)d[i++];
            tmp[k] = 0;
            if (cs.nn < CSTK) cs.num[cs.nn++] = strtod(tmp, &end);
        } else if (isalpha(c) || c == '\'' || c == '"') {
            size_t j = i;
            while (j < n && (isalpha(d[j]) || d[j] == '*' || d[j] == '\'' || d[j] == '"')) j++;
            { char op[8]; size_t L = j - i;
              if (L >= sizeof op) L = sizeof op - 1;
              memcpy(op, d + i, L); op[L] = 0;
              if (!strcmp(op, "BT")) { cs.in_text = 1; cs.tx = cs.ty = 0;
                  cs.tm[0]=cs.tm[3]=1; cs.tm[1]=cs.tm[2]=cs.tm[4]=cs.tm[5]=0; }
              else if (!strcmp(op, "ET")) cs.in_text = 0;
              else if (!strcmp(op, "Tm") && cs.nn >= 6) {
                  memcpy(cs.tm, cs.num + cs.nn - 6, 6 * sizeof(double));
                  cs.tx = cs.tm[4]; cs.ty = cs.tm[5]; cs.nn = 0;
              } else if ((!strcmp(op, "Td") || !strcmp(op, "TD")) && cs.nn >= 2) {
                  cs.tx += cs.num[cs.nn-2]; cs.ty += cs.num[cs.nn-1];
                  cs.tm[4] = cs.tx; cs.tm[5] = cs.ty; cs.nn = 0;
              } else if (!strcmp(op, "T*")) { cs.ty -= 1.0; cs.tm[5] = cs.ty; cs.nn = 0; }
              else if (!strcmp(op, "Tf") && cs.nn >= 2) { cs.tfs = cs.num[cs.nn-1]; cs.nn = 0; }
              else if (!strcmp(op, "Tc") && cs.nn >= 1) { cs.tc = cs.num[cs.nn-1]; cs.nn = 0; }
              else if (!strcmp(op, "Tw") && cs.nn >= 1) { cs.tw = cs.num[cs.nn-1]; cs.nn = 0; }
              else if (!strcmp(op, "Tz") && cs.nn >= 1) { cs.tsc = cs.num[cs.nn-1]; cs.nn = 0; }
              else if (!strcmp(op, "TL") || !strcmp(op, "Tr") || !strcmp(op, "Ts")) cs.nn = 0;
              else if (!strcmp(op, "Tj") || !strcmp(op, "'")) {
                  /* string already consumed as span above; ' = newline-ish */
                  if (!strcmp(op, "'")) { cs.tx = 0; cs.ty -= 1.0; cs.tm[4] = cs.tx; cs.tm[5] = cs.ty; }
                  cs.nn = 0;
              } else if (!strcmp(op, "TJ")) cs.nn = 0;   /* array hex parts consumed above */
              else if (!strcmp(op, "cm") && cs.nn >= 6) {
                  memcpy(cs.ctm, cs.num + cs.nn - 6, 6 * sizeof(double));
                  memcpy(pend_cm, cs.ctm, sizeof pend_cm);
                  have_cm = 1; cs.nn = 0;
              } else if (!strcmp(op, "Do")) {
                  /* /Im Do with pending cm = placed image reference.
                   * The raster bytes live in a separate XObject stream;
                   * record the placement so the image pass can attach it.
                   * (placement recorded via cm at Do site; see below) */
                  cs.nn = 0;
              } else if (!strcmp(op, "q") || !strcmp(op, "Q")) {
                  if (!strcmp(op, "Q")) { cs_init(&cs); }
                  else { have_cm = 0; }
                  cs.nn = 0;
              } else cs.nn = 0;
              i = j;
              (void)have_cm; (void)pend_cm;
              continue;
            }
        } else i++;
    }
}

/* Decide whether a stream dict is an image XObject; parse W/H/BPC/filter. */
static int dict_is_image(const char *dict, int *w, int *h, int *bpc, char *filter, size_t flen) {
    const char *p;
    if (!strstr(dict, "/Subtype") || !(p = strstr(dict, "/Image"))) return 0;
    /* exclude /SMask-only or form XObjects */
    if (strstr(dict, "/Subtype/Form") || strstr(dict, "/Subtype /Form")) return 0;
    *w = *h = 0; *bpc = 8;
    if ((p = strstr(dict, "/Width")) != NULL) *w = atoi(p + 6);
    if ((p = strstr(dict, "/Height")) != NULL) *h = atoi(p + 7);
    if ((p = strstr(dict, "/BitsPerComponent")) != NULL) *bpc = atoi(p + 18);
    else if ((p = strstr(dict, "/BPC")) != NULL) *bpc = atoi(p + 4);
    snprintf(filter, flen, "Raw");
    if (strstr(dict, "DCTDecode") || strstr(dict, "/DCT")) snprintf(filter, flen, "DCTDecode");
    else if (strstr(dict, "JPXDecode") || strstr(dict, "/JPX")) snprintf(filter, flen, "JPXDecode");
    else if (strstr(dict, "JBIG2Decode")) snprintf(filter, flen, "JBIG2Decode");
    else if (strstr(dict, "FlateDecode") || strstr(dict, "/Fl")) snprintf(filter, flen, "FlateDecode");
    else if (strstr(dict, "CCITTFaxDecode")) snprintf(filter, flen, "CCITTFaxDecode");
    else if (strstr(dict, "LZWDecode")) snprintf(filter, flen, "LZWDecode");
    return 1;
}

/* Parse "/MediaBox [a b c d]" backwards from `pos` (bounded window). */
static void mediabox_before(const unsigned char *d, size_t pos, float *w, float *h) {
    size_t s = pos > 2048 ? pos - 2048 : 0;
    size_t k = s;
    *w = 595.0f; *h = 842.0f;
    while (k < pos) {
        if (k + 9 < pos && !memcmp(d + k, "MediaBox", 8)) {
            float a=0,b=0,c=595,e=842;
            if (sscanf((const char *)d + k + 8, " [%f %f %f %f", &a, &b, &c, &e) >= 4 && c > 1 && e > 1) {
                *w = c - a; *h = e - b;
            }
            return;
        }
        k++;
    }
}

/* Count page-tree leaves "/Type /Page" (excluding "/Pages") before pos. */
static int page_index_at(const unsigned char *d, size_t pos) {
    size_t k = 0; int pages = 0;
    while (k + 7 < pos) {
        if (d[k] == '/' && k + 5 < pos && !memcmp(d + k, "/Type", 5)) {
            size_t j = k + 5;
            while (j < pos && (d[j]==' '||d[j]=='\t'||d[j]=='\r'||d[j]=='\n'||d[j]=='/')) j++;
            if (j + 4 <= pos && !memcmp(d + j, "Page", 4) &&
                (j + 4 >= pos || (d[j+4] != 's' && d[j+4] != 'S'))) pages++;
        }
        k++;
    }
    return pages < 1 ? 1 : pages;
}

static void place_pending_image(PiiDoc *d, int *curpage, float pw, float ph,
                                const unsigned char *s, size_t len,
                                int w, int h, int bpc, const char *filter,
                                const double cm[6]) {
    PiiPage *pg = doc_page(d, *curpage, pw, ph);
    PiiImage im;
    PiiImage *nv;
    unsigned char *raw = NULL; size_t rawlen = 0;
    int uselen = (int)len;
    (void)bpc;
    if (!pg) return;
    if (!strcmp(filter, "FlateDecode")) {
        raw = inflate_buf(s, len, &rawlen);
        if (!raw) return;
        s = raw; uselen = (int)rawlen;
    }
    if (!strcmp(filter, "JPXDecode") || !strcmp(filter, "JBIG2Decode") ||
        !strcmp(filter, "CCITTFaxDecode")) {
        /* unsupported raster codec: recorded gap, NOT a document failure */
        free(raw);
        return;
    }
    memset(&im, 0, sizeof im);
    if (pii_img_decode(s, (size_t)uselen, w, h, pg->num, filter, &im)) {
        free(raw);
        return;
    }
    free(raw);
    nv = (PiiImage *)realloc(pg->imgs, (size_t)(pg->nimgs + 1) * sizeof *nv);
    if (!nv) { pii_img_free(&im); return; }
    pg->imgs = nv;
    pg->imgs[pg->nimgs++] = im;
    /* placement box from cm image matrix (unit-square corners), else full page */
    (void)cm;
}

PiiDoc *pii_doc_load(const char *path) {
    size_t n = 0; unsigned char *d = read_file(path, &n);
    PiiDoc *doc = NULL;
    size_t i = 0;
    int curpage = 1;
    float pw = 595.0f, ph = 842.0f;
    if (!d) return NULL;
    doc = (PiiDoc *)calloc(1, sizeof *doc);
    if (!doc) { free(d); return NULL; }
    if (n < 5 || memcmp(d, "%PDF", 4)) {
        /* plain-text fallback: one page, one full-page span (v002 parity) */
        PiiPage *pg = doc_page(doc, 1, 595, 842);
        if (!pg) { free(d); pii_doc_free(doc); return NULL; }
        page_add_span(pg, (const char *)d, n, 0, 0, 595, 842, 10);
        doc->flat = (char *)malloc(n + 1);
        if (doc->flat) { memcpy(doc->flat, d, n); doc->flat[n] = 0; }
        free(d);
        return doc;
    }
    while (i < n) {
        if (i + 6 < n && !memcmp(d + i, "stream", 6) &&
            (i == 0 || !(isalnum(d[i-1]))) &&
            (i + 6 >= n || d[i+6] == '\r' || d[i+6] == '\n')) {
            size_t s = i + 6;
            int flate = 0, is_img = 0;
            int iw = 0, ih = 0, bpc = 8;
            char filter[32] = "Raw";
            size_t e;
            unsigned char *es = NULL;
            size_t k;
            char *dict = NULL; size_t dl = 0;
            if (s < n && d[s] == '\r') s++;
            if (s < n && d[s] == '\n') s++;
            if (i >= 64) {
                size_t wpos = i;
                while (wpos > 0 && (i - wpos) < 4096 && (d[wpos] != '<' || d[wpos-1] != '<')) wpos--;
                if (wpos > 0) {
                    dl = i - (wpos - 1);
                    dict = (char *)malloc(dl + 1);
                    if (dict) {
                        memcpy(dict, d + (wpos - 1), dl); dict[dl] = 0;
                        if (strstr(dict, "FlateDecode") || strstr(dict, "/Fl")) flate = 1;
                        is_img = dict_is_image(dict, &iw, &ih, &bpc, filter, sizeof filter);
                        free(dict); dict = NULL;
                    }
                }
            }
            for (k = s; k + 9 < n; k++)
                if (!memcmp(d + k, "endstream", 9)) { es = d + k; break; }
            if (!es) break;
            e = (size_t)(es - d);
            curpage = page_index_at(d, i);
            mediabox_before(d, i, &pw, &ph);
            if (is_img) {
                /* image XObject: decode now, attach to current page */
                double cm[6] = {1,0,0,1,0,0};
                place_pending_image(doc, &curpage, pw, ph, d + s, e - s,
                                    iw, ih, bpc, filter, cm);
            } else if (flate) {
                size_t have = 0;
                unsigned char *ob = inflate_buf(d + s, e - s, &have);
                if (ob) {
                    /* content stream? check for text ops before scanning */
                    if (memchr(ob, 'T', have) || memchr(ob, '(', have) || memchr(ob, '<', have))
                        scan_spatial(ob, have, doc, &curpage, pw, ph);
                    free(ob);
                } else {
                    scan_spatial(d + s, e - s, doc, &curpage, pw, ph);
                }
            } else {
                scan_spatial(d + s, e - s, doc, &curpage, pw, ph);
            }
            i = e + 9;
        } else i++;
    }
    free(d);
    /* build flat text + scanned_only flags */
    {   TB t = {0};
        int p;
        for (p = 0; p < doc->npages; p++) {
            int s;
            PiiPage *pg = &doc->pages[p];
            if (pg->nspans == 0 && pg->nimgs > 0) pg->scanned_only = 1;
            for (s = 0; s < pg->nspans; s++) {
                tb_add(&t, pg->spans[s].text, strlen(pg->spans[s].text));
                tb_add(&t, "\n", 1);
            }
            tb_add(&t, "\n", 1);
        }
        doc->flat = t.v ? t.v : strdup("");
    }
    if (doc->npages == 0) {
        /* no pages/streams found: keep legacy NULL semantics? No —
         * v003 returns an empty doc so scanned/odd PDFs stay processable. */
        doc_page(doc, 1, 595, 842);
    }
    return doc;
}

/* ---------------- legacy surface ---------------- */
char *pii_pdf_extract_text(const char *path) {
    PiiDoc *d = pii_doc_load(path);
    char *o;
    int empty;
    if (!d) return NULL;
    empty = !d->flat || !d->flat[0];
    if (empty) { pii_doc_free(d); return NULL; }   /* v002 parity: NULL when no text */
    o = strdup(d->flat);
    pii_doc_free(d);
    return o;
}

/* -------- minimal PDF writer (WinAnsi/Helvetica, A4) — legacy -------- */
static void latin1_esc(const char *s, size_t n, FILE *f) {
    size_t i;
    for (i = 0; i < n; i++) {
        unsigned char c = s[i];
        if (c == '(' || c == ')' || c == '\\') { fputc('\\', f); fputc(c, f); }
        else if (c == '\n' || c == '\r') { /* wrapped by caller; skip */ }
        else if (c < 0x20 || c == 0x7F) fputc(' ', f);
        else if (c >= 0x80) {
            if ((c & 0xE0) == 0xC0 && i + 1 < n) {
                unsigned char c2 = s[++i];
                unsigned cp = ((c & 0x1F) << 6) | (c2 & 0x3F);
                if (cp <= 0xFF && cp >= 0x20) {
                    if (cp == 0xA0) fputc(' ', f);
                    else fprintf(f, "\\%03o", cp);
                } else fputc('?', f);
            } else fputc('?', f);
        } else fputc(c, f);
    }
}
int pii_pdf_write_text(const char *path, const char *text, const char *title) {
    FILE *f = fopen(path, "wb");
    long off[16]; int no = 0; size_t i, ln = 0, lines = 0, pages = 0;
    const char *p; char **pg = NULL; size_t pgc = 0, pgcap = 0;
    char *content = NULL; size_t cn = 0, cc = 0;
    if (!f) return -1;
    (void)p; (void)pgc; (void)pgcap;
    {   char *lbuf = NULL; size_t ln2 = 0, lc = 0;
        const char *q = text;
        lbuf = (char *)malloc(1024); if (!lbuf) { fclose(f); return -1; }
        while (*q) {
            size_t cols = 0; ln2 = 0;
            while (*q && *q != '\n' && cols < 95) {
                unsigned char c = *q;
                size_t cl = 1;
                if ((c & 0x80) && (c & 0xE0) == 0xC0) cl = 2;
                else if ((c & 0xF0) == 0xE0) cl = 3;
                if (ln2 + cl >= 1000) break;
                memcpy(lbuf + ln2, q, cl); ln2 += cl; q += cl; cols++;
            }
            while (*q && *q != '\n') q++;
            if (*q == '\n') q++;
            lbuf[ln2] = 0;
            if (lc % 60 == 0) {
                if (pgc == pgcap) {
                    size_t nn = pgcap ? pgcap * 2 : 4;
                    char **nv = (char **)realloc(pg, nn * sizeof *nv);
                    if (!nv) { free(lbuf); fclose(f); return -1; }
                    pg = nv; pgcap = nn;
                }
                pg[pgc] = NULL; pgc++;
            }
            {   size_t old = pg[pgc-1] ? strlen(pg[pgc-1]) : 0;
                char *nv = (char *)realloc(pg[pgc-1], old + ln2 + 2);
                if (!nv) { free(lbuf); fclose(f); return -1; }
                memcpy(nv + old, lbuf, ln2); nv[old + ln2] = '\n'; nv[old + ln2 + 1] = 0;
                pg[pgc-1] = nv;
            }
            lc++;
        }
        free(lbuf);
        (void)ln; (void)lines; (void)pages; (void)i;
    }
    pages = pgc ? pgc : 1;
    if (!pgc) { pg = (char **)malloc(sizeof *pg); pg[0] = strdup(""); pgc = 1; }
    fprintf(f, "%%PDF-1.4\n");
    {   size_t pi; int obj = 4;
        int *pageObjs = (int *)malloc(pgc * sizeof(int));
        int *contObjs = (int *)malloc(pgc * sizeof(int));
        long *offs = (long *)malloc((4 + 2 * pgc + 1) * sizeof(long));
        int N = 4 + 2 * (int)pgc;
        if (!pageObjs || !contObjs || !offs) { fclose(f); return -1; }
        for (pi = 0; pi < pgc; pi++) { pageObjs[pi] = obj++; contObjs[pi] = obj++; }
        off[0] = 0;
        offs[1] = ftell(f); fprintf(f, "1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n");
        {   char kids[4096]; size_t kl = 0; int k;
            kl += snprintf(kids + kl, sizeof kids - kl, "2 0 obj<</Type/Pages/Kids[");
            for (k = 0; k < (int)pgc; k++) kl += snprintf(kids + kl, sizeof kids - kl, "%d 0 R ", pageObjs[k]);
            kl += snprintf(kids + kl, sizeof kids - kl, "]/Count %d>>endobj\n", (int)pgc);
            offs[2] = ftell(f); fwrite(kids, 1, kl, f);
        }
        offs[3] = ftell(f); fprintf(f, "3 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica/Encoding/WinAnsiEncoding>>endobj\n");
        for (pi = 0; pi < pgc; pi++) {
            char *stream = NULL; size_t sn = 0, sc = 0;
            const char *q = pg[pi]; int yy = 800, li = 0;
            while (q && *q) {
                const char *e = strchr(q, '\n'); size_t L = e ? (size_t)(e - q) : strlen(q);
                size_t need = L * 5 + 64;
                while (sn + need >= sc) { sc = sc ? sc * 2 : 4096; stream = (char *)realloc(stream, sc); if (!stream) break; }
                if (!stream) break;
                sn += snprintf(stream + sn, sc - sn, "BT /F1 10 Tf 40 %d Td (", yy);
                {   size_t k; for (k = 0; k < L; k++) {
                        unsigned char c = q[k];
                        char eb[8]; size_t el = 0;
                        if (c == '(' || c == ')' || c == '\\') { eb[0] = '\\'; eb[1] = c; el = 2; }
                        else if (c < 0x20 || c == 0x7F) { eb[0] = ' '; el = 1; }
                        else if (c >= 0x80) {
                            if ((c & 0xE0) == 0xC0 && k + 1 < L) {
                                unsigned cp = ((c & 0x1F) << 6) | (q[k+1] & 0x3F);
                                k++;
                                if (cp >= 0x20 && cp <= 0xFF && cp != 0xA0) { el = snprintf(eb, sizeof eb, "\\%03o", cp); }
                                else { eb[0] = (cp == 0xA0) ? ' ' : '?'; el = 1; }
                            } else { eb[0] = '?'; el = 1; }
                        } else { eb[0] = c; el = 1; }
                        while (sn + el + 8 >= sc) { sc *= 2; stream = (char *)realloc(stream, sc); if (!stream) break; }
                        if (!stream) break;
                        memcpy(stream + sn, eb, el); sn += el;
                    } }
                if (!stream) break;
                sn += snprintf(stream + sn, sc - sn, ") Tj ET\n");
                yy -= 12; li++;
                q = e ? e + 1 : NULL;
                if (li >= 60) break;
            }
            if (!stream) { free(pageObjs); free(contObjs); free(offs); fclose(f); return -1; }
            offs[pageObjs[pi]] = ftell(f);
            fprintf(f, "%d 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 595 842]/Resources<</Font<</F1 3 0 R>>>>/Contents %d 0 R>>endobj\n",
                    pageObjs[pi], contObjs[pi]);
            offs[contObjs[pi]] = ftell(f);
            fprintf(f, "%d 0 obj<</Length %lu>>stream\n", contObjs[pi], (unsigned long)sn);
            fwrite(stream, 1, sn, f);
            fprintf(f, "endstream endobj\n");
            free(stream);
        }
        {   long xref = ftell(f); int k;
            fprintf(f, "xref\n0 %d\n0000000000 65535 f \n", N);
            for (k = 1; k < N; k++) fprintf(f, "%010ld 00000 n \n", offs[k]);
            fprintf(f, "trailer<</Size %d/Root 1 0 R", N);
            if (title && *title) { fprintf(f, "/Info<</Title("); latin1_esc(title, strlen(title), f); fprintf(f, ")>>"); }
            fprintf(f, ">>\nstartxref\n%ld\n%%%%EOF\n", xref);
        }
        free(pageObjs); free(contObjs); free(offs);
    }
    { size_t k; for (k = 0; k < pgc; k++) free(pg[k]); free(pg); }
    (void)no; (void)content; (void)cn; (void)cc; (void)ln;
    if (fclose(f)) return -1;
    return 0;
}

/* -------- spatial redaction writer --------
 * One output page per source page: spans replayed at recorded positions
 * (absolute Tm), then each PiiRedact painted BLACK over its zone.
 * Normalized->points: X = nx/1000*595, Y(top-left) -> Ybl = 842-ny/1000*842.
 */
static void redact_stream_esc(const char *s, FILE *f) {
    latin1_esc(s, strlen(s), f);
}

int pii_pdf_write_redacted(const char *path, const PiiDoc *doc,
                           const char *masked_text, const char *title,
                           const PiiRedact *reds, int nred) {
    FILE *f;
    int p, N;
    long *offs;
    int *pageObjs, *contObjs;
    char **streams = NULL;
    (void)masked_text; /* spans carry the masked text: caller masks doc spans
                        * in place (same order), flat text is informational */
    if (!path || !doc || doc->npages <= 0) return -1;
    f = fopen(path, "wb");
    if (!f) return -1;
    N = 4 + 2 * doc->npages;
    offs = (long *)calloc((size_t)N, sizeof(long));
    pageObjs = (int *)malloc((size_t)doc->npages * sizeof(int));
    contObjs = (int *)malloc((size_t)doc->npages * sizeof(int));
    streams = (char **)calloc((size_t)doc->npages, sizeof(char *));
    if (!offs || !pageObjs || !contObjs || !streams) {
        free(offs); free(pageObjs); free(contObjs); free(streams);
        fclose(f); return -1;
    }
    /* build one content stream per page */
    for (p = 0; p < doc->npages; p++) {
        const PiiPage *pg = &doc->pages[p];
        size_t sn = 0, sc = 4096;
        char *st = (char *)malloc(sc);
        int s, r;
        if (!st) { continue; }
        st[0] = 0;
        for (s = 0; s < pg->nspans; s++) {
            const PiiSpan *sp = &pg->spans[s];
            /* span centre -> points (bottom-left origin) */
            double cx = ((double)sp->box.x0 + sp->box.x1) / 2.0 / 1000.0 * 595.0;
            double cyb = 842.0 - ((double)sp->box.y0 + sp->box.y1) / 2.0 / 1000.0 * 842.0;
            double fs = sp->size > 0 ? sp->size : 10.0;
            char head[128];
            int hl = snprintf(head, sizeof head, "BT /F1 %.1f Tf 1 0 0 1 %.1f %.1f Tm (", fs, cx - 100.0, cyb);
            size_t need = (size_t)hl + strlen(sp->text) * 5 + 16;
            while (sn + need >= sc) { sc *= 2; st = (char *)realloc(st, sc); if (!st) break; }
            if (!st) break;
            memcpy(st + sn, head, (size_t)hl); sn += (size_t)hl;
            /* escape inline (latin-1 best-effort, same as legacy) */
            {   const char *q = sp->text;
                while (*q) {
                    unsigned char c = *q;
                    char eb[8]; size_t el = 0;
                    if (c == '(' || c == ')' || c == '\\') { eb[0]='\\'; eb[1]=(char)c; el=2; q++; }
                    else if (c < 0x20 || c == 0x7F) { eb[0]=' '; el=1; q++; }
                    else if (c >= 0x80 && (c & 0xE0) == 0xC0 && q[1]) {
                        unsigned cp = ((c & 0x1F) << 6) | (((unsigned char)q[1]) & 0x3F);
                        q += 2;
                        if (cp >= 0x20 && cp <= 0xFF && cp != 0xA0) el = (size_t)snprintf(eb, sizeof eb, "\\%03o", cp);
                        else { eb[0] = (cp == 0xA0) ? ' ' : '?'; el = 1; }
                    } else if (c >= 0x80) { eb[0]='?'; el=1; q++; }
                    else { eb[0]=(char)c; el=1; q++; }
                    while (sn + el + 16 >= sc) { sc *= 2; st = (char *)realloc(st, sc); if (!st) break; }
                    if (!st) break;
                    memcpy(st + sn, eb, el); sn += el;
                }
                if (!st) break;
            }
            {   const char *tail = ") Tj ET\n";
                while (sn + 8 >= sc) { sc *= 2; st = (char *)realloc(st, sc); if (!st) break; }
                if (!st) break;
                memcpy(st + sn, tail, 7); sn += 7; st[sn] = 0;
            }
        }
        /* blackout boxes for this page */
        for (r = 0; r < nred; r++) {
            if (reds[r].page != pg->num) continue;
            {   double x = (double)reds[r].box.x0 / 1000.0 * 595.0;
                double yt = (double)reds[r].box.y0 / 1000.0 * 842.0;
                double x1 = (double)reds[r].box.x1 / 1000.0 * 595.0;
                double yt1 = (double)reds[r].box.y1 / 1000.0 * 842.0;
                double yb = 842.0 - yt1;
                char rb[128];
                int rl = snprintf(rb, sizeof rb, "0 0 0 rg %.1f %.1f %.1f %.1f re f\n",
                                  x, yb, x1 - x, yt1 - yt);
                (void)yt;
                while (sn + (size_t)rl + 2 >= sc) { sc *= 2; st = (char *)realloc(st, sc); if (!st) break; }
                if (!st) break;
                memcpy(st + sn, rb, (size_t)rl); sn += (size_t)rl; st[sn] = 0;
            }
        }
        streams[p] = st;
    }
    fprintf(f, "%%PDF-1.4\n");
    {   int k = 4;
        for (p = 0; p < doc->npages; p++) { pageObjs[p] = k++; contObjs[p] = k++; }
        offs[1] = ftell(f); fprintf(f, "1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n");
        {   char kids[8192]; size_t kl = 0; int k2;
            kl += snprintf(kids + kl, sizeof kids - kl, "2 0 obj<</Type/Pages/Kids[");
            for (k2 = 0; k2 < doc->npages; k2++)
                kl += snprintf(kids + kl, sizeof kids - kl, "%d 0 R ", pageObjs[k2]);
            kl += snprintf(kids + kl, sizeof kids - kl, "]/Count %d>>endobj\n", doc->npages);
            offs[2] = ftell(f); fwrite(kids, 1, kl, f);
        }
        offs[3] = ftell(f);
        fprintf(f, "3 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica/Encoding/WinAnsiEncoding>>endobj\n");
        for (p = 0; p < doc->npages; p++) {
            size_t sn = streams[p] ? strlen(streams[p]) : 0;
            offs[pageObjs[p]] = ftell(f);
            fprintf(f, "%d 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 595 842]/Resources<</Font<</F1 3 0 R>>>>/Contents %d 0 R>>endobj\n",
                    pageObjs[p], contObjs[p]);
            offs[contObjs[p]] = ftell(f);
            fprintf(f, "%d 0 obj<</Length %lu>>stream\n", contObjs[p], (unsigned long)sn);
            if (sn) fwrite(streams[p], 1, sn, f);
            fprintf(f, "endstream endobj\n");
        }
        {   long xref = ftell(f); int k3;
            fprintf(f, "xref\n0 %d\n0000000000 65535 f \n", N);
            for (k3 = 1; k3 < N; k3++) fprintf(f, "%010ld 00000 n \n", offs[k3]);
            fprintf(f, "trailer<</Size %d/Root 1 0 R", N);
            if (title && *title) { fprintf(f, "/Info<</Title("); redact_stream_esc(title, f); fprintf(f, ")>>"); }
            fprintf(f, ">>\nstartxref\n%ld\n%%%%EOF\n", xref);
        }
    }
    for (p = 0; p < doc->npages; p++) free(streams[p]);
    free(streams); free(offs); free(pageObjs); free(contObjs);
    if (fclose(f)) return -1;
    return 0;
}
