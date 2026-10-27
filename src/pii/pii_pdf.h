/* PII-AGENT v003 :: pii_pdf.h — SPATIAL PDF model (2D canvas, not 1D text).
 *
 * v002 extracted a flat text stream. v003 parses the page as a canvas:
 * every text-show op keeps its bounding box, every raster XObject keeps
 * its decoded pixels + placement box. Coordinates are NORMALIZED
 * 0..1000 per page, origin TOP-LEFT (x right, y down), page 1-based —
 * resolution- and MediaBox-independent, stable across renders.
 *
 * Legacy API (extract_text / write_text) is KEPT byte-compatible so the
 * CLI, PII.cmd and selftest keep working; extract_text now joins the
 * spatial spans, write_text is used for restore/plain outputs.
 */
#ifndef PII_PDF_H
#define PII_PDF_H

#include "pii_img.h"   /* PiiImage (stb-decoded RGB) */

typedef struct {
    float x0, y0, x1, y1;  /* normalized 0..1000, top-left origin */
    int page;              /* 1-based */
} PiiBBox;

typedef struct {
    char *text;      /* malloc'd UTF-8-ish bytes of one Tj/TJ show op */
    PiiBBox box;     /* span bounds on its page */
    float size;      /* active font size in points (0 = unknown) */
} PiiSpan;

typedef struct {
    int num;             /* 1-based page number */
    float w, h;          /* MediaBox size in points (fallback 595x842) */
    PiiSpan *spans;      /* text spans in content order */
    int nspans;
    PiiImage *imgs;      /* decoded raster XObjects placed on this page */
    int nimgs;
    int scanned_only;    /* 1 when the page has images but no text layer */
} PiiPage;

typedef struct {
    PiiPage *pages;
    int npages;
    char *flat;        /* whole-document text (spans joined, owned) */
} PiiDoc;

/* Load + parse a PDF (or plain-text fallback) into the spatial model.
 * Returns NULL only when the file is missing/unreadable. Scanned-only
 * PDFs yield pages with imgs[] and no spans — never NULL for that
 * reason (v002 failed here; v003 processes them visually). */
PiiDoc *pii_doc_load(const char *path);
void    pii_doc_free(PiiDoc *d);

/* Legacy v002 surface: flat text (joins doc spans). Caller frees. */
char *pii_pdf_extract_text(const char *path);

/* Legacy writer: minimal PDF 1.4 (Helvetica, A4, 60 lines/page). */
int pii_pdf_write_text(const char *path, const char *text, const char *title);

/* Spatial redaction writer: replays each source page's spans at their
 * recorded positions and paints every redaction box BLACK (0 0 0 rg…re f)
 * over the masked zones. `masked_text` is the placeholder-substituted
 * flat text (same span order as the loaded doc); reds[] reference source
 * pages/boxes (normalized). 0 ok, -1 IO error. */
typedef struct {
    int page;      /* 1-based source page */
    PiiBBox box;   /* normalized zone to black out */
} PiiRedact;

int pii_pdf_write_redacted(const char *path, const PiiDoc *doc,
                           const char *masked_text, const char *title,
                           const PiiRedact *reds, int nred);

#endif
