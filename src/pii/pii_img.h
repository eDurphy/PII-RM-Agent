/* PII-AGENT v003 :: pii_img.h — decoded-image wrapper over stb_image.h.
 *
 * Zero-OCR constraint: NO Tesseract / OpenCV / external OCR. PDF-embedded
 * raster objects (DCTDecode/JPEG, FlateDecode/raw, PNG predictors) are
 * decoded with the vendored single-header stb_image.h (public domain,
 * Sean Barrett, v2.30 — see stb_image.h in this directory).
 *
 * Only this TU's .c defines STB_IMAGE_IMPLEMENTATION; everyone else uses
 * the decoder through the tiny API below (RGB, 8 bit/channel).
 */
#ifndef PII_IMG_H
#define PII_IMG_H

#include <stddef.h>

typedef struct {
    unsigned char *px;   /* RGB, 8 bit/channel, size w*h*3 (malloc'd) */
    int w, h;            /* pixels */
    int page;            /* 1-based PDF page the object came from */
    char filter[32];     /* source PDF filter, e.g. "DCTDecode" */
} PiiImage;

void pii_img_free(PiiImage *im);

/* Decode a raw image-object byte buffer (already de-filtered from the PDF
 * stream, i.e. JPEG bytes for DCTDecode, PNG bytes, or raw RGB). Tries
 * stbi_load_from_memory (JPEG/PNG/BMP/GIF); if that fails and len equals
 * w_hint*h_hint*3 (or *1), accepts it as raw RGB/gray. Returns 0 ok,
 * -1 on failure (unsupported e.g. JPXDecode/JBIG2 — caller keeps the
 * text layer and notes the gap, never aborts the whole document). */
int pii_img_decode(const unsigned char *buf, size_t len,
                   int w_hint, int h_hint, int page,
                   const char *filter, PiiImage *out);

#endif
