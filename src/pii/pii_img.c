/* PII-AGENT v003 :: pii_img.c — stb_image implementation TU + wrapper.
 *
 * This is the ONLY file that defines STB_IMAGE_IMPLEMENTATION.
 * No Tesseract / OpenCV / OCR engine anywhere in the build.
 */
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
/* stb compiled quiet: it is a third-party header, silence its warnings
 * under our -Wall build without touching the vendored file. */
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-braces"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Wstringop-overread"
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
#include "stb_image.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "pii_img.h"

void pii_img_free(PiiImage *im)
{
    if (!im) return;
    if (im->px) stbi_image_free(im->px);
    memset(im, 0, sizeof *im);
}

int pii_img_decode(const unsigned char *buf, size_t len,
                   int w_hint, int h_hint, int page,
                   const char *filter, PiiImage *out)
{
    int w = 0, h = 0, comp = 0;
    unsigned char *px = NULL;

    if (!buf || !len || !out) return -1;
    memset(out, 0, sizeof *out);

    /* Primary path: real codec (DCTDecode=JPEG, PNG, BMP, GIF). */
    px = stbi_load_from_memory(buf, (int)len, &w, &h, &comp, 3);
    if (px && w > 0 && h > 0) {
        out->px = px; out->w = w; out->h = h; out->page = page;
        snprintf(out->filter, sizeof out->filter, "%s",
                 filter ? filter : "?");
        return 0;
    }
    if (px) stbi_image_free(px);

    /* Fallback: raw raster (FlateDecode image XObject without a
     * PNG predictor we expand inline). Accept RGB or gray. */
    if (w_hint > 0 && h_hint > 0) {
        size_t rgb = (size_t)w_hint * (size_t)h_hint * 3u;
        size_t gry = (size_t)w_hint * (size_t)h_hint;
        if (len == rgb || len == gry) {
            px = (unsigned char *)malloc(rgb ? rgb : 1);
            if (!px) return -1;
            if (len == rgb) {
                memcpy(px, buf, rgb);
            } else {
                size_t i;
                for (i = 0; i < gry; i++)
                    px[3*i] = px[3*i+1] = px[3*i+2] = buf[i];
            }
            out->px = px; out->w = w_hint; out->h = h_hint;
            out->page = page;
            snprintf(out->filter, sizeof out->filter, "%s",
                     filter ? filter : "Raw");
            return 0;
        }
    }
    /* JPXDecode / JBIG2Decode / anything else stb cannot read:
     * unsupported, NOT fatal — caller degrades to the text layer. */
    return -1;
}
