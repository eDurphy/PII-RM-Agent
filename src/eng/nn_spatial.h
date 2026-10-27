/* PII-AGENT v003 :: nn_spatial.h - spatial-visual embedding module.
 *
 * Unified spatial-visual sequence WITHOUT changing the paper dense-layer
 * math: vision patches (16x16 RGB, flattened to 768 u8) are projected with
 * an INTEGER-ONLY statistic projection; text tokens get 2D coordinate
 * embeddings (quantized grid tables, no float); both live in the same
 * Q16.15 feature space so attention sees layout and pixels together.
 */
#ifndef NN_SPATIAL_H
#define NN_SPATIAL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PII_PATCH 16
#define PII_PATCH_VEC (PII_PATCH * PII_PATCH * 3)
#define PII_SGRID 32
#define PII_SDIM 64
#define PII_MM_D 128

/* deterministic LCG (same stream everywhere, no platform rand) */
static uint32_t mm_lcg(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s >> 8;
}

/* ---- patch projection: 768 u8 -> PII_MM_D int32 (Q16.15-ish scale) ----
 * Integer-only: per-channel mean/std/edge-energy + deterministic mix.
 * Production follow-up: learned 768xD int8 matrix in the weight store;
 * the hook below keeps the call shape so the swap is drop-in. */
static void mm_patch_project(const unsigned char *patch768,
                             int32_t *out, int d) {
    int i, k;
    uint32_t s = 0x50A71A1u;
    long mR = 0, mG = 0, mB = 0;
    long vR = 0, vG = 0, vB = 0, edge = 0;
    for (i = 0; i < 256; i++) {
        mR += patch768[3*i]; mG += patch768[3*i+1]; mB += patch768[3*i+2];
    }
    mR /= 256; mG /= 256; mB /= 256;
    for (i = 0; i < 256; i++) {
        long dr = patch768[3*i]-mR, dg = patch768[3*i+1]-mG, db = patch768[3*i+2]-mB;
        vR += dr*dr >> 8; vG += dg*dg >> 8; vB += db*db >> 8;
        if (i >= 16) {
            long e = dr - ((long)patch768[3*(i-16)]-mR);
            edge += e*e >> 8;
        }
    }
    { long feat[6];
      feat[0]=mR-128; feat[1]=mG-128; feat[2]=mB-128;
      feat[3]=vR; feat[4]=vG; feat[5]=vB+edge;
      for (k = 0; k < d; k++) {
          long acc = 0;
          int j;
          for (j = 0; j < 6; j++) {
              long w = (long)((int)(mm_lcg(&s) % 65536u) - 32768);
              acc += (feat[j] * w) >> 8;
          }
          if (acc > 1073741823L) acc = 1073741823L;
          if (acc < -1073741823L) acc = -1073741823L;
          out[k] = (int32_t)acc;
      }
    }
}

/* ---- 2D coordinate embedding: (nx,ny in 0..1000) -> PII_SDIM int32 ----
 * Quantized grid tables, generated once from a fixed seed (stable across
 * runs, no float, no training state in the inference path). */
static int32_t MM_GX[PII_SGRID][16];
static int32_t MM_GY[PII_SGRID][16];
static int mm_grid_ready = 0;
static void mm_grid_init(void) {
    uint32_t s = 0xC0A71u;
    int i, j;
    if (mm_grid_ready) return;
    for (i = 0; i < PII_SGRID; i++)
        for (j = 0; j < 16; j++) {
            MM_GX[i][j] = (int32_t)((int)(mm_lcg(&s) % 65536u) - 32768);
            MM_GY[i][j] = (int32_t)((int)(mm_lcg(&s) % 65536u) - 32768);
        }
    mm_grid_ready = 1;
}

static void mm_spatial_embed(int nx, int ny, int32_t *out) {
    int xb, yb, j;
    mm_grid_init();
    if (nx < 0) nx = 0;
    if (nx > 1000) nx = 1000;
    if (ny < 0) ny = 0;
    if (ny > 1000) ny = 1000;
    xb = nx * PII_SGRID / 1001;
    yb = ny * PII_SGRID / 1001;
    for (j = 0; j < 16; j++) {
        int32_t a = MM_GX[xb][j] + MM_GY[yb][j];
        out[j*4]   = a;
        out[j*4+1] = MM_GX[xb][(j+5)%16] - MM_GY[yb][(j+11)%16];
        out[j*4+2] = MM_GX[(xb+j)%PII_SGRID][j] + MM_GY[(yb+2*j)%PII_SGRID][j];
        out[j*4+3] = (MM_GX[xb][j] >> 1) + (MM_GY[yb][j] >> 1);
    }
}

/* ---- text token hash: span bytes -> stable 31-bit id (BPE-free path) --- */
static uint32_t mm_hash(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    size_t i;
    for (i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h & 0x7FFFFFFFu;
}

/* ---- layout zones: header band / photo-adjacent / body ---------------- */
typedef enum { MMZ_BODY = 0, MMZ_HEADER = 1, MMZ_PHOTO = 2 } mm_zone_t;
static mm_zone_t mm_classify_zone(int nx0, int ny0, int nx1, int ny1,
                                  int page_has_image) {
    int cy = (ny0 + ny1) / 2;
    int area_w = nx1 - nx0;
    if (ny0 < 150 || cy < 120) return MMZ_HEADER;
    if (page_has_image && (nx0 < 300 || nx1 > 700) && area_w < 400) return MMZ_PHOTO;
    return MMZ_BODY;
}

/* ---- visual-context salience: layout prior added to span scores -------
 * Trainable-hook baseline (integer): header x3, photo-adjacent x2.
 * The full engine learns per-zone weights; flash uses this prior. */
static int32_t mm_zone_boost(mm_zone_t z) {
    if (z == MMZ_HEADER) return 3;
    if (z == MMZ_PHOTO) return 2;
    return 1;
}

#endif
