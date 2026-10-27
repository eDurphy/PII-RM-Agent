# PII-AGENT v003 — spatial-multimodal upgrade (no OCR, no REPL)

Scaffolded strictly on the clean, REPL-free v002 codebase. QATNN becomes a
Spatial & Multimodal VLM for complex PDF CVs (scanned + image-heavy layouts)
with a hard constraint: **ZERO OCR** — no Tesseract, no OpenCV, no external
OCR engine anywhere in the build.

## What changed vs v002

- **Image decoding (Directive 1):** vendored single-header `stb_image.h`
  (public domain, v2.30) in `src/pii/`; the ONLY `STB_IMAGE_IMPLEMENTATION`
  TU is `src/pii/pii_img.c` (`pii_img_decode()`: JPEG/PNG/BMP/GIF via stb,
  raw RGB/gray fallback; JPXDecode/JBIG2 recorded as unsupported WITHOUT
  failing the document).
- **Spatial PDF parser (Directive 2):** `pii_pdf.c` reads the page as a 2D
  canvas — text-show ops tracked through Tm/Td/TF/cm into per-span
  normalized boxes (0..1000, top-left origin, 1-based pages); image XObjects
  (DCTDecode/FlateDecode/…) decoded via `pii_img` and placed per page.
  New model: `PiiDoc/PiiPage/PiiSpan/PiiImage/PiiBBox` (`pii_pdf.h`).
  Legacy `pii_pdf_extract_text` / `pii_pdf_write_text` are KEPT byte-stable
  (CLI, PII.cmd, selftest unaffected).
- **Multimodal embeddings (Directive 3):** `src/eng/nn_spatial.h` —
  16x16 vision patches flattened (768 u8) and projected integer-only into
  the embedding space; 2D coordinate embeddings from quantized grid tables
  (no float in the inference path); text hash-tokens + spatial + visual
  fused in the same Q16.15 latent space. `qatnn_core.c` gains
  `qat_spatial_bias_add()` / `qat_patch_token()` (integer-only, paper
  forward contract untouched).
- **Hybrid forward (Directive 4):** `infer_ctx.c::eng_infer_run_document()`
  routes structured docs (text + boxes + patches) through the LOADED
  I-chunk trits (paper sec-4 shape + SiLUNS r7 probe) fused with the
  visual-context salience; enabled per-TU via `PII_HYBRID_SPATIAL` in
  `eng_infer_tu.c` / `eng_train_tu.c`. Training sees layout through
  `[ZONE:HEADER]` / `[ZONE:PHOTO]` markers injected by `pii_train.c`
  staging (`PII_HYBRID_TRAIN`); scanned pages stage an image inventory
  instead of being skipped.
- **Spatial vault & redaction (Directive 5):** vault payload is now v2 —
  each entry optionally carries `"box":[x0,y0,x1,y1,page]` (v1 payloads
  still parse). `mask` on text-layer PDFs writes the output via
  `pii_pdf_write_redacted()`: spans replayed at recorded positions with
  every identified zone painted BLACK. CLI args, `PII.cmd` (interactive
  menu, identical to v002) and the `pii-agent.exe` / `pii-agent-flash.exe`
  split are UNCHANGED.
- **Build (Directive 6):** `build.cmd` links the new `pii_img.c`; patch
  projection is integer-only in-header + two `qatnn_core.c` helpers, so NO
  new link dependencies (stb is header-only; `-lz -lm -lpsapi` as v002).

## Known scaffolding limits (honest)

1. Text-position tracking is glyph-run approximate (0.5em advance, no TJ
   kerning walk, no rotated-text matrices) — boxes are redaction-grade,
   not typographic-grade.
2. Image placement uses page attribution (nearest preceding `/Type/Page`)
   with a full-page fallback box, not the exact `cm … Do` transform.
3. `mask` on **scanned-only** PDFs (images, no text layer) FAILS CLOSED:
   decoding works, but pixel-level PII localization needs the trained VLM
   head — train on text-layer CVs first (zone markers carry the signal),
   then re-run mask. No vault entry is created on this path.
4. The learned 768→D visual projection matrix is a reserved hook
   (`mm_patch_project` documents the call shape); the statistic projection
   ships as the integer-only baseline.

## Binaries (both `-static`; weights external, never hardcoded)

- `bin/pii-agent-flash.exe` — `mask/unmask/purge/selftest` + hybrid
  spatial-visual forward (`PII_HYBRID_SPATIAL`); train compiled OUT.
- `bin/pii-agent.exe` — same four + `train <path> [X]` with
  spatial-aware staging (`PII_HYBRID_TRAIN`); I+S.

```
PII.cmd                        :: interactive menu (unchanged from v002)
train "dir_path/file_path" [X] :: X omitted/0/inf = infinite until Ctrl+C
```

## Layout (new files marked +)

```
v003/
  PII.cmd                 interactive launcher (IDENTICAL to v002)
  build.cmd               + pii_img.c, spatial flags
  src/pii/stb_image.h   + vendored single-header decoder (public domain)
  src/pii/pii_img.[hc]  + stb wrapper (only IMPLEMENTATION TU)
  src/pii/pii_pdf.[hc]    SPATIAL parser + redaction writer (legacy kept)
  src/pii/pii_mask.[hc]   vault entries + optional 2D boxes (JSON v2, v1 reads)
  src/pii/pii_agent.c     spatial mask (attribute boxes, blackout, fail-closed)
  src/pii/pii_train.c     zone-marker staging + scanned inventory
  src/eng/nn_spatial.h  + patch projection / coord embeds / zone classifier
  src/eng/infer_ctx.c     + eng_infer_run_document hybrid forward
  src/eng/eng_*_tu.c      + PII_HYBRID_SPATIAL(/TRAIN) routing flags
  src/core/qatnn.[hc]     + qat_spatial_bias_add / qat_patch_token
  data/weights/           I_*.bin + S_*.bin seed (external, mutable by train)
```
