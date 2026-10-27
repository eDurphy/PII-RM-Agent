/* PII-AGENT v003 :: eng_infer_tu.c — flash engine translation unit.
 *
 * Inference math + I-weights session loader + HYBRID spatial-visual
 * forward (PII_HYBRID_SPATIAL: fused text/box/patch features are routed
 * through the loaded I-chunk trits in infer_ctx.c::eng_infer_run_document).
 * Training/backprop excluded at compile time via PII_FLASH_NO_TRAIN
 * (train_run, model_backward and the dead diag helpers vanish: verify
 * with nm). No BPE, no train driver.
 */
#define PII_HYBRID_SPATIAL 1
#include "nn_math.c"
#include "infer_ctx.c"
