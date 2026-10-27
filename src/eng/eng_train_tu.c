/* PII-AGENT v003 :: eng_train_tu.c — full engine translation unit.
 *
 * BPE + full neural math (forward AND backward/train) + clean batch
 * train driver + HYBRID spatial-visual forward (PII_HYBRID_SPATIAL: fused
 * text/box/patch features routed through I-chunk trits) and spatial-aware
 * training (PII_HYBRID_TRAIN: PDF staging injects layout-zone markers so
 * backprop sees visual context; see pii_train.c). No REPL, no
 * slash-commands, no legacy shell dispatch: the entries are
 * eng_train_files() and eng_infer_run_document(), called from full_main.c.
 */
#define PII_HYBRID_SPATIAL 1
#define PII_HYBRID_TRAIN 1
#include "nn_bpe.c"
#include "nn_math.c"
#include "train_driver.c"
#include "infer_ctx.c"    /* I-loader + hybrid forward, shared with flash */
