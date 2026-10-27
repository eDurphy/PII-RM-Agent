/* PII-AGENT v002 :: cli_args.h — shared --weights-dir / env resolution.
 * No hardcoded absolute paths: flag > PII_WEIGHTS_DIR > data/weights. */
#ifndef PII_CLI_ARGS_H
#define PII_CLI_ARGS_H

const char *cli_weights_dir(int argc, char **argv);
int cli_strip_weights_dir(int argc, char **argv);
/* Engine (nn_math.c, entrambe le TU): genera il baseline I+S se manca. */
int eng_ensure_seed_weights(void);

#endif
