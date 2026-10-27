/* PII-AGENT v002 :: flash_main.c — pii-agent-flash.exe (inference only).
 *
 * Pure argv CLI, exactly four commands. Startup:
 *   1. `train` aborts explicitly (exit 2) — training does not exist here.
 *   2. External I weights are dynamically LOADED into RAM (headers
 *      validated, trits kept for the session); S files never opened.
 *   3. mask/unmask/purge/selftest dispatch to the shared inference CLI.
 */
#include <stdio.h>
#include <string.h>

#include "cli_args.h"

int pii_agent_main(int argc, char **argv);
int eng_infer_load(const char *dir);
void eng_infer_free(void);
void ooc_dir_set(const char *d);

int main(int argc, char **argv)
{
    const char *wdir;
    int n, ac, rc;

    if (argc >= 2 && !strcmp(argv[1], "train")) {
        fprintf(stderr,
                "TRAIN-UNAVAILABLE: pii-agent-flash.exe is inference-only.\n"
                "  Use pii-agent.exe (full engine) for training.\n");
        return 2;
    }
    wdir = cli_weights_dir(argc, argv);
    ooc_dir_set(wdir);
    n = eng_infer_load(wdir);
    if (n < 0) {
        /* Fresh install: genera il baseline e riprova una volta sola. La
         * free evita duplicati se il primo load era parziale (chunk
         * corrotto in mezzo); il seed non tocca mai i chunk validi. */
        eng_infer_free();
        if (eng_ensure_seed_weights() >= 0)
            n = eng_infer_load(wdir);
    }
    if (n < 0) {
        fprintf(stderr,
                "WEIGHTS-FAIL: cannot load inference (I) weights from '%s'.\n"
                "  Set --weights-dir <dir> or PII_WEIGHTS_DIR (default data/weights).\n",
                wdir);
        return 1;
    }
    ac = cli_strip_weights_dir(argc, argv);
    rc = pii_agent_main(ac, argv);
    eng_infer_free();
    return rc;
}
