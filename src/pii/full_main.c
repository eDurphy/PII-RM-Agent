/* PII-AGENT v002 :: full_main.c — pii-agent.exe (full engine + training).
 *
 * Pure argv CLI, exactly five commands, zero REPL/prompts/stdin loops:
 *   pii-agent.exe mask <pdf> [--vault-dir D] [--weights-dir W]
 *   pii-agent.exe unmask <UUID|pdf> [...] | purge <UUID> [...] | selftest
 *   pii-agent.exe train <file-o-cartella> [X]   (X = 0|inf|omitted = infinite)
 * Bare invocation (or unknown command) prints usage and exits 1.
 *
 * train loads I+S through the engine (chunks spill back to the same
 * external files); inference commands load I only, like flash.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli_args.h"
#include "pii_train.h"

int pii_agent_main(int argc, char **argv);
int eng_train_files(char paths[][560], int n, int epochs);
int eng_infer_load(const char *dir);
void eng_infer_free(void);
void ooc_dir_set(const char *d);

#define FULL_TRAIN_INFINITE (-1)

static int is_inference_cmd(const char *c)
{
    return !strcmp(c, "mask") || !strcmp(c, "unmask") ||
           !strcmp(c, "purge") || !strcmp(c, "selftest");
}

/* FULL_TRAIN_INFINITE, a positive count, or -2 (invalid). */
static long parse_epochs(const char *s)
{
    char *end = NULL;
    long v;
    if (!s || !*s) return FULL_TRAIN_INFINITE;
    if (!strcmp(s, "0") || !strcmp(s, "inf") ||
        !strcmp(s, "run") || !strcmp(s, "forever"))
        return FULL_TRAIN_INFINITE;
    v = strtol(s, &end, 10);
    if (end == s || *end != 0 || v < 1 || v > 1000000) return -2;
    return v;
}

static void usage(const char *argv0)
{
    printf("PII-AGENT full engine — inference + training (no REPL)\n");
    printf("Uso:\n");
    printf("  %s mask <input.pdf> [--vault-dir DIR] [--weights-dir W]\n", argv0);
    printf("  %s unmask <UUID|masked.pdf> [--vault-dir DIR] [--weights-dir W]\n", argv0);
    printf("  %s purge <UUID> [--vault-dir DIR] [--weights-dir W]\n", argv0);
    printf("  %s selftest [--weights-dir W]\n", argv0);
    printf("  %s train <file-o-cartella> [X] [--weights-dir W]\n", argv0);
    printf("      X = epochs; 0 | inf | omitted = infinite until Ctrl+C\n");
}

int main(int argc, char **argv)
{
    if (argc >= 2 && is_inference_cmd(argv[1])) {
        const char *wdir = cli_weights_dir(argc, argv);
        int n, ac, rc;
        ooc_dir_set(wdir);
        n = eng_infer_load(wdir);
        if (n < 0) {
            /* Fresh install: genera il baseline e riprova una volta sola. */
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
    if (argc >= 2 && !strcmp(argv[1], "train")) {
        static char list[64][560];
        const char *path = (argc >= 3) ? argv[2] : NULL;
        const char *wdir;
        long epochs;
        int n;
        if (!path || !strcmp(path, "--help") || !strcmp(path, "-h")) {
            printf("Uso: %s train <file-o-cartella> [X]\n", argv[0]);
            printf("  X = epoch count; 0 | inf | omitted = infinite until Ctrl+C.\n");
            return (!path) ? 1 : 0;
        }
        epochs = parse_epochs((argc >= 4) ? argv[3] : NULL);
        if (epochs == -2) {
            fprintf(stderr,
                    "TRAIN-FAIL: invalid epoch count '%s' (want N>=1, 0, or inf).\n",
                    argv[3]);
            return 1;
        }
        wdir = cli_weights_dir(argc, argv);
        ooc_dir_set(wdir);   /* BEFORE any engine weight I/O */
        n = pii_collect_documents(path, list, 64);
        if (n < 0) return 1;
        if (n == 0) return 1;
        return eng_train_files(list, n, (int)epochs);
    }
    usage(argc > 0 ? argv[0] : "pii-agent.exe");
    return 1;
}
