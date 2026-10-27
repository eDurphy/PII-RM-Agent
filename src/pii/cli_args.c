/* PII-AGENT v002 :: cli_args.c — see cli_args.h. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli_args.h"

const char *cli_weights_dir(int argc, char **argv)
{
    int i;
    const char *e;
    if (argc > 0 && argv) {
        for (i = 1; i + 1 < argc; i++) {
            if (!strcmp(argv[i], "--weights-dir"))
                return argv[i + 1][0] ? argv[i + 1] : NULL;
        }
    }
    e = getenv("PII_WEIGHTS_DIR");
    if (e && *e) return e;
    return "data/weights";
}

int cli_strip_weights_dir(int argc, char **argv)
{
    int r = 0, w = 0;
    if (argc <= 0 || !argv) return argc;
    for (r = 0; r < argc; r++) {
        if (r > 0 && !strcmp(argv[r], "--weights-dir")) { r++; continue; }
        argv[w++] = argv[r];
    }
    argv[w] = NULL;
    return w;
}
