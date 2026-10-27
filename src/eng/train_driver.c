/* PII-AGENT v002 :: train_driver.c — clean batch trainer (no REPL).
 *
 * Compiled ONLY into pii-agent.exe (never into flash). eng_train_files():
 * learns an explicit *.txt list (already PDF-extracted by the caller) with
 * the ported engine: BPE vocab (trained once, saved next to the weights),
 * sliding windows, train_run per file, spill after every file, Ctrl+C-safe
 * infinite loop (epochs < 0). Hyperparameters mirror the measured best
 * (batch 1, grad-scale on, LR 1024).
 *
 * NOTE: same-TU code — #included by eng_train_tu.c AFTER nn_bpe.c and
 * nn_math.c, so Bpe/Model/Data and all engine statics are visible.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#endif

#define ENG_D      1024
#define ENG_NH     8
#define ENG_DFF    2816
#define ENG_T      512
#define ENG_LAYERS 14
#define ENG_LR     1024
#define ENG_SEED   20260930u
#define ENG_FILE_MAX  (8u << 20)   /* biggest single file trained, in bytes */
#define ENG_WIN_MAX   4096         /* windows per file (stride covers file)  */
#define ENG_BPE_HEAD  200000       /* bytes used to learn the BPE vocab      */

static volatile int g_eng_stop = 0;
/* Letto dal loop di train_run (nn_math.c, stessa TU): rende il Ctrl+C
 * immediato anche a meta' epoca; il driver salva comunque alla fine. */
int eng_stop_requested(void) { return g_eng_stop; }
#ifdef _WIN32
static BOOL WINAPI eng_ctrl(DWORD t)
{
    if (t == CTRL_C_EVENT || t == CTRL_BREAK_EVENT || t == CTRL_CLOSE_EVENT) {
        g_eng_stop = 1;
        return TRUE;
    }
    return FALSE;
}
#else
static void eng_ctrl(int s) { (void)s; g_eng_stop = 1; }
#endif

static int eng_steps_effective(void)
{
    const char *e = getenv("PII_TRAIN_STEPS");
    long v;
    char *end = NULL;
    if (!e || !*e) return 60;
    v = strtol(e, &end, 10);
    if (end == e || *end != 0 || v < 1 || v > 100000) return 60;
    return (int)v;
}

static char *eng_read_file(const char *path, size_t *n_out)
{
    FILE *f = NULL;
    long sz;
    char *b = NULL;
    size_t got;
    f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    sz = ftell(f);
    if (sz < 0 || (uint64_t)sz > ENG_FILE_MAX) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    b = (char *)malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    got = fread(b, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { free(b); return NULL; }
    b[sz] = 0;
    if (n_out) *n_out = (size_t)sz;
    return b;
}

/* Build sliding windows over ids (stride chosen to cover the file within
 * ENG_WIN_MAX). Returns window count, or -1 on OOM. */
static int eng_make_windows(const int32_t *ids, int32_t n_ids, Data *D)
{
    int32_t nwin_full, stride, nwin, i;
    if (n_ids <= ENG_T + 1) return 0;
    nwin_full = n_ids - ENG_T - 1;
    stride = (nwin_full + ENG_WIN_MAX - 1) / ENG_WIN_MAX;
    if (stride < 1) stride = 1;
    nwin = nwin_full / stride;
    if (nwin < 1) nwin = 1;
    if (nwin > ENG_WIN_MAX) nwin = ENG_WIN_MAX;
    memset(D, 0, sizeof *D);
    D->V = 0; D->T = ENG_T; D->nwin = nwin;
    D->x = (int32_t *)malloc((size_t)nwin * ENG_T * 4u);
    D->y = (int32_t *)malloc((size_t)nwin * ENG_T * 4u);
    if (!D->x || !D->y) { free(D->x); free(D->y); return -1; }
    for (i = 0; i < nwin; i++) {
        int64_t src = (int64_t)i * stride;
        if (src + ENG_T + 1 > n_ids) { nwin = i; D->nwin = nwin; break; }
        memcpy(D->x + (size_t)i * ENG_T, ids + src, (size_t)ENG_T * 4u);
        memcpy(D->y + (size_t)i * ENG_T, ids + src + 1, (size_t)ENG_T * 4u);
    }
    return nwin;
}

/* Train the explicit file list. epochs < 0 = infinite until Ctrl+C.
 * Returns 0 ok, 1 on error / nothing trainable. */
int eng_train_files(char paths[][560], int n, int epochs)
{
    static Bpe bpe;
    static int bpe_ready = 0;
    char bpe_path[600];
    Model *M = NULL;
    int f, e = 0, infinite, steps, rc = 1, trained_any = 0;
    uint64_t total_win = 0;

    if (n <= 0 || !paths) { printf("  * niente da allenare: lista vuota.\n"); return 1; }
    infinite = (epochs < 0);
    if (!infinite && epochs < 1) epochs = 1;

    {
        const char *wd = getenv("PII_WEIGHTS_DIR");
        int r;
        if (!wd || !*wd) wd = "data/weights";
        r = snprintf(bpe_path, sizeof bpe_path, "%s/pii_bpe.txt", wd);
        if (r < 0 || (size_t)r >= sizeof bpe_path) {
            printf("TRAIN-FAIL: weights dir path too long.\n");
            return 1;
        }
    }
    ooc_dir_ensure();   /* weights dir exists BEFORE bpe_save (fail-open safe) */
    if (!bpe_ready) {
        if (bpe_load(&bpe, bpe_path)) {
            bpe_ready = 1;
            printf("  vocabolario caricato: %s.\n", bpe_path);
        }
    }
    steps = eng_steps_effective();
    llm_set_batch(1);
    llm_set_grad_scale(1);
#ifdef _WIN32
    SetConsoleCtrlHandler(eng_ctrl, TRUE);
#else
    signal(SIGINT, eng_ctrl);
#endif
    if (infinite)
        printf("\n  train: %d documenti, SENZA FINE fino a Ctrl+C (%d passi/file).\n",
               n, steps);
    else
        printf("\n  train: %d documenti, %d %s (%d passi/file).\n",
               n, epochs, epochs == 1 ? "epoca" : "epoche", steps);
    for (f = 0; f < n; f++) printf("    %d. %s\n", f + 1, paths[f]);

    for (e = 0; infinite || e < epochs; e++) {
        if (g_eng_stop) break;
        printf("  epoca %d%s su %d file.\n", e + 1,
               infinite ? " (senza fine)" : "", n);
        for (f = 0; f < n; f++) {
            char *text = NULL;
            size_t tlen = 0;
            int32_t *ids = NULL, n_ids = 0;
            Data D;
            if (g_eng_stop) break;
            printf("  file %d di %d: %s ...\n", f + 1, n, paths[f]);
            text = eng_read_file(paths[f], &tlen);
            if (!text || tlen == 0) {
                printf("    salto (non leggibile o vuoto).\n");
                free(text);
                continue;
            }
            /* Documenti brevi (CV tipici): ripeto il testo fino a coprire
             * qualche finestra T=512 (le epoche ripetono comunque i dati).
             * Senza questo, file < ~514 token non formano nemmeno una
             * finestra x/y e non c'e' nulla da imparare. */
            if (tlen < 8u * (ENG_T + 1)) {
                size_t want = 8u * (ENG_T + 1), have = tlen, cap = want + 1;
                char *rep;
                if (cap > ENG_FILE_MAX) cap = ENG_FILE_MAX;
                rep = (char *)malloc(cap);
                if (rep) {
                    memcpy(rep, text, tlen);
                    while (have + tlen + 1 < cap) {
                        rep[have++] = '\n';
                        memcpy(rep + have, text, tlen);
                        have += tlen;
                    }
                    rep[have] = 0;
                    free(text);
                    text = rep;
                    tlen = have;
                }
            }
            if (!bpe_ready) {
                size_t learn = tlen > ENG_BPE_HEAD ? ENG_BPE_HEAD : tlen;
                bpe_init(&bpe);
                bpe_train(&bpe, text, learn, 768);
                bpe_ready = 1;
                bpe_save(&bpe, bpe_path);
                printf("    vocabolario imparato sul primo file: %s.\n", bpe_path);
            }
            ids = (int32_t *)malloc((tlen + 1) * sizeof(int32_t));
            if (!ids) { printf("    memoria esaurita.\n"); free(text); break; }
            n_ids = bpe_encode(&bpe, text, tlen, ids, (int32_t)(tlen + 1));
            free(text);
            text = NULL;
            if (!M) {
                M = model_new_reuse(bpe.V, ENG_D, ENG_NH, ENG_DFF,
                                    ENG_T, ENG_LAYERS, ENG_SEED);
                lr_set_all(M, ENG_LR);
                printf("  modello pronto: V=%d d=%d L=%d T=%d.\n",
                       bpe.V, ENG_D, ENG_LAYERS, ENG_T);
            }
            {
                int nw = eng_make_windows(ids, n_ids, &D);
                free(ids);
                ids = NULL;
                if (nw < 0) { printf("    memoria esaurita (finestre).\n"); break; }
                if (nw == 0) {
                    printf("    troppo corto, salto.\n");
                    free(D.x); free(D.y);
                    continue;
                }
                D.V = bpe.V;
                printf("    %d finestre da %d token: %d passi...\n", nw, ENG_T, steps);
                train_run(M, &D, steps, ENG_SEED + (uint32_t)e, 0);
                total_win += (uint64_t)steps;
                free(D.x); free(D.y);
            }
            trained_any = 1;
            if (!model_sync_chunks(M))
                printf("    ATTENZIONE: sync chunk incompleta.\n");
            else
                printf("    file salvato nei pesi esterni.\n");
        }
        if (g_eng_stop) break;
        if (!infinite)
            printf("  epoca %d finita. Pesi salvati.\n", e + 1);
    }
    if (g_eng_stop) {
        printf("  interrotto: salvo e esco.\n");
        g_eng_stop = 0;
    }
    if (M) {
        model_sync_chunks(M);
        model_free(M);
    }
    if (trained_any)
        printf("  fatto: %llu passi totali. Pesi esterni aggiornati.\n",
               (unsigned long long)total_win);
    else
        printf("  fatto: nessun passo (documenti troppo brevi o illeggibili).\n");
    rc = 0;
    return rc;
}
