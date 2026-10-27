/* PII-AGENT v002 :: pii_train.c — file-or-folder batch discovery.
 *
 * Pure collection: discovery + PDF text extraction live here (PII layer).
 * The caller hands the explicit *.txt list to eng_train_files(); no dataset
 * dirs, no select lists. Text files train in place (no copy); PDFs are
 * staged once into data/train_stage/<base>.txt.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define MKDIR(d) _mkdir(d)
#else
#include <dirent.h>
#include <sys/types.h>
#define MKDIR(d) mkdir(d, 0700)
#endif

#include "pii_pdf.h"
#include "pii_train.h"

#define TRAIN_MAX_FILES 64
#define TRAIN_PATH_CAP  560


static void ensure_dir(const char *d) { MKDIR(d); }

static int has_ext_ci(const char *name, const char *ext)
{
    size_t n = strlen(name), e = strlen(ext);
    size_t i;
    if (n < e) return 0;
    for (i = 0; i < e; i++) {
        char a = name[n - e + i], b = ext[i];
        if (a >= 'A' && a <= 'Z') a = (char)(a + 32);
        if (b >= 'A' && b <= 'Z') b = (char)(b + 32);
        if (a != b) return 0;
    }
    return 1;
}

static int is_pdf(const char *p) { return has_ext_ci(p, ".pdf"); }

static int is_trainable(const char *p)
{
    static const char *exts[] = { ".txt", ".md", ".pdf", ".csv", ".json", ".log", ".text", NULL };
    int i;
    for (i = 0; exts[i]; i++)
        if (has_ext_ci(p, exts[i])) return 1;
    return 0;
}

static const char *base_name(const char *p)
{
    const char *s = strrchr(p, '/'), *b = strrchr(p, '\\');
    const char *m = s > b ? s : b;
    return m ? m + 1 : p;
}

static void strip_ext(const char *base, char *out, size_t cap)
{
    const char *dot = strrchr(base, '.');
    size_t n = dot ? (size_t)(dot - base) : strlen(base);
    if (n >= cap) n = cap - 1;
    memcpy(out, base, n);
    out[n] = 0;
}

static int cmp_str(const void *a, const void *b)
{
    return strcmp((const char *)a, (const char *)b);
}

static int is_regular_file(const char *p)
{
    struct stat st;
    if (stat(p, &st) != 0) return 0;
#ifdef _WIN32
    return (st.st_mode & _S_IFMT) == _S_IFREG;
#else
    return S_ISREG(st.st_mode);
#endif
}

static int is_dir(const char *p)
{
    struct stat st;
    if (stat(p, &st) != 0) return 0;
#ifdef _WIN32
    return (st.st_mode & _S_IFMT) == _S_IFDIR;
#else
    return S_ISDIR(st.st_mode);
#endif
}

/* v003: stage one PDF as SPATIALLY-ANNOTATED text. Layout-zone markers
 * ([ZONE:HEADER]/[ZONE:PHOTO]) are injected ahead of spans in the header
 * band / photo-adjacent zones so the existing windowed trainer (which
 * consumes plain text) learns visual context through the network.
 * Scanned-only PDFs (images, no text layer) stage an image inventory
 * line instead of being skipped (v002 dropped them). Returns 0 ok. */
static int stage_pdf(const char *pdf_path, const char *stage_dir, char *out, size_t cap)
{
    char base[TRAIN_PATH_CAP], stem[TRAIN_PATH_CAP], staged[TRAIN_PATH_CAP];
    PiiDoc *doc = NULL;
    FILE *o;
    int p, s;
    snprintf(base, sizeof base, "%s", base_name(pdf_path));
    strip_ext(base, stem, sizeof stem);
    if (!stem[0]) snprintf(stem, sizeof stem, "doc");
    snprintf(staged, sizeof staged, "%s/%s.txt", stage_dir, stem);
    if (is_regular_file(staged)) {
        int k;
        for (k = 2; k < 10000; k++) {
            snprintf(staged, sizeof staged, "%s/%s_%d.txt", stage_dir, stem, k);
            if (!is_regular_file(staged)) break;
        }
    }
    doc = pii_doc_load(pdf_path);
    if (!doc) {
        printf("  * salto '%s': illeggibile.\n", pdf_path);
        return -1;
    }
    o = fopen(staged, "wb");
    if (!o) { printf("  * non scrivo '%s'.\n", staged); pii_doc_free(doc); return -1; }
    for (p = 0; p < doc->npages; p++) {
        PiiPage *pg = &doc->pages[p];
        if (pg->nspans == 0 && pg->nimgs > 0) {
            int m;
            /* scanned page: stage inventory so training still sees it */
            fprintf(o, "[PAGE %d SCANNED-IMAGE", pg->num);
            for (m = 0; m < pg->nimgs; m++)
                fprintf(o, " %dx%d/%s", pg->imgs[m].w, pg->imgs[m].h,
                        pg->imgs[m].filter[0] ? pg->imgs[m].filter : "?");
            fprintf(o, "]\n");
            continue;
        }
        for (s = 0; s < pg->nspans; s++) {
            PiiSpan *sp = &pg->spans[s];
            int cx = (int)(sp->box.x0 + sp->box.x1) / 2;
            int cy = (int)(sp->box.y0 + sp->box.y1) / 2;
            /* header/photo zone markers: the spatial signal for training */
            if (sp->box.y0 < 150 || cy < 120) fputs("[ZONE:HEADER] ", o);
            else if (pg->nimgs > 0 &&
                     (((int)sp->box.x0 < 300 || (int)sp->box.x1 > 700) &&
                      ((int)sp->box.x1 - (int)sp->box.x0) < 400))
                fputs("[ZONE:PHOTO] ", o);
            (void)cx;
            fwrite(sp->text, 1, strlen(sp->text), o);
            fputc('\n', o);
        }
        fputc('\n', o);
    }
    fclose(o);
    printf("  + '%s' -> '%s' (%d pagine, annotazione spaziale).\n",
           pdf_path, staged, doc->npages);
    pii_doc_free(doc);
    snprintf(out, cap, "%s", staged);
    return 0;
}

int pii_collect_documents(const char *path, char out[][560], int max)
{
    char found[TRAIN_MAX_FILES][TRAIN_PATH_CAP];
    int nfound = 0, n = 0, i, skipped = 0;
    const char *stage_dir = "data/train_stage";

    if (!path || !*path) {
        printf("TRAIN-FAIL: indica un file o una cartella.\n");
        return -1;
    }
    if (max > TRAIN_MAX_FILES) max = TRAIN_MAX_FILES;

    if (is_regular_file(path)) {
        snprintf(found[nfound++], TRAIN_PATH_CAP, "%s", path);
    } else if (is_dir(path)) {
#ifdef _WIN32
        char pattern[TRAIN_PATH_CAP + 4];
        WIN32_FIND_DATAA fd;
        HANDLE h;
        snprintf(pattern, sizeof pattern, "%s/*", path);
        h = FindFirstFileA(pattern, &fd);
        if (h == INVALID_HANDLE_VALUE) {
            printf("TRAIN-FAIL: non leggo la cartella '%s'.\n", path);
            return 1;
        }
        do {
            char full[TRAIN_PATH_CAP];
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
            if (fd.cFileName[0] == '.') continue;
            snprintf(full, sizeof full, "%s/%s", path, fd.cFileName);
            if (!is_trainable(full)) { skipped++; continue; }
            if (nfound < TRAIN_MAX_FILES) snprintf(found[nfound++], TRAIN_PATH_CAP, "%s", full);
        } while (FindNextFileA(h, &fd));
        FindClose(h);
#else
        DIR *d = opendir(path);
        struct dirent *e;
        if (!d) { printf("TRAIN-FAIL: non leggo la cartella '%s'.\n", path); return 1; }
        while ((e = readdir(d)) != NULL) {
            char full[TRAIN_PATH_CAP];
            if (e->d_name[0] == '.') continue;
            snprintf(full, sizeof full, "%s/%s", path, e->d_name);
            if (!is_regular_file(full)) continue;
            if (!is_trainable(full)) { skipped++; continue; }
            if (nfound < TRAIN_MAX_FILES) snprintf(found[nfound++], TRAIN_PATH_CAP, "%s", full);
        }
        closedir(d);
#endif
        qsort(found, (size_t)nfound, sizeof found[0], cmp_str);
    } else {
        printf("TRAIN-FAIL: '%s' non esiste (ne' file ne' cartella).\n", path);
        return -1;
    }

    if (nfound == 0) {
        printf("TRAIN-FAIL: nessun documento allenabile in '%s'%s.\n", path,
               skipped ? " (solo estensioni non supportate)" : "");
        printf("  estensioni accettate: .txt .md .pdf .csv .json .log\n");
        return 0;
    }

    ensure_dir("data");
    ensure_dir(stage_dir);

    for (i = 0; i < nfound && n < max; i++) {
        if (is_pdf(found[i])) {
            char staged[TRAIN_PATH_CAP];
            if (stage_pdf(found[i], stage_dir, staged, sizeof staged) == 0)
                snprintf(out[n++], 560, "%s", staged);
        } else {
            snprintf(out[n++], 560, "%s", found[i]);
        }
    }
    if (skipped)
        printf("  (%d file con estensione non supportata saltati.)\n", skipped);
    if (n == 0)
        printf("TRAIN-FAIL: nessun file utilizzabile dopo l'estrazione.\n");
    qsort(out, (size_t)n, sizeof out[0], cmp_str); /* ordine stabile finale */
    return n;
}
