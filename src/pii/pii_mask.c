/* PII-AGENT :: pii_mask.c — see pii_mask.h. Pure C99, no dependencies. */
#include "pii_mask.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* ---------------- map ---------------- */
void pii_map_init(PiiVaultMap *m) {
    memset(m, 0, sizeof *m);
}
void pii_map_free(PiiVaultMap *m) {
    free(m->items); memset(m, 0, sizeof *m);
}
const char *pii_map_intern(PiiVaultMap *m, const char *type,
                           const char *original, size_t len) {
    int i; PiiEntry *e; int *ctr;
    if (!original || len == 0 || len > PII_ORIG_MAX) return NULL;
    for (i = 0; i < m->n; i++)
        if (!strcmp(m->items[i].type, type) &&
            strlen(m->items[i].original) == len &&
            !memcmp(m->items[i].original, original, len))
            return m->items[i].placeholder;
    if (m->n == m->cap) {
        int nc = m->cap ? m->cap * 2 : 32;
        PiiEntry *ni = (PiiEntry *)realloc(m->items, (size_t)nc * sizeof *ni);
        if (!ni) return NULL;
        m->items = ni; m->cap = nc;
    }
    e = &m->items[m->n++];
    memset(e, 0, sizeof *e);
    if (!strcmp(type, "NAME")) ctr = &m->c_name;
    else if (!strcmp(type, "EMAIL")) ctr = &m->c_email;
    else if (!strcmp(type, "PHONE")) ctr = &m->c_phone;
    else if (!strcmp(type, "FISCAL")) ctr = &m->c_fiscal;
    else ctr = &m->c_addr;
    snprintf(e->placeholder, sizeof e->placeholder, "[%s_%d]", type, ++(*ctr));
    memcpy(e->original, original, len);
    e->original[len] = 0;
    snprintf(e->type, sizeof e->type, "%s", type);
    return e->placeholder;
}

/* ---------------- small helpers ---------------- */
static int is_wordch(char c) {
    return isalnum((unsigned char)c) || c == '_' || c == '\'' ||
           ((unsigned char)c >= 0xC0); /* latin-1 letter bytes */
}
static int ci_eq(const char *a, const char *b, size_t n) {
    size_t i;
    for (i = 0; i < n; i++)
        if (tolower((unsigned char)a[i]) != tolower((unsigned char)b[i]))
            return 0;
    return 1;
}
/* lowercase copy of [s, s+n) into dst (dst size >= n+1) */
static void lower_copy(char *dst, const char *s, size_t n) {
    size_t i; for (i = 0; i < n; i++) dst[i] = tolower((unsigned char)s[i]);
    dst[n] = 0;
}

/* KEEP words: any capitalised-pair candidate containing one of these
 * (case-insensitive, substring on the lowered pair) is NOT a person name.
 * Covers institutions, degrees, job titles, skills, months/days, company
 * suffixes, CV section words. Also blocks "Curriculum Vitae" etc. */
static const char *KEEP_SUB[] = {
    "universit", "university", "politecnic", "college", "school", "istituto",
    "institute", "faculty", "facolt", "dipartimento", "department",
    "laurea", "master", "bachelor", "phd", "dottorato", "diploma", "mba",
    "engineer", "ingegner", "developer", "sviluppator", "manager",
    "director", "direttor", "consultant", "consulent", "analyst", "analista",
    "designer", "architect", "architetto", "technician", "tecnico",
    "specialist", "specialista", "assistant", "assistent", "officer",
    "accountant", "commercialista", "avvocato", "lawyer", "medico", "doctor",
    "nurse", "insegnante", "teacher", "professor", "ricercator",
    "project", "progetto", "product", "prodotto", "sales", "vendite",
    "marketing", "human resources", "risorse umane", "customer", "cliente",
    "software", "python", "java", "javascript", "typescript", "sql", "excel",
    "linux", "windows", "azure", "aws", "docker", "kubernetes", "git",
    "sap", "oracle", "tableau", "powerbi", "power bi", "machine learning",
    "data science", "curriculum", "vitae", "resume", "profilo", "profile",
    "esperienza", "experience", "esperienze", "istruzione", "education",
    "formazione", "competenze", "skills", "skill", "lingue", "languages",
    "hobby", "interessi", "patente", "driving", "licence", "license",
    "certificat", "attestato", "corso", "course", "lavoro", "work",
    "impiego", "employment", "tirocinio", "internship", "stage",
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "uned", "luned", "marted", "mercoled", "gioved", "venerd",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "s.p.a", "s.r.l", "srl", "spa", "s.a.s", "snc", "sdf", "s.s.d",
    "ltd", "limited", "inc", "corp", "gmbh", "sarl", "llc", "bv", "pty",
    "group", "gruppo", "holding", "partner", "associat", "studio",
    "banca", "bank", "assicurazion", "insurance", "telecom", "media",
    "sistemi", "systems", "solution", "soluzion", "service", "servizi",
    "codice", "fiscale", "partita", "iva", "telefono", "telefon",
    "tel.", "cell", "fax", "e-mail", "indirizzo", "residenza",
    "domicilio", "nascita", "nato", "nata", "cognome", "cittadinanza",
    "civile", "professione", "nazionalit", "nationality",
    "liceo", "maturita", "maturit", "scuola", "elementari", "medie",
    "superiori", "asilo", "voto", "votazione", "lode", "tesi",
    "relatore", "erasmus", "borsa di studio", "scholarship",
    "technolog", "tecnolog", "digital", "informatica", "consulting",
    "hospital", "ospedale", "clinica", "clinic", "comune", "municip",
    "ministero", "ministry", "agenzia", "agency", "ente", "fondazione",
    "foundation", "cooperativa", "onlus", "universita", "docente", NULL
};
static int keep_hit(const char *s, size_t n) {
    char buf[256]; int k;
    if (n >= sizeof buf) return 1; /* too long -> keep (not a name) */
    lower_copy(buf, s, n);
    for (k = 0; KEEP_SUB[k]; k++)
        if (strstr(buf, KEEP_SUB[k])) return 1;
    return 0;
}

/* Titles that force a following capitalised pair to be a person name. */
static int title_before(const char *line_start, const char *pos) {
    static const char *T[] = { "sig.", "sig.ra", "sig.ra.", "dott.", "dott.ssa",
        "dott.ssa.", "ing.", "avv.", "dr.", "prof.", "prof.ssa", "geom.",
        "rag.", "arch.", "egr.", NULL };
    int t; (void)line_start;
    for (t = 0; T[t]; t++) {
        size_t L = strlen(T[t]);
        if ((size_t)(pos - line_start) >= L + 1) {
            const char *q = pos - L - 1;
            if ((*q == ' ' || *q == '\t') && ci_eq(q + 1, T[t], L))
                return 1;
        }
        if (pos > line_start && ci_eq(pos - L, T[t], L) &&
            (pos - L == line_start || *(pos - L - 1) == ' ' || *(pos - L - 1) == '\t'))
            return 1; /* tight: "Dott.Rossi Mario" */
    }
    return 0;
}
/* Reference/emergency cue anywhere on the line (before the pair). */
static int ref_cue_on_line(const char *line_start, const char *pos) {
    static const char *C[] = { "referen", "referee", "emergen", "contatto",
        "contact", "a cura di", "redatto da", "coniuge", "figlio", "figlia",
        "madre", "padre", "sorella", "fratello", "parente", "kin",
        "presentazione di", "su segnalazione", NULL };
    char buf[512]; size_t n = (size_t)(pos - line_start);
    int c; if (n >= sizeof buf) n = sizeof buf - 1;
    lower_copy(buf, line_start, n);
    for (c = 0; C[c]; c++) if (strstr(buf, C[c])) return 1;
    return 0;
}

/* ---------------- span collection ---------------- */
typedef struct { size_t st, len; char type[PII_TYPE_MAX]; } Span;
typedef struct { Span *v; int n, cap; } SpanList;
static int spans_add(SpanList *L, size_t st, size_t len, const char *type) {
    if (!len) return 1;
    if (L->n == L->cap) {
        int nc = L->cap ? L->cap * 2 : 64;
        Span *nv = (Span *)realloc(L->v, (size_t)nc * sizeof *nv);
        if (!nv) return 0;
        L->v = nv; L->cap = nc;
    }
    L->v[L->n].st = st; L->v[L->n].len = len;
    snprintf(L->v[L->n].type, sizeof L->v[L->n].type, "%s", type);
    L->n++;
    return 1;
}
static int span_prio(const char *t) {
    if (!strcmp(t, "EMAIL")) return 0;
    if (!strcmp(t, "FISCAL")) return 1;
    if (!strcmp(t, "PHONE")) return 2;
    if (!strcmp(t, "ADDRESS")) return 3;
    return 4;
}
static int span_cmp(const void *a, const void *b) {
    const Span *x = (const Span *)a, *y = (const Span *)b;
    if (x->st != y->st) return x->st < y->st ? -1 : 1;
    return span_prio(x->type) - span_prio(y->type);
}

/* ---- EMAIL ---- */
static void find_emails(const char *s, size_t n, SpanList *L) {
    size_t i;
    for (i = 0; i < n; i++) {
        if (s[i] != '@') continue;
        size_t a = i, b = i + 1;
        while (a > 0 && (isalnum((unsigned char)s[a-1]) || strchr("._%+-", s[a-1]))) a--;
        while (b < n && (isalnum((unsigned char)s[b]) || strchr(".-", s[b]))) b++;
        /* need dot in domain part with TLD>=2 */
        {   size_t dot = b; int ok = 0;
            while (dot > i + 2) { dot--; if (s[dot] == '.') {
                if (b - dot - 1 >= 2 && b - dot - 1 <= 10) ok = 1; break; } }
            if (ok && i - a >= 1) spans_add(L, a, b - a, "EMAIL");
        }
    }
}
/* ---- FISCAL: CF16 | 11-digit P.IVA | US SSN ---- */
static int cf_at(const char *s, size_t n, size_t i) {
    /* 6L 2D 1L 2D 1L 3D 1L, word boundaries */
    if (i + 16 > n) return 0;
    {   int k;
        for (k = 0; k < 6; k++) if (!isalpha((unsigned char)s[i+k])) return 0;
        if (!isdigit((unsigned char)s[i+6]) || !isdigit((unsigned char)s[i+7])) return 0;
        if (!isalpha((unsigned char)s[i+8])) return 0;
        if (!isdigit((unsigned char)s[i+9]) || !isdigit((unsigned char)s[i+10])) return 0;
        if (!isalpha((unsigned char)s[i+11])) return 0;
        if (!isdigit((unsigned char)s[i+12]) || !isdigit((unsigned char)s[i+13]) ||
            !isdigit((unsigned char)s[i+14])) return 0;
        if (!isalpha((unsigned char)s[i+15])) return 0;
        if (i > 0 && is_wordch(s[i-1])) return 0;
        if (i + 16 < n && is_wordch(s[i+16])) return 0;
        return 1;
    }
}
static void find_fiscal(const char *s, size_t n, SpanList *L) {
    size_t i;
    for (i = 0; i < n; i++) {
        if (cf_at(s, n, i)) { spans_add(L, i, 16, "FISCAL"); i += 15; continue; }
        /* US SSN ddd-dd-dddd */
        if (i + 11 <= n && isdigit((unsigned char)s[i]) && isdigit((unsigned char)s[i+1]) &&
            isdigit((unsigned char)s[i+2]) && s[i+3] == '-' &&
            isdigit((unsigned char)s[i+4]) && isdigit((unsigned char)s[i+5]) && s[i+6] == '-' &&
            isdigit((unsigned char)s[i+7]) && isdigit((unsigned char)s[i+8]) &&
            isdigit((unsigned char)s[i+9]) && isdigit((unsigned char)s[i+10]) &&
            (i == 0 || !is_wordch(s[i-1])) && (i + 11 >= n || !is_wordch(s[i+11]))) {
            spans_add(L, i, 11, "FISCAL"); i += 10; continue;
        }
        /* P.IVA: exactly 11 digits, word boundaries, no '/' nearby (=date guard) */
        if (isdigit((unsigned char)s[i]) && (i == 0 || !is_wordch(s[i-1]))) {
            size_t j = i;
            while (j < n && isdigit((unsigned char)s[j])) j++;
            if (j - i == 11 && (j >= n || !is_wordch(s[j]))) {
                /* date guard: ddmmyyyy-like 8-digit handled (not 11); slash check */
                int slash = 0;
                if (i >= 1 && s[i-1] == '/') slash = 1;
                if (j < n && s[j] == '/') slash = 1;
                if (!slash) { spans_add(L, i, 11, "FISCAL"); i = j - 1; continue; }
            }
        }
    }
}
/* ---- PHONE: runs with >=7 digits, separators, no '/' (date guard) ---- */
static void find_phones(const char *s, size_t n, SpanList *L) {
    size_t i = 0;
    while (i < n) {
        if (isdigit((unsigned char)s[i]) || s[i] == '+' || s[i] == '(') {
            size_t j = i; int digits = 0, slash = 0, paren = 0, plus = 0;
            while (j < n && (isdigit((unsigned char)s[j]) || strchr(" +-.()/	", s[j]))) {
                if (isdigit((unsigned char)s[j])) digits++;
                if (s[j] == '/') slash = 1;
                if (s[j] == '(' || s[j] == ')') paren = 1;
                if (s[j] == '+') plus = 1;
                j++;
            }
            /* trim trailing separators */
            while (j > i && strchr(" +-.()	", s[j-1]) && !isdigit((unsigned char)s[j-1])) j--;
            /* year-range guard: employment dates like 2015-2022 are NOT phones */
            {   size_t a = i, b = j;
                while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '(')) a++;
                while (b > a && (s[b-1] == ' ' || s[b-1] == '\t' || s[b-1] == ')')) b--;
                if (b - a == 9 &&
                    ((s[a] == '1' && s[a+1] == '9') || (s[a] == '2' && s[a+1] == '0')) &&
                    isdigit((unsigned char)s[a+2]) && isdigit((unsigned char)s[a+3]) &&
                    (s[a+4] == '-' || s[a+4] == '\xe2') &&
                    isdigit((unsigned char)s[b-4]) && isdigit((unsigned char)s[b-3]) &&
                    isdigit((unsigned char)s[b-2]) && isdigit((unsigned char)s[b-1])) {
                    /* allow 2015-22 form too */
                    i = j; continue;
                }
                if (b - a == 7 &&
                    ((s[a] == '1' && s[a+1] == '9') || (s[a] == '2' && s[a+1] == '0')) &&
                    isdigit((unsigned char)s[a+2]) && isdigit((unsigned char)s[a+3]) &&
                    s[a+4] == '-' &&
                    isdigit((unsigned char)s[b-2]) && isdigit((unsigned char)s[b-1])) {
                    i = j; continue;
                }
            }
            if (!slash && digits >= 7 && digits <= 15 && (j - i) >= 7) {
                int has_sep = paren || plus || (j - i > (size_t)digits);
                /* plain digit run (mobile "3331234567") always counts */
                if (has_sep || (size_t)digits == j - i) {
                    /* word-ish boundaries */
                    if ((i == 0 || !isalpha((unsigned char)s[i-1])) &&
                        (j >= n || !isalpha((unsigned char)s[j])))
                        spans_add(L, i, j - i, "PHONE");
                    i = j; continue;
                }
            }
            i = (j > i) ? j : i + 1;
        } else i++;
    }
}
/* ---- ADDRESS: street keyword + following tokens incl. civic number ---- */
static const char *STREET_KW[] = { "via", "viale", "vicolo", "piazza",
    "piazzale", "corso", "largo", "strada", "borgo", "contrada", "frazione",
    "localita", "località", "street", "avenue", "road", "boulevard",
    "lane", "drive", "place", "court", "allee", "allée", NULL };
static int street_kw_at(const char *s, size_t n, size_t i, size_t *klen) {
    int k;
    for (k = 0; STREET_KW[k]; k++) {
        size_t L = strlen(STREET_KW[k]);
        if (i + L <= n && ci_eq(s + i, STREET_KW[k], L) &&
            (i == 0 || !is_wordch(s[i-1])) &&
            (i + L >= n || !is_wordch(s[i+L]))) {
            /* skip abbreviations with dot: "St." handled since '.' not wordch */
            *klen = L; return 1;
        }
    }
    return 0;
}
static void find_addresses(const char *s, size_t n, SpanList *L) {
    size_t i;
    for (i = 0; i < n; i++) {
        size_t kl = 0;
        if (!street_kw_at(s, n, i, &kl)) continue;
        {   size_t j = i + kl, commas = 0; int has_digit = 0;
            /* consume up to 110 chars / newline / second comma */
            while (j < n && s[j] != '\n' && (j - i) < 110) {
                if (s[j] == ',') { commas++; if (commas >= 2) break; }
                if (isdigit((unsigned char)s[j])) has_digit = 1;
                j++;
            }
            while (j > i && strchr(" ,;	", s[j-1])) j--;
            if (has_digit && j - i > kl + 2)
                spans_add(L, i, j - i, "ADDRESS");
        }
    }
}

/* ---- NAME helpers ---- */
static int cap_word_at(const char *s, size_t n, size_t i, size_t *wlen) {
    /* Capitalised latin word, len>=2 (first upper incl. accented, rest lower) */
    size_t j = i; unsigned char c = s[i];
    int up = (c >= 'A' && c <= 'Z') || (c >= 0xC0 && c <= 0xDE && c != 0xD7);
    if (!up || i > 0) { /* need start-of-word */
        if (i > 0 && is_wordch(s[i-1])) return 0;
    }
    if (!up) return 0;
    j++;
    while (j < n && (unsigned char)s[j] >= 0x80) j++; /* skip accent tails byte-wise */
    while (j < n && islower((unsigned char)s[j])) j++;
    /* allow internal apostrophe: D'Angelo */
    if (j < n && s[j] == '\'' && j + 1 < n && isupper((unsigned char)s[j+1])) {
        j += 2;
        while (j < n && islower((unsigned char)s[j])) j++;
    }
    if (j - i < 2) return 0;
    if (j < n && is_wordch(s[j])) return 0; /* ALL-CAPS or longer */
    *wlen = j - i; return 1;
}
static const char *line_start_of(const char *s, size_t pos) {
    while (pos > 0 && s[pos-1] != '\n') pos--;
    return s + pos;
}
static int line_no(const char *s, size_t pos) {
    int ln = 1; size_t k;
    for (k = 0; k < pos; k++) if (s[k] == '\n') ln++;
    return ln;
}
/* collect email-local-part token pairs for candidate-name derivation */
typedef struct { char a[64], b[64]; } NamePair;
static int derive_from_emails(const char *s, const SpanList *L,
                              NamePair *out, int cap) {
    int cnt = 0, i;
    for (i = 0; i < L->n && cnt < cap; i++) {
        if (strcmp(L->v[i].type, "EMAIL")) continue;
        {   const char *e = s + L->v[i].st; size_t el = L->v[i].len;
            size_t at = 0; while (at < el && e[at] != '@') at++;
            size_t p = 0, t0 = 0, t1 = (size_t)-1, t2 = (size_t)-1, tl = 0;
            int t = 0;
            for (p = 0; p <= at; p++) {
                int sep = (p == at) || e[p] == '.' || e[p] == '_' || e[p] == '-' || e[p] == '+';
                if (!sep) { if (!tl) t0 = p; tl++; continue; }
                if (tl >= 2 && t == 0) { t1 = t0; }
                if (tl >= 2) { t2 = t0; t++; t0 = 0; tl = 0; }
                else { tl = 0; }
            }
            if (t >= 2) {
                /* take first two tokens of len>=2: boundaries within local part */
                size_t q = 0, lens[8]; size_t starts[8]; int nt = 0;
                size_t u = 0, us = 0, ul = 0;
                for (u = 0; u <= at && nt < 8; u++) {
                    int sep = (u == at) || e[u] == '.' || e[u] == '_' || e[u] == '-' || e[u] == '+';
                    if (!sep) { if (!ul) us = u; ul++; continue; }
                    if (ul >= 2) { starts[nt] = us; lens[nt] = ul; nt++; }
                    ul = 0;
                }
                (void)q; (void)t1; (void)t2;
                if (nt >= 2 && lens[0] < sizeof out[0].a && lens[1] < sizeof out[0].b) {
                    memcpy(out[cnt].a, e + starts[0], lens[0]); out[cnt].a[lens[0]] = 0;
                    memcpy(out[cnt].b, e + starts[1], lens[1]); out[cnt].b[lens[1]] = 0;
                    cnt++;
                }
            }
        }
    }
    return cnt;
}
/* First CV section header line (1-based), or -1. The header zone (candidate
 * names, free-form) ends where the structured body begins. */
static int section_line(const char *s, size_t n) {
    static const char *SEC[] = { "istruzione", "education", "esperienza",
        "experience", "employment", "formazione", "competenze", "skills",
        "lingue", "languages", "referenze", "references", "pubblicazioni",
        "corsi", "certificazioni", "interessi", "hobby", "autorizzo",
        "patente", NULL };
    size_t i = 0; int ln = 1, k;
    char low[128]; size_t ll = 0;
    for (i = 0; i <= n; i++) {
        char c = i < n ? s[i] : '\n';
        if (c == '\n') {
            /* trim line, lowercase it */
            size_t a = 0, b = ll;
            while (a < b && (low[a] == ' ' || low[a] == '\t')) a++;
            while (b > a && (low[b-1] == ' ' || low[b-1] == '\t' || low[b-1] == '\r')) b--;
            if (b - a >= 4 && b - a <= 30) {
                char t[32]; size_t tl = b - a; int kk;
                if (tl >= sizeof t) tl = sizeof t - 1;
                for (kk = 0; kk < (int)tl; kk++)
                    t[kk] = tolower((unsigned char)low[a + kk]);
                t[tl] = 0;
                for (k = 0; SEC[k]; k++)
                    if (strstr(t, SEC[k])) return ln;
            }
            ln++; ll = 0;
        } else if (ll < sizeof low - 1) low[ll++] = c;
    }
    return -1;
}
static void find_names(const char *s, size_t n, const SpanList *L, SpanList *NL) {
    NamePair pairs[16]; int np = 0, pi; size_t i;
    int sec = section_line(s, n);
    int header_end = (sec > 0 && sec - 1 <= 12) ? sec - 1 : 12;
    if (header_end < 4) header_end = 4;
    np = derive_from_emails(s, L, pairs, 16);
    for (i = 0; i < n; i++) {
        size_t w1 = 0, w2 = 0, w3 = 0, p = i;
        if (!cap_word_at(s, n, i, &w1)) continue;
        p = i + w1;
        while (p < n && (s[p] == ' ' || s[p] == '\t')) p++;
        if (!cap_word_at(s, n, p, &w2)) continue;
        {   size_t pend = p + w2, q = pend;
            while (q < n && (s[q] == ' ' || s[q] == '\t')) q++;
            if (cap_word_at(s, n, q, &w3) && w3 >= 2 && q - pend <= 2) {
                /* 3-token name only with title/ref cue (avoid "Laurea Magistrale X") */
                const char *ls = line_start_of(s, i);
                if (!(title_before(ls, s + i) || ref_cue_on_line(ls, s + i)))
                    w3 = 0;
            } else w3 = 0;
            {   size_t tot = (w3 ? (q + w3) : pend) - i;
                char low[256];
                if (tot >= sizeof low) continue;
                lower_copy(low, s + i, tot);
                if (keep_hit(s + i, tot)) continue;
                {   const char *ls = line_start_of(s, i);
                    int header = line_no(s, i) <= header_end;
                    int title = title_before(ls, s + i);
                    int ref = ref_cue_on_line(ls, s + i);
                    int derived = 0;
                    for (pi = 0; pi < np && !derived; pi++) {
                        char cand[140];
                        snprintf(cand, sizeof cand, "%s %s", pairs[pi].a, pairs[pi].b);
                        if (ci_eq(s + i, cand, strlen(cand)) &&
                            (tot == strlen(cand) || (w3 && tot > strlen(cand))))
                            derived = 1;
                    }
                    if (header || title || ref || derived) {
                        size_t fl = w3 ? (q + w3 - i) : (pend - i);
                        spans_add(NL, i, fl, "NAME");
                        i += fl - 1;
                    }
                }
            }
        }
    }
}

/* ---------------- public mask/unmask ---------------- */
char *pii_mask_text(const char *in, PiiVaultMap *map) {
    size_t n = strlen(in);
    SpanList L = {0}, NL = {0};
    Span *all = NULL; int na = 0, i;
    size_t out_cap, out_len = 0; char *out;
    find_emails(in, n, &L);
    find_fiscal(in, n, &L);
    find_phones(in, n, &L);
    find_addresses(in, n, &L);
    find_names(in, n, &L, &NL);
    for (i = 0; i < NL.n; i++) spans_add(&L, NL.v[i].st, NL.v[i].len, NL.v[i].type);
    free(NL.v);
    if (!L.n) {
        free(L.v);
        out = (char *)malloc(n + 1);
        if (out) memcpy(out, in, n + 1);
        return out;
    }
    qsort(L.v, (size_t)L.n, sizeof(Span), span_cmp);
    /* drop overlaps: keep earliest start, then priority */
    all = (Span *)malloc((size_t)L.n * sizeof(Span));
    if (!all) { free(L.v); return NULL; }
    for (i = 0; i < L.n; i++) {
        if (na > 0 && L.v[i].st < all[na-1].st + all[na-1].len) continue;
        all[na++] = L.v[i];
    }
    free(L.v);
    out_cap = n + (size_t)na * 16 + 1;
    out = (char *)malloc(out_cap);
    if (!out) { free(all); return NULL; }
    {   size_t cur = 0;
        for (i = 0; i < na; i++) {
            const char *ph = pii_map_intern(map, all[i].type,
                                            in + all[i].st, all[i].len);
            size_t pl;
            if (!ph) { free(all); free(out); return NULL; }
            pl = strlen(ph);
            memcpy(out + out_len, in + cur, all[i].st - cur);
            out_len += all[i].st - cur;
            memcpy(out + out_len, ph, pl); out_len += pl;
            cur = all[i].st + all[i].len;
        }
        memcpy(out + out_len, in + cur, n - cur);
        out_len += n - cur;
        out[out_len] = 0;
    }
    free(all);
    return out;
}
char *pii_unmask_text(const char *in, const PiiVaultMap *map) {
    /* replace longest placeholders first ([NAME_10] before [NAME_1]) */
    int *ord = NULL, i, j;
    size_t n = strlen(in), cap = n + 1, len = 0;
    char *out; int k;
    for (k = 0; k < map->n; k++) cap += strlen(map->items[k].original);
    out = (char *)malloc(cap);
    if (!out) return NULL;
    ord = (int *)malloc((map->n ? (size_t)map->n : 1) * sizeof(int));
    if (!ord) { free(out); return NULL; }
    for (i = 0; i < map->n; i++) ord[i] = i;
    for (i = 0; i < map->n; i++)
        for (j = i + 1; j < map->n; j++)
            if (strlen(map->items[ord[j]].placeholder) >
                strlen(map->items[ord[i]].placeholder)) {
                int t = ord[i]; ord[i] = ord[j]; ord[j] = t;
            }
    {   const char *p = in;
        while (*p) {
            int hit = -1;
            for (i = 0; i < map->n; i++) {
                size_t pl = strlen(map->items[ord[i]].placeholder);
                if (!strncmp(p, map->items[ord[i]].placeholder, pl)) {
                    hit = ord[i]; break;
                }
            }
            if (hit >= 0) {
                size_t ol = strlen(map->items[hit].original);
                memcpy(out + len, map->items[hit].original, ol);
                len += ol; p += strlen(map->items[hit].placeholder);
            } else out[len++] = *p++;
        }
        out[len] = 0;
    }
    free(ord);
    return out;
}

/* ---------------- JSON payload ---------------- */
static void json_esc(const char *s, char **p, size_t *cap, size_t *len) {
    /* appends "escaped" (with quotes) */
    const unsigned char *u = (const unsigned char *)s;
    if (*len + 2 >= *cap) return;
    (*p)[(*len)++] = '"';
    for (; *u; u++) {
        char tmp[8]; size_t tl = 0;
        if (*u == '"' || *u == '\\') { tmp[0] = '\\'; tmp[1] = *u; tl = 2; }
        else if (*u == '\n') { tmp[0] = '\\'; tmp[1] = 'n'; tl = 2; }
        else if (*u == '\r') { tmp[0] = '\\'; tmp[1] = 'r'; tl = 2; }
        else if (*u == '\t') { tmp[0] = '\\'; tmp[1] = 't'; tl = 2; }
        else if (*u < 0x20) { snprintf(tmp, sizeof tmp, "\\u%04x", *u); tl = 6; }
        else { tmp[0] = *u; tl = 1; }
        while (*len + tl + 2 >= *cap) {
            *cap *= 2;
            *p = (char *)realloc(*p, *cap);
            if (!*p) return;
        }
        memcpy(*p + *len, tmp, tl); *len += tl;
    }
    (*p)[(*len)++] = '"'; (*p)[*len] = 0;
}
char *pii_map_to_json(const PiiVaultMap *m, const char *uuid,
                      const char *source_file) {
    size_t cap = 1024, len = 0; char *p; int i;
    p = (char *)malloc(cap); if (!p) return NULL;
    len = snprintf(p, cap, "{\"v\":2,\"uuid\":");
    json_esc(uuid, &p, &cap, &len);
    {   char tmp[512]; int tl = snprintf(tmp, sizeof tmp, ",\"source\":");
        while (len + (size_t)tl + 1 >= cap) { cap *= 2; p = realloc(p, cap); if (!p) return NULL; }
        memcpy(p + len, tmp, (size_t)tl); len += (size_t)tl;
    }
    json_esc(source_file, &p, &cap, &len);
    {   char tmp[64]; int tl = snprintf(tmp, sizeof tmp, ",\"map\":[");
        while (len + (size_t)tl + 1 >= cap) { cap *= 2; p = realloc(p, cap); if (!p) return NULL; }
        memcpy(p + len, tmp, (size_t)tl); len += (size_t)tl;
    }
    for (i = 0; i < m->n; i++) {
        char tmp[64];
        int tl = snprintf(tmp, sizeof tmp, "%s{\"ph\":", i ? "," : "");
        while (len + (size_t)tl + 1 >= cap) { cap *= 2; p = realloc(p, cap); if (!p) return NULL; }
        memcpy(p + len, tmp, (size_t)tl); len += (size_t)tl;
        json_esc(m->items[i].placeholder, &p, &cap, &len);
        tl = snprintf(tmp, sizeof tmp, ",\"orig\":");
        while (len + (size_t)tl + 1 >= cap) { cap *= 2; p = realloc(p, cap); if (!p) return NULL; }
        memcpy(p + len, tmp, (size_t)tl); len += (size_t)tl;
        json_esc(m->items[i].original, &p, &cap, &len);
        tl = snprintf(tmp, sizeof tmp, ",\"type\":");
        while (len + (size_t)tl + 1 >= cap) { cap *= 2; p = realloc(p, cap); if (!p) return NULL; }
        memcpy(p + len, tmp, (size_t)tl); len += (size_t)tl;
        json_esc(m->items[i].type, &p, &cap, &len);
        /* v003: spatial box, present only when known (v1 readers ignore it) */
        if (m->items[i].has_box) {
            tl = snprintf(tmp, sizeof tmp,
                          ",\"box\":[%.1f,%.1f,%.1f,%.1f,%d]",
                          m->items[i].bx0, m->items[i].by0,
                          m->items[i].bx1, m->items[i].by1,
                          m->items[i].bpage);
            while (len + (size_t)tl + 1 >= cap) { cap *= 2; p = realloc(p, cap); if (!p) return NULL; }
            memcpy(p + len, tmp, (size_t)tl); len += (size_t)tl;
        }
        tl = snprintf(tmp, sizeof tmp, "}");
        while (len + (size_t)tl + 1 >= cap) { cap *= 2; p = realloc(p, cap); if (!p) return NULL; }
        memcpy(p + len, tmp, (size_t)tl); len += (size_t)tl;
    }
    {   const char *tail = "]}";
        while (len + 3 >= cap) { cap *= 2; p = realloc(p, cap); if (!p) return NULL; }
        memcpy(p + len, tail, 2); len += 2; p[len] = 0;
    }
    return p;
}
/* minimal JSON string-field scanner: finds "key":"value" with unescaping */
static int jfield(const char **pp, const char *key, char *out, size_t cap) {
    char pat[64]; const char *f, *p;
    snprintf(pat, sizeof pat, "\"%s\"", key);
    f = strstr(*pp, pat); if (!f) return 0;
    p = strchr(f + strlen(pat), ':'); if (!p) return 0;
    p++; while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return 0;
    p++; {   size_t o = 0;
        while (*p && *p != '"' && o + 1 < cap) {
            if (*p == '\\' && p[1]) {
                p++;
                if (*p == 'n') out[o++] = '\n';
                else if (*p == 'r') out[o++] = '\r';
                else if (*p == 't') out[o++] = '\t';
                else if (*p == 'u' && isxdigit((unsigned char)p[1])) {
                    /* \u00XX -> latin-1 byte (payload is latin-1/utf8 text) */
                    char hb[5]; int v = 0;
                    memcpy(hb, p + 1, 4); hb[4] = 0;
                    v = (int)strtol(hb, NULL, 16);
                    if (v < 0x80) out[o++] = (char)v;
                    else if (v < 0x800 && o + 2 < cap) {
                        out[o++] = (char)(0xC0 | (v >> 6));
                        out[o++] = (char)(0x80 | (v & 0x3F));
                    }
                    p += 4;
                } else out[o++] = *p;
                p++;
            } else out[o++] = *p++;
        }
        out[o] = 0;
        if (*p == '"') p++;
        *pp = p; return 1;
    }
}
int pii_map_from_json(const char *json, PiiVaultMap *m,
                      char *uuid_out, size_t uuid_cap) {
    const char *p = json;
    char ub[128] = {0};
    pii_map_free(m); pii_map_init(m);
    if (!jfield(&p, "uuid", ub, sizeof ub)) return 0;
    if (uuid_out && uuid_cap) snprintf(uuid_out, uuid_cap, "%s", ub);
    while (1) {
        char ph[64] = {0}, og[PII_ORIG_MAX + 1] = {0}, ty[32] = {0};
        const char *q = strstr(p, "\"ph\"");
        if (!q) break;
        p = q;
        if (!jfield(&p, "ph", ph, sizeof ph)) break;
        if (!jfield(&p, "orig", og, sizeof og)) break;
        if (!jfield(&p, "type", ty, sizeof ty)) break;
        if (!pii_map_intern(m, ty, og, strlen(og))) return 0;
        /* v003: optional spatial box [x0,y0,x1,y1,page]; absent in v1. */
        {   const char *nph = strstr(p, "\"ph\"");
            const char *bx = strstr(p, "\"box\"");
            PiiEntry *e = &m->items[m->n - 1];
            e->has_box = 0;
            if (bx && (!nph || bx < nph)) {
                float x0, y0, x1, y1; int pg;
                if (sscanf(bx, "\"box\":[%f,%f,%f,%f,%d",
                           &x0, &y0, &x1, &y1, &pg) == 5) {
                    e->bx0 = x0; e->by0 = y0;
                    e->bx1 = x1; e->by1 = y1;
                    e->bpage = pg; e->has_box = 1;
                }
            }
        }
        /* fix placeholder numbering (intern assigns new numbers): overwrite */
        snprintf(m->items[m->n - 1].placeholder,
                 sizeof m->items[m->n - 1].placeholder, "%s", ph);
        /* keep counters monotonic */
        {   char *us = strrchr(ph, '_'); int v = us ? atoi(us + 1) : 0;
            if (!strcmp(ty, "NAME") && v > m->c_name) m->c_name = v;
            if (!strcmp(ty, "EMAIL") && v > m->c_email) m->c_email = v;
            if (!strcmp(ty, "PHONE") && v > m->c_phone) m->c_phone = v;
            if (!strcmp(ty, "FISCAL") && v > m->c_fiscal) m->c_fiscal = v;
            if (!strcmp(ty, "ADDRESS") && v > m->c_addr) m->c_addr = v;
        }
    }
    return m->n > 0 || strstr(json, "\"map\":[") != NULL;
}
