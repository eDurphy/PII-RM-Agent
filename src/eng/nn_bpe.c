/* ===========================================================================
 * qatnn_bpe.c - tokenizer byte-level BPE per l'autobuilder.
 *
 * PERCHE' ESISTE
 * Il modello di qatnn_llm.c e' a CARATTERI, e data_build() blocca il
 * vocabolario a 40 simboli (`while (V < 40 ...)`, qatnn_llm.c:1248). Un
 * dataset vero di Hugging Face e' scritto in byte BPE: 128256 token per
 * Llama-3, 32000 per Mistral, 151k per Qwen. Con V che non puo' passare 40,
 * il dataset non entra. Non e' una questione di velocita': la matrice di
 * embedding e' [V x d], e con V=128256 servirebbero 8.2 milioni di colonne per
 * una rete che oggi ha 85mila parametri in totale.
 *
 * Il limite di 40 era una scelta di data_build(), non un limite della rete:
 * model_new(V, d, ...) accetta qualsiasi V e la testa e' gia' allocata
 * [V x d]. Basta passargli un V vero.
 *
 * PERCHE' BYTE-LEVEL E NON SUBCARATTERE
 * Il byte-level e' l'unica base che non perde niente: 256 byte coprono
 * l'intero spazio Unicode, quindi cinese, arabo ed emoji entrano intatti,
 * mentre 40 caratteri ASCII getterebbero il resto. Il costo e' un alfabeto
 * iniziale di 256 simboli invece di ~40.
 *
 * L'addestramento e' il BPE standard: conta le coppie adiacenti, fusiona la
 * piu' frequente, riconta. Ogni fusione aggiunge UN simbolo, quindi con
 * BPE_MERGE_MAX fusioni si arriva a 256 + BPE_MERGE_MAX simboli.
 *
 * Compile : incluso da qatnn_autobuilder.c, non compilato da solo.
 * ========================================================================= */
#ifndef QATNN_BPE_H
#define QATNN_BPE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* 256 byte + MERGE_MAX fusioni. Ogni fusione e' UN simbolo in piu'. La
 * testa e' [V x d] : con V=1024 e d=64 sono 65k parametri solo per la
 * testa, quindi V grande costa memoria E tempo per passo.                */
#define BPE_MERGE_MAX  768      /* 256 + 768 = 1024 simboli massimi        */
#define BPE_TOKEN_MAX   32      /* byte massimi per simbolo               */

/* Una fusione: due simboli diventano uno. I simboli sono indici in [0,V),
 * quindi una fusione e' una coppia di interi piu' il nuovo id.            */
typedef struct { int32_t a, b, id; uint32_t freq; } BpeMerge;

/* Il tokenizer. `sym[k]` e' il testo del simbolo k (non termina in \0, ha
 * lunghezza `len[k]`), cosi' un simbolo puo' contenere qualsiasi byte,
 * incluso \0, cosa che una stringa C non saprebbe rappresentare.           */
typedef struct {
    int32_t  V;                        /* 256..256+MERGE_MAX                 */
    uint16_t len[BPE_MERGE_MAX + 256];
    uint8_t  sym[BPE_MERGE_MAX + 256][BPE_TOKEN_MAX];
    BpeMerge mrg[BPE_MERGE_MAX];
    int32_t  n_merge;
} Bpe;

static void bpe_init(Bpe *B)
{
    int i;
    memset(B, 0, sizeof *B);
    for (i = 0; i < 256; i++) { B->sym[i][0] = (uint8_t)i; B->len[i] = 1; }
    B->V = 256;
}

/* ---------------------------------------------------------------------------
 * La tabella e' aperta, con indici a potenza di due e risoluzione lineare:
 * O(1) in media, nessuna libreria, nessun malloc nel percorso caldo.
 * ------------------------------------------------------------------------- */
#define BPE_HT_BITS  16
#define BPE_HT_SIZE (1 << BPE_HT_BITS)
#define BPE_HT_MASK (BPE_HT_SIZE - 1)

typedef struct {
    int32_t key[BPE_HT_SIZE];      /* coppia codificata in un intero      */
    int32_t rank[BPE_HT_SIZE];
    uint8_t used[BPE_HT_SIZE];
} BpeHt;

/* entrambi < 2^20 nella pratica (V <= 1024), quindi 40 bit bastano.       */
static int32_t bpe_key(int32_t a, int32_t b)
{
    return (int32_t)(((uint32_t)a << 20) | (uint32_t)b);
}

static uint32_t bpe_hash(int32_t k)
{
    uint32_t x = (uint32_t)k;
    x ^= x >> 15; x *= 0x2545F491u; x ^= x >> 13;
    return x & BPE_HT_MASK;
}

/* cerca (a,b). Ritorna lo slot, e *found dice se era gia' presente.        */
static int32_t bpe_ht_slot(BpeHt *H, int32_t a, int32_t b, int *found)
{
    int32_t k = bpe_key(a, b);
    uint32_t h = bpe_hash(k);
    uint32_t n = 0;
    while (H->used[h]) {
        if (H->key[h] == k) { if (found) *found = 1; return (int32_t)h; }
        h = (h + 1) & BPE_HT_MASK;
        if (++n >= BPE_HT_SIZE) return -1; /* tabella piena: scarta la coppia */
    }
    if (found) *found = 0;
    return (int32_t)h;
}

/* =========================================================================
 * 1. addestramento del BPE sul corpus
 * ======================================================================= */
/* Restituisce il numero di fusioni apprese. `cap_sym` e' il tetto di byte
 * per simbolo: una fusione che lo supererebbe viene saltata, altrimenti un
 * documento molto ripetitivo produrrebbe un simbolo da 300 byte.          */
static int32_t bpe_train(Bpe *B, const char *txt, size_t n, int cap_sym)
{
    int32_t *ids;
    size_t i, k;
    int32_t n_ids, merges = 0;
    BpeHt *H;
    int32_t banned[128];          /* coppie scartate, per sempre            */
    int n_banned = 0;

    if (n < 2) return 0;
    ids = (int32_t *)malloc(n * sizeof(int32_t));
    H   = (BpeHt *)calloc(1, sizeof(BpeHt));
    if (!ids || !H) { free(ids); free(H); return 0; }
    for (i = 0; i < n; i++) ids[i] = (unsigned char)txt[i];
    n_ids = (int32_t)n;

    while (merges < BPE_MERGE_MAX) {
        uint32_t best = 0; int32_t ba = -1, bb = -1, s;

        /* --- 1. conta le coppie adiacenti, davvero tutte ------------------
         * Il conteggio e' esatto: a differenza di molte implementazioni
         * veloci, qui non si tiene "una coppia per primo elemento" che
         * sottovaluta le frequenze. E' un hashing su 16k slot e una
         * passata sul testo, quindi O(n) vero per iterazione.              */
        memset(H->used, 0, sizeof H->used);
        for (i = 0; i + 1 < (size_t)n_ids; i++) {
            int found;
            s = bpe_ht_slot(H, ids[i], ids[i + 1], &found);
            if (s < 0) continue; /* tabella piena: coppia scartata, no hang */
            if (!found) { H->used[s] = 1; H->key[s] = bpe_key(ids[i], ids[i + 1]);
                          H->rank[s] = 0; }
            H->rank[s]++;
        }
        for (i = 0; i < BPE_HT_SIZE; i++) {
            if (H->used[i] && (uint32_t)H->rank[i] > best) {
                best = (uint32_t)H->rank[i];
                ba = H->key[i] >> 20;
                bb = H->key[i] & 0xFFFFF;
            }
        }
        if (ba < 0 || best < 2) break;            /* niente da fondere       */
        {   /* salta le coppie gia' scartate per la lunghezza            */
            int b;
            for (b = 0; b < n_banned; b++)
                if (banned[b * 2] == ba && banned[b * 2 + 1] == bb) {
                    best = 0; ba = -1; break;
                }
            if (ba < 0) {
                /* la vincitrice e' scartata : ne cerco un'altra valida.
                 * Se non ne resta nessuna, finisce qui il ciclo.          */
                int found_any = 0;
                for (i = 0; i < BPE_HT_SIZE && !found_any; i++) {
                    int32_t ca, cb; int b2;
                    if (!H->used[i] || (uint32_t)H->rank[i] < 2) continue;
                    ca = H->key[i] >> 20; cb = H->key[i] & 0xFFFFF;
                    for (b2 = 0; b2 < n_banned; b2++)
                        if (banned[b2 * 2] == ca && banned[b2 * 2 + 1] == cb) break;
                    if (b2 < n_banned) continue;
                    ba = ca; bb = cb; best = (uint32_t)H->rank[i]; found_any = 1;
                }
                if (!found_any) break;
            }
        }

        /* --- 2. la fusione ci sta nel tetto di byte ? --------------------- */
        if ((int)B->len[ba] + (int)B->len[bb] > cap_sym) {
            /* La coppia e' troppo lunga : la scarto per sempre. Il mio primo
             * tentativo la escludeva solo dalla tabella di questa iterazione,
             * ma la tabella viene RICOSTRUITA in cima al ciclo, quindi la
             * stessa coppia tornava come vincitrice e il ciclo non finiva
             * piu'. Le coppie scartate vanno tenute in una lista che dura. */
            if (n_banned < 64) {
                banned[n_banned * 2]     = ba;
                banned[n_banned * 2 + 1] = bb;
                n_banned++;
                continue;
            }
            break;      /* troppe scartate : mi fermo, dichiarato           */
        }

        /* --- 3. crea il simbolo e applicalo a tutto il testo -------------- */
        {   int32_t nid = B->V;
            uint8_t *ns = B->sym[nid];
            int nl = B->len[ba] + B->len[bb];
            memcpy(ns, B->sym[ba], B->len[ba]);
            memcpy(ns + B->len[ba], B->sym[bb], B->len[bb]);
            B->len[nid] = (uint16_t)nl;
            B->mrg[merges].a = ba; B->mrg[merges].b = bb;
            B->mrg[merges].id = nid; B->mrg[merges].freq = best;
            merges++; B->V++;

            k = 0;
            for (i = 0; i < (size_t)n_ids; ) {
                if (i + 1 < (size_t)n_ids && ids[i] == ba && ids[i + 1] == bb) {
                    ids[k++] = nid; i += 2;
                } else {
                    ids[k++] = ids[i]; i++;
                }
            }
            n_ids = (int32_t)k;
        }
        /* Se il testo e' diventato un simbolo solo non c'e' piu' nulla da
         * fondere, e senza questo controllo il ciclo continuerebbe a
         * fondere il simbolo con se' stesso su un corpus omogeneo.          */
        if (n_ids < 2) break;
    }
    B->n_merge = merges;
    free(ids); free(H);
    return merges;
}

/* =========================================================================
 * 2. codifica : testo -> id
 * ======================================================================= */
static int32_t bpe_encode(const Bpe *B, const char *txt, size_t n,
                          int32_t *out, int32_t out_max)
{
    int32_t *ids, n_ids, i, nout = 0;
    if (n == 0) return 0;
    ids = (int32_t *)malloc(n * sizeof(int32_t));
    if (!ids) return 0;
    for (i = 0; i < (int32_t)n; i++) ids[i] = (unsigned char)txt[i];
    n_ids = (int32_t)n;

    if (B->n_merge > 0) {
        /* Una passata per ogni fusione, in ordine di rango, e ogni passata
         * scrive in un buffer nuovo invece di spostare il resto dell'array
         * a ogni fusione.
         *
         * Prima facevo cosi': un ciclo che ri-scansionava tutta la sequenza
         * finche' non succedeva piu' niente, e a ogni fusione riusava
         * memmove sul TUTTO di cio' che restava. Con un corpus da 20 MB
         * quello e' 80 MB spostati per ogni singola fusione, centinaia di
         * volte : quadratico. Non e' un difetto di prestazione, e' il
         * motivo per cui il /load di un CSV da 3,2 GB non terminava mai.
         * Il BPE classico applica le fusioni per rango, quindi una passata
         * per rango con un doppio buffer e' la stessa cosa con un costo
         * O(n) per rango.                                                   */
        int32_t lvl;
        int32_t *tmp = (int32_t *)malloc((size_t)n * sizeof(int32_t));
        if (!tmp) { free(ids); return 0; }
        for (lvl = 0; lvl < B->n_merge; lvl++) {
            int32_t a = B->mrg[lvl].a, b = B->mrg[lvl].b;
            int32_t id = B->mrg[lvl].id, k = 0;
            for (i = 0; i < n_ids; ) {
                if (i + 1 < n_ids && ids[i] == a && ids[i + 1] == b)
                    { tmp[k++] = id; i += 2; }
                else tmp[k++] = ids[i++];
            }
            if (k != n_ids) { memcpy(ids, tmp, (size_t)k * sizeof(int32_t));
                              n_ids = k; }
            if (n_ids < 2) break;
        }
        free(tmp);
    }
    for (i = 0; i < n_ids && nout < out_max; i++) out[nout++] = ids[i];
    free(ids);
    return nout;
}

/* =========================================================================
 * 3. salvataggio : il tokenizer va nel registro, non solo i pesi
 * ======================================================================= */
/* =========================================================================
 * PERCHE' bpe_load() NON FUNZIONAVA MAI, E PERCHE' IL SEPARATORE E' UN ':'
 *
 * Il formato scriveva la lunghezza del simbolo come "%u " e la leggeva con
 * "%u ". Ma quello spazio dentro una stringa di formato di scanf NON e' un
 * separatore esatto : e' "zero o piu' caratteri di spaziatura", e quindi
 * si mangia TUTTI quelli che seguono. E il primo byte del simbolo e' spesso
 * uno spazio, perche' il vocabolario e' byte-level e lo spazio e' il
 * simbolo numero 32, sempre presente per definizione.
 *
 * Quindi: leggevo la lunghezza, la spaziatura mangiava il separatore E il
 * primo byte del simbolo, il simbolo risultava corto di uno e da li' in
 * poi tutto il file era fuori passo. bpe_load tornava 0 SEMPRE. Non era
 * codice morto per disattenzione : era codice che non funziona, e nessuno
 * lo aveva notato perche' nessuno lo chiamava.
 *
 * Il ':' e' esatto. Un byte che capita a essere ':' non confonde niente,
 * perche' la lunghezza e' gia' stata consumata prima di leggere i byte.
 * La magia passa a QATNNBPE2 cosi' un file scritto con le regole vecchie
 * non viene letto con quelle nuove e viceversa.
 * ---------------------------------------------------------------------- */
static void bpe_save(const Bpe *B, const char *path)
{
    FILE *f = fopen(path, "wb");
    int32_t i, j;
    if (!f) return;
    fprintf(f, "QATNNBPE2 %d %d\n", B->V, B->n_merge);
    for (i = 0; i < B->V; i++) {
        /* lunghezza esplicita : un simbolo puo' contenere \0 e spazi, che
         * una riga di testo non distinguerebbe da un campo.               */
        fprintf(f, "%u:", (unsigned)B->len[i]);
        for (j = 0; j < B->len[i]; j++) fputc(B->sym[i][j], f);
        fputc('\n', f);
    }
    for (i = 0; i < B->n_merge; i++)
        fprintf(f, "M %d %d %d %u\n", B->mrg[i].a, B->mrg[i].b,
                B->mrg[i].id, B->mrg[i].freq);
    fclose(f);
}

static int bpe_load(Bpe *B, const char *path)
{
    FILE *f = fopen(path, "rb");
    char magic[16];
    int V, nm, i, j;
    if (!f) return 0;
    if (fscanf(f, "%15s %d %d", magic, &V, &nm) != 3 ||
        strcmp(magic, "QATNNBPE2") != 0) { fclose(f); return 0; }
    if (V < 256 || V > 256 + BPE_MERGE_MAX || nm < 0 || nm > BPE_MERGE_MAX) {
        fclose(f); return 0;
    }
    bpe_init(B);
    for (i = 0; i < V; i++) {
        unsigned L = 0;
        if (fscanf(f, "%u:", &L) != 1 || L > BPE_TOKEN_MAX) { fclose(f); return 0; }
        for (j = 0; j < (int)L; j++) {
            int c = fgetc(f);
            if (c == EOF) { fclose(f); return 0; }
            B->sym[i][j] = (uint8_t)c;
        }
        B->len[i] = (uint16_t)L;
        fgetc(f);                       /* il '\n' finale                    */
    }
    for (i = 0; i < nm; i++) {
        int a, b, id; unsigned fr;
        if (fscanf(f, " M %d %d %d %u", &a, &b, &id, &fr) != 4) break;
        B->mrg[i].a = a; B->mrg[i].b = b; B->mrg[i].id = id; B->mrg[i].freq = fr;
    }
    fclose(f);
    B->V = V; B->n_merge = nm;
    return 1;
}

/* decodifica un id nel testo che rappresenta */
static int32_t bpe_decode1(const Bpe *B, int32_t id, char *out, int32_t cap)
{
    int32_t L;
    if (id < 0 || id >= B->V) return 0;
    L = B->len[id];
    if (L > cap) L = cap;
    memcpy(out, B->sym[id], (size_t)L);
    return L;
}

#endif /* QATNN_BPE_H */