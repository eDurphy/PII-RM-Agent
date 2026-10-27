/* ===========================================================================
 * qatnn.h - QATNN reference implementation, revisione 7 del paper
 * "QATNN Research Paper.docx" (1 029 448 B, md5 9ed8eb3720a1f95f0c977b1a7897c7b4)
 *
 * Scritto interamente in C99, senza librerie esterne, senza floating point
 * nel percorso di inferenza. Compilatore di riferimento: gcc 4.8.1 (tdm64-2).
 *
 * Ogni formula del paper e' presente qui come funzione nominata, e ogni
 * scelta deliberatamente dubbia e' resa selezionabile da un enum, cosi' che
 * il programma possa misurare sia la versione scritta sia la versione
 * corretta invece di asserts silenziosamente.
 * ======================================================================== */
#ifndef QATNN_H
#define QATNN_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- formati
 * Il paper dichiara questi formati. Ogni formato porta con se' il suo shift
 * frazionario, e la funzione q_<fmt>_round() converte un intero grezzo nel
 * formato. Sono usate nei test di conformita'.                                */
#define Q_FRAC        15      /* Q16.15 : 1 segno, 16 interi, 15 decimali   */
#define QG_FRAC       11      /* Q4.11  : 1 segno,  4 interi, 11 decimali   */
#define LR_FRAC        0      /* UQ0.16 : 0 segno,  0 interi, 16 decimali   */
#define ACC_FRAC      25      /* UQ7.25 : 0 segno,  7 interi, 25 decimali   */
#define DECAY_FRAC    12      /* UQ4.12 : 0 segno,  4 interi, 12 decimali   */

/* Limiti dichiarati dal paper (sezione 1b, revisione 7):
 * "il range di saturazione asintotica e' limitato fisicamente all'intervallo
 *  [-2.0,+1.99997]". Q16.15 su int32_t e' esattamente questo.               */
#define Q16_15_MIN    (-2147483648LL / (1LL << Q_FRAC))       /* -2.0      */
#define Q16_15_MAX    ((2147483647LL / (1LL << Q_FRAC)))       /* +1.99997  */

/* ------------------------------------------------- costanti del paper */
#ifndef THRESHOLD
#define THRESHOLD        213      /* sez. 2, soglia di isteresi             */
#endif
#define SILUNS_XMIN   (-41900)     /* sez. 1b, Xmin = -1.28 in Q16.15       */
#define SILUNS_OFF    (-20950)     /* sez. 1b, offset di raccordo           */
#define GRAD_SHIFT       19        /* sez. 5, Error (x) Input >> 19         */
#define ACC_SHIFT         9        /* sez. 8, AccError >> 9                 */
#define WINDOW            8        /* sez. 8, ciclo adattivo a 8 inferenze */
#define LR_FLOOR         64        /* sez. 8, pavimento strutturale         */
#define DECAY_CLIP       15        /* sez. 6, clipping del decay 0-15       */
#define LAYER_CUT_MAX 42949672u    /* sez. 9, limite massimo dichiarato     */
#define LAYER_CUT_MIN      40      /* sez. 9, soglia minima dichiarata      */
#define TRITS_PER_BYTE      5      /* sez. 3, 5 trit per byte               */

/* ===================================================================== */
/*  Le due letture di gamma-shift che il paper non distingue            */
/* ===================================================================== */
/*  REVISIONE 7 scrive testualmente:
 *      gammashift = 31 - ((gammraw==0) ? 31 : -__builtin_clz(gammraw)) - 15
 *  La guardia e' stata introdotta per evitare clz(0). Ma il ramo NON-zero
 *  contiene un segno negativo davanti a clz, e clz() conta gli ZERI iniziali:
 *  la guardia ha quindi INVERTITO l'intera scala, non l'ha protetta.
 *  Per gammraw>0:  scritto = 16 + clz(gammraw)
 *                  corretto = 16 - clz(gammraw)
 *  errore = 2*clz(gammraw), che vale 0 solo per gammraw >= 2^31.          */
typedef enum { GAMMA_AS_WRITTEN = 0, GAMMA_CORRECTED = 1 } gamma_mode_t;

/*  Analogamente per il raffreddamento della sezione 8. La prosa dice che
 *  Cooling_Shift e' il divisore dello shift (>>9, >>10, >>11) ma la formula
 *  scritta lo SOMMA al learning rate, il che lo aumenta.                  */
typedef enum { COOL_AS_WRITTEN = 0, COOL_AS_DENOM = 1 } cool_mode_t;

/*  La sezione 8 della revisione 6 aveva il termine di inerzia (LRold>>3).
 *  La revisione 7 lo dichiara rimosso ("rimuove l'inerzia autoreferenziale")
 *  ma lo lascia in una riga residua. Lo si puo' riattivare.              */
typedef enum { INERTIA_OFF = 0, INERTIA_ON = 1 } inertia_mode_t;

typedef struct {
    gamma_mode_t  gamma;
    cool_mode_t   cool;
    inertia_mode_t inertia;
    int           siluns_r7;   /* 1 = r7 a due rami, 0 = r6 a tre gradini   */
    int           ste_zero;    /* 1 = STE sui trit nulli (sez. 2)          */
} qatnn_cfg_t;

extern const qatnn_cfg_t QATNN_R7;      /* letterale del paper, r7           */
extern const qatnn_cfg_t QATNN_FIXED;   /* r7 con i tre difetti corretti     */

/* ===================================================================== */
/*  Tipi                                                                   */
/* ===================================================================== */
typedef int32_t shadow_t;      /* Q16.15  */
typedef int8_t  trit_t;        /* {-1,0,+1}                                 */
typedef int16_t grad_t;        /* Q4.11                                    */
typedef uint16_t lr_t;         /* UQ0.16                                   */
typedef uint16_t decay_w_t;    /* UQ4.12                                   */
typedef uint32_t acc_t;        /* UQ7.25                                   */

/* Un layer: matrice [N_in x N_out] memorizzata per colonne, cosi' che la
 * riduzione su un output e' un passo lineare di memoria.                   */
typedef struct {
    int32_t  n_in;
    int32_t  n_out;
    trit_t  *trit;        /* n_in * n_out valori in {-1,0,+1}, riga = input  */
    uint8_t *packed;      /* ceil(n_in*n_out/5) byte, 5 trit per byte      */
    shadow_t *shadow;     /* n_in * n_out pesi ombra Q16.15 (file S_X.bin) */
    uint8_t  gauss;       /* 1 se gli shadow sono stati inizializzati      */
} qat_layer_t;

typedef struct {
    qatnn_cfg_t cfg;
    int32_t  n_layers;
    qat_layer_t *L;
    /* stato per-layer derivato dal training */
    decay_w_t *decay_w;   /* accumulatore UQ4.12, ottimizzato               */
    int32_t   *decay;     /* decay = clip(decay_w >> 12, 0, 15)             */
    int8_t    *gshift;    /* gamma-shift calcolato a ogni epoca             */
    int32_t   *graw;      /* gamma grezzo, per diagnostica                  */
    /* accumulatori di sezione 8 */
    acc_t     *acc_err;   /* uno per layer                                 */
    uint64_t  *acc_grad;  /* uno per layer, int64 per la finestra a 8       */
    lr_t      *lr;        /* uno per layer                                 */
    uint32_t   cool;      /* Cooling_Shift globale                         */
    /* rete */
    int32_t   n_seed;
} qat_net_t;

/* ===================================================================== */
/*  API                                                                    */
/* ===================================================================== */

/* --- sezione 1b: SiLUNS ------------------------------------------------ */
int32_t siluns_r7(int32_t x);
int32_t siluns_r6(int32_t x);          /* la versione a tre gradini       */
double  silu_exact(double x);           /* solo per il confronto nei test   */
double  q_to_d(int32_t q);

/* --- sezione 1: gamma ------------------------------------------------- */
int32_t gamma_shift_r7(uint64_t gammraw);
int32_t gamma_shift_correct(uint64_t gammraw);
int32_t gamma_of_layer(const qat_layer_t *L);
int32_t apply_gamma(int32_t raw, int32_t gshift);

/* --- sezione 2: mappa di isteresi -------------------------------------- */
trit_t  hysteresis_map(shadow_t s, trit_t prev);

/* --- sezione 3: packing ------------------------------------------------ */
void     trit_pack(const trit_t *src, size_t n, uint8_t *dst);
size_t   qat_packed_bytes(size_t n);
void     trit_unpack(const uint8_t *src, size_t n, trit_t *dst);
void     trit_lut_init(void);
extern trit_t TRIT_LUT[256][5];

/* --- forward pass ------------------------------------------------------ */
void qat_forward(qat_net_t *net, int32_t layer, const int32_t *in,
                 int32_t n_in, int32_t *out);

/* --- backpropagation, sezione 5 --------------------------------------- */
void qat_backprop_error(qat_net_t *net, int32_t layer,
                        const int32_t *err_next, int32_t *err_cur);
void qat_gradient(const int32_t *err_next, const int32_t *input,
                  int32_t n_out, int32_t n_in, grad_t *g);

/* --- sezione 8: ciclo adattivo ---------------------------------------- */
lr_t  qat_learning_rate(const qat_net_t *net, int64_t gavg, acc_t acc_err,
                        lr_t lr_old);
int32_t layer_cut(uint32_t totlayer, uint32_t pct);

/* --- addestramento, sezioni 6/7/8 -------------------------------------- */
typedef struct {
    double   acc; int32_t correct; double loss; double lr;
    int32_t  gshift0, graw0, decay0; uint32_t lr_min, lr_max;
} qat_train_report_t;

void qat_refresh(qat_net_t *net);
void qat_apply_hysteresis(qat_net_t *net, int32_t layer);
int  qat_train(qat_net_t *net, const int32_t *x, const int32_t *y,
               const int32_t *t, int32_t n, int32_t cycles, int verb,
               qat_train_report_t *rep);
int  qat_write_chunks(const qat_net_t *net, const char *dir);
size_t qat_chunk_bytes(const qat_layer_t *L);
size_t qat_opt_bytes(const qat_layer_t *L);

/* --- costruzione / distruzione ---------------------------------------- */
qat_net_t *qat_create(int32_t n_layers, int32_t *shapes, int32_t n_shapes,
                      uint32_t seed);
void       qat_free(qat_net_t *net);
void       qat_gauss_init(qat_net_t *net);
void       qat_trit_gauss_init(qat_net_t *net);
void       qat_repack(qat_net_t *net);
double     qat_zero_fraction(const qat_net_t *net);
int        qat_fit_bitcos(const qat_net_t *net);   /* >= 51% ? */

/* --- benchmark -------------------------------------------------------- */
typedef struct {
    double  sec_ternary, sec_int8_mac, sec_int8_shift, sec_int8_addsub;
    double  go_ternary, go_int8_mac, go_int8_shift;
    long long ops;
    size_t  bytes_packed, bytes_int8;
} qat_bench_t;

void qat_bench(int32_t M, int32_t N, int32_t K, int rep, qat_bench_t *out);

/* --- v003 spatial-multimodal helpers (integer-only, inference-safe) -----
 * qat_spatial_bias_add: feat[i] += semb[i % sdim] (saturating, Q16.15).
 * qat_patch_token: 768-u8 patch -> stable 31-bit visual token id.
 * Both keep the paper forward contract: no float, no allocation. */
void qat_spatial_bias_add(int32_t *feat, const int32_t *semb,
                          int32_t d, int32_t sdim);
uint32_t qat_patch_token(const unsigned char *patch768);

#endif /* QATNN_H */
