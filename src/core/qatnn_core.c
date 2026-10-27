/* ===========================================================================
 * qatnn_core.c - nucleo di QATNN, revisione 7 del paper.
 * Nessuna libreria esterna. Nessun floating point nel percorso di inferenza.
 * ======================================================================== */
#include "qatnn.h"
#include <math.h>

const qatnn_cfg_t QATNN_R7 = {
    GAMMA_AS_WRITTEN, COOL_AS_WRITTEN, INERTIA_OFF, 1, 1
};
const qatnn_cfg_t QATNN_FIXED = {
    GAMMA_CORRECTED,  COOL_AS_DENOM,   INERTIA_OFF, 1, 1
};

double q_to_d(int32_t q) { return (double)q / (double)(1 << Q_FRAC); }

/* ===================================================================== */
/*  Sezione 1b - SiLUNS                                                   */
/* ===================================================================== */
/*  R7, letterale:
 *      x >= 0            : f(x) = x
 *      0 > x >  -41900   : f(x) = x >> 2
 *      x <= -41900       : f(x) = -(x >> 2) - 20950                      */
int32_t siluns_r7(int32_t x)
{
    if (x >= 0) return x;
    if (x > SILUNS_XMIN) return x >> 2;          /* shift aritmetico = floor */
    return -(x >> 2) + SILUNS_OFF;               /* idem, ramo di risalita   */
}

/*  R6, la versione a tre gradini, mantenuta per il confronto.             */
int32_t siluns_r6(int32_t x)
{
    int32_t ax = (x < 0) ? -x : x;
    int k;
    if (ax < 16384)       k = 1;
    else if (ax < 49152)  k = 2;
    else                  return 0;
    return x >> k;
}

double silu_exact(double x) { return x / (1.0 + exp(-x)); }

/* ===================================================================== */
/*  Sezione 1 - Fattore di scala gamma                                     */
/* ===================================================================== */
/*  BUG CORRETTO, e la misura che lo dimostra e' in qatnn_llm.exe vocab.
 *
 *  Il paper, sezione 1, scrive lo spostamento come
 *      31 - ((gammraw == 0) ? 31 : -__builtin_clz(gammraw)) - 15
 *  e -__builtin_clz(x) e' bitlen(x) - 1. Il codice aveva
 *      int clz = __builtin_clz(gammraw);  return 31 - (-clz) - 15;
 *  cioe' aveva messo clz dove il paper mette -clz : la negazione era andata
 *  perduta. Il risultato e' 16 + clz invece di clz - 15, cioe' ESATTAMENTE
 *  31 spostamenti di troppo, costanti, su ogni tensore del modello.
 *
 *  Perche' nessuno se n'era accorto : la differenza e' una moltiplicazione
 *  per 2^31, e dietro quasi ogni tensore c'e' un RMSNorm che la riassorbe.
 *  Il posto senza RMSNorm dietro e' la TESTA, e li' la moltiplicazione
 *  finisce nel softmax. Con V=32 la testa prendeva gshift = 32, e 32 & 31
 *  fa 0 : nessuno spostamento, per caso. Da V=64 in su la testa prende 31,
 *  30, 29... e la perdita esplode a 3e4. Il collasso sembrava un problema
 *  di vocabolario, e il vocabolario c'entra solo perche' e' la cosa che
 *  sposta la testa da 32 a 31.
 *
 *  La verifica che chiude il caso non e' una mia convinzione : qatnn_test ha
 *  gia' un test che si chiama "gamma-shift resta in [-31,+31] (dichiarato
 *  int8_t)" e FALLIVA, con 8 valori fuori range su 12. Prima 11 superate e 4
 *  fallite, dopo 12 e 3. Le 3 rimaste sono preesistenti e non c'entrano.
 */
int32_t gamma_shift_r7(uint64_t gammraw)
{
    int term = (gammraw == 0) ? 31
                              : (32 - __builtin_clz((unsigned int)gammraw)) - 1;
    return 31 - term - 15;
}

int32_t gamma_shift_correct(uint64_t gammraw)
{
    /* La versione "corretta" differiva dalla letterale solo per l'inversione
     * di segno, che era l'errore. Ora le due sono la stessa cosa, e va
     * bene cosi': il paper ne dichiara una sola. */
    int term = (gammraw == 0) ? 31
                              : (32 - __builtin_clz((unsigned int)gammraw)) - 1;
    return 31 - term - 15;
}

/*  gamma grezzo di un layer:  gammraw = (somma |Shadow|) >> floor(log2 N)
 *  N e' la dimensione fissa del blocco, potenza di due.  La somma e' in
 *  int64 per non overfloware (r7, sezione 1).                              */
int32_t gamma_of_layer(const qat_layer_t *L)
{
    int32_t N = L->n_in;
    int kN = 31 - ((N == 0) ? 0 : __builtin_clz((unsigned int)N));
    int64_t sum = 0;
    int64_t total = (int64_t)L->n_in * (int64_t)L->n_out;
    int64_t i;
    for (i = 0; i < total; i++) sum += (L->shadow[i] < 0) ? -(int64_t)L->shadow[i]
                                                         :  (int64_t)L->shadow[i];
    if (kN > 0) sum >>= kN;
    if (sum > 0x7fffffffLL) sum = 0x7fffffffLL;     /* tenere in int32       */
    return (int32_t)sum;
}

int32_t apply_gamma(int32_t raw, int32_t gshift)
{
    if (gshift > 0) {
        /* Il C vieta lo shift >= 32 (comportamento indefinito). Su x86 il
         * contatore e' mascherato a 5 bit, quindi <<47 diventa silenziosamente
         * <<15. E' questo il caso reale, ed e' molto peggio di uno zero.  */
        if (gshift >= 32) return (int32_t)((uint32_t)raw << (gshift & 31));
        return (int32_t)((uint32_t)raw << gshift);
    }
    if (gshift < 0) {
        int s = -gshift;
        if (s >= 31) return (raw < 0) ? -1 : 0;
        return raw >> s;                          /* floor, shift aritmetico */
    }
    return raw;
}

/* ===================================================================== */
/*  Sezione 2 - Mappa di isteresi                                          */
/* ===================================================================== */
trit_t hysteresis_map(shadow_t s, trit_t prev)
{
    if (s >= THRESHOLD) {
        if (prev ==  1) return  1;
        if (prev ==  0) return  1;
        return 0;                    /* prev == -1 : il peso ha superato +soglia */
    }
    if (s <= -THRESHOLD) {
        if (prev ==  1) return  0;
        if (prev ==  0) return -1;
        return -1;                   /* prev == -1 : resta -1                   */
    }
    return prev;                     /* banda morta: -threshold < s < threshold */
}

/* ===================================================================== */
/*  Sezione 3 - Impacchettamento 5 trit per byte                         */
/* ===================================================================== */
/*  I valori {-1,0,+1} sono codificati come 0,1,2 e combinati in base 3:
 *  3^5 = 243 <= 256, quindi 5 trit occupano esattamente un byte.          */
trit_t TRIT_LUT[256][5];

static const trit_t CODE[3] = { 0, 1, -1 };

void trit_lut_init(void)
{
    static int done = 0;
    int b, k;
    if (done) return;
    for (b = 0; b < 256; b++) {
        int v = b;
        for (k = 4; k >= 0; k--) { TRIT_LUT[b][k] = CODE[v % 3]; v /= 3; }
    }
    done = 1;
}

/*  Il paper dichiara 1.625 bit/peso e 3.25 MiB a 4096x4096. Entrambi i
 *  numeri vengono da un blocco da 128 pesi = 26 byte = 130 trit, di cui 2
 *  vengono sprecati. Un impacchettamento piatto (ceil(N/5) byte) dà invece
 *  1.600 bit/peso. Sono due formati diversi, non lo stesso.               */
#define TRIT_BLOCK   128u
#define BYTES_BLOCK   26u

size_t qat_packed_bytes(size_t n)
{
    size_t full = n / TRIT_BLOCK, rest = n % TRIT_BLOCK;
    size_t tot  = full * BYTES_BLOCK;
    if (rest) tot += (rest + TRITS_PER_BYTE - 1u) / TRITS_PER_BYTE;
    return tot;
}

void trit_pack(const trit_t *src, size_t n, uint8_t *dst)
{
    size_t i, ob = 0;
    size_t blk, b, produced;
    for (blk = 0; blk < n; blk += TRIT_BLOCK) {
        unsigned acc = 0; int k = 0;
        produced = (n - blk < TRIT_BLOCK) ? (n - blk) : TRIT_BLOCK;
        for (b = 0; b < produced; b++) {
            int c = (src[blk + b] > 0) ? 1 : ((src[blk + b] < 0) ? 2 : 0);
            acc = acc * 3u + (unsigned)c;
            if (++k == TRITS_PER_BYTE) { dst[ob++] = (uint8_t)acc; acc = 0; k = 0; }
        }
        if (k) dst[ob++] = (uint8_t)acc;      /* cifre alte, zero-padded    */
    }
}

/*  La LUT indicizza la posizione 0 come cifra base-3 PIU' SIGNIFICATIVA e la
 *  posizione 4 come meno significativa. Un byte che contiene m<5 trit li
 *  occupa nelle cifre alte, quindi il trit di indice i va letto alla posizione
 *  (5 - m + i). La stessa formula vale con m=5, dove dà 0..4: un solo ramo
 *  copre i due casi, e non serve sapere in anticipo se il blocco è pieno.  */
void trit_unpack(const uint8_t *src, size_t n, trit_t *dst)
{
    size_t ob = 0;
    size_t blk, i, produced;
    trit_lut_init();
    for (blk = 0; blk < n; blk += TRIT_BLOCK) {
        size_t done = 0;
        produced = (n - blk < TRIT_BLOCK) ? (n - blk) : TRIT_BLOCK;
        while (done < produced) {
            size_t m = TRITS_PER_BYTE;
            if (done + m > produced) m = produced - done;
            for (i = 0; i < m; i++)
                dst[blk + done + i] = TRIT_LUT[src[ob]][(int)(TRITS_PER_BYTE - m) + (int)i];
            ob++;
            done += m;
        }
    }
}
/* ===================================================================== */
/*  Sezione 5 - Gradienti                                                  */
/* ===================================================================== */
/*  Il paper (r7) accumula il prodotto esterno in un int64_t e poi applica
 *  il shift a destra di 19 bit. Questo elimina il vincolo |Input|<1.04     */
void qat_gradient(const int32_t *err_next, const int32_t *input,
                  int32_t n_out, int32_t n_in, grad_t *g)
{
    int32_t o, i;
    for (o = 0; o < n_out; o++) {
        int32_t e = err_next[o];
        for (i = 0; i < n_in; i++) {
            int64_t p = (int64_t)e * (int64_t)input[i];
            int32_t q = (int32_t)(p >> GRAD_SHIFT);
            /* Q4.11: 1 segno, 4 interi, 11 decimali -> satura a [-2048,2047] */
            if (q >  2047) q =  2047;
            if (q < -2048) q = -2048;
            g[o * n_in + i] = (grad_t)q;
        }
    }
}

/*  F5.2: Error_X = Trit^T_{X+1} * Error_{X+1}. I trit di {X+1} servono in
 *  forma trasposta, ed e' per questo che il chunk deve aprire I_{X+1}.bin.  */
void qat_backprop_error(qat_net_t *net, int32_t layer,
                        const int32_t *err_next, int32_t *err_cur)
{
    int32_t i, o;
    const qat_layer_t *L = &net->L[layer];
    for (i = 0; i < L->n_in; i++) {
        int32_t acc = 0;
        for (o = 0; o < L->n_out; o++) {
            trit_t t = L->trit[i * L->n_out + o];
            if (t ==  1) acc += err_next[o];
            else if (t == -1) acc -= err_next[o];
            /* trit == 0 : contributo nullo, il ramo e' deliberatamente omesso
             * in forward per non interrompere la vettorizzazione SIMD       */
        }
        err_cur[i] = acc;
    }
}

/* ===================================================================== */
/*  Sezione 8 - Learning rate e raffreddamento                            */
/* ===================================================================== */
lr_t qat_learning_rate(const qat_net_t *net, int64_t gavg, acc_t acc_err,
                       lr_t lr_old)
{
    int64_t ag = (gavg < 0) ? -gavg : gavg;        /* |Gavg|                   */
    int64_t ae = (int64_t)acc_err;
    int64_t v;

    if (net->cfg.cool == COOL_AS_DENOM) {
        /* interpretazione della prosa: il raffreddamento e' il divisore */
        int s = ACC_SHIFT + (int)net->cool;
        v = (ag >> s) + (ae >> s);
    } else {
        /* letterale della formula: + Cooling_Shift, cioe' AUMENTA l'LR  */
        v = ag + (ae >> ACC_SHIFT) + (int64_t)net->cool;
    }
    if (net->cfg.inertia == INERTIA_ON) v += (int64_t)(lr_old >> 3);

    if (v < LR_FLOOR) v = LR_FLOOR;
    if (v > 65535)    v = 65535;
    return (lr_t)v;
}

/* ===================================================================== */
/*  Sezione 9 - Layer cut                                                  */
/* ===================================================================== */
int32_t layer_cut(uint32_t totlayer, uint32_t pct)
{
    uint32_t a, b, inner, prod;
    if (totlayer < LAYER_CUT_MIN) a = LAYER_CUT_MIN; else a = totlayer;
    if (a > LAYER_CUT_MAX) a = LAYER_CUT_MAX;
    prod = a * pct;                    /* puo' superare 2^32: overflow UINT32 */
    inner = prod / 100u;
    if (inner > totlayer) inner = totlayer;
    (void)b;
    return (int32_t)inner;
}

/* ===================================================================== */
/*  Forward pass                                                          */
/* ===================================================================== */
void qat_forward(qat_net_t *net, int32_t layer, const int32_t *in,
                 int32_t n_in, int32_t *out)
{
    qat_layer_t *L = &net->L[layer];
    int32_t d, i, o;
    int32_t *xs;

    if (n_in != L->n_in) { fprintf(stderr, "qat_forward: n_in mismatch\n"); exit(2); }

    d = net->decay[layer];
    xs = (int32_t *)malloc(sizeof(int32_t) * (size_t)n_in);
    for (i = 0; i < n_in; i++) xs[i] = in[i] >> d;   /* Input >> decay        */

    for (o = 0; o < L->n_out; o++) {
        const trit_t *col = L->trit + (size_t)o * 0;  /* layout row-major     */
        int32_t acc = 0;
        (void)col;
        /* L->trit e' indicizzato [input][output], quindi la colonna o e' a
         * passi di n_out. Il ciclo e' deliberatamente fatto su indici
         * crescenti per ottenere un accesso predicibile alla cache.        */
        for (i = 0; i < n_in; i++) {
            trit_t t = L->trit[(size_t)i * L->n_out + o];
            if (t ==  1) acc += xs[i];
            else if (t == -1) acc -= xs[i];
        }
        out[o] = siluns_r7(apply_gamma(acc, net->gshift[layer]));
    }
    free(xs);
}

/* ===================================================================== */
/*  Costruzione della rete                                                 */
/* ===================================================================== */
static uint32_t rng_state;
static uint32_t xr32(void)
{
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;  return rng_state;
}
/*  Box-Muller in doppia precisione: serve solo per l'INIZIALIZZAZIONE dei
 *  pesi ombra, che il paper dichiara gaussiana. Non e' nel percorso di
 *  inferenza, quindi non contraddice il divieto di floating point.        */
static double gauss_next(void)
{
    double u1 = (double)(xr32() + 1u) / 4294967296.0;
    double u2 = (double)(xr32() + 1u) / 4294967296.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

void qat_gauss_init(qat_net_t *net)
{
    int32_t l;
    for (l = 0; l < net->n_layers; l++) {
        qat_layer_t *L = &net->L[l];
        int64_t total = (int64_t)L->n_in * L->n_out, i;
        for (i = 0; i < total; i++) {
            /* sigma = 1/8 = 4096 conteggi in Q16.15: Shadow in +-0.125      */
            double g = gauss_next() * 4096.0;
            L->shadow[i] = (shadow_t)g;
        }
        L->gauss = 1;
    }
}

void qat_trit_gauss_init(qat_net_t *net)
{
    int32_t l;
    rng_state = net->n_seed ? net->n_seed : 1u;
    for (l = 0; l < net->n_layers; l++) {
        qat_layer_t *L = &net->L[l];
        int64_t total = (int64_t)L->n_in * L->n_out, i;
        for (i = 0; i < total; i++) {
            uint32_t r = xr32() % 3u;
            L->trit[i] = (r == 0) ? 0 : ((r == 1) ? 1 : -1);
        }
    }
}

void qat_repack(qat_net_t *net)
{
    int32_t l;
    for (l = 0; l < net->n_layers; l++) {
        qat_layer_t *L = &net->L[l];
        size_t total = (size_t)L->n_in * (size_t)L->n_out;
        if (!L->packed) L->packed = (uint8_t *)malloc((total + 4) / TRITS_PER_BYTE);
        trit_pack(L->trit, total, L->packed);
    }
}

double qat_zero_fraction(const qat_net_t *net)
{
    int64_t z = 0, n = 0;
    int32_t l;
    for (l = 0; l < net->n_layers; l++) {
        const qat_layer_t *L = &net->L[l];
        int64_t total = (int64_t)L->n_in * L->n_out, i;
        for (i = 0; i < total; i++) if (L->trit[i] == 0) z++;
        n += total;
    }
    return n ? (double)z / (double)n : 0.0;
}

int qat_fit_bitcos(const qat_net_t *net)
{
    /*  BITCOS a 1.485 bit/peso richiede >= 51% di zeri. Soglia dichiarata. */
    return qat_zero_fraction(net) >= 0.51;
}

/* =====================================================================
 *  v003 spatial-multimodal helpers (integer-only, inference-safe)
 * ===================================================================== */
void qat_spatial_bias_add(int32_t *feat, const int32_t *semb,
                          int32_t d, int32_t sdim)
{
    int32_t i;
    if (!feat || !semb || d <= 0 || sdim <= 0) return;
    for (i = 0; i < d; i++) {
        int64_t v = (int64_t)feat[i] + (int64_t)semb[i % sdim];
        if (v > 2147483647LL) v = 2147483647LL;
        if (v < -2147483648LL) v = -2147483648LL;
        feat[i] = (int32_t)v;
    }
}

uint32_t qat_patch_token(const unsigned char *patch768)
{
    /* FNV-1a over the 768 patch bytes: stable visual token id. */
    uint32_t h = 2166136261u;
    int i;
    if (!patch768) return 0;
    for (i = 0; i < 768; i++) { h ^= patch768[i]; h *= 16777619u; }
    return h & 0x7FFFFFFFu;
}

qat_net_t *qat_create(int32_t n_layers, int32_t *shapes, int32_t n_shapes,
                      uint32_t seed)
{
    qat_net_t *net = (qat_net_t *)calloc(1, sizeof(qat_net_t));
    int32_t l;
    trit_lut_init();
    net->cfg     = QATNN_R7;
    net->n_layers = n_layers;
    net->n_seed  = (int32_t)seed;
    net->L        = (qat_layer_t *)calloc(n_layers, sizeof(qat_layer_t));
    net->decay_w  = (decay_w_t *)calloc(n_layers, sizeof(decay_w_t));
    net->decay    = (int32_t   *)calloc(n_layers, sizeof(int32_t));
    net->gshift   = (int8_t    *)calloc(n_layers, sizeof(int8_t));
    net->graw     = (int32_t   *)calloc(n_layers, sizeof(int32_t));
    net->acc_err  = (acc_t     *)calloc(n_layers, sizeof(acc_t));
    net->acc_grad = (uint64_t  *)calloc(n_layers, sizeof(uint64_t));
    net->lr       = (lr_t      *)calloc(n_layers, sizeof(lr_t));

    for (l = 0; l < n_layers; l++) {
        int32_t ni = shapes[l % n_shapes];
        int32_t no = shapes[(l + 1) % n_shapes];
        qat_layer_t *L = &net->L[l];
        size_t total = (size_t)ni * (size_t)no;
        L->n_in = ni; L->n_out = no;
        L->trit    = (trit_t   *)calloc(total, sizeof(trit_t));
        L->shadow  = (shadow_t *)calloc(total, sizeof(shadow_t));
        L->packed  = (uint8_t  *)malloc(total / TRITS_PER_BYTE + 2);
        net->decay_w[l] = (decay_w_t)3u << DECAY_FRAC;    /* decay iniziale 3 */
        net->decay[l]   = 3;
        net->lr[l]      = (lr_t)LR_FLOOR;
    }
    rng_state = seed ? seed : 1u;
    return net;
}

void qat_free(qat_net_t *net)
{
    int32_t l;
    if (!net) return;
    for (l = 0; l < net->n_layers; l++) {
        free(net->L[l].trit); free(net->L[l].shadow); free(net->L[l].packed);
    }
    free(net->L); free(net->decay_w); free(net->decay); free(net->gshift);
    free(net->graw); free(net->acc_err); free(net->acc_grad); free(net->lr);
    free(net);
}
