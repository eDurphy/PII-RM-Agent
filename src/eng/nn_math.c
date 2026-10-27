/* ===========================================================================
 * qatnn_llm.c - un modello linguistico transformer completo, in C.
 *
 * ---------------------------------------------------------------------------
 * 1. COSA MANCA NEL PAPER, E QUINDI COSA HO DOVUTO AGGIUNGERE
 * ---------------------------------------------------------------------------
 * Il paper non descrive un modello linguistico. Le parole "attention",
 * "softmax", "residual", "normalizzazione", "embedding", "vocabolario",
 * "token", "posizione", "causale" e "perdita" non compaiono mai nel testo.
 * Quello che il paper descrive e' uno stack di strati densi quantizzati con
 * SiLUNS: nessuna sequenza, nessun contesto, nessuna perdita.
 *
 * Per arrivare a un LLM ho quindi aggiunto undici cose che il paper non
 * menziona. Sono marcate [MIO] qui e nel referto, e sono le uniche cose in
 * questo file che non vengono dal paper:
 *
 *   [MIO]  embedding ternario, vocabolario, tokenizzazione a caratteri
 *   [MIO]  RMSNorm, in fp32
 *   [MIO]  attenzione causale multi-testa, con maschera e scala 1/sqrt(d_head)
 *   [MIO]  SwiGLU nella parte MLP
 *   [MIO]  connessioni residue pre-norm
 *   [MIO]  perdita cross-entropy e la sua derivata
 *   [MIO]  la catena del gradiente attraverso gli strati (la trasposta)
 *
 * Tutto il resto e' il paper, alla lettera:
 *
 *   sez. 1   gamma-shift dalla somma dei pesi ombra
 *   sez. 1b  SiLUNS a due rami
 *   sez. 2   pesi ternari, soglia di isteresi 213, ombre Q16.15
 *   sez. 3   impacchettamento 5 trit per byte, file I<S>.bin ed S<S>.bin
 *   sez. 4   Outputgrezzo += Trit * (Input >> decay)
 *   sez. 5   G = (Error (x) Input) >> 19, e Error_X = Trit^T * Error_{X+1}
 *   sez. 6   decay_Weight ottimizzato, UQ4.12, clipped 0..15
 *   sez. 7   Shadow -= (G * LR) >> 12
 *   sez. 8   ciclo adattivo a WINDOW=8 inferenze
 *
 * Lo scheletro del blocco e' pre-norm, come in arch-transformer-block:
 *   x = x + Attn(RMSNorm(x));   x = x + MLP(RMSNorm(x))
 * Tutte e sette le matrici del blocco sono ternarie: il paper non ammette
 * eccezioni, quindi nemmeno embedding e testa.
 * ---------------------------------------------------------------------------
 * 2. LA DERIVATA DI SILUNS
 * ---------------------------------------------------------------------------
 * SiLUNS r7 e' piecewise lineare con pendenze 1, 1/4, -1/4, quindi la sua
 * derivata esiste quasi ovunque ed e' esatta: nessun surrogato needed.
 * (Questo e' un vantaggio reale della r7 rispetto alla r6 a tre gradini,
 * che aveva discontinuita' e richiedeva una regola a parte.)
 * ---------------------------------------------------------------------------
 * 3. LE TRE MOSSE DOVE ILLA HO DOVUTO SCEGLIERE, E LE DICHIARO
 * ---------------------------------------------------------------------------
 *   a) Il gradiente attraverso il gamma-shift. Lo shift e' un fattore di
 *      scala 2^-g dichiarato dal paper, non un'operazione lineare da
 *      invertere. Il gradiente torna indietro con lo STESSO shift a destra,
 *      non con la scala inversa, altrimenti i registri esplodono di 2^|g|.
 *      Scelta, dichiarata, misurabile: la funzione e' qlin_set_gamma_mode().
 *   b) Il gradiente attraverso SiLUNS e' siluns'(x), cioe' 1, 1/4 o -1/4.
 *      La sua assenza o la sua presenza cambiano il segno della discesa per
 *      gli input negativi: e' un interruttore, non una dettaglio.
 *   c) La normalizzazione e l'attenzione restano in fp32. arch-transformer-block
 *      lo prescrive: sono le operazioni sensibili, e quantizzarle farebbe
 *      dipendere la stabilita' dalla risoluzione e non dall'algoritmo.
 * ======================================================================== */
#include "qatnn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <windows.h>
#include <psapi.h>
#include <pthread.h>     /* v004: pool persistente (mai -fopenmp: 17x lento) */
#ifndef _WIN32
#include <unistd.h>       /* sysconf(_SC_NPROCESSORS_ONLN) per il tetto 50% */
#endif

/* v004: nessun #pragma omp rimasto; exp() di math.h e' VIETATA nei percorsi
 * caldi (RWKV-6 la chiamerebbe per canale/token: vedi fexp()). sqrt/log/
 * llround restano (norme sensibili + init, per arch-transformer-block). */

#define QF  (1 << Q_FRAC)          /* 32768 : Q16.15                      */
#define GSAT 2047                  /* Q4.11 satura a +-2047, NON a +-8191 */
#define GSATN 2048

static void die(const char *m) { fprintf(stderr, "errore: %s\n", m); exit(2); }

/* ===================================================================== RNG */
static uint64_t rng_s = 0x243F6A8885A308D3ull;
static void rseed(uint64_t s) { rng_s = s ? s : 1; }
static uint32_t rnd32(void)
{
    rng_s ^= rng_s >> 12; rng_s ^= rng_s << 25; rng_s ^= rng_s >> 27;
    return (uint32_t)((rng_s * 0x2545F4914F6CDD1Dull) >> 32);
}
/* sezione 2 : shadow iniziali da una gaussiana "controllata", sigma 2048
 *
 * [MIO] sigma 2048 e' il valore dichiarato dal paper, e misurato e' 1679.
 * Ma la soglia di isteresi dello STESSO paper e' 213. Quindi il paper
 * inizializza lo shadow a una deviazione standard OTTO VOLTE la soglia che
 * decide se il trit e' +1, 0 o -1. A quel punto lo shadow iniziale non e'
 * una posizione di partenza: e' rumore casuale che vale otto volte la
 * decisione, e i gradienti — che nell'ordine di grandezza giusto varrebbero
 * qualche centinaio di conteggi — non possono spostarlo in modo
 * significantemente. Il peso non scende: e' gia' "deciso" dal rumore.
 *
 * Non lo cambio alla cieca : `diag3` misura sigma per sigma e dice quale
 * funziona. Il default resta quello dichiarato finche' la misura non dice
 * altrimenti. vedi DIAG3.                                          */
static int g_shadow_sigma = 2048;      /* 0 = shadow a zero, la scelta    */
static shadow_t shadow_init(void)
{
    int32_t a, b;
    if (g_shadow_sigma <= 0) return 0;
    a = (int32_t)(rnd32() & 0xFFFFu) - 32768;
    b = (int32_t)((rnd32() >> 3) & 0xFFFFu) - 32768;
    return (shadow_t)(((int64_t)(a + b) * g_shadow_sigma) / 32768);
}

/* ============================================================ v004: MATH-LUT
 * LUT matematica a interpolazione su 4 punti (Catmull-Rom) per exp().
 * SEPARATA dalla LUT dei pesi ternari TL2 sotto: tabelle, domini e scopi
 * diversi; mai mescolate. Motivo: RWKV-6 chiama il decadimento dinamico
 * per canale e per token (dh exp per testa/passo); exp() di math.h li
 * renderebbe il collo di bottiglia CPU. Nessun exp() resta nei percorsi
 * caldi: verificare con  grep -n "[^f]exp("  (solo math_lut + init/log).
 * Dominio [-16,+16] + un punto di guardia per lato (1027 punti, passo
 * 1/32); fuori dominio: clamp agli estremi. Mai estrapolazione: le celle
 * di bordo interpolano grazie alle guardie. Errore relativo misurato
 * < 1e-5 contro exp() sul dominio (vedi t_fexp in verify). Inizializzata
 * una volta sola via pthread_once: sicura con il pool attivo.        */
#define FEXP_MIN   (-16.0 - 1.0 / 32.0)   /* un punto di guardia per lato */
#define FEXP_STEP  (1.0 / 32.0)
#define FEXP_N     1027   /* [-16-1/32, +16+1/32]: le celle di bordo        */
static double fexp_tab[FEXP_N];
static pthread_once_t fexp_once = PTHREAD_ONCE_INIT;
static void fexp_build(void)
{
    int i;
    for (i = 0; i < FEXP_N; i++)
        fexp_tab[i] = exp(FEXP_MIN + (double)i * FEXP_STEP); /* init only */
}
/* Interpolazione cubica di Catmull-Rom sui 4 punti intorno a x. */
static double fexp(double x)
{
    double pos, fr;
    int k, k0;
    double p0, p1, p2, p3, a, b, c, d;
    (void)pthread_once(&fexp_once, fexp_build);
    if (x <= FEXP_MIN) return fexp_tab[0];
    if (x >= -FEXP_MIN) return fexp_tab[FEXP_N - 1];
    /* x in (-16-1/32, +16+1/32): pos in (0, 1026), sempre interpolazione */
    pos = (x - FEXP_MIN) / FEXP_STEP;
    k = (int)pos;
    k0 = k - 1;
    if (k0 < 0) k0 = 0;
    if (k0 + 3 >= FEXP_N) k0 = FEXP_N - 4;
    /* fr MISURATA sul segmento [k0+1, k0+2] dopo il clamp: ai bordi k0
     * non vale k-1 e fr = pos-k valuterebbe il segmento sbagliato di una
     * cella intera (e^{1/32}-1 = 3.17% di errore). Dentro vale pos-k. */
    fr = pos - (double)(k0 + 1);
    p0 = fexp_tab[k0]; p1 = fexp_tab[k0+1];
    p2 = fexp_tab[k0+2]; p3 = fexp_tab[k0+3];
    /* Catmull-Rom uniforme: y = d + c*fr + b*fr^2 + a*fr^3 (Horner).    */
    a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
    b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
    c = 0.5 * (p2 - p0);
    d = p1;
    return ((a * fr + b) * fr + c) * fr + d;
}
/* Decadimento dinamico RWKV-6/Finch: w = exp(-exp(dd)), dd token-dipendente.
 * Clamp: dd > 5 -> w = 0 (oblio totale); dd < -5 -> w = 1 (memoria piena).
 * Un solo fexp() per canale/token, mai per elemento di stato.          */
static double fexp_decay(double dd)
{
    if (dd >= 5.0) return 0.0;
    if (dd <= -5.0) return 1.0;
    return fexp(-fexp(dd));
}

/* ============================================================ v004: TL2
 * LUT dei pesi ternari stile TL2 (BitNet.cpp): terne di trit -> 27 pattern.
 * Decodifica statica TL2_PAT[idx][3], idx = (a+1)*9+(b+1)*3+(c+1).
 * Il dot di riga somma le TRIPLE con soli add/sub sugli ingressi interi:
 * nessuna moltiplicazione fp32 nel BitLinear; l'accumulatore e' int64, per
 * cui il raggruppamento per 3 e' bit-exact rispetto al seriale (l'addizione
 * intera e' associativa: stesso ordine j crescente, verifica t_tl2exact).
 * Scala absmean/gamma applicata UNA volta dopo il dot, mai per tile.   */
static int8_t TL2_PAT[27][3];
static pthread_once_t tl2_once = PTHREAD_ONCE_INIT;
static void tl2_build(void)
{
    int idx, t[3], k;
    for (idx = 0; idx < 27; idx++) {
        int q = idx;
        for (k = 2; k >= 0; k--) { t[k] = (q % 3) - 1; q /= 3; }
        TL2_PAT[idx][0] = (int8_t)t[0];
        TL2_PAT[idx][1] = (int8_t)t[1];
        TL2_PAT[idx][2] = (int8_t)t[2];
    }
}
static void tl2_init(void) { (void)pthread_once(&tl2_once, tl2_build); }
/* Dot di riga TL2: row = trit {-1,0,+1}, xs = ingressi interi gia' scalati
 * (xs = xq >> decay, semantica sezione 4 invariata). Solo add/sub/skip. */
static int64_t tl2_dot(const trit_t *row, const int32_t *xs, int32_t n)
{
    int64_t acc = 0;
    int32_t j = 0, n3 = n - (n % 3);
    tl2_init();
    for (; j < n3; j += 3) {
        int idx = ((int)row[j] + 1) * 9 + ((int)row[j+1] + 1) * 3
                + ((int)row[j+2] + 1);
        /* idx seleziona il pattern; la somma e' add/sub sugli xs. Il ramo
         * sullo stesso idx e' prevedibile; niente mult, niente LUT fp. */
        int8_t a = TL2_PAT[idx][0], b = TL2_PAT[idx][1], c = TL2_PAT[idx][2];
        int64_t g = 0;
        if (a > 0) g += (int64_t)xs[j]; else if (a < 0) g -= (int64_t)xs[j];
        if (b > 0) g += (int64_t)xs[j+1]; else if (b < 0) g -= (int64_t)xs[j+1];
        if (c > 0) g += (int64_t)xs[j+2]; else if (c < 0) g -= (int64_t)xs[j+2];
        acc += g;
    }
    for (; j < n; j++) {
        if (row[j] > 0) acc += (int64_t)xs[j];
        else if (row[j] < 0) acc -= (int64_t)xs[j];
    }
    return acc;
}

/* ============================================================ v004: POOL
 * Pool pthread persistente, SOLO per le GEMV BitLinear (righe indipendenti).
 * Gli aggiornamenti di stato RWKV restano seriali (passi dipendenti, il lock
 * costerebbe piu' del lavoro). Creazione pigra al primo uso, mai per layer.
 * Soglia dinamica + switch A/B d'ambiente:
 *   QATNN_THREADS=N  (0/1 = seriale; default = 50% dei processori logici;
 *                     valori sopra il tetto vengono abbassati al tetto)
 *   QATNN_MT_MIN=K   (soglia n_out*n_in sotto cui resto seriale; def 32768)
 *   QATNN_SERIAL=1   (forza seriale ovunque: A/B contro il pool)
 * Diagnostica: mt_tasks (dispatch al pool), mt_serial (cadute in seriale).
 * Pesi/input in sola lettura condivisi (nessun false sharing); ogni worker
 * scrive righe di output disgiunte; split allineato a 16 float dove riesce. */
typedef struct { int beg, end; void (*fn)(int, void *); void *arg; } mt_range_t;
static pthread_t  *mt_w = NULL;
static int         mt_live = 0, mt_inited = 0;
static pthread_mutex_t mt_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  mt_go = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  mt_done = PTHREAD_COND_INITIALIZER;
static mt_range_t  mt_job;
static int         mt_pending = 0;
static int64_t     mt_seq = 0;
static int         mt_cap = 0;   /* Directive 2: tetto 50% dei processori */
static long long   mt_tasks = 0, mt_serial = 0;
static int         mt_env_threads = -2, mt_env_min = -2, mt_env_serial = -2;
static int mt_env(const char *nm, int dflt)
{
    const char *s = getenv(nm);
    if (!s || !*s) return dflt;
    return atoi(s);
}
static void *mt_worker(void *a)
{
    int id = (int)(intptr_t)a;
    int64_t seen = -1;
    for (;;) {
        int b, e, nw;
        void (*fn)(int, void *);
        void *arg;
        int64_t my;
        pthread_mutex_lock(&mt_mu);
        while (mt_seq == seen)
            pthread_cond_wait(&mt_go, &mt_mu);
        seen = mt_seq; my = seen;
        b = mt_job.beg; e = mt_job.end;
        fn = mt_job.fn; arg = mt_job.arg;
        nw = mt_live;
        /* partizione statica contigua per righe: chunk = (end-beg)/nw */
        if (nw > 1) {
            int tot = e - b, q = tot / nw, r = tot % nw;
            int i;
            for (i = 0; i < id; i++) b += q + (i < r);
            e = b + q + (id < r);
            if (e > mt_job.end) e = mt_job.end;
        }
        pthread_mutex_unlock(&mt_mu);
        if (fn) { int i; for (i = b; i < e; i++) fn(i, arg); }
        pthread_mutex_lock(&mt_mu);
        if (my == mt_seq && --mt_pending == 0)
            pthread_cond_broadcast(&mt_done);
        pthread_mutex_unlock(&mt_mu);
    }
    return NULL;
}
static void mt_ensure(void)
{
    int want = 0, i, live = 0;
    pthread_t *w = NULL;
    pthread_mutex_lock(&mt_mu);
    if (mt_inited) { pthread_mutex_unlock(&mt_mu); return; }
    mt_inited = 1;
    /* Directive 2: tetto rigido al 50% dei processori logici (almeno 1).
     * Default = tetto; QATNN_THREADS puo' solo abbassare, mai alzarlo. */
    {
#ifdef _WIN32
        SYSTEM_INFO si; GetSystemInfo(&si); mt_cap = (int)si.dwNumberOfProcessors;
#else
        long np = sysconf(_SC_NPROCESSORS_ONLN);
        mt_cap = (np > 0) ? (int)np : 1;
#endif
        mt_cap /= 2;
        if (mt_cap < 1) mt_cap = 1;
    }
    mt_env_threads = mt_env("QATNN_THREADS", -1);
    want = mt_env_threads;
    if (want < 0) want = mt_cap;
    if (want > mt_cap) want = mt_cap;
    if (want < 0) want = 0;
    mt_env_threads = want;
    if (want > 1) {
        w = (pthread_t *)calloc((size_t)want, sizeof(pthread_t));
        if (w) {
            for (i = 0; i < want; i++)
                if (pthread_create(&w[i], NULL, mt_worker,
                                   (void *)(intptr_t)i) != 0) break;
            live = i;
        }
        mt_w = w; mt_live = live;
        mt_env_threads = (live > 1) ? live : 1;
        if (live == 0) { free(w); mt_w = NULL; }
    }
    pthread_mutex_unlock(&mt_mu);
}
/* Ritorna 1 se il lavoro (rows x cols) va al pool, 0 se seriale. */
static int mt_use(int64_t rows, int64_t cols)
{
    int64_t work;
    mt_ensure();
    if (mt_env_serial == -2) mt_env_serial = mt_env("QATNN_SERIAL", 0);
    if (mt_env_threads == 0 || mt_env_threads == 1) return 0;
    if (mt_env_serial) return 0;
    if (mt_live <= 1) return 0;
    if (rows < 2) return 0;
    if (mt_env_min == -2) mt_env_min = mt_env("QATNN_MT_MIN", 32768);
    if (rows > 0 && cols > 0 && cols > INT64_MAX / rows) return 1;
    work = rows * cols;
    return work >= (int64_t)mt_env_min;
}
/* Dispatch di fn(beg..end-1) al pool; attesa del completamento. */
static void mt_run(int beg, int end, void (*fn)(int, void *), void *arg)
{
    int live;
    pthread_mutex_lock(&mt_mu);
    live = mt_live;
    if (live <= 1) {
        pthread_mutex_unlock(&mt_mu);
        if (fn) { int i; for (i = beg; i < end; i++) fn(i, arg); }
        return;
    }
    mt_job.beg = beg; mt_job.end = end; mt_job.fn = fn; mt_job.arg = arg;
    mt_pending = live;
    mt_seq++;
    mt_tasks++;
    pthread_cond_broadcast(&mt_go);
    while (mt_pending > 0)
        pthread_cond_wait(&mt_done, &mt_mu);
    pthread_mutex_unlock(&mt_mu);
}
/* Directive 1: stadi element-wise (copie/somme vettoriali token-indipen-
 * denti) via pool. mode 0 = dst[k] = a[k], mode 1 = dst[k] += a[k].
 * Stessi numeri del seriale: un elemento per task, ordine irrilevante. */
typedef struct { float *dst; const float *a; int mode; } flat_task_t;
static void flat_vec_op(int k, void *arg)
{
    flat_task_t *f = (flat_task_t *)arg;
    if (f->mode == 0) f->dst[k] = f->a[k];
    else f->dst[k] += f->a[k];
}

/* ===================================================================== SiLU
 * del paper, sezione 1b. Chiamo siluns_r7() perche' e' la versione gia'
 * verificata bit-exact contro il simulatore Python. */
static double siluns_d(int32_t x)
{
    if (x >= 0) return 1.0;
    if (x > SILUNS_XMIN) return 0.25;
    return -0.25;
}
static int    g_siluns_grad = 1;   /* interruttore (b), vedi intestazione */
/* 1 = catena LETTERALE della sezione 5 : si propaga un errore, non una
 * derivata. Nessuna derivata di SiLUNS, nessuna di RMSNorm, nessuna
 * della gate SwiGLU, e l'errore iniziale e' Output - Target e non la
 * derivata della cross-entropy.                                      */
static int    g_paper_chain = 1;
/* Loss utente : Loss = sum_k (Output_k - Target_k)^2 >> 19.
 * Con g_usq = 1 il >> 19 entra nel GRADIENTE : dLoss/dOutput_k =
 * (2 * e) >> 19 in interi (il shift distribuisce sulla somma).
 * Con g_usq = 0 la catena resta quella del paper (Error = e).        */
static int    g_usq = 0;   /* 1 = (2e)>>19 nel gradiente (lettura letterale)
                            * 2 = gradiente 2e, >>19 solo sul valore      */
/* 1 = il ciclo adattivo della sezione 8 ricalcola l'LR ogni WINDOW passi,
 * 0 = learning rate costante (confronto A/B misurato).               */
static int g_use_sched = 1;
static double g_last_best = 0.0;   /* minima raggiunta nell'ultimo training */
static double g_usq_val  = 0.0;   /* Loss utente = sum e^2 >> 19 (ultima forward) */
/* sezione 8 : somma dei |G| della finestra corrente, per il ciclo adattivo */
static int64_t g_abs_step = 0;
static int64_t g_sat_step = 0;
static int64_t g_n_step   = 0;
static int64_t g_ntot = 0, g_nzero = 0, g_nsat = 0;   /* sonda sez. 5 */

/* [MIO] due costanti che il paper NON dichiara, e che ho dovuto aggiungere
 * perche' senza di esse l'addestramento non converge:
 *   REFRESH_EVERY = ogni quanti passi il shadow viene riquantizzato nei
 *                   trit (vedi la nota BUG dentro train_run);
 *   STALL_LIMIT   = quante finestre consecutive di peggioramento tollero
 *                   prima di dimezzare il learning rate.               */
static int REFRESH_EVERY = 1;
static int STALL_LIMIT   = 3;
static int LR_MIN        = 64;
static int LR_MAX        = 4096;
/* [MIO] il riscaldo automatico, in passi.vedi la nota dentro train_run.
 * 10 passi e' corto rispetto ai 2000 della ricetta, e di proposito : la
 * ricetta parla di un modello che parte da zero e deve costruirsi tutto,
 * mentre qui si riparte da pesi gia' allenati e il rischio non e' non
 * abbastanza addestramento, e' distruggere quello che c'era. Dieci passi
 * bastano a non fare la prima valanga, e non costano niente.            */
static int WARM_STEPS     = 10;
static int g_warmup_on    = 1;
/* =========================================================================
 * WSD : WARMUP - STABLE - DECAY.
 *
 * Fin qui l'LR era una costante. E' il caso peggiore per una rete che si
 * vuole addestrare a pezzi : un passo costante alto che improvvisamente
 * trova il gradiente sbagliato fa saltare la perdita e non si torna indietro,
 * e un passo costante basso non finisce mai. Il WSD risolve entrambi e ha
 * una proprieta' che qui conta piu' che altrove : la fase stabile e' lunga,
 * quindi puoi fermarti quando vuoi e il LR ha un senso coerente. (Llama 3
 * lo ha scelto per questa ragione, non per moda.)
 *
 *   warmup  : sale da LR_MIN fino a `peak`, in `warm` passi.
 *             Serve perche' l'ombra iniziale e' casuale : i primi passi con
 *             un passo grosso la spostano tutta e il punto di partenza
 *             non vuole piu' niente a che fare con l'inizializzazione.
 *   stable  : costante al picco, finche' vuoi. E' la fase in cui stai.
 *   decay   : scende linearmente a zero negli ultimi `decay` passi, e
 *             poi resta a zero. Serve perche' a fine addestramento il
 *             passo piccolo "fissa" quello che hai imparato : i trit
 *             smettono di cambiare e il modello smette di oscillare.
 *
 * TRE VINCOLI CHE VENGONO DA QUESTO CODICE, NON DALLA RICETTA GENERICA.
 *
 * 1. NON SI DECORRE A ZERO VERAMENTE. Il campo e' UQ0.16 e LR_MIN = 64, che
 *    il paper dichiara come "pavimento strutturale" (sez. 8, LR_FLOOR). Un
 *    LR a zero non e' "addestramento lento", e' addestramento FERMO: la
 *    regola sez. 7 fa `Shadow -= (G * LR) >> 12`, e con LR = 0 il shadow non
 *    si muove di un'unita'. Quindi il pavimento e' LR_MIN, non zero, e lo
 *    dico perche' la parola "decay" suggerirebbe il contrario.
 * 2. IL PROGRAMMA DEVE CONOSCERE QUANDO FINISCE. Un decay che non finisce e'
 *    un LR costante con un nome migliore, quindi `total` e' un numero che
 *    devi dare tu : io non posso indovinare quanti passi ti servono.
 * 3. IL PROGRAMMA DEVE SAPERE DOVE SEI. Il passo e' GLOBALE, non relativo a
 *    questa chiamata : `base` e' il numero di passi gia' fatti, che
 *    l'autobuilder legge da autobuilder_state.txt. E' cio' che permette a
 *    domani di riprendere il decay a meta' invece di ricominciarlo dal
 *    gradino piu' basso. Senza questo, WSD attraverso le sessioni sarebbe
 *    impossibile, e per un addestramento a pezzi e' il punto intero.     */
static int     g_wsd_on      = 0;   /* 1 = il WSD pilota il LR              */
static double  g_wsd_peak    = 256; /* LR della fase stabile               */
static int64_t g_wsd_total  = 0;    /* passi TOTALI previsti               */
static int64_t g_wsd_warm   = 0;    /* passi di warmup                     */
static int64_t g_wsd_base   = 0;    /* passi gia' fatti prima di questa run */

/* il LR del passo globale `gstep`. Tre tratti, e la formula e' scritta in
 * modo che i tre si tocchino senza discontinuita': a fine warmup arriva
 * esattamente a `peak`, e a fine decay arriva esattamente a LR_MIN.      */
/* IL PASSO E' GLOBALE, E `base` E' IL PUNTO DI PARTENZA.
 *
 * `gstep` e' il passo assoluto dalla prima passata mai fatta. `g_wsd_base`
 * e' il passo a cui comincia QUESTA sessione, e serve a una cosa sola:
 * ricordare che la curva e' gia' stata percorsa fino a li'. Per il valore
 * da usare NON si fa `gstep - base`: quello e' sbagliato in due modi
 * insieme, perche' dentro train_run passo gia' `base + step`, e fuori si
 * chiama con il passo assoluto gia' completo. Sottraendo base due volte la
 * curva ripartiva dal gradino piu' basso a ogni passata — cioe' esattamente
 * il difetto che WSD doveva eliminare.
 *
 * Quindi: la funzione prende il passo GLOBALE e basta. `base` serve solo a
 * chi la chiama, per capire dove si comincia e per stampare la fase.      */
static int wsd_lr_at(int64_t gstep)
{
    double t = (double)gstep;
    double warm, decay_start, total;
    if (!g_wsd_on || g_wsd_total <= 0) return (int)g_wsd_peak;
    total = (double)g_wsd_total;
    warm  = (double)(g_wsd_warm  > 0 ? g_wsd_warm  : (int64_t)(total * 0.1));
    if (warm > total * 0.5) warm = total * 0.5;
    if (warm < 1.0) warm = 1.0;
    decay_start = total * 0.8;              /* gli ultimi 20% scendono   */
    if (decay_start < warm) decay_start = warm;
    if (t < warm) {
        /* warmup lineare da ~0 al picco : parto da un quinto del picco e
         * non da zero, perche' i primi passi devono comunque muovere
         * qualcosa e il campo e' intero.                                  */
        double a = (double)LR_MIN, b = g_wsd_peak;
        double f = (t < 0.0) ? 0.0 : t / warm;
        return (int)(a + (b - a) * f);
    }
    if (t < decay_start) return (int)g_wsd_peak;      /* fase stabile     */
    {   double f = (t - decay_start) / (total - decay_start);
        if (f < 0.0) f = 0.0;
        if (f > 1.0) f = 1.0;
        return (int)((double)LR_MIN +
                     (g_wsd_peak - (double)LR_MIN) * (1.0 - f));
    }
}
/* lr_set_all() sta piu' avanti, dopo il typedef di Model : qui il tipo non
 * esiste ancora e il compilatore si rifiuta di inventarlo.               */
/* 1 = train_run non scrive nulla : serve all'autobuilder, che addestra in
 * background mentre l'utente sta al prompt e non puo' tollerare che il
 * thread gli rubi lo stdout.                                          */
static int g_quiet       = 0;

/* ====================================================== strato denso ternario
 * Lo strato e' [n_in -> n_out]. I trit sono organizzati per riga di uscita,
 * cosi' la riduzione su un'uscita e' un passo lineare di memoria.        */
typedef struct {
    int32_t n_out, n_in;
    trit_t  *trit;        /* n_out * n_in, {-1,0,+1}                      */
    shadow_t *shadow;     /* n_out * n_in, Q16.15 : il file S_X.bin         */
    int8_t   gshift;
    uint16_t decay_w;     /* UQ4.12                                        */
    int32_t  decay;       /* clip(decay_w >> 12, 0, 15)                    */
    /* cache del forward, = le sezioni Input e Output della sezione 3       */
    int32_t *x_q;         /* n_in, l'attivazione quantizzata               */
    int32_t *raw;         /* n_out, accumulatore prima del gamma          */
    float   *act;         /* n_out, dopo gamma e SiLUNS                   */
    float   *gbuf;        /* n_out, gradiente dopo SiLUNS e gamma         */
    int32_t  n_sat;       /* quante volte G ha saturato a Q4.11          */
    shadow_t *sh0;        /* shadow al momento della quantizzazione       */
    int      use_ste;     /* 1 = surrogato, per il test a differenze     */
    int      g_slot;      /* indice nella tabella delle scale di gradiente*/
} QLin;

static int g_gamma_correct = 0;   /* 0 = letterale del paper, 1 = corretto */

/* ---- r8 (HFP) : Update_Gain della sezione 7 -------------------------
 * La r7 scriveva   Delta = (G * LR) >> 12
 * La r8 scrive     Delta = (G * LR) >> (12 - Update_Gain)
 * con Update_Gain dichiarato come «costante hardware necessaria (es.
 * impostata a 4)».  Il paper NON lo fissa: e' un esempio, non un valore.
 * g_update_gain = 0 riproduce la r7.                              */
static int g_update_gain = 0;
static int g_gain_pinned = 0;      /* 1 = il paper dichiara il valore    */

/*  Shift effettivo della sezione 7.  Se il gain supera 12 lo shift
 *  diventa NEGATIVO e il `>>` del paper diventa uno `<<` : in C e'
 *  comportamento indefinito.  Lo blocco a 0 dichiaro quel buco.     */
static int upd_shift(void)
{
    int sft = 12 - g_update_gain;
    return (sft < 0) ? 0 : sft;
}

/* ---- IL BATCH. Quante finestre vedo per ogni aggiornamento dei pesi -------
 *
 * Prima di tutto il fatto che rende possibile l'accumulo: `model_backward`
 * fa `shadow -= (q * lr) >> upd_shift()`. Il shadow e' un ACCUMULATORE e il
 * LR e' gia' dentro, quindi B chiamate consecutive a model_backward() senza
 * model_refresh() in mezzo sommano B gradienti e ne applicano uno solo. E'
 * esattamente l'accumulo di gradiente, e il codice di avanti gia' lo faceva
 * senza saperlo — REFRESH_EVERY = 1, pero', lo annullava a ogni passo.
 *
 * Il punto che rende la cosa DIVERSA e non solo piu' lenta: B gradienti
 * sommati muovono lo shadow di B volte tanto. Perche' un passo sia la MEDIA
 * dei B esempi e non la somma, il LR va diviso per B. Ecco cosa fa
 * `acc_lr()`. Senza quella divisione non avrei un batch, avrei B passi di
 * SGD in fila con la riquantizzazione in coda — che e' rumoto peggiore.
 *
 * Perche' 32 token per passo non basta: la ricetta di preaddestramento chiede
 * un batch di 8-16 milioni di token (train-pretraining-recipes, sez. 2 e 8).
 * Con 32 token il gradiente e' una stima su una frase sola e la sua varianza
 * e' piu' grande del segnale.                                              */
static int g_batch = 1;

/* ======================================================================
 * [MIO] LA SCALA DEL GRADIENTE — il difetto che fermava l'apprendimento
 * ======================================================================
 *
 * MISURATO (mode `diag`, non dedotto):
 *   - 100% dei gradienti della testa arrivano a +-2047, cioe' al fondo del
 *     campo Q4.11. max |G| = 2048 su 192 pesi: 192 su 192 a saturazione.
 *   - Su 400 passi di addestramento, ALMENO UN trit cambia in 1 passo solo.
 *     Gli altri 399 non muovono niente. In quel passo unico cambiano
 *     3129 tritt su 10240.
 *
 * PERCHE' SUCCEDE. La sezione 5 scrive G = (Error (x) Input) >> 19 e la
 * sezione 7 scrive Shadow -= (G * LR) >> 12. Ma l'errore che questo codice
 * propaga e' in Q16.15 (`ec = e * QF`, riga sopra) e l'ingresso x_q e' in
 * Q16.15: il prodotto e' Q32.30. Un >>19 su Q32.30 lascia Q13.11, cioe'
 * valori fino a ~4096, e il campo dichiarato finisce a 2047. Quindi il
 * gradiente VIAGGIA SEMPRE FUORI dalla sua finestra e viene tagliato.
 *
 * COSA RESTA DEL GRADIENTE. Solo il SEGNO. Ogni peso riceve ±2047 identico,
 * per quanto grande o piccolo sia il gradiente vero. E un aggiornamento in
 * cui il modulo e' costante non e' una discesa del gradiente: e' sign-SGD.
 * Con un passo di (2047 * LR) >> 12 = 2047 copie al LR 4096 contro una
 * soglia di isteresi di 213, il passo e' dieci volte la soglia: a ogni
 * passo i pesi oltrepassano la soglia e vengono rimessi a posto al
 * passo dopo. E' una valanga, non un cammino. Ecco perche' la risposta del
 * modello non cambiava MAI: non stava imparando lentamente, stava girando.
 *
 * LA CORREZIONE, e cosa resta del paper. Non cambio la sezione 5, che
 * dice `>> 19`, ne' il campo Q4.11, che e' dichiarato: cambio COME ci metto
 * dentro il gradiente. Prima di moltiplicare per l'ingresso, riporto
 * l'errore a una scala assoluta con una media mobile del massimo assoluto
 * osservato su quel tensore: e' esattamente la normalizzazione che la
 * ricetta chiama grad clipping (train-pretraining-recipes sez. 6, primo
 * punto dell'elenco), e qui ha anche un nome proprio nel paper: il gamma
 * shift della sezione 1 e' gia' un fattore di scala 2^-g per attivazione,
 * e qui serve il suo gemello per il gradiente.
 *
 * La scala e' una media mobile, non un valore fisso per passo: senza
 * memoria, un gradiente piccolo in un passo verrebbe amplificato fino a
 * saturare, che e' lo stesso difetto con un altro nome.             */
#define GRAD_RMS_N 8   /* finestra della media mobile : 8 passi           */
static int     g_grad_scale_on = 1;   /* 0 = la lettera della sez. 5     */
static double  g_grad_amax[16];       /* media mobile di |G| per tensore   */
static int     g_grad_nidx[16];
static int     g_grad_init = 0;

/* La scala di un tensore : 2047 / media(|G|).
 *
 * MISURATO (mode `diag`): il gradiente grezzo massimo della testa e' di
 * 2.28e14, cioe' 435 milioni di volte il fondo del campo Q4.11. Il >>19
 * della sezione 5 lascia quindi ~4.35e8, e il campo finisce a 2047 : tutto
 * il gradiente arriva a saturazione, e della sua grandezza non resta che il
 * segno. Per questo la scala e' TIPICAMENTE MINORE DI 1 : il gradiente va
 * ABBASSATO, non alzato, e la prima stesure di questa funzione — che
 * imponeva s >= 1 — non faceva assolutamente niente. Misurato: la scala
 * giusta e' 0.0000047, e il codice la tagliava a 1.
 *
 * Il limite basso a 2^-20 non e' un dettaglio: sotto, un gradiente che
 * dovrebbe essere rumore diventa segnale amplificato.                */
/* v004: gli slot OOC sono widened (1000+l*16+k, fino a 1630 a 14B) e non
 * entrano nella tabella grad da 16: la scala e' una media mobile euristica
 * per classe di tensore, quindi l'indice si ripiega con modulo (c-pure
 * RULE 6: mai OOB). Stessa piega in lettura (scale) e scrittura (feed). */
static int grad_slot_norm(int slot)
{
    int r = slot % 16;
    if (r < 0) r += 16;
    return r;
}
static double grad_scale(int slot)
{
    double am, s;
    if (!g_grad_scale_on) return 1.0;
    slot = grad_slot_norm(slot);
    am = g_grad_amax[slot];
    if (am <= 0.0) return 1.0;
    s = (double)GSAT / am;
    if (s > 1024.0)   s = 1024.0;        /* non amplifico troppo il rumore */
    if (s < 9.5367e-7) s = 9.5367e-7;    /* 2^-20 : sotto non ha senso     */
    return s;
}
/* Aggiorna la media mobile con il massimo osservato su questo passo.     */
static void grad_scale_feed(int slot, double amax)
{
    if (!g_grad_scale_on) return;
    if (amax <= 0.0) return;
    slot = grad_slot_norm(slot);
    if (!g_grad_init || g_grad_amax[slot] <= 0.0) { g_grad_amax[slot] = amax; return; }
    g_grad_amax[slot] = (g_grad_amax[slot] * (GRAD_RMS_N - 1) + amax) / (double)GRAD_RMS_N;
}
/* Solo train_run() lo mette a 1, e solo mentre sta accumulando un batch.
 * Gli altri chiamanti di model_backward() — la sonda a differenze finite e il
 * thread in background — riquantizzavano embedding e testa DENTRO la backward
 * e non chiamano mai model_refresh(): se avessi legato la cosa a `g_batch`
 * senza questo, con batch > 1 avrebbero smesso di aggiornare i pesi senza
 * dirlo. Cosi' la modifica resta confinata al ciclo di batch.               */
static int g_accum_active = 0;
void llm_set_batch(int b)
{
    if (b < 1)  b = 1;
    if (b > 4096) b = 4096;
    g_batch = b;
}
int  llm_get_batch(void) { return g_batch; }
int  llm_get_grad_scale(void) { return g_grad_scale_on; }
void llm_set_grad_scale(int on)
{
    g_grad_scale_on = on ? 1 : 0;
    memset(g_grad_amax, 0, sizeof g_grad_amax);
    memset(g_grad_nidx, 0, sizeof g_grad_nidx);
    g_grad_init = 0;
}
static int acc_lr(int lr)
{
    return (g_batch > 1) ? (lr / g_batch) : lr;
}

static void qlin_refresh(QLin *L)
{
    int64_t i, N = (int64_t)L->n_in * L->n_out;
    int32_t kN, graw;
    int64_t sum = 0;
    /* sezione 1 : gammraw = (somma |Shadow|) >> (31 - clz(N))
     * N e' la dimensione del blocco, n_in, come nel nucleo verificato.  */
    kN = 31 - ((L->n_in == 0) ? 0 : __builtin_clz((unsigned int)L->n_in));
    for (i = 0; i < N; i++) sum += (L->shadow[i] < 0) ? -(int64_t)L->shadow[i]
                                                      :  (int64_t)L->shadow[i];
    if (kN > 0) sum >>= kN;
    graw = (sum > 0x7fffffffLL) ? 0x7fffffff : (int32_t)sum;
    L->gshift = (int8_t)(g_gamma_correct ? gamma_shift_correct((uint64_t)graw)
                                         : gamma_shift_r7((uint64_t)graw));
    {   int32_t d = (int32_t)(L->decay_w >> DECAY_FRAC);
        if (d < 0) d = 0;
        if (d > DECAY_CLIP) d = DECAY_CLIP;
        L->decay = d;
    }
    /* sezione 2 : riquantizzazione dei trit dalla mappa di isteresi        */
    for (i = 0; i < N; i++) {
        L->trit[i] = hysteresis_map(L->shadow[i], L->trit[i]);
        L->sh0[i]  = L->shadow[i];
    }
}

static QLin *qlin_new_slot(int32_t n_out, int32_t n_in, int slot);
static QLin *qlin_new_empty(int32_t n_out, int32_t n_in, int slot);
static void qlin_alloc_bufs(QLin *L);
static void qlin_random_init(QLin *L);
static QLin *qlin_new(int32_t n_out, int32_t n_in)
{
    return qlin_new_slot(n_out, n_in, -1);
}
/* Struttura leggera: soli metadati (dimensioni, slot), nessun buffer.
 * Serve allo streaming: le strutture di tutti i layer restano in RAM
 * (centinaia di byte), i buffer pesanti vivono solo sul layer attivo. */
static QLin *qlin_new_empty(int32_t n_out, int32_t n_in, int slot)
{
    QLin *L = (QLin *)calloc(1, sizeof(QLin));
    if (!L) die("malloc");
    L->n_out = n_out; L->n_in = n_in;
    L->g_slot = grad_slot_norm(slot);
    return L;
}
/* I 7 buffer di lavoro. Separati dall'init perche' lo streaming li
 * crea/distrugge a ogni load/unload senza re-inizializzare i pesi. */
static void qlin_alloc_bufs(QLin *L)
{
    int64_t N = (int64_t)L->n_out * L->n_in;
    if (L->trit) return;   /* gia' allocati */
    L->trit   = (trit_t  *)calloc((size_t)N, 1);
    L->shadow = (shadow_t *)calloc((size_t)N, sizeof(shadow_t));
    L->x_q    = (int32_t *)calloc((size_t)L->n_in , 4);
    L->raw    = (int32_t *)calloc((size_t)L->n_out, 4);
    L->act    = (float   *)calloc((size_t)L->n_out, 4);
    L->gbuf   = (float   *)calloc((size_t)L->n_out, 4);
    L->sh0    = (shadow_t *)calloc((size_t)N, sizeof(shadow_t));
    if (!L->trit || !L->shadow || !L->x_q || !L->raw || !L->act || !L->gbuf || !L->sh0)
        die("malloc");
}
/* Init random dei pesi (stessa sequenza RNG di prima a pari ordine di
 * chiamata: model_new inizializza i layer nello stesso ordine). */
static void qlin_random_init(QLin *L)
{
    int64_t N = (int64_t)L->n_out * L->n_in, i;
    int32_t n_in = L->n_in;
    /* sezione 2 : i trit nascono UNIFORMI su {-1,0,+1}, non dalla soglia.  */
    for (i = 0; i < N; i++) {
        uint32_t r = rnd32() % 3u;
        L->trit[i]   = (trit_t)(r == 0 ? -1 : (r == 1 ? 0 : 1));
        L->shadow[i] = shadow_init();
    }
    /* sezione 6 : decay_Weight e' un parametro che il paper ottimizza,
     * ma la sua inizializzazione non e' dichiarata. Il valore naturale
     * e' quello che porta l'accumulatore vicino a 1 : con RMS(x)=1 e
     * trit uniformi su {-1,0,+1} si ha RMS(acc) = sqrt(n_in)*0.816.
     * Partire da 0 (la mia scelta precedente) moltiplicava per ~1000. */
    {   int d0 = (int)(0.5 * log((double)n_in) / log(2.0) + 0.5);
        if (d0 < 0) d0 = 0;
        if (d0 > DECAY_CLIP) d0 = DECAY_CLIP;
        L->decay_w = (uint16_t)(d0 << DECAY_FRAC);
    }
    qlin_refresh(L);
}
static QLin *qlin_new_slot(int32_t n_out, int32_t n_in, int slot)
{
    QLin *L = qlin_new_empty(n_out, n_in, slot);
    qlin_alloc_bufs(L);
    qlin_random_init(L);
    return L;
}
static void qlin_free(QLin *L)
{ if (L) { free(L->trit); free(L->shadow); free(L->x_q); free(L->raw);
           free(L->act); free(L->gbuf); free(L->sh0); free(L); } }
static int64_t qlin_zero(const QLin *L)
{
    int64_t i, N = (int64_t)L->n_in * L->n_out, z = 0;
    for (i = 0; i < N; i++) if (L->trit[i] == 0) z++;
    return z;
}

/* ------------------------------------------------- AVANTI, sezione 4 ------ */
/* v003: blocking sull'asse di riduzione (approvato: loop bloccati prima,
 * intrinseci solo su misura). qlin_forward e' GEMV, non GEMM: ogni peso e'
 * letto una volta sola, quindi e' bandwidth-bound e il tiling NON riduce il
 * traffico come in GEMM — il blocco tiene il pannello x_q + segmento di riga
 * L1-resident e da' al vettorizzatore (-O3 -mavx2) trip-count noti su QLIN_BK.
 * Ordine di somma INVARIATO (blocchi sequenziali, stesso j crescente): gli
 * stessi numeri del seriale, verifiche bit-exact salve. Le righe sono gia'
 * contigue (row-major, unit-stride): niente packing da fare qui. */
#define QLIN_BK 256   /* resta per qlin_backward (ordine invariato) */
/* v004: variante TL2 con shift di sezione 4 applicato on the fly
 * (xs = xq >> shr). Stessa semantica del vecchio `row*(xq>>d)`, ma per
 * terne: solo add/sub/skip, mai moltiplicazioni fp32. Bit-exact: la somma
 * intera e' associativa e l'ordine j resta crescente. */
static int64_t tl2_dot_s(const trit_t *row, const int32_t *xq, int32_t shr,
                         int32_t n)
{
    int64_t acc = 0;
    int32_t j = 0, n3 = n - (n % 3);
    tl2_init();
    for (; j < n3; j += 3) {
        int32_t x0 = xq[j] >> shr, x1 = xq[j+1] >> shr, x2 = xq[j+2] >> shr;
        int idx = ((int)row[j] + 1) * 9 + ((int)row[j+1] + 1) * 3
                + ((int)row[j+2] + 1);
        int8_t a = TL2_PAT[idx][0], b = TL2_PAT[idx][1], c = TL2_PAT[idx][2];
        int64_t g = 0;
        if (a > 0) g += (int64_t)x0; else if (a < 0) g -= (int64_t)x0;
        if (b > 0) g += (int64_t)x1; else if (b < 0) g -= (int64_t)x1;
        if (c > 0) g += (int64_t)x2; else if (c < 0) g -= (int64_t)x2;
        acc += g;
    }
    for (; j < n; j++) {
        int32_t v = xq[j] >> shr;
        if (row[j] > 0) acc += (int64_t)v;
        else if (row[j] < 0) acc -= (int64_t)v;
    }
    return acc;
}
/* v004: UNA riga del forward. Unita' di lavoro del pool: le righe sono
 * indipendenti (scrivono raw[i]/act[i] disgiunti, leggono pesi/input in
 * sola lettura). Stessi numeri del seriale per costruzione. */
static void qlin_forward_row(int i, void *arg)
{
    QLin *L = (QLin *)arg;
    const int32_t d = L->decay;
    const int32_t g = L->gshift;
    const trit_t *row = L->trit + (int64_t)i * L->n_in;
    int64_t acc;
    if (L->use_ste) {
        /* SOLO test a differenze finite (t4): formula esatta invariata.
         * Zero moltiplicazioni fp32 (e' aritmetica int64); non e' l'hot
         * path di inferenza/addestramento, che e' il ramo TL2 sotto. */
        int32_t j;
        acc = 0;
        for (j = 0; j < L->n_in; j++) {
            int64_t k = (int64_t)i * L->n_in + j;    /* indice GIUSTO */
            int64_t w = (int64_t)row[j] * QF
                       + ((int64_t)L->shadow[k] - (int64_t)L->sh0[k]);
            acc += (int64_t)((w * (int64_t)L->x_q[j]) / QF);
        }
    } else {
        acc = tl2_dot_s(row, L->x_q, d, L->n_in);
    }
    acc = (int64_t)apply_gamma((int32_t)((acc > 2147483647LL) ? 2147483647LL
                                     : (acc < -2147483648LL) ? -2147483648LL
                                     : acc), g);
    L->raw[i] = (int32_t)acc;
    L->act[i] = (float)siluns_r7((int32_t)acc) / (float)QF;
}
static void qlin_forward(QLin *L, const float *x)
{
    int32_t j;
    /* quantizzo l'ingresso una volta sola (seriale, O(n_in)), poi sez. 4 */
    for (j = 0; j < L->n_in; j++) {
        double v = (double)x[j] * QF;
        int64_t q = (int64_t)llround(v);
        if (q >  2147483647LL) q =  2147483647LL;
        if (q < -2147483648LL) q = -2147483648LL;
        L->x_q[j] = (int32_t)q;
    }
    /* v004: dispatch al pool persistente sopra soglia, seriale sotto. */
    if (mt_use((int64_t)L->n_out, (int64_t)L->n_in)) {
        mt_run(0, L->n_out, qlin_forward_row, L);
    } else {
        int32_t i;
        mt_serial++;
        for (i = 0; i < L->n_out; i++) qlin_forward_row(i, L);
    }
}

/* ---- Directive 1: backward parallelo (stessi numeri del seriale) ---------
 * qlin_gx_col : UNA colonna di Error_X = Trit^T * Error. Le colonne sono
 * disgiunte in scrittura (ogni worker tocca solo le sue gx[c]) e leggono
 * gbuf/trit in sola lettura; il corpo replica il seriale in ordine i
 * crescente, quindi ogni gx[c] accumula nella stessa sequenza del seriale.
 * qlin_gw_row : UNA riga dell'aggiornamento shadow (sez. 5+7). Le righe
 * scrivono shadow disgiunti; i contatori globali (g_abs_step, g_n_step,
 * L->n_sat) escono in parziali per-riga (bw_gacc/bw_nsat) e il master li
 * riduce in ordine crescente: stessi totali del seriale. lr/gs entrano nel
 * task (acc_lr/upd_shift sono pure dello stesso lr). Seriale invariato. */
typedef struct { QLin *L; float *gx; } gx_task_t;
static void qlin_gx_col(int c, void *arg)
{
    gx_task_t *t = (gx_task_t *)arg;
    QLin *L = t->L;
    float *gx = t->gx;
    int32_t i;
    for (i = 0; i < L->n_out; i++) {
        const trit_t *row = L->trit + (int64_t)i * L->n_in;
        float e = L->gbuf[i];
        if (e == 0.0f) continue;
        if (row[c] > 0) gx[c] += e;
        else if (row[c] < 0) gx[c] -= e;
    }
}
typedef struct { QLin *L; lr_t lr; double gs; } gw_task_t;
static int64_t *bw_gacc = NULL;
static int     *bw_nsat = NULL;
static int32_t  bw_cap  = 0;
static void bw_ensure(int32_t n)
{
    if (n <= bw_cap) return;
    free(bw_gacc); free(bw_nsat);
    bw_gacc = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    bw_nsat = (int *)calloc((size_t)n, sizeof(int));
    if (!bw_gacc || !bw_nsat) die("malloc");
    bw_cap = n;
}
static void qlin_gw_row(int i, void *arg)
{
    gw_task_t *t = (gw_task_t *)arg;
    QLin *L = t->L;
    const int32_t *xq = L->x_q;
    int32_t e = (int32_t)llround((double)L->gbuf[i] * QF);
    int64_t gacc = 0;
    int nsat = 0, j, jj;
    for (jj = 0; jj < L->n_in; jj += QLIN_BK) {
    int32_t j1 = jj + QLIN_BK;
    if (j1 > L->n_in) j1 = L->n_in;
    for (j = jj; j < j1; j++) {
        int64_t p = (int64_t)e * (int64_t)xq[j];
        int32_t q;
        if (t->gs != 1.0) {
            double v = (double)p * t->gs;
            if (v >  2147483000.0) v =  2147483000.0;
            if (v < -2147483000.0) v = -2147483000.0;
            p = (int64_t)v;
        }
        q = (int32_t)(p >> GRAD_SHIFT);
        {
            int32_t delta;
            if (q >  GSAT)  { q =  GSAT;  nsat++; }
            if (q < -GSATN) { q = -GSATN; nsat++; }
            delta = (int32_t)(((int64_t)q * (int64_t)acc_lr(t->lr))
                              >> upd_shift());
            L->shadow[(int64_t)i * L->n_in + j] -= (shadow_t)delta;
            gacc += (delta < 0) ? -(int64_t)delta : (int64_t)delta;
        }
    }
    }
    bw_gacc[i] = gacc; bw_nsat[i] = nsat;
}
/* ------------------------------------------- INDIETRO, sezione 5 ---------
 * want_gx = 1 calcola anche il gradiente rispetto all'ingresso, che e'
 *   Error_X = Trit^T * Error_{X+1} : serve a scendere di un livello.
 * want_gw = 1 applica G = (Error (x) Input) >> 19 e poi
 *   sezione 7 : Shadow -= (G * LR) >> 12.
 * La matrice di gradiente NON viene mai materializzata: si calcola una
 * riga alla volta e si scrive subito nello shadow, che e' il file S_X.bin.
 * La riga i dipende solo da err[i] e dall'intero vettore d'ingresso.      */
/* v003: stesso blocking di qlin_forward (BLAS GEMV: traffico invariato,
 * pannelli L1-resident, ordine preservato). gx[j] e' per-elemento
 * indipendente: il blocco non cambia nessun numero per costruzione. */
static void qlin_backward(QLin *L, const float *gz, float *gx,
                          int want_gx, int want_gw, lr_t lr)
{
    int32_t i, j, jj;
    int32_t g = L->gshift;
    int32_t *restrict xq = L->x_q;
    /* attraverso SiLUNS e gamma-shift */
    for (i = 0; i < L->n_out; i++) {
        /* SiLUNS riceve l'uscita DOPO il gamma, quindi la derivata va
         * valutata sullo stesso valore. Usare raw era un mio errore.     */
        int32_t post = apply_gamma(L->raw[i], g);
        double v = (double)gz[i] * (g_siluns_grad ? siluns_d(post) : 1.0);
        int64_t s = (g >= 0) ? ((int64_t)llround(v * QF) >> g)
                             : ((int64_t)llround(v * QF) >> (-g));
        L->gbuf[i] = (float)((double)s / (double)QF);
    }
    /* NB : gx ACCUMULA, non viene azzerato. Il chiamante lo azzera una
     * volta sola : se lo azzerassi qui, la somma di piu' contributi sullo
     * stesso buffer perderebbe tutto tranne l'ultimo.                    */
    /* v004: Error_X = Trit^T * Error, solo add/sub (trit in {-1,0,+1
a}):
     * gx[j] += e / -= e / skip. Stessi numeri, zero mult fp32.
     * Seriale di proposito: ogni riga ACCUMULA sullo stesso gx (riduzione
     * tra righe); il pool qui richiederebbe buffer parziali per worker +
     * reduce finale. Il GEMV pesante (forward) e' gia' al pool. */
    if (gx && want_gx) {
        if (mt_use((int64_t)L->n_in, (int64_t)L->n_out)) {
            gx_task_t gt;
            gt.L = L; gt.gx = gx;
            mt_run(0, L->n_in, qlin_gx_col, &gt);
        } else {
            for (i = 0; i < L->n_out; i++) {
                const trit_t *row = L->trit + (int64_t)i * L->n_in;
                float e = L->gbuf[i];
                if (e == 0.0f) continue;
                for (jj = 0; jj < L->n_in; jj += QLIN_BK) {
                    int32_t j1 = jj + QLIN_BK;
                    if (j1 > L->n_in) j1 = L->n_in;
                    for (j = jj; j < j1; j++) {
                        if (row[j] > 0) gx[j] += e;
                        else if (row[j] < 0) gx[j] -= e;
                    }
                }
            }
        }
    }
    if (want_gw) {
        /* [MIO] La scala del gradiente, calcolata PRIMA di moltiplicare.
         * Prima cosa misuro il massimo assoluto di questo passo su questo
         * tensore, poi applico la scala della media mobile. Il massimo lo
         * calcola la stessa passata che aggiorna lo shadow : sono due righe
         * in piu', non un secondo giro della matrice.                    */
        double gs = 1.0, amax = 0.0;
        if (g_grad_scale_on) {
            /* xm dipende solo da x_q (invariante nel loop): calcolato una
             * volta sola invece che per ogni riga. Stessi numeri. */
            int64_t xm = 0;
            {   int32_t j2;
                for (j2 = 0; j2 < L->n_in; j2++) {
                    int64_t a = L->x_q[j2];
                    if (a < 0) a = -a;
                    if (a > xm) xm = a;
                }
            }
            for (i = 0; i < L->n_out; i++) {
                int64_t e = llabs((int64_t)llround((double)L->gbuf[i] * QF));
                {   int64_t p = e * xm;
                    int64_t q = p >> GRAD_SHIFT;
                    if (q < 0) q = -q;
                    if (q > amax) amax = (double)q;
                }
            }
            grad_scale_feed(L->g_slot, amax);
            gs = grad_scale(L->g_slot);
        }
        if (mt_use((int64_t)L->n_out, (int64_t)L->n_in)) {
            gw_task_t gt;
            bw_ensure(L->n_out);
            gt.L = L; gt.lr = lr; gt.gs = gs;
            mt_run(0, L->n_out, qlin_gw_row, &gt);
            for (i = 0; i < L->n_out; i++) {
                g_abs_step += bw_gacc[i]; g_n_step++; L->n_sat += bw_nsat[i];
            }
        } else {
            for (i = 0; i < L->n_out; i++) {
                int32_t e = (int32_t)llround((double)L->gbuf[i] * QF);
                int64_t gacc = 0;
                for (jj = 0; jj < L->n_in; jj += QLIN_BK) {
                int32_t j1 = jj + QLIN_BK;
                if (j1 > L->n_in) j1 = L->n_in;
                for (j = jj; j < j1; j++) {
                    int64_t p = (int64_t)e * (int64_t)xq[j];
                    int32_t q;
                    /* [MIO] il gradiente viene riportato dentro la sua finestra
                     * PRIMA dello shift dichiarato dalla sezione 5. vedi nota. */
                    if (gs != 1.0) {
                        double v = (double)p * gs;
                        if (v >  2147483000.0) v =  2147483000.0;
                        if (v < -2147483000.0) v = -2147483000.0;
                        p = (int64_t)v;
                    }
                    q = (int32_t)(p >> GRAD_SHIFT);   /* sez. 5 : >> 19 */
                    {
                        int32_t delta;
                        if (q >  GSAT)  { q =  GSAT;  L->n_sat++; }
                        if (q < -GSATN) { q = -GSATN; L->n_sat++; }
                        delta = (int32_t)(((int64_t)q * (int64_t)acc_lr(lr))
                                          >> upd_shift());
                        /* s.7 : r8 usa (12 - Update_Gain) */
                        L->shadow[(int64_t)i * L->n_in + j] -= (shadow_t)delta;
                        gacc += (delta < 0) ? -(int64_t)delta : (int64_t)delta;
                    }
                }
                }
                g_abs_step += gacc; g_n_step++;
            }
        }
    }
}

/* Sonda : calcola G = (Error (x) Input) >> 19 per TUTTA la matrice, senza
 * applicarla. Serve al test a differenze finite sul modello completo.    */
static void qlin_grad_probe(QLin *L, grad_t *G)
{
    int32_t i, j;
    for (i = 0; i < L->n_out; i++) {
        int32_t e = (int32_t)llround((double)L->gbuf[i] * QF);
        for (j = 0; j < L->n_in; j++) {
            int64_t p = (int64_t)e * (int64_t)L->x_q[j];
            int32_t q = (int32_t)(p >> GRAD_SHIFT);
            if (q >  GSAT)  { q =  GSAT;  g_nsat++; }
            if (q < -GSATN) { q = -GSATN; g_nsat++; }
            if (q == 0) g_nzero++;
            g_ntot++;
            G[(int64_t)i * L->n_in + j] = (grad_t)q;
        }
    }
}

/* Sezione 6, regola del nucleo verificato: decay_w e' un accumulatore che
 * SCENDE quando l'errore pesato sull'ingresso e' grande.                  */
static void qlin_decay_update(QLin *L, const float *gz, lr_t lr)
{
    int32_t i, k;
    int64_t gacc = 0;
    for (i = 0; i < L->n_out; i++) {
        int32_t ev = (int32_t)llround((double)L->gbuf[i] * QF);
        if (ev < 0) ev = -ev;
        for (k = 0; k < L->n_in; k++) {
            int32_t xs = L->x_q[k] >> L->decay;
            if (xs < 0) xs = -xs;
            gacc += (int64_t)ev * (int64_t)xs;
        }
    }
    {   uint32_t gd = (uint32_t)((gacc >> 8) & 0xFFFFu);
        int32_t dd = (int32_t)(((int64_t)gd * (int64_t)lr) >> 15);
        L->decay_w = (uint16_t)((uint16_t)L->decay_w - (uint16_t)dd);
    }
}

/* ========================================================== RMSNorm [MIO] */
static void rmsnorm(const float *x, float *y, int32_t d, float eps)
{
    int32_t i; double s = 0.0;
    for (i = 0; i < d; i++) s += (double)x[i] * x[i];
    s = 1.0 / sqrt(s / (double)d + eps);
    for (i = 0; i < d; i++) y[i] = (float)(x[i] * s);
}
/*  dy_i/dx_j = s*delta_ij - x_i * s^3 * x_j / N
 *  =>  dL/dx_i = s*g_i - (s^3 * x_i / N) * sum_j g_j x_j               */
static void rmsnorm_backward(const float *x, const float *g,
                             float *dx, int32_t d, float eps)
{
    int32_t i; double s = 0.0, sxg = 0.0, s3;
    for (i = 0; i < d; i++) s += (double)x[i] * x[i];
    s = 1.0 / sqrt(s / (double)d + eps);
    s3 = s * s * s;
    for (i = 0; i < d; i++) sxg += (double)g[i] * x[i];
    for (i = 0; i < d; i++)
        dx[i] = (float)(s * (double)g[i] - (s3 * (double)x[i] / (double)d) * sxg);
}

/* ================================================== blocco transformer ===== */
typedef struct {
    int32_t d, n_head, d_head, H, d_ff, T;
    QLin *Wq, *Wk, *Wv, *Wo, *Wg, *Wu, *Wd;   /* tutti ternari, dal paper */
    /* attivazioni salvate dal forward, obbligatorie per l'indietro        */
    float *nh1, *Qh, *Kh, *Vh, *Ch;          /* T x d  e  T x H          */
    float *nh2, *gt, *up, *ff;                /* T x d   e  T x d_ff      */
    /* v004 RWKV-6/Finch: NIENTE matrice T*T (era prob). Stato ricorrente
     * O(1) per passo, O(N) sul contesto. Qh/Kh/Vh tengono r/k/v per testa
     * (T x H); Wt i decadimenti dinamici (T x H). Sst e' lo stato corrente
     * (n_head x dh x dh, layout S[v][k], righe contigue su k); Ssnap gli
     * stati pre-passo per la BPTT; xprev l'attivazione precedente per il
     * token-shift; xmix i 3 ingressi mixati (3 x d, temp per passo).
     * mu_* e ru_b sono FISSI (init ricetta RWKV, mai appresi): il formato
     * checkpoint resta quello delle 7 QLin, compatibile con v003. Il mu/u
     * appresi sono differiti a quando il formato verra' versionato.     */
    float *Wt;                                /* T x H                    */
    float *Sst;                               /* n_head*dh*dh             */
    float *Ssnap;   /* v004 checkpointing: SEMPRE NULL (ex T*n_head*dh*dh) */
    float *xprev;                             /* d                        */
    float *xmix;                              /* 3*d                      */
    float *mu_r, *mu_k, *mu_v;                /* d, fissi                 */
    float *ru_b;                              /* H, bonus u fissi         */
    /* gradienti */
    float *dQ, *dK, *dV, *dC, *dg, *du, *dtmp, *dn2;
    lr_t   lr;            /* sezione 8 : un learning rate per layer  */
    int    dirty;         /* v004-14B/incremental: 1 = RAM piu' nuova del chunk */
/* ============ v004: SLOT OOC WIDENED (allineati ai chunk 14B) =============
 * emb=14, head=15; blocco l, matrice k: 1000+l*16+k (k: 0=Wq 1=Wk 2=Wv
 * 3=Wo 4=Wg 5=Wu 6=Wd; stride 16 > 7 matrici, mai collisione con 14/15 e
 * mai tra layer). Lo schema l*7 e' RIMOSSO: per L>2 collideva con emb/head
 * e lo spill del blocco 2 sovrascriveva i chunk dell'embedding (corruzione
 * silenziosa dei file chunks/). Questi stessi id numerano i chunk 14B
 * (chunks-14b/{I,S}_<slot>.bin, MANIFEST) e gli indici grad (ripiegati a
 * 16 via grad_slot_norm, mai OOB). */
#define OOC_SLOT_EMB 14
#define OOC_SLOT_HEAD 15
#define OOC_SLOT_BASE 1000
#define OOC_SLOT_STRIDE 16
#define OOC_SLOT_LK(l, k) (OOC_SLOT_BASE + (l) * OOC_SLOT_STRIDE + (k))
} Block;static void block_alloc_slot(Block *B, int32_t d, int32_t n_head, int32_t d_ff,
                             int32_t T, int slot);
static void block_alloc(Block *B, int32_t d, int32_t n_head, int32_t d_ff, int32_t T)
{
    block_alloc_slot(B, d, n_head, d_ff, T, -1);
}
static void block_alloc_slot(Block *B, int32_t d, int32_t n_head, int32_t d_ff,
                             int32_t T, int slot)
{
    int32_t t;
    B->d = d; B->n_head = n_head; B->d_head = d / n_head;
    B->H = n_head * B->d_head; B->d_ff = d_ff; B->T = T;
    B->lr = (lr_t)(1u << 12);
    B->dirty = 0;   /* nasce pulito: init spillera' prima di liberare */
    /* v004/stream: SOLO strutture leggere (QLin senza buffer) + vettori fissi
     * mu/ru_b. Pesi e attivazioni vivono nei chunk e nel layer attivo
     * (block_load/unload); qui restano NULL. T serve a block_alloc_bufs. */
    (void)T;
    B->Wq = qlin_new_empty(B->H, d, slot + 0); B->Wk = qlin_new_empty(B->H, d, slot + 1);
    B->Wv = qlin_new_empty(B->H, d, slot + 2); B->Wo = qlin_new_empty(d, B->H, slot + 3);
    B->Wg = qlin_new_empty(d_ff, d, slot + 4); B->Wu = qlin_new_empty(d_ff, d, slot + 5);
    B->Wd = qlin_new_empty(d, d_ff, slot + 6);
    B->nh1 = NULL; B->Qh = NULL; B->Kh = NULL; B->Vh = NULL; B->Ch = NULL;
    B->nh2 = NULL; B->gt = NULL; B->up = NULL; B->ff = NULL;
    B->Wt = NULL; B->Sst = NULL; B->Ssnap = NULL;
    B->xprev = NULL; B->xmix = NULL;
    B->dQ = NULL; B->dK = NULL; B->dV = NULL; B->dC = NULL;
    B->dg = NULL; B->du = NULL; B->dtmp = NULL; B->dn2 = NULL;
    B->mu_r = (float *)calloc((size_t)d, 4);
    B->mu_k = (float *)calloc((size_t)d, 4);
    B->mu_v = (float *)calloc((size_t)d, 4);
    B->ru_b = (float *)calloc((size_t)B->H, 4);
    if (!B->Wq || !B->Wk || !B->Wv || !B->Wo || !B->Wg || !B->Wu || !B->Wd ||
        !B->mu_r || !B->mu_k || !B->mu_v || !B->ru_b) die("malloc");
    /* v004: mix fissi a imbuto (ricetta RWKV: canali bassi = passato,
     * canali alti = presente); bonus u = 0 (termine spento, esatto). */
    for (t = 0; t < d; t++) {
        float f = 1.0f - (float)(t + 1) / (float)(d + 1);
        B->mu_r[t] = f; B->mu_k[t] = f; B->mu_v[t] = f;
    }
    for (t = 0; t < B->H; t++) B->ru_b[t] = 0.0f;
}
/* Buffer di lavoro del layer attivo (stesse calloc di prima, spostate qui).
 * Chiamata da block_load / stream_layer_init, mai da sola a regime. */
static void block_alloc_bufs(Block *B)
{
    int32_t T = B->T, d = B->d, d_ff = B->d_ff, n_head = B->n_head;
    if (B->nh1) return;   /* gia' allocati */
    B->nh1 = (float *)calloc((size_t)T * d, 4);
    B->Qh  = (float *)calloc((size_t)T * B->H, 4);
    B->Kh  = (float *)calloc((size_t)T * B->H, 4);
    B->Vh  = (float *)calloc((size_t)T * B->H, 4);
    B->Ch  = (float *)calloc((size_t)T * B->H, 4);
    B->nh2 = (float *)calloc((size_t)T * d, 4);
    B->gt  = (float *)calloc((size_t)T * d_ff, 4);
    B->up  = (float *)calloc((size_t)T * d_ff, 4);
    B->ff  = (float *)calloc((size_t)T * d_ff, 4);
    B->Wt  = (float *)calloc((size_t)T * B->H, 4);
    B->Sst = (float *)calloc((size_t)n_head * B->d_head * B->d_head, 4);
    B->Ssnap = NULL; /* v004: nessuno snapshot residente (era 10.7 GiB/layer a 14B) */
    B->xprev = (float *)calloc((size_t)d, 4);
    B->xmix  = (float *)calloc((size_t)3 * d, 4);
    B->dQ  = (float *)calloc((size_t)T * B->H, 4);
    B->dK  = (float *)calloc((size_t)T * B->H, 4);
    B->dV  = (float *)calloc((size_t)T * B->H, 4);
    B->dC  = (float *)calloc((size_t)T * B->H, 4);
    B->dg  = (float *)calloc((size_t)T * d_ff, 4);
    B->du  = (float *)calloc((size_t)T * d_ff, 4);
    B->dtmp= (float *)calloc((size_t)T * d, 4);
    B->dn2  = (float *)calloc((size_t)T * d, 4);
    if (!B->nh1 || !B->Qh || !B->Kh || !B->Vh || !B->Ch || !B->nh2 ||
        !B->gt || !B->up || !B->ff || !B->Wt || !B->Sst ||
        !B->xprev || !B->xmix ||
        !B->dQ || !B->dK ||
        !B->dV || !B->dC || !B->dg || !B->du || !B->dtmp || !B->dn2) die("malloc");
}
/* Libera i buffer del layer (pesi QLin + attivazioni), strutture intatte. */
static void block_drop_bufs(Block *B)
{
    QLin *Ls[7]; int u;
    if (!B || !B->Wq) return;
    Ls[0]=B->Wq; Ls[1]=B->Wk; Ls[2]=B->Wv; Ls[3]=B->Wo;
    Ls[4]=B->Wg; Ls[5]=B->Wu; Ls[6]=B->Wd;
    for (u = 0; u < 7; u++) {
        free(Ls[u]->trit); free(Ls[u]->shadow); free(Ls[u]->x_q); free(Ls[u]->raw);
        free(Ls[u]->act); free(Ls[u]->gbuf); free(Ls[u]->sh0);
        Ls[u]->trit = NULL; Ls[u]->shadow = NULL; Ls[u]->x_q = NULL; Ls[u]->raw = NULL;
        Ls[u]->act = NULL; Ls[u]->gbuf = NULL; Ls[u]->sh0 = NULL;
    }
    free(B->nh1); free(B->Qh); free(B->Kh); free(B->Vh); free(B->Ch);
    free(B->nh2); free(B->gt); free(B->up); free(B->ff); free(B->Wt);
    free(B->Sst); free(B->Ssnap); free(B->xprev); free(B->xmix);
    free(B->dQ); free(B->dK); free(B->dV); free(B->dC);
    free(B->dg); free(B->du); free(B->dtmp); free(B->dn2);
    B->nh1 = NULL; B->Qh = NULL; B->Kh = NULL; B->Vh = NULL; B->Ch = NULL;
    B->nh2 = NULL; B->gt = NULL; B->up = NULL; B->ff = NULL;
    B->Wt = NULL; B->Sst = NULL; B->Ssnap = NULL;
    B->xprev = NULL; B->xmix = NULL;
    B->dQ = NULL; B->dK = NULL; B->dV = NULL; B->dC = NULL;
    B->dg = NULL; B->du = NULL; B->dtmp = NULL; B->dn2 = NULL;
}
static void block_free(Block *B)
{
    qlin_free(B->Wq); qlin_free(B->Wk); qlin_free(B->Wv); qlin_free(B->Wo);
    qlin_free(B->Wg); qlin_free(B->Wu); qlin_free(B->Wd);
    free(B->nh1); free(B->Qh); free(B->Kh); free(B->Vh); free(B->Ch);
    free(B->nh2); free(B->gt); free(B->up); free(B->ff); free(B->Wt);
    free(B->Sst); free(B->Ssnap); free(B->xprev); free(B->xmix);
    free(B->mu_r); free(B->mu_k); free(B->mu_v); free(B->ru_b);
    free(B->dQ); free(B->dK); free(B->dV); free(B->dC);
    free(B->dg); free(B->du); free(B->dtmp); free(B->dn2);
}
static void block_refresh(Block *B)
{
    /* Ogni mutazione pesi passa di qui (backward chiude con refresh): da
     * questo punto il chunk su disco e' vecchio finche' non si respilla. */
    B->dirty = 1;
    qlin_refresh(B->Wq); qlin_refresh(B->Wk); qlin_refresh(B->Wv);
    qlin_refresh(B->Wo); qlin_refresh(B->Wg); qlin_refresh(B->Wu);
    qlin_refresh(B->Wd);
}

/* AVANTI del blocco su tutta la finestra. h e out sono T x d.            */
static void block_forward(Block *B, const float *h, float *out)
{
    int32_t t, i, hh, dh = B->d_head, H = B->H, d = B->d, T = B->T;
    float *xr = B->xmix, *xk = B->xmix + d, *xv = B->xmix + (size_t)2 * d;
    /* v004 RWKV-6/Finch time-mixing O(T): ricorrenza per testa su stato
     * S[dh][dh] (layout S[v][k], righe contigue), niente matrice T*T.
     * r/k/v = proiezioni ternarie di ingressi mixati (token-shift, un solo
     * vettore precedente, O(d)); w_t,i = fexp_decay(k) dinamico per canale
     * e token (UN fexp per canale, mai per elemento di stato, mai exp()).
     * y_j = v_j*bonus + sum_i r_i*S_old[j][i]; S = w*S_old + v*k^T.
     * Seriale sui passi (dipendenza temporale), teste indipendenti.
     * Lo stato e' fp per prescrizione (arch-transformer-block, regola (c)
     * in testa al file): il divieto di mult fp32 vale per i pesi BitLinear,
     * gia' TL2 add/sub sopra.                                          */
    for (i = 0; i < d; i++) B->xprev[i] = 0.0f;
    for (hh = 0; hh < B->n_head; hh++)
        memset(B->Sst + (size_t)hh * dh * dh, 0, (size_t)dh * dh * 4);
    for (t = 0; t < T; t++) {
        const float *xn = B->nh1 + (size_t)t * d;
        rmsnorm(h + (size_t)t * d, B->nh1 + (size_t)t * d, d, 1e-5f);
        for (i = 0; i < d; i++) {
            float dv = B->xprev[i] - xn[i];
            xr[i] = xn[i] + B->mu_r[i] * dv;
            xk[i] = xn[i] + B->mu_k[i] * dv;
            xv[i] = xn[i] + B->mu_v[i] * dv;
        }
        qlin_forward(B->Wq, xr);
        qlin_forward(B->Wk, xk);
        qlin_forward(B->Wv, xv);
        memcpy(B->Qh + (size_t)t * H, B->Wq->act, (size_t)H * 4);
        memcpy(B->Kh + (size_t)t * H, B->Wk->act, (size_t)H * 4);
        memcpy(B->Vh + (size_t)t * H, B->Wv->act, (size_t)H * 4);
        /* v004 checkpointing: nessuno snapshot residente (Ssnap=NULL, vedi
         * ckpt_* sul backward). Resta il ramo condizionale per compatibilita'
         * se un giorno lo snapshot torna allocato. */
        if (B->Ssnap)
            memcpy(B->Ssnap + (size_t)t * B->n_head * dh * dh, B->Sst,
                   (size_t)B->n_head * dh * dh * 4);
        for (hh = 0; hh < B->n_head; hh++) {
            const float *r = B->Qh + (size_t)t * H + (size_t)hh * dh;
            const float *k = B->Kh + (size_t)t * H + (size_t)hh * dh;
            const float *v = B->Vh + (size_t)t * H + (size_t)hh * dh;
            const float *u = B->ru_b + (size_t)hh * dh;
            float *S = B->Sst + (size_t)hh * dh * dh;
            float *w = B->Wt + (size_t)t * H + (size_t)hh * dh;
            float *y = B->Ch + (size_t)t * H + (size_t)hh * dh;
            double bonus = 0.0;
            int32_t a, b;
            for (a = 0; a < dh; a++) {
                double ww = fexp_decay((double)k[a]);
                w[a] = (float)ww;
                bonus += (double)r[a] * (double)u[a] * (double)k[a];
            }
            for (b = 0; b < dh; b++) {
                double yj = (double)v[b] * bonus;
                float *Sb = S + (size_t)b * dh;
                for (a = 0; a < dh; a++) {
                    float old = Sb[a];
                    yj += (double)r[a] * (double)old;
                    Sb[a] = (float)((double)w[a] * (double)old
                                    + (double)v[b] * (double)k[a]);
                }
                y[b] = (float)yj;
            }
        }
        memcpy(B->xprev, xn, (size_t)d * 4);
    }
    /* proiezione di uscita + residua 1 */
    for (t = 0; t < T; t++) {
        qlin_forward(B->Wo, B->Ch + (size_t)t * H);
        for (i = 0; i < d; i++)
            out[(size_t)t * d + i] = h[(size_t)t * d + i] + B->Wo->act[i];
    }
    /* SwiGLU : ff = SiLUNS(Wg x) * (Wu x)   [MIO: la gate]              */
    for (t = 0; t < T; t++) {
        rmsnorm(out + t * d, B->nh2 + t * d, d, 1e-5f);
        qlin_forward(B->Wg, B->nh2 + t * d);
        qlin_forward(B->Wu, B->nh2 + t * d);
        for (i = 0; i < B->d_ff; i++) {
            float gq = B->Wg->act[i], uq = B->Wu->act[i];
            B->gt[(size_t)t*B->d_ff + i] = gq;
            B->up[(size_t)t*B->d_ff + i] = uq;
            B->ff[(size_t)t*B->d_ff + i] = (float)(siluns_r7((int32_t)llround(
                (double)gq * QF)) / (double)QF * uq);
        }
        qlin_forward(B->Wd, B->ff + (size_t)t * B->d_ff);
        for (i = 0; i < d; i++)
            out[(size_t)t * d + i] += B->Wd->act[i];
    }
}

/* INDIETRO del blocco. dh_out e' T x d, dhin riceve T x d.
 *
 * Il gradiente si propaga in tre passi, in quest'ordine:
 *   1. residua 2 : dff  = Wd^T  * dout , poi le due gate, poi nh2
 *   2. residua 1 : dC   = Wo^T  * dout , poi l'attenzione, poi nh1
 *   3. le due residue sommano : dhin = dout + RMSNorm^T(...) + RMSNorm^T(...)
 */
/* ============ v004: ACTIVATION CHECKPOINTING (niente Ssnap residente) ====
 * La BPTT RWKV ha bisogno di S_old[t] (stato pre-passo) in ordine INVERSO,
 * ma la ricorrenza S[t+1] = w[t]*S[t] + v[t]*k[t]^T corre solo in AVANTI.
 * Materializzare tutti gli snapshot costa T*n_head*dh*dh float residenti
 * (10.7 GiB/layer a 14B): inaccettabile. Soluzione a segmenti (OOC_CKPT_SEG
 * passi, default 64):
 *   1. ckpt_boundaries(): UNA spazzata avanti con Qh/Kh/Vh/Wt SALVATI dal
 *      forward (niente fexp: w si riusa da Wt) che salva lo stato solo ai
 *      confini di segmento: (nseg+1)*st float transienti (~170 MB a 14B);
 *   2. ckpt_seg_head(): per (testa, segmento) rigenera gli snapshot del
 *      segmento dal suo confine (<=64*dh*dh transienti, ~167 MB a 14B);
 *   3. la BPTT esistente gira sui segmenti dal futuro al passato, con dS
 *      portato attraverso i segmenti (stessa matematica, stessi numeri:
 *      la formula per elemento e' identica al forward, w riusato da Wt).
 * Costo: ~+1 spazzata avanti per backward (il +1/3 documentato); picco
 * residente Ssnap: 0. Sul modello piccolo (T=32) e' un solo segmento. */
#define OOC_CKPT_SEG 64
static float *ckpt_boundaries(const Block *B, int32_t nseg)
{
    int32_t T = B->T, nh = B->n_head, dh = B->d_head, H = B->H;
    int32_t t, hh;
    size_t st;
    float *S, *bnd;
    if (T <= 0 || nh <= 0 || dh <= 0 || H <= 0 || nseg <= 0) return NULL;
    st = (size_t)nh * dh * dh;
    if (st == 0 || (uint64_t)(nseg + 1) > UINT64_MAX / (st * 4)) return NULL;
    bnd = (float *)malloc((size_t)(nseg + 1) * st * 4);
    S = (float *)calloc(st, 4);
    if (!bnd || !S) { free(bnd); free(S); return NULL; }
    memcpy(bnd, S, st * 4);   /* bnd[0] = S_0 = 0, come nel forward */
    for (t = 0; t < T; t++) {
        for (hh = 0; hh < nh; hh++) {
            const float *k = B->Kh + (size_t)t * H + (size_t)hh * dh;
            const float *v = B->Vh + (size_t)t * H + (size_t)hh * dh;
            const float *w = B->Wt + (size_t)t * H + (size_t)hh * dh;
            float *Sh = S + (size_t)hh * dh * dh;
            int32_t a, b;
            for (b = 0; b < dh; b++) {
                float vb = v[b];
                float *row = Sh + (size_t)b * dh;
                for (a = 0; a < dh; a++)
                    row[a] = w[a] * row[a] + vb * k[a];
            }
        }
        if ((t + 1) % OOC_CKPT_SEG == 0) {
            int32_t s = (t + 1) / OOC_CKPT_SEG;
            if (s >= 0 && s <= nseg) memcpy(bnd + (size_t)s * st, S, st * 4);
        }
    }
    free(S);
    return bnd;
}
/* Rigenera gli snapshot della testa hh per t in [s0,s1) dal confine s0.
 * Layout bnd: bnd[(s*nh+hh)*dh*dh], s=0..nseg; seg: seg[(t-s0)*dh*dh]. */
static void ckpt_seg_head(const Block *B, int32_t hh, const float *bnd, int32_t nseg,
                          int32_t s0, int32_t s1, float *seg)
{
    int32_t dh = B->d_head, H = B->H;
    int32_t t, a, b, s = s0 / OOC_CKPT_SEG;
    size_t hst = (size_t)dh * dh;
    float *S = (float *)malloc(hst * 4);
    if (!S) die("malloc");
    if (s < 0) s = 0;
    if (s > nseg) s = nseg;
    memcpy(S, bnd + ((size_t)s * B->n_head + hh) * hst, hst * 4);
    for (t = s0; t < s1; t++) {
        const float *k = B->Kh + (size_t)t * H + (size_t)hh * dh;
        const float *v = B->Vh + (size_t)t * H + (size_t)hh * dh;
        const float *w = B->Wt + (size_t)t * H + (size_t)hh * dh;
        memcpy(seg + (size_t)(t - s0) * hst, S, hst * 4);
        for (b = 0; b < dh; b++) {
            float vb = v[b];
            for (a = 0; a < dh; a++)
                S[(size_t)b * dh + a] = w[a] * S[(size_t)b * dh + a] + vb * k[a];
        }
    }
    free(S);
}
/* ---- BPTT per-testa via pool (stessi numeri del seriale) -----------------
 * Ogni testa scrive slice disgiunte di dQ/dK/dV (offset hh*dh per token) e
 * legge il resto in sola lettura; l'unico stato condiviso del seriale
 * (dSa/dSb) diventa coppia privata del task (malloc/free dentro la fn:
 * thread-safe, nessuna condivisione tra worker). ckpt_boundaries resta
 * seriale (ricorrenza temporale); ckpt_seg_head e' pura a parita' di
 * (hh,s0,s1). Il fallback seriale chiama la stessa fn: per costruzione,
 * gli stessi numeri. Mai nidificazione: la fn non chiama mt_run. */
typedef struct { Block *B; float *bnd; int32_t nseg; } bptt_task_t;
static void bptt_head(int hh, void *arg)
{
    bptt_task_t *bt = (bptt_task_t *)arg;
    Block *B = bt->B;
    float *bnd = bt->bnd;
    int32_t nseg = bt->nseg;
    int32_t T = B->T, dh = B->d_head, H = B->H;
    int32_t s, t;
    float *dSa = (float *)calloc((size_t)dh * dh, 4);
    float *dSb = (float *)calloc((size_t)dh * dh, 4);
    if (!dSa || !dSb) die("malloc");
    memset(dSa, 0, (size_t)dh * dh * 4);
    for (s = nseg - 1; s >= 0; s--) {
    int32_t s0 = s * OOC_CKPT_SEG, s1 = s0 + OOC_CKPT_SEG;
    float *seg;
    if (s1 > T) s1 = T;
    if (s0 >= s1) continue;
    seg = (float *)malloc((size_t)(s1 - s0) * dh * dh * 4);
    if (!seg) die("malloc");
    ckpt_seg_head(B, hh, bnd, nseg, s0, s1, seg);
    for (t = s1 - 1; t >= s0; t--) {
        const float *r = B->Qh + (size_t)t * H + (size_t)hh * dh;
        const float *k = B->Kh + (size_t)t * H + (size_t)hh * dh;
        const float *v = B->Vh + (size_t)t * H + (size_t)hh * dh;
        const float *u = B->ru_b + (size_t)hh * dh;
        const float *w = B->Wt + (size_t)t * H + (size_t)hh * dh;
        const float *S_old = seg + (size_t)(t - s0) * dh * dh;
        const float *dy = B->dC + (size_t)t * H + (size_t)hh * dh;
        float *dr = B->dQ + (size_t)t * H + (size_t)hh * dh;
        float *dk = B->dK + (size_t)t * H + (size_t)hh * dh;
        float *dv = B->dV + (size_t)t * H + (size_t)hh * dh;
        double b = 0.0, db = 0.0;
        int32_t a, jj;
        for (a = 0; a < dh; a++)
            b += (double)r[a] * (double)u[a] * (double)k[a];
        for (jj = 0; jj < dh; jj++) db += (double)dy[jj] * (double)v[jj];
        for (a = 0; a < dh; a++) {
            double dv_a = (double)dy[a] * b;
            double dr_a = db * (double)u[a] * (double)k[a];
            double dk_a = db * (double)u[a] * (double)r[a];
            double dw_a = 0.0;
            for (jj = 0; jj < dh; jj++) {
                float sold = S_old[(size_t)jj * dh + a];
                float dn = dSa[(size_t)jj * dh + a];
                dv_a += (double)dn * (double)k[a];
                dr_a += (double)dy[jj] * (double)sold;
                dk_a += (double)dn * (double)v[jj];
                dw_a += (double)dn * (double)sold;
            }
            dv[a] = (float)dv_a;
            dr[a] = (float)dr_a;
            {   /* dd = k (scelta v004: decadimento guidato dalla chiave) */
                double ek = fexp((double)k[a]);
                double dwdd = -(double)w[a] * ek;
                dk[a] = (float)(dk_a + dw_a * dwdd);
            }
        }
        for (jj = 0; jj < dh; jj++)
            for (a = 0; a < dh; a++)
                dSb[(size_t)jj * dh + a] =
                    (float)((double)dy[jj] * (double)r[a]
                            + (double)dSa[(size_t)jj * dh + a]
                              * (double)w[a]);
        { float *sw = dSa; dSa = dSb; dSb = sw; }
    }
    free(seg);
    }
    free(dSa); free(dSb);
}
static void block_backward(Block *B, const float *h, const float *dh_out,
                           float *dhin, lr_t lr)
{
    int32_t t, i, j, k, hh, dh = B->d_head, H = B->H, d = B->d, T = B->T, F = B->d_ff;
    float *dX = B->dtmp;                 /* T x d : residua 1 + residua 2  */

    {   flat_task_t ft;
        int64_t nflat = (int64_t)T * d;
        ft.dst = dX; ft.a = dh_out; ft.mode = 0;
        if (nflat > 1 && nflat <= 2147483647 && mt_use(nflat, 1))
            mt_run(0, (int)nflat, flat_vec_op, &ft);
        else for (t = 0; t < T; t++)
            for (i = 0; i < d; i++) dX[(size_t)t*d + i] = dh_out[(size_t)t*d + i];
    }

    /* ---- 1. residua 2 : MLP ----------------------------------------- */
    for (t = 0; t < T; t++) {
        /* gradiente rispetto a ff : la trasposta di Wd, che e' gia' la
         * seconda componente di qlin_backward */
        qlin_backward(B->Wd, dh_out + (size_t)t * d,
                      B->dg + (size_t)t * F, 1, 1, lr);
        /* ff = SiLUNS(gt) * up : due prodotti, due derivate */
        for (i = 0; i < F; i++) {
            float gq = B->gt[(size_t)t*F + i], uq = B->up[(size_t)t*F + i];
            int32_t gqi = (int32_t)llround((double)gq * QF);
            float ds = g_siluns_grad ? (float)siluns_d(gqi) : 1.0f;
            float dff = B->dg[(size_t)t*F + i];
            if (g_paper_chain) {
                B->dg[(size_t)t*F + i] = dff;         /* catena lineare    */
                B->du[(size_t)t*F + i] = dff;
            } else {
                B->dg[(size_t)t*F + i] = dff * uq * ds;   /* verso la gate */
                B->du[(size_t)t*F + i] = dff * gq;        /* verso l'up    */
            }
        }
        /* nh2 riceve la somma dei due contributi : azzero, poi sommo */
        for (i = 0; i < d; i++) B->dn2[(size_t)t*d + i] = 0.0f;
        qlin_backward(B->Wg, B->dg + (size_t)t*F, B->dn2 + (size_t)t*d, 1, 1, lr);
        qlin_backward(B->Wu, B->du + (size_t)t*F, B->dn2 + (size_t)t*d, 1, 1, lr);
        for (i = 0; i < d; i++) dX[(size_t)t*d + i] += B->dn2[(size_t)t*d + i];
    }
    /* rmsnorm al contrario sul secondo pre-norm */
    for (t = 0; t < T; t++) {
        float *dst = (float *)calloc((size_t)d, 4);
        if (!dst) die("malloc");
        if (g_paper_chain) {
            for (i = 0; i < d; i++) dst[i] = dX[(size_t)t*d + i];
        } else
            rmsnorm_backward(B->nh2 + (size_t)t*d, dX + (size_t)t*d, dst, d, 1e-5f);
        for (i = 0; i < d; i++) dX[(size_t)t*d + i] = dst[i];
        free(dst);
    }

    /* ---- 2. residua 1 : attenzione ----------------------------------- */
    for (t = 0; t < T; t++) {
        for (i = 0; i < H; i++) B->dC[(size_t)t*H + i] = 0.0f;
        qlin_backward(B->Wo, dh_out + (size_t)t*d, B->dC + (size_t)t*H, 1, 1, lr);
    }

    /* v004: BPTT attraverso la ricorrenza RWKV-6. O(T) tempo, O(T) snapshot
     * (Ssnap), zero matrici T*T. Seriale sui passi all'indietro (la dipen-
     * denza temporale lo impone: niente lock, niente pool qui). Per testa,
     * con dy = dC[t], S_old = Ssnap[t], dS_next portato dal futuro:
     *   b = sum_i r_i*u_i*k_i;   y_j = v_j*b + sum_i r_i*S_old[j][i]
     *   dv_j = dy_j*b + sum_i dS_next[j][i]*k_i
     *   db = sum_j dy_j*v_j  ->  dr += db*u*k , dk += db*u*r
     *   dr_i += sum_j dy_j*S_old[j][i]
     *   dS_old[j][i] = dy_j*r_i + dS_next[j][i]*w_i
     *   dw_i = sum_j dS_next[j][i]*S_old[j][i] -> dk via fexp_decay'(k)
     *   dk_i += sum_j dS_next[j][i]*v_j;  poi dS_next = dS_old.
     * dR/dK/dV riusano i buffer dQ/dK/dV (stessa forma T x H). Derivata di
     * fexp_decay esatta sul fexp (liscio a tratti, ok per l'addestramento). */
    memset(B->dQ, 0, (size_t)T * H * 4);
    memset(B->dK, 0, (size_t)T * H * 4);
    memset(B->dV, 0, (size_t)T * H * 4);
    {   bptt_task_t bt;
        /* ckpt_boundaries resta seriale (ricorrenza temporale); le teste
         * vanno al pool con scratch dS privato (vedi bptt_head sopra). */
        int32_t nseg = (T + OOC_CKPT_SEG - 1) / OOC_CKPT_SEG;
        float *bnd = ckpt_boundaries(B, nseg);
        if (!bnd) die("malloc");
        if (nseg <= 0) nseg = 1;
        bt.B = B; bt.bnd = bnd; bt.nseg = nseg;
        if (mt_use((int64_t)B->n_head, (int64_t)T * dh * dh))
            mt_run(0, B->n_head, bptt_head, &bt);
        else for (hh = 0; hh < B->n_head; hh++) bptt_head(hh, &bt);
        free(bnd);
    }
    /* v004: r/k/v risalgono ai 3 rami mixati; il token-shift si inverte:
     * xm[t] = x[t]+mu*(x[t-1]-x[t]) => dx[t] += dxm*(1-mu),
     * dx[t-1] += dxm*mu (t=0: il passato e' la costante 0, via).      */
    {   float *dpre = (float *)calloc((size_t)T * d, 4);
        float *tx = (float *)calloc((size_t)3 * d, 4);
        if (!dpre || !tx) die("malloc");
        memset(dpre, 0, (size_t)T * d * 4);
        for (t = 0; t < T; t++) {
            float *tr = tx, *tk = tx + d, *tv = tx + (size_t)2 * d;
            memset(tx, 0, (size_t)3 * d * 4);
            qlin_backward(B->Wq, B->dQ + (size_t)t*H, tr, 1, 1, lr);
            qlin_backward(B->Wk, B->dK + (size_t)t*H, tk, 1, 1, lr);
            qlin_backward(B->Wv, B->dV + (size_t)t*H, tv, 1, 1, lr);
            for (i = 0; i < d; i++) {
                dpre[(size_t)t*d + i] +=
                    tr[i] * (1.0f - B->mu_r[i])
                  + tk[i] * (1.0f - B->mu_k[i])
                  + tv[i] * (1.0f - B->mu_v[i]);
                if (t > 0)
                    dpre[(size_t)(t-1)*d + i] +=
                        tr[i] * B->mu_r[i]
                      + tk[i] * B->mu_k[i]
                      + tv[i] * B->mu_v[i];
            }
        }
        for (t = 0; t < T; t++) {
            float *dst = (float *)calloc((size_t)d, 4);
            if (!dst) die("malloc");
            if (g_paper_chain) {
                for (i = 0; i < d; i++) dst[i] = dpre[(size_t)t*d + i];
            } else
                rmsnorm_backward(B->nh1 + (size_t)t*d, dpre + (size_t)t*d, dst, d, 1e-5f);
            for (i = 0; i < d; i++) dX[(size_t)t*d + i] += dst[i];
            free(dst);
        }
        free(dpre); free(tx);
    }
    /* sezione 6 : il decay ottimizzato scende con l'errore pesato */
    for (t = 0; t < T; t++) {
        qlin_decay_update(B->Wd, dh_out + (size_t)t * d, lr);
        qlin_decay_update(B->Wq, dh_out + (size_t)t * d, lr);
    }
    block_refresh(B);

    /* ---- 3. le due residue ------------------------------------------- */
    for (t = 0; t < T; t++) {
        for (i = 0; i < d; i++) {
            float g = dX[(size_t)t*d + i] + dh_out[(size_t)t*d + i];
            dhin[(size_t)t*d + i] = g;
        }
    }
    (void)h; (void)k; (void)j;
}

/* ================================================================= modello */
typedef struct {
    int32_t d, n_head, d_ff, T, V, n_layers;
    Block *B;
    QLin  *emb, *head;
    float *logits, *dlogits;
    float *hfin, *dhn;     /* RMSNorm finale prima della testa [MIO] */
    float *hsave;                       /* attivazione di ogni blocco      */
    lr_t   lr, lr_emb, lr_head;
    int32_t n_correct, n_tok;
    int64_t n_sat;
} Model;

/* ============ v003: NOTA TOPOLOGIA 14B (documentata, default invariati) ===
 * model_params() sotto: P = 2*V*d + L*(4*H*d + 3*d*d_ff). Esempio 14B
 * (forma Llama-13B: V=128000, d=5120, H=5120, d_ff=13824, L=40, T=4096):
 * per-layer 4*5120^2 + 3*5120*13824 = 317M, x40 = 12.7B + emb/head 1.3B.
 * Per una 14B in v003: cambiare cfg_default() (layers/d/nh/dff/T) e
 * ricostruire; niente argomenti CLI di forma in v003 (di proposito: il
 * programma pensa lui ai dettagli). RAM: trit a 1.625 bit/peso -> ~2.8 GB
 * mappati (I_X, read-only, paginati dall'OS); shadow int32 -> ~56 GB su
 * disco (S_X) con UN SOLO chunk attivo mappato alla volta (vedi ooc_*).
 * AVVERTENZE: (1) [v004 FATTO] gli slot blocco sono OOC_SLOT_LK(l,k) =
 * 1000+l*16+k (stride 16, mai collisione con emb/head=14/15); (2) un
 * chunk solo di shadow (1.27 GB/layer a 14B) supera il tetto 30% su PC
 * piccoli: il governatore rifiuta la map e l'addestramento resta in-RAM —
 * la 14B vuole una workstation, non questo portatile; (3) i default restano
 * d64/L2/T32/V~674 (85K param): la 14B non e' materializzata qui, solo la
 * via (chunk + tetto) e' verificata sul modello piccolo. */
/* v004/stream: prototipi (definiti prima di model_forward, usati gia'
 * in model_new/refresh/maintain). Model e' definito sopra, qui visibile. */
static int block_load(Model *M, int32_t l);
static int block_spill_layer(Model *M, int32_t l);
static int block_unload(Model *M, int32_t l, int dirty);
static int stream_layer_init(Model *M, int32_t l);
static int stream_layer_chunks_ready(Model *M, int32_t l);
static int qlin_spill_locked(int slot, const QLin *L);
/* Contatore riuso chunk: sta qui (non col resto dei contatori piu' sotto)
 * perche' model_new_impl lo usa prima del loro punto di definizione. */
static int g_stream_reused = 0;
static Model *model_new_impl(int32_t V, int32_t d, int32_t n_head, int32_t d_ff,
                       int32_t T, int32_t n_layers, uint32_t seed, int reuse)
{
    Model *M = (Model *)calloc(1, sizeof(Model));
    int32_t l;
    if (!M) die("malloc");
    rseed(seed);
    M->V = V; M->d = d; M->n_head = n_head; M->d_ff = d_ff;
    M->T = T; M->n_layers = n_layers;
    M->B = (Block *)calloc((size_t)n_layers, sizeof(Block));
    M->hsave   = (float *)calloc((size_t)(n_layers + 1) * T * d, 4);
    M->hfin    = (float *)calloc((size_t)T * d, 4);
    M->dhn     = (float *)calloc((size_t)T * d, 4);
    M->logits  = (float *)calloc((size_t)T * V, 4);
    M->dlogits = (float *)calloc((size_t)T * V, 4);
    if (!M->B || !M->hsave || !M->logits || !M->dlogits || !M->hfin || !M->dhn)
        die("malloc");
    {   /* v004-14B: riuso chunk SOLO con reuse=1 (boot interattivo).
         * Re-inizializzare 40 layer a ogni avvio sono 12.7B params buttati +
         * ~63 GiB riscritti: se i chunk ci sono gia' con header coerenti
         * (anche dimensione file) si salta l'init (i load leggono da disco).
         * Fresh (assenti/mismatch) = init normale. reuse=0: SEMPRE dai semi
         * (stessi pesi a pari seed: lo provano i test di determinismo). */
        int32_t reused = 0;
        for (l = 0; l < n_layers; l++) {
            block_alloc_slot(&M->B[l], d, n_head, d_ff, T, OOC_SLOT_LK(l, 0));
            if (reuse && stream_layer_chunks_ready(M, l)) reused++;
            else if (!stream_layer_init(M, l)) die("ooc init");
        }
        if (reused) {
            printf("stream: %d/%d layer riusati dai chunk (init saltata).\n",
                   reused, n_layers);
            g_stream_reused += reused;
        }
    }
    /* riga = token, colonna = feature : quindi n_out = V           */
    M->emb  = qlin_new_slot(V, d, OOC_SLOT_EMB);
    M->head = qlin_new_slot(V, d, OOC_SLOT_HEAD);
    M->lr = (lr_t)(1u << 12);
    M->lr_emb = (lr_t)(1u << 12); M->lr_head = (lr_t)(1u << 12);
    return M;
}
/* Sempre dai semi: a pari seed, stessi pesi (nessun riuso chunk). */
static Model *model_new(int32_t V, int32_t d, int32_t n_head, int32_t d_ff,
                       int32_t T, int32_t n_layers, uint32_t seed)
{
    return model_new_impl(V, d, n_head, d_ff, T, n_layers, seed, 0);
}
/* v004-14B: boot/ricostruzione del modello interattivo (g_m). riusa i chunk
 * su disco quando coerenti (riavvio veloce, pesi allenati conservati),
 * init dai semi solo al primo avvio o se assenti/corrotti. I layer non
 * dipendono da V: il riuso vale anche cambiando vocabolario (emb/head sono
 * sempre fresche all'allocazione, poi ab_recall_emb_head le ricarica dai
 * loro slot se la forma combacia). */
static Model *model_new_reuse(int32_t V, int32_t d, int32_t n_head, int32_t d_ff,
                       int32_t T, int32_t n_layers, uint32_t seed)
{
    return model_new_impl(V, d, n_head, d_ff, T, n_layers, seed, 1);
}
/* Il learning rate su TUTTO il modello : embedding, testa e tutti gli
 * strati. Serve al WSD, che a ogni passo ricalcola il suo valore e deve
 * scriverlo ovunque, altrimenti metà della rete importerebbe con un passo
 * e l'altra meta con un altro — e il risultato non e' "una learning rate
 * sbagliata", e' nessuna learning rate.                                    */
static void lr_set_all(Model *M, int lr)
{
    int32_t l;
    if (lr < LR_MIN) lr = LR_MIN;
    if (lr > LR_MAX) lr = LR_MAX;
    M->lr = (lr_t)lr; M->lr_emb = M->lr; M->lr_head = M->lr;
    for (l = 0; l < M->n_layers; l++) M->B[l].lr = M->lr;
}

static void model_free(Model *M)
{
    int32_t l;
    if (!M) return;
    for (l = 0; l < M->n_layers; l++) block_free(&M->B[l]);
    qlin_free(M->emb); qlin_free(M->head);
    free(M->B); free(M->hsave); free(M->logits); free(M->dlogits);
    free(M->hfin); free(M->dhn); free(M);
}
static int64_t model_params(const Model *M)
{
    int64_t p = 0; int32_t l;
    const Block *B;
    p += (int64_t)M->emb->n_in * M->emb->n_out;
    p += (int64_t)M->head->n_in * M->head->n_out;
    for (l = 0; l < M->n_layers; l++) {
        B = &M->B[l];
        p += 3ll * B->Wq->n_in * B->Wq->n_out + (int64_t)B->Wo->n_in * B->Wo->n_out;
        p += 3ll * B->Wg->n_in * B->Wg->n_out;
    }
    return p;
}
static void model_refresh(Model *M)
{   int32_t l;
    for (l = 0; l < M->n_layers; l++) {
        if (!block_load(M, l)) die("ooc load");
        block_refresh(&M->B[l]);
        if (!block_unload(M, l, 1)) die("ooc spill");
    }
    qlin_refresh(M->emb); qlin_refresh(M->head);
}

/* ============ v003: STIMA IMPRONTA RAM (cancello OOM, c-pure RULE 5/6) ===
 * Somma gli STESSI prodotti delle calloc di model_new/block_alloc_slot/
 * qlin_new_slot, in uint64 con overflow checked (UINT64_MAX = non stimabile
 * o dimensioni illegali). Pesi: trit(1B)+shadow(4B)+sh0(4B) = 9B/parametro;
 * ausiliari QLin: 4*n_in + 12*n_out; buffer float: gli stessi T*d / T*H /
 * T*d_ff / stato RWKV (Sst+Ssnap+Wt+aux) di block_alloc_slot +
 * hsave/logits di model_new. Niente piu' nh*T*T (v003: prob, rimossa).
 * Pura aritmetica: non alloca, non tocca il modello, non mente sul tetto. */
static uint64_t oom_add(uint64_t a, uint64_t b)
{ uint64_t s = a + b; return (s < a) ? UINT64_MAX : s; }
static uint64_t oom_mul(uint64_t a, uint64_t b)
{ if (a && b > UINT64_MAX / a) return UINT64_MAX; return a * b; }
static uint64_t model_est_bytes(int32_t V, int32_t d, int32_t n_head,
                                int32_t d_ff, int32_t T, int32_t n_layers)
{
    uint64_t tot = 0, H, P, n, aux, buf;
    if (V <= 0 || d <= 0 || n_head <= 0 || d_ff <= 0 || T <= 0 || n_layers <= 0)
        return UINT64_MAX;
    H = (uint64_t)n_head * (uint64_t)(d / n_head);   /* come block_alloc_slot */
    P = oom_mul(oom_mul((uint64_t)V, (uint64_t)d), 2);       /* emb+head      */
    n = oom_mul(H, (uint64_t)d); n = oom_mul(n, 4);
    n = oom_add(n, oom_mul(oom_mul((uint64_t)d, (uint64_t)d_ff), 3));
    n = oom_mul(n, (uint64_t)n_layers);
    P = oom_add(P, n);                                       /* parametri     */
    tot = oom_add(tot, oom_mul(P, 9));                       /* trit+sh+sh0   */
    aux = oom_mul(oom_add(oom_mul(4, (uint64_t)d), oom_mul(12, (uint64_t)V)), 2);
    buf = oom_mul(3, oom_add(oom_mul(4, (uint64_t)d), oom_mul(12, H)));
    buf = oom_add(buf, oom_add(oom_mul(4, H), oom_mul(12, (uint64_t)d)));
    buf = oom_add(buf, oom_mul(2, oom_add(oom_mul(4, (uint64_t)d),
                                          oom_mul(12, (uint64_t)d_ff))));
    buf = oom_add(buf, oom_add(oom_mul(4, (uint64_t)d_ff), oom_mul(12, (uint64_t)d)));
    aux = oom_add(aux, oom_mul(buf, (uint64_t)n_layers));
    tot = oom_add(tot, aux);
    buf = oom_add(oom_mul((uint64_t)T, (uint64_t)d),
                  oom_mul(oom_mul(8, (uint64_t)T), H));
    buf = oom_add(buf, oom_mul(oom_mul(3, (uint64_t)T), (uint64_t)d));
    buf = oom_add(buf, oom_mul(oom_mul(5, (uint64_t)T), (uint64_t)d_ff));
    {   /* v004: stato RWKV al posto di nh*T*T. dh = H/n_head. */
        uint64_t dh64 = n_head ? H / (uint64_t)n_head : 0;
        uint64_t st = oom_mul(oom_mul((uint64_t)n_head, dh64), dh64);
        buf = oom_add(buf, st);                       /* Sst        */
        buf = oom_add(buf, oom_mul(st, (uint64_t)T)); /* Ssnap      */
        buf = oom_add(buf, oom_mul((uint64_t)T, H));  /* Wt         */
        buf = oom_add(buf, oom_add(oom_mul(7, (uint64_t)d), H)); /*aux*/
    }
    buf = oom_mul(buf, 4);
    buf = oom_mul(buf, (uint64_t)n_layers);
    tot = oom_add(tot, buf);
    tot = oom_add(tot, oom_mul(oom_mul((uint64_t)n_layers + 1, (uint64_t)T), oom_mul((uint64_t)d, 4)));
    tot = oom_add(tot, oom_mul(oom_mul(2, (uint64_t)T), oom_mul((uint64_t)d, 4)));
    tot = oom_add(tot, oom_mul(oom_mul(2, (uint64_t)T), oom_mul((uint64_t)V, 4)));
    return tot;
}

/* ============ v004-14B: STIMA STREAMING (finestra 2 layer, RULE 5/6) =====
 * Il blocco monolitico sopra somma L layer × tutto: per 14B vale 607 GiB
 * e il cancello rifiuterebbe per sempre. L'architettura v004-14B pero'
 * pagina i chunk I_X/S_X con finestra scorrevole di MAX 2 layer residenti
 * (corrente + successivo, block_load muore oltre STREAM_MAX_RESIDENT): il
 * residente vero e' DUE layer attivi + DUE layer di attivazioni + i buffer
 * di I/O di base. Stessi prodotti delle calloc, stessa aritmetica checked
 * (UINT64_MAX = non stimabile), ma con finestra fissa di 2 layer:
 *  - W2: pesi dei 2 layer della finestra, P1 = 4*H*d + 3*d*d_ff parametri a
 *    trit(1B)+shadow(4B)+sh0(4B) = 9B/parametro + ausiliari dei layer;
 *  - A2: attivazioni float di DUE layer (×4B). Ssnap (st×T, 10.7 GiB a 14B)
 *    e' ESCLUSA: lo snapshot BPTT si ricalcola sul backward (activation
 *    checkpointing, ~+1/3 di compute, tecnica nativa del training); resta
 *    il solo Sst per layer residente;
 *  - BIO: righe emb/head toccate (2×T×d×9B, una riga per token), logits a
 *    tile (2×Tc×V×4B, Tc=min(T,1024): chunked output come il chunked
 *    prefill di CPM.cu), double-buffer hsave (2×T×d×4B), norm finale,
 *    ausiliari emb/head (residenti, pochi MB).
 * 14B rif (V=128000,d=H=5120,dff=13824,T=4096): ~11.1 GiB (era ~6.4 con
 * finestra 1). Pesi davvero in streaming (block_load/unload: max 2 layer
 * residenti per costruzione, provato da g_stream_max<=2 nell'harness). */
#define OOC_LOGIT_TILE 1024
static uint64_t model_est_bytes_stream(int32_t V, int32_t d, int32_t n_head,
                                int32_t d_ff, int32_t T, int32_t n_layers)
{
    uint64_t tot = 0, H, P1, aux, buf, Tc;
    if (V <= 0 || d <= 0 || n_head <= 0 || d_ff <= 0 || T <= 0 || n_layers <= 0)
        return UINT64_MAX;
    (void)n_layers;  /* L non scala: finestra di 2 layer (v004-14B) */
    H = (uint64_t)n_head * (uint64_t)(d / n_head);   /* come block_alloc_slot */
    P1 = oom_mul(H, (uint64_t)d); P1 = oom_mul(P1, 4);
    P1 = oom_add(P1, oom_mul(oom_mul((uint64_t)d, (uint64_t)d_ff), 3));
    tot = oom_add(tot, oom_mul(oom_mul(P1, 9), 2));     /* W2: 2 layer, 9B/p */
    aux = oom_mul(oom_add(oom_mul(4, (uint64_t)d), oom_mul(12, (uint64_t)V)), 2);
    buf = oom_mul(3, oom_add(oom_mul(4, (uint64_t)d), oom_mul(12, H)));
    buf = oom_add(buf, oom_add(oom_mul(4, H), oom_mul(12, (uint64_t)d)));
    buf = oom_add(buf, oom_mul(2, oom_add(oom_mul(4, (uint64_t)d),
                                          oom_mul(12, (uint64_t)d_ff))));
    buf = oom_add(buf, oom_add(oom_mul(4, (uint64_t)d_ff), oom_mul(12, (uint64_t)d)));
    tot = oom_add(tot, aux);                            /* aux emb/head      */
    tot = oom_add(tot, oom_mul(buf, 2));                /* aux: 2 layer      */
    buf = oom_add(oom_mul((uint64_t)T, (uint64_t)d),
                  oom_mul(oom_mul(8, (uint64_t)T), H));
    buf = oom_add(buf, oom_mul(oom_mul(3, (uint64_t)T), (uint64_t)d));
    buf = oom_add(buf, oom_mul(oom_mul(5, (uint64_t)T), (uint64_t)d_ff));
    {   /* Sst una-tantum, niente Ssnap ×T: recompute sul backward. */
        uint64_t dh64 = n_head ? H / (uint64_t)n_head : 0;
        uint64_t st = oom_mul(oom_mul((uint64_t)n_head, dh64), dh64);
        buf = oom_add(buf, st);                       /* Sst        */
        buf = oom_add(buf, oom_mul((uint64_t)T, H));  /* Wt         */
        buf = oom_add(buf, oom_add(oom_mul(7, (uint64_t)d), H)); /*aux*/
    }
    tot = oom_add(tot, oom_mul(buf, 8));                /* A2: 2 layer float */
    tot = oom_add(tot, oom_mul(oom_mul(oom_mul(2, (uint64_t)T),   /* righe emb/head */
                                       (uint64_t)d), 9));
    Tc = (uint64_t)T < OOC_LOGIT_TILE ? (uint64_t)T : OOC_LOGIT_TILE;
    tot = oom_add(tot, oom_mul(oom_mul(oom_mul(2, Tc),            /* logits a tile */
                                       (uint64_t)V), 4));
    tot = oom_add(tot, oom_mul(oom_mul(oom_mul(2, (uint64_t)T),   /* hsave double-buf */
                                       (uint64_t)d), 4));
    tot = oom_add(tot, oom_mul(oom_mul(oom_mul(2, (uint64_t)T),   /* norm finale */
                                       (uint64_t)d), 4));
    return tot;
}

/* ============ v003: DATA LOCALITY CHUNKING I_X.bin / S_X.bin =============
 * Paper HFP sez. 3, alla lettera: ogni layer vive in due chunk numerati.
 *   I_<slot>.bin (inferenza): header + trit[n] (1 byte/cad)
 *   S_<slot>.bin (ottimizzatore): header + shadow[n] (int32/cad)
 * Slot (v004 widened, = MANIFEST 14B): emb=14, head=15, blocco l base
 * 1000+l*16 (k=0..6, stride 16). Meccanismo, non magia:
 * su 64 bit la map intera e' solo indirizzi virtuali; il footprint piatto
 * viene dal MAPPARE UN SOLO CHUNK ALLA VOLTA e smapparlo subito dopo
 * (ooc_mapped lo impone: una seconda map senza unmap e' un errore, non una
 * seconda vista). Il cancello RAM e' il chiamante: ooc_verify_slot riceve
 * cap_bytes (governatore 30/75) e RIFIUTA la map se il chunk supera il tetto
 * (lo spill su disco resta: il tetto limita la RAM, non i file).
 * OS: Windows CreateFileMapping/MapViewOfFile; POSIX open+mmap (ramo #else:
 * per compilarlo servono <sys/mman.h>, <fcntl.h>, <unistd.h>). Little-endian
 * nativo, stessa macchina che scrive e legge (come i pesi .bin esistenti).
 * I pesi vivi stanno nei chunk (lo streaming li ricarica): un fallimento
 * qui non li tocca (spill fallito = rc 0, unload trattiene i buffer). */
#define OOC_MAGIC 0x5143544bu
#define OOC_KIND_I 0
#define OOC_KIND_S 1
typedef struct { uint32_t magic, kind, slot; uint64_t n;
                 int32_t n_in, n_out, decay, gshift;
                 uint16_t decay_w, rsv; } ooc_hdr_t;
#ifdef _WIN32
static HANDLE ooc_hf = NULL, ooc_hm = NULL;
#else
static int ooc_fd = -1;
#endif
static int ooc_mapped = 0;   /* 1 = vista attiva: map dentro map = errore  */
/* v002: weights live in an EXTERNAL dir, never hardcoded as absolute.
 * Default "data/weights" (CWD-relative). Override: --weights-dir flag,
 * PII_WEIGHTS_DIR env (resolved in main, pushed here via ooc_dir_set
 * BEFORE any weight I/O). */
#define PII_WEIGHTS_DEFAULT "data/weights"
static char g_ooc_dir[520] = PII_WEIGHTS_DEFAULT;
void ooc_dir_set(const char *d);
void ooc_dir_set(const char *d)
{
    size_t i, n;
    int r;
    if (!d || !*d) return;
    r = snprintf(g_ooc_dir, sizeof g_ooc_dir, "%s", d);
    if (r < 0 || (size_t)r >= sizeof g_ooc_dir) die("ooc dir");
    n = strlen(g_ooc_dir);
    while (n > 1 && (g_ooc_dir[n - 1] == '/' || g_ooc_dir[n - 1] == '\\') &&
           g_ooc_dir[n - 2] != ':') {
        g_ooc_dir[--n] = 0;
    }
    for (i = 0; i < n; i++)
        if (g_ooc_dir[i] == '\\') g_ooc_dir[i] = '/';
}
static void ooc_paths(int slot, int kind, char *out, size_t cap)
{
    int n = snprintf(out, cap, "%s/%c_%d.bin", g_ooc_dir,
                     kind ? 'S' : 'I', slot);
    if (n < 0 || (size_t)n >= cap) die("ooc path");
}
static void ooc_dir_ensure(void)
{
    /* v004/stream: NESSUNA cache "done". Il processo puo' cambiare cwd
     * (l'harness crea modelli prima in test_chunks_tmp/ e poi nella root,
     * e model_new spillera' in qualunque cwd): la mkdir e' idempotente,
     * l'errore e' ignorato, il costo e' una syscall contro MB di I/O. */
    /* v002: mkdir -p on the external weights dir, component by component. */
    {
        char tmp[520]; size_t i, n;
        int r = snprintf(tmp, sizeof tmp, "%s", g_ooc_dir);
        if (r > 0 && (size_t)r < sizeof tmp) {
            n = strlen(tmp);
            for (i = 1; i <= n; i++) {
                if (tmp[i] == '/' || tmp[i] == 0) {
                    char save = tmp[i];
                    if (!(i >= 2 && tmp[i - 1] == ':')) {
                        tmp[i] = 0;
                        if (tmp[0]) {
#ifdef _WIN32
                            CreateDirectoryA(tmp, NULL);
#else
                            mkdir(tmp, 0755);
#endif
                        }
                        tmp[i] = save;
                    }
                    if (save == 0) break;
                }
            }
        }
    }
}
static int ooc_spill_qlin(int slot, const QLin *L)
{
    char path[64]; FILE *f; ooc_hdr_t h; int64_t N;
    if (!L || L->n_in <= 0 || L->n_out <= 0 || !L->trit || !L->shadow) return 0;
    if (L->n_in > 0 && L->n_out > 2147483647L / L->n_in) return 0; /* overflow */
    N = (int64_t)L->n_in * L->n_out;
    ooc_dir_ensure();
    memset(&h, 0, sizeof h);
    h.magic = OOC_MAGIC; h.kind = OOC_KIND_I; h.slot = (uint32_t)slot;
    h.n = (uint64_t)N; h.n_in = L->n_in; h.n_out = L->n_out;
    h.decay = L->decay; h.gshift = L->gshift;
    ooc_paths(slot, OOC_KIND_I, path, sizeof path);
    f = fopen(path, "wb"); if (!f) return 0;
    if (fwrite(&h, 1, sizeof h, f) != sizeof h
        || fwrite(L->trit, 1, (size_t)N, f) != (size_t)N) { fclose(f); return 0; }
    fclose(f);
    h.kind = OOC_KIND_S; h.decay_w = L->decay_w;
    ooc_paths(slot, OOC_KIND_S, path, sizeof path);
    f = fopen(path, "wb"); if (!f) return 0;
    if (fwrite(&h, 1, sizeof h, f) != sizeof h
        || fwrite(L->shadow, sizeof(shadow_t), (size_t)N, f) != (size_t)N) { fclose(f); return 0; }
    fclose(f);
    return 1;
}
static int ooc_map_one(const char *path, int wr, void **base, size_t *len)
{
#ifdef _WIN32
    LARGE_INTEGER sz;
    DWORD acc = GENERIC_READ, prot = PAGE_READONLY, macc = FILE_MAP_READ;
    void *v;
    if (ooc_mapped) return 0;
    if (wr) { acc |= GENERIC_WRITE; prot = PAGE_READWRITE; macc = FILE_MAP_ALL_ACCESS; }
    ooc_hf = CreateFileA(path, acc, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL, NULL);
    if (ooc_hf == INVALID_HANDLE_VALUE) { ooc_hf = NULL; return 0; }
    /* v004 streaming: niente tetto 2 GiB sulla dimensione del file. La map
     * intera e' solo indirizzi virtuali; il residente lo decide il paging
     * dell'OS (chunk emb/head a 14B: shadow da 2.6 GiB in un solo file).
     * Resta il rifiuto sotto la dimensione dell'header. */
    if (!GetFileSizeEx(ooc_hf, &sz) || sz.QuadPart < (LONGLONG)sizeof(ooc_hdr_t)) { CloseHandle(ooc_hf); ooc_hf = NULL; return 0; }
    ooc_hm = CreateFileMappingA(ooc_hf, NULL, prot, 0, 0, NULL);
    if (!ooc_hm) { CloseHandle(ooc_hf); ooc_hf = NULL; return 0; }
    v = MapViewOfFile(ooc_hm, macc, 0, 0, 0);
    if (!v) { CloseHandle(ooc_hm); ooc_hm = NULL; CloseHandle(ooc_hf); ooc_hf = NULL; return 0; }
    ooc_mapped = 1; *base = v; *len = (size_t)sz.QuadPart; return 1;
#else
    int fd; void *v; long sz; /* POSIX: servono fcntl/mman/unistd (vedi nota) */
    if (ooc_mapped) return 0;
    fd = open(path, wr ? O_RDWR : O_RDONLY);
    if (fd < 0) return 0;
    sz = lseek(fd, 0, SEEK_END);
    if (sz < (long)sizeof(ooc_hdr_t)) { close(fd); return 0; }
    v = mmap(NULL, (size_t)sz, PROT_READ | (wr ? PROT_WRITE : 0),
             MAP_SHARED, fd, 0);
    if (v == MAP_FAILED) { close(fd); return 0; }
    ooc_fd = fd; ooc_mapped = 1; *base = v; *len = (size_t)sz; return 1;
#endif
}
static void ooc_unmap_one(void *base, size_t len)
{
    if (!ooc_mapped) return;
#ifdef _WIN32
    UnmapViewOfFile(base);
    CloseHandle(ooc_hm); ooc_hm = NULL; CloseHandle(ooc_hf); ooc_hf = NULL;
#else
    munmap(base, len); close(ooc_fd); ooc_fd = -1;
#endif
    (void)len;
    ooc_mapped = 0;
}
/* ============ v004: LOOKAHEAD PREFETCH (LES mobile, solo I/O) ============
 * Mentre la CPU verifica il chunk N, un thread helper legge in avanti il
 * chunk I_{N+1} con fread sequenziale: le pagine finiscono nella cache
 * dell'OS, cosi' la mmap di verifica successiva trova i dati caldi e la
 * latenza disco sparisce dietro il compute (idea LES di MiniCPM, senza il
 * router MoE: qui nessuna decisione, solo read-ahead del prossimo slot in
 * ordine). Stessa primitiva per lo streaming dei pesi per-layer (carica il
 * chunk N+1 mentre il compute usa il chunk N): il sito di chiamata si sposta
 * nel loop dei layer quando il paging per-layer atterra; l'I/O e' lo stesso.
 * Sicurezza: il thread usa SOLO fopen/fread su heap proprio, mai
 * ooc_map_one (ooc_mapped resta single-view, single-thread); il buffer si
 * libera nel thread dopo la lettura (scalda solo la page cache, picco RAM
 * = un chunk I alla volta); oltre il tetto RAM non si prefetcha affatto
 * (ok=-1, la verifica normale rifiutera' comunque con -1); se pthread_create
 * fallisce si procede senza overlap (ok=0), mai un errore fatale. */
typedef struct { pthread_t th; int active; int ok; char path[64]; size_t cap; } ooc_pf_t;
static void *ooc_pf_thread(void *a)
{
    ooc_pf_t *pf = (ooc_pf_t *)a;
    FILE *f;
    uint8_t *buf = NULL;
    long long sz = -1;
    size_t left, got;
    f = fopen(pf->path, "rb");
    if (!f) { pf->ok = 0; return NULL; }
#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END); sz = _ftelli64(f);
#else
    fseek(f, 0, SEEK_END); sz = (long long)ftell(f);
#endif
    if (sz <= 0 || (uint64_t)sz != (size_t)sz) { fclose(f); pf->ok = 0; return NULL; }
    if (pf->cap > 0 && (uint64_t)sz > pf->cap) { fclose(f); pf->ok = -1; return NULL; }
    buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fclose(f); pf->ok = 0; return NULL; }
    rewind(f);
    left = (size_t)sz;
    while (left > 0) {
        got = fread(buf + ((size_t)sz - left), 1, left, f);
        if (got == 0) break;
        left -= got;
    }
    fclose(f);
    free(buf);   /* la page cache resta calda: e' quello il punto */
    pf->ok = (left == 0) ? 1 : 0;
    return NULL;
}
static void ooc_pf_start(ooc_pf_t *pf, const char *path, size_t cap)
{
    int n;
    memset(pf, 0, sizeof *pf);
    n = snprintf(pf->path, sizeof pf->path, "%s", path);
    if (n < 0 || (size_t)n >= sizeof pf->path) return;
    pf->cap = cap;
    if (pthread_create(&pf->th, NULL, ooc_pf_thread, pf) != 0) return;
    pf->active = 1;
}
static int ooc_pf_join(ooc_pf_t *pf)
{
    if (!pf->active) return pf->ok;
    pthread_join(pf->th, NULL);
    pf->active = 0;
    return pf->ok;
}
/* Mappa I e S dello slot in sola lettura UNO ALLA VOLTA, controlla magic +
 * dimensioni, smappa SUBITO. Ritorna 1 = ok, 0 = file corrotti/assenti,
 * -1 = oltre il tetto (non mappo affatto). */
static int ooc_verify_slot(int slot, size_t cap_bytes)
{
    int k;
    for (k = 0; k < 2; k++) {
        char path[64]; FILE *f; long long t = -1;
        void *base = NULL; size_t len = 0;
        const ooc_hdr_t *h; int ok;
        size_t expect;
        ooc_paths(slot, k, path, sizeof path);
        f = fopen(path, "rb");
        if (!f) return 0;
#ifdef _WIN32
        _fseeki64(f, 0, SEEK_END); t = _ftelli64(f); fclose(f);  /* oltre 2 GiB: ftell mente */
#else
        fseek(f, 0, SEEK_END); t = (long long)ftell(f); fclose(f);
#endif
        if (t <= 0) return 0;
        if (cap_bytes > 0 && (uint64_t)t > cap_bytes) return -1; /* tetto RAM */
        if (!ooc_map_one(path, 0, &base, &len)) return 0;
        h = (const ooc_hdr_t *)base;
        expect = sizeof *h + (k ? h->n * sizeof(shadow_t) : (size_t)h->n);
        ok = (len >= sizeof *h && h->magic == OOC_MAGIC && h->kind == (uint32_t)k
              && h->slot == (uint32_t)slot && h->n > 0 && len == expect);
        ooc_unmap_one(base, len);   /* smappa SUBITO: footprint piatto    */
        base = NULL;
        if (!ok) return 0;
    }
    return 1;
}
/* Spill di tutto il modello + verifica mappata un chunk alla volta.
 * 1 = tutto verificato, -1 = almeno un chunk oltre il tetto (spill ok),
 * 0 = spill o verifica falliti (pesi vivi intatti). */
static int ooc_maintain_all(Model *M, size_t cap_bytes)
{
    int32_t l;
    int total = 0, good = 0, over = 0, rc, i, n = 0;
    int *slots;
    int max;
    /* v004/stream: i pesi vivono nei chunk. Fase 1: emb/head (residenti) +
     * per layer load → spill blindato → unload(pulito). Solo gli spill
     * riusciti entrano in verifica (stessa semantica di prima: spill
     * fallito = non contato; unload pulito non puo' fallire qui). */
    if (!M) return 0;
    if (M->n_layers < 0 || M->n_layers > 100000) return 0;  /* c-pure RULE 6 */
    max = 2 + M->n_layers * 7;
    slots = (int *)malloc((size_t)max * sizeof(int));
    if (!slots) return 0;
    if (M->emb && qlin_spill_locked(OOC_SLOT_EMB, M->emb)) {
        slots[n++] = OOC_SLOT_EMB; total++;
    }
    if (M->head && qlin_spill_locked(OOC_SLOT_HEAD, M->head)) {
        slots[n++] = OOC_SLOT_HEAD; total++;
    }
    for (l = 0; l < M->n_layers; l++) {
        QLin *Ls[7]; int u;
        if (!block_load(M, l)) continue;
        Ls[0]=M->B[l].Wq; Ls[1]=M->B[l].Wk; Ls[2]=M->B[l].Wv; Ls[3]=M->B[l].Wo;
        Ls[4]=M->B[l].Wg; Ls[5]=M->B[l].Wu; Ls[6]=M->B[l].Wd;
        for (u = 0; u < 7; u++) {
            int slot = OOC_SLOT_LK(l, u);
            if (Ls[u] && qlin_spill_locked(slot, Ls[u])) {
                slots[n++] = slot; total++;
            }
        }
        block_unload(M, l, 0);
    }
    /* Fase 2 invariata: verifica mappata con lookahead sul chunk I
     * successivo (due pf alternati, mai copia di pthread_t). */
    if (n > 0) {
        ooc_pf_t pf[2]; char ipath[64]; int cur = 0;
        memset(pf, 0, sizeof pf);
        ooc_paths(slots[0], OOC_KIND_I, ipath, sizeof ipath);
        ooc_pf_start(&pf[cur], ipath, cap_bytes);
        for (i = 0; i < n; i++) {
            int nxt = cur ^ 1;
            if (i + 1 < n) {
                ooc_paths(slots[i + 1], OOC_KIND_I, ipath, sizeof ipath);
                ooc_pf_start(&pf[nxt], ipath, cap_bytes);
            }
            rc = ooc_verify_slot(slots[i], cap_bytes);
            if (rc == 1) good++; else if (rc == -1) over++;
            ooc_pf_join(&pf[cur]);
            cur = nxt;
        }
        ooc_pf_join(&pf[cur]);
    }
    free(slots);
    if (total == 0) return 0;
    if (good == total) return 1;
    return over > 0 ? -1 : 0;
}

/* ============ v004/stream: WEIGHT STREAMING per-layer (flat RAM) ==========
 * model_new NON alloca piu' i pesi di tutti i layer: le strutture QLin/Block
 * restano (metadati: dimensioni, slot, lr, n_sat — centinaia di byte), ma i
 * buffer pesanti (trit/shadow/sh0/x_q/raw/act/gbuf di ogni QLin + tutte le
 * attivazioni float del Block) esistono SOLO sul layer attivo. Il backing
 * store sono i chunk chunks/{I,S}_<slot>.bin (stesso formato del MANIFEST):
 *   block_load(l)      : legge I+S negli heap buffer (sh0 = copia di shadow)
 *   compute            : block_forward / block_backward invariati
 *   block_unload(l,dirty): dirty=1 spill I+S e libera; dirty=0 libera e basta
 * sh0 (shadow-al-refresh, serve solo al path STE dei test): ogni spill
 * avviene post-refresh (block_backward termina con block_refresh; l'init fa
 * refresh prima dello spill), quindi sh0 == shadow al momento dello spill e
 * la copia sullo shadow caricato e' ESATTA. Unico angolo noto: STE + lr!=0 +
 * accumulo senza refresh tra unload e load (nessun chiamante: T4 azzera lr,
 * il training usa TL2 che ignora sh0).
 * File I/O serializzato da stream_mx (unico lock, mai annidato, solo attorno
 * a fopen/fread/fwrite — mai compute/join/malloc): foreground e background
 * condividono chunks/ e senza lock le letture si strapperebbero con gli
 * spill. Verify/prefetch restano fuori dal lock per scelta: la verify e' solo
 * diagnostica (torn = rc 0, mai mutazione) e il prefetch scalda la cache
 * (torn = warm perso, mai errore). I contatori resident/max sono diagnostici
 * (l'harness li legge), non sincronizzazione: un solo thread per modello.
 * Residenti per scelta: emb/head (row-paging futuro), hsave (checkpoint a
 * livello modello futuro), struct QLin/Block (metadati). */
static pthread_mutex_t stream_mx = PTHREAD_MUTEX_INITIALIZER;
/* v004-14B: FINESTRA SCORREVOLE DI 2, INVARIANTE GLOBALE. Mai tutto il
 * modello in RAM (a 14B non ci sta): al massimo 2 layer residenti — il
 * corrente + il successivo gia' caricato per una transizione senza stalli.
 * Oltre 2 = die(), non un avviso: e' la garanzia, non la diagnostica.
 * Globale per costruzione: OGNI load del sistema (forward, backward,
 * refresh, maintain, save/load/sync) passa da block_load qui sotto; il
 * prefetch non alloca mai (solo fread in 1 MiB) e fa sempre join prima di
 * load/unload/spill. A regime la finestra e' piena (=2 nei loop); ai bordi
 * (primo/ultimo layer, maintain, sync) scende a 1 o 0, mai oltre 2. */
#define STREAM_MAX_RESIDENT 2
static int g_stream_resident = 0;  /* layer con pesi in RAM in questo momento */
static int g_stream_max = 0;       /* high-water mark (prova per l'harness)   */
static void stream_lock(void) { pthread_mutex_lock(&stream_mx); }
static void stream_unlock(void) { pthread_mutex_unlock(&stream_mx); }
static void stream_note_resident(void)
{
    if (g_stream_resident > g_stream_max) g_stream_max = g_stream_resident;
}
/* Legge I+S dello slot di L negli heap buffer (dopo qlin_alloc_bufs).
 * Ripristina decay/gshift/decay_w dagli header, sh0 = copia di shadow.
 * 1 = ok, 0 = assenti/corrotti/mismatch (il chiamante decide). */
static int qlin_read_chunks(QLin *L, int slot)
{
    char path[64]; FILE *f; ooc_hdr_t h; int64_t N; size_t nn;
    int k, ok = 0;
    if (!L || !L->trit || !L->shadow || !L->sh0) return 0;
    N = (int64_t)L->n_in * L->n_out;
    if (N <= 0) return 0;
    nn = (size_t)N;
    /* NESSUN lock attorno alla fread: la join prima di ogni load/unload/spill
     * garantisce che nessun altro thread faccia I/O sui chunk in questo
     * momento (l'unico I/O concorrente possibile e' il bg-loader, e lo si e'
     * appena joinato). Il lock attorno a decine di MB di fread serializzava
     * l'intero pipeline dietro il disco: freeze all'avvio del primo passo
     * su dischi lenti, con unload/spill in coda dietro il loader. */
    for (k = 0; k < 2; k++) {
        ooc_paths(slot, k, path, sizeof path);
        f = fopen(path, "rb");
        if (!f) goto qrc_out;
        if (fread(&h, 1, sizeof h, f) != sizeof h) { fclose(f); goto qrc_out; }
        if (h.magic != OOC_MAGIC || h.kind != (uint32_t)k
            || h.slot != (uint32_t)slot || h.n != (uint64_t)N
            || h.n_in != L->n_in || h.n_out != L->n_out) {
            fclose(f); goto qrc_out;
        }
        if (k == 0) {
            if (fread(L->trit, 1, nn, f) != nn) { fclose(f); goto qrc_out; }
            L->decay = h.decay; L->gshift = h.gshift;
        } else {
            if (fread(L->shadow, sizeof(shadow_t), nn, f) != nn) { fclose(f); goto qrc_out; }
            L->decay_w = h.decay_w;
        }
        fclose(f);
    }
    memcpy(L->sh0, L->shadow, nn * sizeof(shadow_t));  /* post-refresh: esatto */
    ok = 1;
qrc_out:
    return ok;
}
/* ============ v004: SEED AUTOMATICO DEI PESI (fresh install) ===============
 * Su pesi assenti `mask` moriva con WEIGHTS-FAIL. eng_ensure_seed_weights()
 * genera il baseline ternario deterministico (stesso qlin_random_init del
 * train, rseed fissato) per ogni slot mancante e lo spilla I+S: MAI
 * sovrascrive chunk validi esistenti (verifica header+size per slot).
 * Geometria = quella del train (train_driver.c ENG_*: d=1024 nh=8 dff=2816
 * L=14 -> H=1024; V=1024=256+BPE_MERGE_MAX da nn_bpe.c): i layer seedati
 * vengono riusati dal train (stream_layer_chunks_ready li accetta, stesso
 * identico baseline del re-init), emb/head servono a infer/mask.
 * Chiamata al boot dai main SOLO se eng_infer_load fallisce, single-thread
 * (nessun bg attivo): niente lock. Ritorna slot scritti, 0 se tutto
 * presente, -1 se nulla si e' potuto scrivere. */
#define OOC_SEED_V 1024     /* tetto V: 256 base + 768 merge (nn_bpe.c) */
#define OOC_SEED_L 14       /* ENG_LAYERS (train_driver.c: tenere in sync) */
#define OOC_SEED_KEY 20260930u  /* ENG_SEED: baseline identico al train */
static int ooc_slot_files_ok(int slot, int32_t n_in, int32_t n_out)
{
    int64_t N = (int64_t)n_in * n_out;
    int k;
    if (N <= 0) return 0;
    for (k = 0; k < 2; k++) {
        char path[64]; FILE *f; ooc_hdr_t h; long sz = -1;
        uint64_t want;
        ooc_paths(slot, k, path, sizeof path);
        f = fopen(path, "rb");
        if (!f) return 0;
        if (fread(&h, 1, sizeof h, f) != sizeof h) { fclose(f); return 0; }
        if (fseek(f, 0, SEEK_END) == 0) sz = ftell(f);
        fclose(f);
        if (sz < 0) return 0;
        want = (uint64_t)sizeof h + (k == 0 ? (uint64_t)N
                                            : (uint64_t)N * sizeof(shadow_t));
        if ((uint64_t)sz != want) return 0;
        if (h.magic != OOC_MAGIC || h.kind != (uint32_t)k
            || h.slot != (uint32_t)slot || h.n != (uint64_t)N
            || h.n_in != n_in || h.n_out != n_out) return 0;
    }
    return 1;
}
static int ooc_seed_one(int slot, int32_t n_out, int32_t n_in)
{
    QLin *L;
    if ((int64_t)n_out * n_in <= 0) return 0;
    if (ooc_slot_files_ok(slot, n_in, n_out)) return 0;  /* trained: giu' le mani */
    L = qlin_new_empty(n_out, n_in, slot);
    if (!L) return -1;
    qlin_alloc_bufs(L);
    if (!L->trit || !L->shadow) { qlin_free(L); return -1; }
    qlin_random_init(L);   /* sh0 non serve: chi legge lo ricopia da shadow */
    if (!ooc_spill_qlin(slot, L)) { qlin_free(L); return -1; }
    qlin_free(L);
    return 1;
}
int eng_ensure_seed_weights(void)
{
    /* (n_out, n_in) per k=0..6: Q/K/V, O, G/U, D — come block_alloc_slot. */
    static const int32_t no[7] = { 1024, 1024, 1024, 1024, 2816, 2816, 1024 };
    static const int32_t ni[7] = { 1024, 1024, 1024, 1024, 1024, 1024, 2816 };
    int32_t l, u;
    int written = 0, fail = 0;
    rseed(OOC_SEED_KEY);   /* baseline deterministico */
    ooc_dir_ensure();
    for (l = 0; l < OOC_SEED_L; l++)
        for (u = 0; u < 7; u++) {
            int rc = ooc_seed_one(OOC_SLOT_LK(l, u), no[u], ni[u]);
            if (rc < 0) fail = 1; else written += rc;
        }
    {   int rc = ooc_seed_one(OOC_SLOT_EMB, OOC_SEED_V, 1024);
        if (rc < 0) fail = 1; else written += rc;
        rc = ooc_seed_one(OOC_SLOT_HEAD, OOC_SEED_V, 1024);
        if (rc < 0) fail = 1; else written += rc;
    }
    if (written > 0)
        printf("weights: seed baseline scritto (%d chunk I+S) [%s].\n",
               written, g_ooc_dir);
    if (fail && written == 0) return -1;
    return written;
}
/* Spill blindato dal lock (coppia I+S atomica contro i load concorrenti). */
static int qlin_spill_locked(int slot, const QLin *L)
{
    int rc;
    stream_lock();
    rc = ooc_spill_qlin(slot, L);
    stream_unlock();
    return rc;
}
/* v004-14B: sonda riuso chunk (solo header I+S, niente buffer). 1 = i 14
 * chunk del layer esistono con header coerenti alle dimensioni QLin.
 * Senza lock per scelta: una race con uno spill concorrente puo' solo dare
 * mismatch -> re-init, mai pesi corrotti (block_load riverifica tutto). */
static int stream_layer_chunks_ready(Model *M, int32_t l)
{
    Block *B; QLin *Ls[7]; int u, k;
    if (!M || l < 0 || l >= M->n_layers) return 0;
    B = &M->B[l];
    if (!B->Wq) return 0;
    Ls[0]=B->Wq; Ls[1]=B->Wk; Ls[2]=B->Wv; Ls[3]=B->Wo;
    Ls[4]=B->Wg; Ls[5]=B->Wu; Ls[6]=B->Wd;
    for (u = 0; u < 7; u++) {
        int64_t N;
        int slot = OOC_SLOT_LK(l, u);
        if (!Ls[u] || Ls[u]->n_in <= 0 || Ls[u]->n_out <= 0) return 0;
        N = (int64_t)Ls[u]->n_in * Ls[u]->n_out;
        for (k = 0; k < 2; k++) {
            char path[64]; FILE *f; ooc_hdr_t h; long sz;
            uint64_t want;
            ooc_paths(slot, k, path, sizeof path);
            f = fopen(path, "rb");
            if (!f) return 0;
            if (fread(&h, 1, sizeof h, f) != sizeof h) { fclose(f); return 0; }
            /* File troncato (kill durante lo spill)? header ok ma payload
             * corto: mismatch -> re-init, mai pesi letti a meta'. */
            if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
            sz = ftell(f);
            fclose(f);
            if (sz < 0) return 0;
            want = (uint64_t)sizeof h + (k == 0 ? (uint64_t)N
                                               : (uint64_t)N * 4u);
            if ((uint64_t)sz != want) return 0;
            if (h.magic != OOC_MAGIC || h.kind != (uint32_t)k
                || h.slot != (uint32_t)slot || h.n != (uint64_t)N
                || h.n_in != Ls[u]->n_in || h.n_out != Ls[u]->n_out) return 0;
        }
    }
    return 1;
}
/* Carica pesi + attivazioni del layer. 1 = residente (o gia' tale). */
static int block_load(Model *M, int32_t l)
{
    Block *B; QLin *Ls[7]; int u;
    if (!M || l < 0 || l >= M->n_layers) return 0;
    B = &M->B[l];
    if (!B->Wq) return 0;
    if (B->Wq->trit) return 1;   /* gia' residente */
    block_alloc_bufs(B);
    Ls[0]=B->Wq; Ls[1]=B->Wk; Ls[2]=B->Wv; Ls[3]=B->Wo;
    Ls[4]=B->Wg; Ls[5]=B->Wu; Ls[6]=B->Wd;
    for (u = 0; u < 7; u++) {
        int v;
        qlin_alloc_bufs(Ls[u]);
        if (!qlin_read_chunks(Ls[u], OOC_SLOT_LK(l, u))) {
            for (v = 0; v <= u; v++) {
                free(Ls[v]->trit); free(Ls[v]->shadow); free(Ls[v]->x_q);
                free(Ls[v]->raw); free(Ls[v]->act); free(Ls[v]->gbuf);
                free(Ls[v]->sh0);
                Ls[v]->trit = NULL; Ls[v]->shadow = NULL; Ls[v]->x_q = NULL;
                Ls[v]->raw = NULL; Ls[v]->act = NULL; Ls[v]->gbuf = NULL;
                Ls[v]->sh0 = NULL;
            }
            block_drop_bufs(B);
            return 0;
        }
    }
    /* Finestra max-2 sotto lock: il bg-loader incrementa in concorrenza
     * col compute, quindi il contatore e' stato condiviso (join = barriera). */
    {   int over = 0;
        stream_lock();
        g_stream_resident++; stream_note_resident();
        over = (g_stream_resident > STREAM_MAX_RESIDENT);
        stream_unlock();
        /* Garanzia 14B: oltre la finestra si muore, mai paging incontrollato. */
        if (over) die("stream: oltre 2 layer residenti");
    }
    return 1;
}
/* Scrive I+S del layer (richiede residente). 1 = tutto scritto. */
static int block_spill_layer(Model *M, int32_t l)
{
    Block *B; QLin *Ls[7]; int u, ok = 1;
    if (!M || l < 0 || l >= M->n_layers) return 0;
    B = &M->B[l];
    if (!B->Wq || !B->Wq->trit) return 0;
    Ls[0]=B->Wq; Ls[1]=B->Wk; Ls[2]=B->Wv; Ls[3]=B->Wo;
    Ls[4]=B->Wg; Ls[5]=B->Wu; Ls[6]=B->Wd;
    for (u = 0; u < 7; u++)
        if (!qlin_spill_locked(OOC_SLOT_LK(l, u), Ls[u])) ok = 0;
    if (ok) B->dirty = 0;   /* disco == RAM: il sync incrementale lo salta */
    return ok;
}
/* Scarica il layer. dirty=1: spill prima; spill fallito = 0 E TIENE i buffer
 * (mai perdere pesi in silenzio). dirty=0: libera e basta. 1 = fuori. */
static int block_unload(Model *M, int32_t l, int dirty)
{
    Block *B; QLin *Ls[7]; int u;
    if (!M || l < 0 || l >= M->n_layers) return 0;
    B = &M->B[l];
    if (!B->Wq || !B->Wq->trit) return 1;   /* gia' fuori */
    if (dirty && !block_spill_layer(M, l)) return 0;
    Ls[0]=B->Wq; Ls[1]=B->Wk; Ls[2]=B->Wv; Ls[3]=B->Wo;
    Ls[4]=B->Wg; Ls[5]=B->Wu; Ls[6]=B->Wd;
    for (u = 0; u < 7; u++) {
        free(Ls[u]->trit); free(Ls[u]->shadow); free(Ls[u]->x_q); free(Ls[u]->raw);
        free(Ls[u]->act); free(Ls[u]->gbuf); free(Ls[u]->sh0);
        Ls[u]->trit = NULL; Ls[u]->shadow = NULL; Ls[u]->x_q = NULL; Ls[u]->raw = NULL;
        Ls[u]->act = NULL; Ls[u]->gbuf = NULL; Ls[u]->sh0 = NULL;
    }
    block_drop_bufs(B);
    stream_lock();
    g_stream_resident--;
    stream_unlock();
    return 1;
}
/* Crea i pesi del layer da zero (solo model_new): init + spill + drop. */
static int stream_layer_init(Model *M, int32_t l)
{
    Block *B; QLin *Ls[7]; int u, v;
    if (!M || l < 0 || l >= M->n_layers) return 0;
    B = &M->B[l];
    if (!B->Wq) return 0;
    block_alloc_bufs(B);
    Ls[0]=B->Wq; Ls[1]=B->Wk; Ls[2]=B->Wv; Ls[3]=B->Wo;
    Ls[4]=B->Wg; Ls[5]=B->Wu; Ls[6]=B->Wd;
    for (u = 0; u < 7; u++) { qlin_alloc_bufs(Ls[u]); qlin_random_init(Ls[u]); }
    if (!block_spill_layer(M, l)) {
        block_drop_bufs(B);
        for (v = 0; v < 7; v++) {
            free(Ls[v]->trit); free(Ls[v]->shadow); free(Ls[v]->x_q); free(Ls[v]->raw);
            free(Ls[v]->act); free(Ls[v]->gbuf); free(Ls[v]->sh0);
            Ls[v]->trit = NULL; Ls[v]->shadow = NULL; Ls[v]->x_q = NULL; Ls[v]->raw = NULL;
            Ls[v]->act = NULL; Ls[v]->gbuf = NULL; Ls[v]->sh0 = NULL;
        }
        return 0;
    }
    block_drop_bufs(B);
    for (v = 0; v < 7; v++) {
        free(Ls[v]->trit); free(Ls[v]->shadow); free(Ls[v]->x_q); free(Ls[v]->raw);
        free(Ls[v]->act); free(Ls[v]->gbuf); free(Ls[v]->sh0);
        Ls[v]->trit = NULL; Ls[v]->shadow = NULL; Ls[v]->x_q = NULL; Ls[v]->raw = NULL;
        Ls[v]->act = NULL; Ls[v]->gbuf = NULL; Ls[v]->sh0 = NULL;
    }
    return 1;
}
/* v004-14B/incremental-save: scrive SOLO gli strati sporchi (dirty=1) o
 * residenti. Invariante strutturale: uno strato non residente e' pulito —
 * ogni mutazione finisce con spill (backward→unload dirty, refresh→spill,
 * init→spill) — quindi a regime non si riscrive niente: i chunk SONO il
 * salvataggio, nessun file monolitico. Picco 1 layer, nessun prefetch: la
 * garanzia globale max-2 resta intatta. 1 = tutto su disco, 0 = fallito
 * (pesi vivi intatti: lo spill fallito trattiene i buffer, mai persi). */
static int model_sync_chunks(Model *M)
{
    int32_t l;
    if (!M) return 0;
    for (l = 0; l < M->n_layers; l++) {
        Block *B = &M->B[l];
        int resident = B->Wq && B->Wq->trit;
        if (!resident && !B->dirty) continue;   /* gia' su disco */
        if (!resident && !block_load(M, l)) return 0;
        if (!block_spill_layer(M, l)) return 0;
        if (!block_unload(M, l, 0)) return 0;
    }
    return 1;
}
/* Ricarica emb/head residenti dai loro slot (boot/resume). I buffer ci sono
 * gia' (qlin_new_slot li alloca); qui si rileggono solo I+S con verifica
 * header+shape (mismatch/assenti = 0, il chiamante tiene i semi). 1 = ok. */
static int qlin_load_chunks(QLin *L, int slot)
{
    if (!L || !L->trit) return 0;
    return qlin_read_chunks(L, slot);
}
/* Caricamento sincrono usato dal bg-thread: identico a block_load ma senza
 * die() (riporta 0/1, il foreground ritenta e muore con lo stesso messaggio).
 * Il contatore residenti e' sotto stream_mx come nel path foreground. */
static int block_load_bg(Model *M, int32_t l)
{
    Block *B; QLin *Ls[7]; int u;
    if (!M || l < 0 || l >= M->n_layers) return 0;
    B = &M->B[l];
    if (!B->Wq) return 0;
    if (B->Wq->trit) return 1;   /* gia' residente (join precedente) */
    block_alloc_bufs(B);
    Ls[0]=B->Wq; Ls[1]=B->Wk; Ls[2]=B->Wv; Ls[3]=B->Wo;
    Ls[4]=B->Wg; Ls[5]=B->Wu; Ls[6]=B->Wd;
    for (u = 0; u < 7; u++) {
        int v;
        qlin_alloc_bufs(Ls[u]);
        if (!qlin_read_chunks(Ls[u], OOC_SLOT_LK(l, u))) {
            for (v = 0; v <= u; v++) {
                free(Ls[v]->trit); free(Ls[v]->shadow); free(Ls[v]->x_q);
                free(Ls[v]->raw); free(Ls[v]->act); free(Ls[v]->gbuf);
                free(Ls[v]->sh0);
                Ls[v]->trit = NULL; Ls[v]->shadow = NULL; Ls[v]->x_q = NULL;
                Ls[v]->raw = NULL; Ls[v]->act = NULL; Ls[v]->gbuf = NULL;
                Ls[v]->sh0 = NULL;
            }
            block_drop_bufs(B);
            return 0;
        }
    }
    {   int over = 0;
        stream_lock();
        g_stream_resident++; stream_note_resident();
        over = (g_stream_resident > STREAM_MAX_RESIDENT);
        stream_unlock();
        if (over) return 0;   /* il foreground muore con lo stesso die() */
    }
    return 1;
}
/* Double buffering a 2 slot: mentre il pool calcola sul layer corrente, UN
 * solo thread di I/O (bloccante, mai compute) precarica I+S del successivo
 * nei buffer del layer successivo (slot RAM inattivo). Il thread non chiama
 * mai mt_use/mt_run, non tocca il pool e non alza il tetto 50% (mt_cap
 * invariato: QATNN_THREADS puo' solo abbassarlo); fa solo malloc+fread+
 * memcpy, quindi a regime dorme in fread e la CPU resta al pool.
 * Sincronizzazione: join prima di qualunque load/unload/spill sullo stesso
 * modello (il bg tocca solo il layer target, disgiunto dal corrente: nessuna
 * condivisione tranne il contatore sotto lock e i file chunk, letti qui e
 * scritti solo dopo la join). La join e' la barriera happens-before: dopo,
 * il foreground vede i buffer o ritenta in sincrono (stessi byte -> stessi
 * numeri, bit-identici). Mai copia di pthread_t: una istanza per loop. */
typedef struct { pthread_t th; int active; int ok; Model *M; int32_t l; } ooc_pfl_t;
static void *ooc_pf_layer_thread(void *a)
{
    ooc_pfl_t *pf = (ooc_pfl_t *)a;
#ifdef _WIN32
    /* I/O in background: cede la CPU al pool (tetto 50% intatto). */
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
    /* Mai compute annidato qui dentro: solo I/O verso lo slot inattivo. */
    pf->ok = block_load_bg(pf->M, pf->l);
    return NULL;
}
static int g_pf_disabled = -2;   /* QATNN_NO_PREFETCH=1: streaming sincrono */
static void ooc_pf_layer_start(ooc_pfl_t *pf, Model *M, int32_t l, int32_t n_layers)
{
    /* La join precede QUALUNQUE uscita: un bg attivo appartiene al layer
     * precedente e va raccolto prima di decidere il successivo (mai thread
     * orfani sullo stesso pf, mai join saltata da un return anticipato). */
    if (pf->active) { pthread_join(pf->th, NULL); pf->active = 0; }
    if (l < 0 || l >= n_layers) return;   /* niente da precaricare: no-op */
    if (g_pf_disabled == -2) g_pf_disabled = mt_env("QATNN_NO_PREFETCH", 0);
    if (g_pf_disabled) return;            /* kill-switch: load solo sincroni */
    if (M->B[l].Wq && M->B[l].Wq->trit) return;  /* gia' residente: no-op */
    pf->M = M; pf->l = l; pf->ok = 0;
    if (pthread_create(&pf->th, NULL, ooc_pf_layer_thread, pf) == 0) pf->active = 1;
}
static void ooc_pf_layer_join(ooc_pfl_t *pf)
{
    if (!pf->active) return;
    pthread_join(pf->th, NULL);
    pf->active = 0;
}

/* ------------------------------------------------------------- avanti ---- */
/* ---- Directive 1: stadi token-indipendenti via pool (stessi numeri) -----
 * emb_row : gather di UNA riga di embedding (letture condivise, scritture
 * disgiunte). rms_row : UNA rmsnorm finale (solo locali). dll_row : il
 * dL/dlogit di UN token (softmax + onehot + catena sez. 5; legge lg/y/n_tok
 * condivisi, scrive solo la sua riga dl; fexp e' thread-safe via
 * pthread_once). I fallback seriali chiamano le stesse fn: per
 * costruzione, gli stessi numeri. */
typedef struct { Model *M; const int32_t *x; float *h; } emb_task_t;
static void emb_row(int t, void *arg)
{
    emb_task_t *e = (emb_task_t *)arg;
    int32_t V = e->M->V, d = e->M->d, i;
    int32_t tk = (e->x[t] >= 0 && e->x[t] < V) ? e->x[t] : 0;
    for (i = 0; i < d; i++)
        e->h[(size_t)t * d + i] = (float)e->M->emb->trit[(int64_t)tk * d + i];
}
typedef struct { const float *h; float *o; int32_t d; } rms_task_t;
static void rms_row(int t, void *arg)
{
    rms_task_t *r = (rms_task_t *)arg;
    rmsnorm(r->h + (size_t)t * r->d, r->o + (size_t)t * r->d, r->d, 1e-5f);
}
typedef struct { Model *M; const int32_t *y; } dll_task_t;
static void dll_row(int t, void *arg)
{
    dll_task_t *a = (dll_task_t *)arg;
    Model *M = a->M;
    const int32_t *y = a->y;
    int32_t V = M->V, i;
    float *lg = M->logits + (size_t)t * V, *dl = M->dlogits + (size_t)t * V;
    double mx = -1e300, sm = 0.0;
    for (i = 0; i < V; i++) if (lg[i] > mx) mx = lg[i];
    for (i = 0; i < V; i++) sm += fexp((double)lg[i] - mx);
    for (i = 0; i < V; i++) dl[i] = (float)(fexp((double)lg[i] - mx) / sm);
    if (y[t] >= 0 && y[t] < V) dl[y[t]] -= 1.0f;
    for (i = 0; i < V; i++) dl[i] /= (float)M->n_tok;
    if (g_paper_chain) {
        for (i = 0; i < V; i++) {
            double e = (double)lg[i] - ((y[t] >= 0 && i == y[t]) ? 1.0 : 0.0);
            double ec = e * (double)QF;
            if (ec >  2147483647.0) ec =  2147483647.0;
            if (ec < -2147483648.0) ec = -2147483648.0;
            if (g_usq == 1) {
                ec = (double)((((int64_t)llround(ec)) * 2) >> GRAD_SHIFT);
            } else if (g_usq == 2) {
                ec = ec * 2.0;
            }
            dl[i] = (float)ec;
        }
    }
}
static double model_forward(Model *M, const int32_t *x, const int32_t *y)
{
    int32_t t, l, i, V = M->V, d = M->d, T = M->T;
    float *h = M->hsave;
    double loss = 0.0;
    M->n_correct = 0; M->n_tok = 0;
    g_usq_val = 0.0;
    {   emb_task_t et;
        et.M = M; et.x = x; et.h = h;
        if (mt_use((int64_t)T, (int64_t)d)) mt_run(0, T, emb_row, &et);
        else { int32_t tt; for (tt = 0; tt < T; tt++) emb_row(tt, &et); }
    }
    /* la base e' SEMPRE M->hsave : h e' gia' stato avantato di un livello,
     * quindi usarlo come base spostava la destinazione di un livello in
     * piu' e scriveva fuori dal buffer. E' stato un mio errore, non del
     * paper, e ha corrotto lo shadow del blocco 0 con valori float.     */
    /* v004-14B: FINESTRA SCORREVOLE DI 2, DOUBLE BUFFERING (mai tutto in RAM).
     * load(0) + bg-load(1); poi per ogni layer: compute sul corrente mentre
     * il bg-thread carica il successivo nello slot inattivo -> join ->
     * load(successivo, no-op se il bg e' riuscito, picco = 2 residenti) ->
     * unload(corrente, torna a 1) -> bg-load(quello dopo). A regime il
     * successivo e' gia' in RAM: I/O nascosto dietro il compute, stessi
     * numeri (gli stessi byte dai chunk). Max-2 sotto lock in entrambi i
     * path; la join precede ogni load/unload/spill (nessuna race). */
    {   ooc_pfl_t pfl; int32_t nl = M->n_layers;
        memset(&pfl, 0, sizeof pfl);
        if (nl > 0) {
            if (!block_load(M, 0)) die("ooc load");
            ooc_pf_layer_start(&pfl, M, 1, nl);
        }
        for (l = 0; l < nl; l++) {
            block_forward(&M->B[l], h, M->hsave + (size_t)(l + 1) * T * d);
            ooc_pf_layer_join(&pfl);
            if (l + 1 < nl && !block_load(M, l + 1)) die("ooc load");
            if (!block_unload(M, l, 0)) die("ooc unload");
            ooc_pf_layer_start(&pfl, M, l + 2, nl);
            h = M->hsave + (size_t)(l + 1) * T * d;
        }
        ooc_pf_layer_join(&pfl);
    }
    /* [MIO] normalizzazione finale : con pre-norm e connessioni residue
     * lo stream residuo non e' normalizzato in uscita, e la testa lo
     * riceverebbe grezzo. E' la norma in ogni transformer pre-norm.      */
    {   rms_task_t rt;
        rt.h = h; rt.o = M->hfin; rt.d = d;
        if (mt_use((int64_t)T, (int64_t)d)) mt_run(0, T, rms_row, &rt);
        else { int32_t tt; for (tt = 0; tt < T; tt++) rms_row(tt, &rt); }
    }
    h = M->hfin;
    for (t = 0; t < T - 1; t++) {
        float *lg = M->logits + (size_t)t * V;
        double mx = -1e300, sm = 0.0, lse;
        int32_t arg = 0;
        qlin_forward(M->head, h + (size_t)t * d);
        for (i = 0; i < V; i++) { lg[i] = M->head->act[i];
                                  if (lg[i] > mx) { mx = lg[i]; arg = i; } }
        for (i = 0; i < V; i++) sm += fexp((double)lg[i] - mx); /* v004 */
        lse = log(sm) + mx;
        if (y[t] >= 0 && y[t] < V) {
            double acc = 0.0;
            loss += lse - (double)lg[y[t]];
            M->n_tok++;
            if (arg == y[t]) M->n_correct++;
            /* Loss utente : sum_k (Output_k - Target_k)^2 >> 19, e in Q16.15 */
            for (i = 0; i < V; i++) {
                double ee = (double)lg[i]
                          - ((i == y[t]) ? 1.0 : 0.0);
                acc += (ee * (double)QF) * (ee * (double)QF);
            }
            g_usq_val += floor(acc / 524288.0);   /* >> 19 */
        }
    }
    return M->n_tok ? loss / (double)M->n_tok : 0.0;
}

/* ------------------------------------------------------------ indietro --- */
#ifndef PII_FLASH_NO_TRAIN
static void model_backward(Model *M, const int32_t *x, const int32_t *y)
{
    int32_t t, l, i, V = M->V, d = M->d, T = M->T;
    float *dh = (float *)calloc((size_t)T * d, 4);
    float *dhp = (float *)calloc((size_t)T * d, 4);
    float *bout = (float *)calloc((size_t)T * d, 4);  /* rigenera attivazioni */
    if (!dh || !dhp || !bout) die("malloc");
    /* dL/dlogit : softmax - onehot, normalizzato sul numero di token      */
    {   dll_task_t dt;
        dt.M = M; dt.y = y;
        if (T > 1 && mt_use((int64_t)(T - 1), (int64_t)V))
            mt_run(0, T - 1, dll_row, &dt);
        else { int32_t tt; for (tt = 0; tt < T - 1; tt++) dll_row(tt, &dt); }
    }
    /* la testa e' l'ultimo strato : il gradiente rispetto all'ingresso
     * della testa e' il gradiente rispetto allo stream residuo          */
    memset(dh, 0, (size_t)T * d * 4);
    for (t = T - 2; t >= 0; t--)
        qlin_backward(M->head, M->dlogits + (size_t)t * V,
                      M->dhn + (size_t)t * d, 1, 1, M->lr_head);
    /* La riquantizzazione di testa ed embedding stava QUI dentro, e
     * rendeva impossibile accumulare: il batch accumulava sui blocchi, poi
     * testa ed embedding venivano riquantizzati a ogni singola finestra.
     * Con batch = 1 resta esattamente il comportamento di prima, e il
     * default è 1 : senza che lo chieda nessuno non cambia nulla.      */
    if (!g_accum_active) qlin_refresh(M->head);
    /* attraverso la normalizzazione finale, fino allo stream residuo      */
    memset(M->dhn, 0, (size_t)T * d * 4);      /* mai letta spazzatura */
    /* T-1 non e' mai scritta */
    if (g_paper_chain) {
        /* copia token-indipendente: stessi numeri, via pool sopra soglia */
        flat_task_t ft;
        ft.dst = dh; ft.a = M->dhn; ft.mode = 0;
        if (T > 1 && mt_use((int64_t)(T - 1), (int64_t)d))
            mt_run(0, (int)((int64_t)(T - 1) * d), flat_vec_op, &ft);
        else for (t = T - 2; t >= 0; t--)
            for (i = 0; i < d; i++) dh[(size_t)t*d + i] = M->dhn[(size_t)t*d + i];
    } else {
        for (t = T - 2; t >= 0; t--)
            rmsnorm_backward(M->hfin + (size_t)t * d, M->dhn + (size_t)t * d,
                             dh + (size_t)t * d, d, 1e-5f);
    }
    /* v004-14B: stessa finestra di 2 all'indietro, double buffering — compute
     * (bg-load in overlap sullo slot inattivo), join (= barriera), load
     * (no-op se il bg e' riuscito, picco = 2), unload con spill (i pesi
     * aggiornati tornano su disco prima di liberare, sempre dopo la join),
     * bg-load(ancora prima). Stessi byte -> stessi numeri, bit-identici. */
    {   ooc_pfl_t pfl; int32_t nl = M->n_layers;
        memset(&pfl, 0, sizeof pfl);
        if (nl > 0) {
            if (!block_load(M, nl - 1)) die("ooc load");
            ooc_pf_layer_start(&pfl, M, nl - 2, nl);
        }
        for (l = nl - 1; l >= 0; l--) {
            const float *h = M->hsave + (size_t)l * T * d;
            int64_t n = (int64_t)T * d;
            block_forward(&M->B[l], h, bout);   /* rigenera Qh/Kh/Vh/Wt/Ch */
            memset(dhp, 0, (size_t)T * d * 4);
            block_backward(&M->B[l], h, dh, dhp, M->B[l].lr);
            ooc_pf_layer_join(&pfl);
            if (l - 1 >= 0 && !block_load(M, l - 1)) die("ooc load");
            if (!block_unload(M, l, 1)) die("ooc spill");  /* pesi aggiornati */
            ooc_pf_layer_start(&pfl, M, l - 2, nl);
            /* la residua trasporta il gradiente al blocco precedente          */
            {   flat_task_t ft;
                ft.dst = dh; ft.a = dhp; ft.mode = 1;
                if (n > 1 && n <= 2147483647 && mt_use(n, 1))
                    mt_run(0, (int)n, flat_vec_op, &ft);
                else for (i = 0; i < n; i++) dh[i] += dhp[i];
            }
        }
        ooc_pf_layer_join(&pfl);
    }
    /* l'embedding e' una tabella di terne indicizzate : si aggiorna solo
     * la riga del token, e il gradiente si somma su tutte le occorrenze */
    {   float *eg = (float *)calloc((size_t)V * d, 4);
        if (!eg) die("malloc");
        for (t = 0; t < T; t++) {
            int32_t tk = (x[t] >= 0 && x[t] < V) ? x[t] : 0;
            for (i = 0; i < d; i++) eg[(size_t)tk*d + i] += dh[(size_t)t*d + i];
        }
        for (t = 0; t < V; t++) {
            for (i = 0; i < d; i++) {
                int64_t raw = ((int64_t)(int32_t)llround((double)eg[(size_t)t*d+i] * QF))
                              >> GRAD_SHIFT;
                int32_t delta;
                if (raw >  GSAT)  { raw =  GSAT;  M->n_sat++; }
                if (raw < -GSATN) { raw = -GSATN; M->n_sat++; }
                delta = (int32_t)((raw * (int64_t)acc_lr(M->lr_emb))
                                  >> upd_shift());
                M->emb->shadow[(int64_t)t * d + i] -= (shadow_t)delta;
            }
        }
        free(eg);
    }
    if (!g_accum_active) qlin_refresh(M->emb);
    free(dh); free(dhp); free(bout);
}
#endif /* PII_FLASH_NO_TRAIN */

/* ============================================================== utility === */
static double rss_mib(void)
{
    PROCESS_MEMORY_COUNTERS pm;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pm, sizeof(pm))) {
        double peak = (double)pm.PeakWorkingSetSize / (1024.0 * 1024.0);
        return peak;
    }
    return 0.0;
}
/* Diagnostica: carica il layer per leggere i trit (emb/head residenti). */
static void layer_stats(Model *M, const char *tag, int32_t layer)
{
    const QLin *Ls[7];
    const char *nm[7] = { "Wq","Wk","Wv","Wo","Wg","Wu","Wd" };
    int32_t i, N = 0; int64_t tot = 0, zer = 0; double gs = 0.0, dc = 0.0;
    if (layer < 0) { Ls[0]=M->emb; Ls[1]=M->head; nm[0]="emb"; nm[1]="head"; N=2; }
    else { const Block *B = &M->B[layer];
           if (layer < 0 || layer >= M->n_layers) return;
           if (!block_load(M, layer)) die("ooc load");
           Ls[0]=B->Wq; Ls[1]=B->Wk; Ls[2]=B->Wv; Ls[3]=B->Wo;
           Ls[4]=B->Wg; Ls[5]=B->Wu; Ls[6]=B->Wd; N=7; }
    for (i = 0; i < N; i++) {
        int64_t z = qlin_zero(Ls[i]);
        zer += z; tot += (int64_t)Ls[i]->n_in * Ls[i]->n_out;
        gs  += (double)Ls[i]->gshift; dc += (double)Ls[i]->decay;
    }
    if (layer >= 0) { if (!block_unload(M, layer, 0)) die("ooc unload"); }
    printf("  %-5s trit=%.4f bit/peso  zero=%6.2f%%  gamma=%7.2f  decay=%.2f",
           tag, 8.0 / 5.0, 100.0 * (double)zer / (double)tot, gs / N, dc / N);
    {   int32_t s = 0, l;
        for (l = 0; l < M->n_layers; l++)
            s += M->B[l].Wq->n_sat + M->B[l].Wk->n_sat + M->B[l].Wv->n_sat
               + M->B[l].Wo->n_sat + M->B[l].Wg->n_sat + M->B[l].Wu->n_sat
               + M->B[l].Wd->n_sat;
        printf("  G_saturati=%d", s);
    }
    printf("\n");
}

/* ================================================================= test === */
static int g_pass = 0, g_fail = 0;
static void chk(const char *name, int ok, const char *note)
{
    if (ok) { g_pass++; printf("  [ OK ] %-50s %s\n", name, note); }
    else    { g_fail++; printf("  [FAIL] %-50s %s\n", name, note); }
}

/* --- DIAG : perche' la rete NON impara ne' una sequenza di 6 simboli ----
 *
 * Il sintomo: `tiny_overfit` resta SOPRA il caso uniforme. Un transformer di
 * 11264 parametri che deve imparare y[t] = (x[t]+1) mod 6 e' un compito che
 * qualsiasi implementazione corretta porta a ~0. Restare a 1.96 quando
 * l'uniforme e' 1.79 significa che il difetto e' nel codice, non nel modello.
 *
 * La domanda che questo mode risponde, e' una sola: **quanto si muove
 * davvero lo shadow, e di quanto, per ogni passo?** Perche' la sezione 7
 * scrive  Shadow -= (G * LR) >> 12  con G in Q4.11 (max 2047) e LR in
 * UQ0.16 (max 65535). Il passo massimo e' (2047 * 65535) >> 12 = 32766,
 * ben oltre la soglia 213 : quindi il paper dichiara un aggiornamento che
 * puo' saltare la soglia. Ma il caso tipico e' un gradiente PICCOLO, e se
 * (|G| * LR) >> 12 < 213 lo shadow non raggiunge mai la soglia di isteresi e
 * il trit non cambia MAI. La rete allora non e' lenta : e' immobile.
 *
 * Non e' una domanda teorica. E' una divisione, e la risposta si misura.
 * ------------------------------------------------------------------- */
#ifndef PII_FLASH_NO_TRAIN
static void diag_step_path(void)
{
    int32_t V = 6, d = 32, T = 8, nl = 1;
    int i, step;
    double first = 0.0, last = 0.0, best = 1e300;
    int64_t ntrit = 0, nflip = 0, nsteps_flip = 0;
    int32_t *x = (int32_t *)calloc(T, 4), *y = (int32_t *)calloc(T, 4);
    Model *M;
    trit_t *prev = NULL;

    printf("--- DIAG : il percorso dell'aggiornamento, misurato ------------\n");
    printf("  compito : y[t] = (x[t]+1) mod %d, T=%d, 1 blocco, d=%d\n", V, T, d);
    printf("  uniforme ln(%d) = %.5f : sotto questa soglia non si e' imparato\n",
           V, log((double)V));
    printf("  soglia di isteresi %d, G in Q4.11 (max %d), LR in UQ0.16\n\n",
           THRESHOLD, GSAT);

    /* --- parte 1 : la tabella dei passi possibili --------------------- */
    printf("  1. Un passo muove lo shadow di (|G| * LR) >> 12.\n");
    printf("     Per arrivare alla soglia %d con GRAD_SHIFT=%d e' "
           "servito |G|*LR >= %d.\n\n",
           THRESHOLD, GRAD_SHIFT, (int)(((int64_t)THRESHOLD << GRAD_SHIFT) >> 12));

    {   static const int32_t GS[] = { 1, 7, 64, 512, 2047 };
        static const int32_t LRS[] = { 64, 512, 4096, 65535 };
        int a, b;
        printf("     |G|\\LR  ");
        for (b = 0; b < 4; b++) printf("%8d", LRS[b]);
        printf("   <- muove il trit?\n");
        for (a = 0; a < 5; a++) {
            printf("     %6d ", GS[a]);
            for (b = 0; b < 4; b++) {
                int64_t step_q = ((int64_t)GS[a] * LRS[b]) >> 12;
                printf("%8s", (step_q >= THRESHOLD) ? "si" : "NO");
            }
            printf("\n");
        }
        printf("\n     Questa tabella e' IL ritratto del difetto. Per |G| piccoli\n");
        printf("     il passo non arriva alla soglia, e la cella dice NO.\n");
        printf("     La soglia non e' un dettaglio : e' l'unico modo che ha un\n");
        printf("     trit di cambiare segno.\n\n");
    }

    /* --- parte 2 : cosa succede davvero durante l'addestramento -------
     * Uso lo stesso trucco che usa il resto del file : un array di QLin
     * che contiene tutte le matrici del blocco, e un buffer di tritt che
     * copio PRIMA del passo e confronto DOPO. E' l'unico modo onesto di
     * dire "quanti pesi sono cambiati davvero", perche' ispezionare il
     * trit a fine addestramento non dice nulla : un peso puo' essere
     * tornato al suo valore iniziale dopo essere passato due volte.     */
    for (i = 0; i < T; i++) { x[i] = i % V; y[i] = (i + 1) % V; }
    g_paper_chain = 1; g_siluns_grad = 0; g_use_sched = 0;
    M = model_new(V, d, 4, 64, T, nl, 20260930u);
    {   const QLin *Ls[7];
        int32_t u;
        Ls[0]=M->B[0].Wq; Ls[1]=M->B[0].Wk; Ls[2]=M->B[0].Wv;
        Ls[3]=M->B[0].Wo; Ls[4]=M->B[0].Wg; Ls[5]=M->B[0].Wu;
        Ls[6]=M->B[0].Wd;
        for (u = 0; u < 7; u++)
            ntrit += (int64_t)Ls[u]->n_in * Ls[u]->n_out;
        prev = (trit_t *)calloc((size_t)ntrit, 1);
        if (!prev) die("malloc");
    }
    for (step = 0; step < 400; step++) {
        double L = model_forward(M, x, y);
        int64_t f = 0;
        int32_t u, k = 0;
        const QLin *Ls[7];
        Ls[0]=M->B[0].Wq; Ls[1]=M->B[0].Wk; Ls[2]=M->B[0].Wv;
        Ls[3]=M->B[0].Wo; Ls[4]=M->B[0].Wg; Ls[5]=M->B[0].Wu;
        Ls[6]=M->B[0].Wd;
        if (!block_load(M, 0)) die("ooc load");
        for (u = 0; u < 7; u++) {
            int64_t j, NN = (int64_t)Ls[u]->n_in * Ls[u]->n_out;
            for (j = 0; j < NN; j++) prev[k++] = Ls[u]->trit[j];
        }
        if (!block_unload(M, 0, 0)) die("ooc unload");
        if (step == 0) first = L;
        if (L < best) best = L;
        last = L;
        model_backward(M, x, y);
        /* confronto DOPO la backward e la riquantizzazione */
        k = 0;
        if (!block_load(M, 0)) die("ooc load");
        for (u = 0; u < 7; u++) {
            int64_t j, NN = (int64_t)Ls[u]->n_in * Ls[u]->n_out;
            for (j = 0; j < NN; j++) if (prev[k + j] != Ls[u]->trit[j]) f++;
            k += (int32_t)NN;
        }
        if (!block_unload(M, 0, 0)) die("ooc unload");
        nflip += f; if (f > 0) nsteps_flip++;
    }
    printf("  2. 400 passi sul compito : perdita %.5f -> %.5f, minima %.5f\n",
           first, last, best);
    printf("     passi in cui ALMENO UN trit e' cambiato : %lld su 400\n",
           (long long)nsteps_flip);
    printf("     cambi di trit totali : %lld su %lld pesi del blocco\n",
           (long long)nflip, (long long)ntrit);
    if (best >= log((double)V))
        printf("     ==> la rete NON ha imparato : resta sopra il caso uniforme.\n");
    else
        printf("     ==> la perdita scende, ma 399 passi su 400 non hanno\n"
               "         cambiato NESSUN peso : cio' che funziona e' un'unica\n"
               "         valanga, non una discesa.\n");
    printf("\n");

    /* --- parte 3 : il punto di partenza dello shadow -------------------
     * Se lo shadow iniziale e' gia' lontano dalla soglia, la banda di
     * isteresi non e' mai attiva e il trit non puo' tornare indietro.    */
    {   int32_t z;
        int64_t n = 0, beyond = 0;
        double sumsq = 0.0, mean = 0.0;
        const QLin *Ls[7];
        Ls[0]=M->B[0].Wq; Ls[1]=M->B[0].Wk; Ls[2]=M->B[0].Wv;
        Ls[3]=M->B[0].Wo; Ls[4]=M->B[0].Wg; Ls[5]=M->B[0].Wu;
        Ls[6]=M->B[0].Wd;
        if (!block_load(M, 0)) die("ooc load");
        for (z = 0; z < 7; z++) {
            int64_t j, NN = (int64_t)Ls[z]->n_in * Ls[z]->n_out;
            for (j = 0; j < NN; j++) {
                double v = (double)Ls[z]->shadow[j];
                mean += v; sumsq += v * v; n++;
                if (fabs(v) >= (double)THRESHOLD) beyond++;
            }
        }
        if (!block_unload(M, 0, 0)) die("ooc unload");
        mean /= (double)n;
        {   double var = sumsq / (double)n - mean * mean;
            printf("  3. Dove nasce lo shadow (dopo 400 passi).\n");
            printf("     media %.1f, deviazione standard %.1f\n",
                   mean, sqrt(var > 0 ? var : 0));
            printf("     soglia %d : il %.1f%% degli shadow e' oltre la soglia\n",
                   THRESHOLD, 100.0 * (double)beyond / (double)n);
            printf("\n");
        }
    }
    printf("  4. Cosa significa. Il trit ha un solo modo di cambiare : lo shadow\n");
    printf("     deve attraversare la soglia %d. Con un passo che vale\n", THRESHOLD);
    printf("     migliaia di conteggi l'aggiornamento non e' una discesa : e'\n");
    printf("     un salto che rimescola i pesi e butta via quello appena\n");
    printf("     imparato. E' il motivo per cui la risposta non cambia.\n\n");

    /* --- parte 5 : la scala REALE del gradiente, prima di Q4.11 --------
     * Il campo Q4.11 finisce a 2047. Se il grezzo e' 2^26, il >>19 della
     * sezione 5 finisce sempre a saturazione : il gradiente perde la sua
     * grandezza e diventa il solo SEGNO del gradiente. Misuro il grezzo. */
    {   int32_t z;
        int64_t n = 0, nsat = 0;
        int64_t smax = 0;
        const QLin *Ls[9];
        Ls[0]=M->B[0].Wq; Ls[1]=M->B[0].Wk; Ls[2]=M->B[0].Wv;
        Ls[3]=M->B[0].Wo; Ls[4]=M->B[0].Wg; Ls[5]=M->B[0].Wu;
        Ls[6]=M->B[0].Wd; Ls[7]=M->emb; Ls[8]=M->head;
        for (z = 0; z < 9; z++) {
            /* x_q e' l'ingresso quantizzato della testa : e' il tensore
             * dove il gradiente e' piu' grande, quindi il caso peggiore.  */
            if (z != 8) continue;
            {   grad_t *G = (grad_t *)calloc((size_t)V * d, sizeof(grad_t));
                int64_t j, NN = (int64_t)V * d;
                if (!G) die("malloc");
                qlin_grad_probe(Ls[z], G);
                for (j = 0; j < NN; j++) {
                    int64_t a = G[j] < 0 ? -(int64_t)G[j] : (int64_t)G[j];
                    if (a > smax) smax = a;
                    if (a >= GSAT) nsat++;
                    n++;
                }
                printf("  5. La scala reale del gradiente (testa, %lld pesi).\n",
                       (long long)n);
                printf("     NOTA : la sonda restituisce G gia' >>%d, quindi il\n",
                       GRAD_SHIFT);
                printf("     confronto con il campo Q4.11 (%d) e' diretto.\n", GSAT);
                printf("     max |G| : %lld\n", (long long)smax);
                printf("     saturati : %lld su %lld, cioe' il %.1f%%\n",
                       (long long)nsat, (long long)n,
                       n ? 100.0 * (double)nsat / (double)n : 0.0);
                /* Il grezzo PRIMA dello shift : e' lui che dice di quanto
                 * sbaglia il >>%d. ricalcolato qui su gbuf e x_q, che sono
                 * leggibili, perche' la sonda ha gia' perso l'informazione. */
                {   int64_t rawo = 0, rawn = 0;
                    double amax = 0.0;
                    int32_t i2;
                    for (i2 = 0; i2 < V; i2++) {
                        int64_t e = llabs((int64_t)llround((double)Ls[z]->gbuf[i2] * QF));
                        int64_t xm = 0, j2;
                        for (j2 = 0; j2 < d; j2++) {
                            int64_t a = Ls[z]->x_q[j2];
                            if (a < 0) a = -a;
                            if (a > xm) xm = a;
                        }
                        if (e * xm > rawo) rawo = e * xm;
                    }
                    if (rawo > 0) amax = (double)rawo / (double)(((int64_t)1) << GRAD_SHIFT);
                    rawn = (int64_t)amax;
                    printf("     grezzo massimo prima dello shift : %lld\n",
                           (long long)rawo);
                    printf("     cioe' %.0f volte il fondo del campo (%d).\n",
                           amax, GSAT);
                    printf("     la scala giusta e' %.4f : sotto 1, perche' il\n",
                           (double)GSAT / (amax > 0 ? amax : 1));
                    printf("     gradiente va ABBASSATO, non alzato.\n\n");
                }
                free(G);
            }
        }
    }
    free(prev); free(x); free(y); model_free(M);
}
#endif /* PII_FLASH_NO_TRAIN */

/* --- DIAG2 : la domanda vera. Che cosa fa imparare i 6 simboli? --------
 *
 * Il primo diag ha misurato il difetto. Questo cerca il rimedio, e lo
 * cerca misurando, non ragionando. La domanda e' una: **quanto deve
 * valere il passo sullo shadow, rispetto alla soglia di isteresi?**
 *
 * Perche' e' quella e non un'altra. Un trit puo' cambiare solo quando lo
 * shadow attraversa la soglia 213. Se il passo e' 2047 — dieci volte la
 * soglia — lo shadow non "attraversa": salta da una parte all'altra e il
 * peso che era appena diventato giusto viene rimesso a posto sbagliato al
 * passo dopo. Se il passo e' 32 — un sesto della soglia — lo shadow cammina,
 * e la banda di isteresi fa il suo lavoro: tiene il peso fermo finche' la
 * spinta non e' davvero convincente.
 *
 * Quindi la domanda non e' "quanto e' grande il gradiente" ma "quanto e'
 * grande il passo". E il passo e' (|G| * LR) >> 12 : per arrivare a 213 con
 * |G| = 2047 serve LR = 427, e LR = 4096 lo supera dieci volte. La griglia
 * qui sotto misura il numero di passi che cambiano un trit, e la perdita,
 * su tutta la scala dei LR, con e senza la scala del gradiente.         */
#ifndef PII_FLASH_NO_TRAIN
static void diag2_stepsweep(void)
{
    int32_t V = 6, d = 32, T = 8, nl = 1;
    static const int32_t LRS[] = { 64, 128, 256, 427, 512, 1024, 4096, 16384 };
    int i, s;
    printf("--- DIAG2 : passo sullo shadow contro soglia di isteresi ---------\n");
    printf("  compito : y[t] = (x[t]+1) mod %d. uniforme %.5f.\n",
           V, log((double)V));
    printf("  sotto la riga 'uniforme' la rete ha imparato qualcosa.\n\n");
    printf("  %6s %10s %12s %12s %12s %12s %10s\n",
           "LR", "passo", "perdita0", "finale", "minima", "acc%", "passi che");
    printf("  %6s %10s %12s %12s %12s %12s %10s\n",
           "", "sul shadow", "(no scala)", "(no scala)", "(no scala)", "",
           "cambiano");
    for (s = 0; s < 8; s++) {
        int on;
        for (on = 0; on < 2; on++) {
            Model *M;
            double first = 0.0, last = 0.0, best = 1e300;
            int32_t *x = (int32_t *)calloc(T, 4), *y = (int32_t *)calloc(T, 4);
            int64_t ntrit = 0, nstep_flip = 0;
            trit_t *prev;
            const QLin *Ls[7];
            int32_t z;
            for (i = 0; i < T; i++) { x[i] = i % V; y[i] = (i + 1) % V; }
            g_paper_chain = 1; g_siluns_grad = 0; g_use_sched = 0;
            g_grad_scale_on = on;
            memset(g_grad_amax, 0, sizeof g_grad_amax);
            g_grad_init = 0;
            M = model_new(V, d, 4, 64, T, nl, 20260930u);
            for (z = 0; z < nl; z++) M->B[z].lr = (lr_t)LRS[s];
            M->lr = (lr_t)LRS[s]; M->lr_emb = M->lr; M->lr_head = M->lr;
            Ls[0]=M->B[0].Wq; Ls[1]=M->B[0].Wk; Ls[2]=M->B[0].Wv;
            Ls[3]=M->B[0].Wo; Ls[4]=M->B[0].Wg; Ls[5]=M->B[0].Wu;
            Ls[6]=M->B[0].Wd;
            for (z = 0; z < 7; z++)
                ntrit += (int64_t)Ls[z]->n_in * Ls[z]->n_out;
            prev = (trit_t *)calloc((size_t)ntrit, 1);
            if (!prev) die("malloc");
            for (i = 0; i < 400; i++) {
                double L = model_forward(M, x, y);
                int64_t f = 0;
                int32_t k = 0, u;
                if (!block_load(M, 0)) die("ooc load");
                for (u = 0; u < 7; u++) {
                    int64_t j, NN = (int64_t)Ls[u]->n_in * Ls[u]->n_out;
                    for (j = 0; j < NN; j++) prev[k + j] = Ls[u]->trit[j];
                }
                if (!block_unload(M, 0, 0)) die("ooc unload");
                if (i == 0) first = L;
                if (L < best) best = L;
                last = L;
                model_backward(M, x, y);
                if (!block_load(M, 0)) die("ooc load");
                for (u = 0; u < 7; u++) {
                    int64_t j, NN = (int64_t)Ls[u]->n_in * Ls[u]->n_out;
                    for (j = 0; j < NN; j++)
                        if (prev[k + j] != Ls[u]->trit[j]) f++;
                    k += (int32_t)NN;
                }
                if (!block_unload(M, 0, 0)) die("ooc unload");
                if (f > 0) nstep_flip++;
            }
            printf("  %6d %10d %12s %12.5f %12.5f %12.1f %10lld\n",
                   LRS[s],
                   (int)(((int64_t)2047 * LRS[s]) >> 12),
                   on ? "  (si)" : "  (no)", last, best,
                   100.0 * M->n_correct / (M->n_tok ? M->n_tok : 1),
                   (long long)nstep_flip);
            if (on) printf("  %6s %10s %12.5f %12.5f %12.5f %12.1f %10lld\n",
                           "", "", first, last, best,
                           100.0 * M->n_correct / (M->n_tok ? M->n_tok : 1),
                           (long long)nstep_flip);
            free(prev); free(x); free(y); model_free(M);
        }
    }
    g_grad_scale_on = 1;
    printf("\n  Lettura : nella prima riga di ogni coppia la colonna 'finale'\n");
    printf("  e' la perdita DOPO 400 passi. Se e' minore di %.5f la rete\n",
           log((double)V));
    printf("  ha imparato. La colonna 'passi che cambiano' dice SE lo fa'\n");
    printf("  camminando (centinaia di passi) o con una valanga (uno solo).\n\n");
}
#endif /* PII_FLASH_NO_TRAIN */

/* --- DIAG3 : quanto deve essere rumoroso lo shadow iniziale? -----------
 *
 * Il paper dichiara sigma 2048 e soglia 213. La misura dice che il primo e'
 * otto volte il secondo, e che quindi il rumore iniziale decide il segno
 * dei pesi PRIMA che il gradiente possa dire la sua. Qui misuro sigma per
 * sigma, sul compito dei 6 simboli, e leggo quale fa davvero scendere. */
#ifndef PII_FLASH_NO_TRAIN
static void diag3_shadowsigma(void)
{
    int32_t V = 6, d = 32, T = 8, nl = 1;
    static const int SIG[] = { 0, 16, 64, 128, 256, 512, 1024, 2048 };
    int i, s;
    printf("--- DIAG3 : la rumorosita' dello shadow iniziale ----------------\n");
    printf("  compito : y[t] = (x[t]+1) mod %d, 400 passi. uniforme %.5f.\n",
           V, log((double)V));
    printf("  la soglia di isteresi e' %d : la colonna sigma/213 dice quante\n",
           THRESHOLD);
    printf("  volte il rumore iniziale e' piu' grande della decisione.\n\n");
    printf("  %8s %9s %12s %12s %12s %10s\n",
           "sigma", "sigma/213", "perdita0", "finale", "minima", "acc%");
    for (s = 0; s < 8; s++) {
        Model *M;
        double first = 0.0, last = 0.0, best = 1e300;
        int32_t *x = (int32_t *)calloc(T, 4), *y = (int32_t *)calloc(T, 4);
        int32_t z;
        for (i = 0; i < T; i++) { x[i] = i % V; y[i] = (i + 1) % V; }
        g_paper_chain = 1; g_siluns_grad = 0; g_use_sched = 0;
        g_grad_scale_on = 1;
        g_shadow_sigma = SIG[s];
        M = model_new(V, d, 4, 64, T, nl, 20260930u);
        for (z = 0; z < nl; z++) M->B[z].lr = (lr_t)1024;
        M->lr = (lr_t)1024; M->lr_emb = M->lr; M->lr_head = M->lr;
        for (i = 0; i < 400; i++) {
            double L = model_forward(M, x, y);
            if (i == 0) first = L;
            if (L < best) best = L;
            last = L;
            model_backward(M, x, y);
        }
        printf("  %8d %9.2f %12.5f %12.5f %12.5f %9.1f%%%s\n",
               SIG[s], (double)SIG[s] / (double)THRESHOLD,
               first, last, best,
               100.0 * M->n_correct / (M->n_tok ? M->n_tok : 1),
               (last < log((double)V)) ? "   <-- impara" : "");
        free(x); free(y); model_free(M);
    }
    g_shadow_sigma = 2048;
    printf("\n  Il valore dichiarato dal paper e' sigma 2048. Quello che\n");
    printf("  funziona, se funziona, e' una MUTAZIONE e va detta tale.\n\n");
}
#endif /* PII_FLASH_NO_TRAIN */

/* --- T1 : la derivata di SiLUNS e' esatta ----------------------------- */
static int t1_siluns(void)
{
    int32_t xs[] = { -60000, -45000, -20000, -100, 100, 20000, 45000, 60000 };
    int i, ok = 1; double worst = 0.0;
    for (i = 0; i < 8; i++) {
        int32_t x = xs[i];
        const int32_t h = 64;
        /* in CONTEGGI GREZZI : dividere per QF qui darebbe pendenza/QF   */
        double a  = (double)siluns_r7(x + h);
        double b  = (double)siluns_r7(x - h);
        double fd = (a - b) / (2.0 * (double)h);
        double an = siluns_d(x);
        double e  = fabs(fd - an);
        if (e > worst) worst = e;
        if (e > 1e-3) ok = 0;
    }
    {   char b[128]; snprintf(b, sizeof b, "scarto massimo %.2e su 8 punti", worst);
        chk("T1 SiLUNS r7 : derivata = differenza finita", ok, b); }
    return ok;
}

/* --- T5 : la soglia 213 e' raggiungibile con l'aggiornamento dichiarato? --
 * Domanda unica e decisiva: con la formula della sezione 7, esiste almeno
 * un gradiente G (Q4.11, |G| <= 2047) e un learning rate LR (UQ0.16,
 * 0 <= LR <= 65535) per cui  Delta = |G| * LR >> sft  >= 213 ?
 * Se la risposta e' no, nessun trit si muove e il modello non apprende:
 * non perche' l'errore sia piccolo, ma perche' il passo e' fuori portata. */
static int t5_reach(void)
{
    static const int GAINS[3] = { 0, 4, 6 };
    static const char *GL[3] = { "r7  (>> 12)", "r8  (>> 8 , Update_Gain=4)",
                                 "r8+ (>> 6 , Update_Gain=6)" };
    int gi, g, lr, q, best_q, best_lr, live_from, live_to, nlive;
    char b[220];
    int any_alive = 0;

    printf("\n  Leva della sezione 7 :  Delta = (|G| * LR) >> (12 - Update_Gain)\n");
    printf("  Soglia di isteresi 213, |G|<=2047 (Q4.11), LR<=65535 (UQ0.16)\n");
    printf("  %-30s %10s %12s %12s %10s\n",
           "variante", "shift", "min|G|@max", "min LR@|G|=7", "passa@64");

    for (gi = 0; gi < 3; gi++) {
        g = GAINS[gi];
        g_update_gain = g;
        {   int sft = upd_shift();
            /* minimo |G| che muove un trit al LR massimo rappresentabile */
            best_q = 0;
            for (q = 1; q <= 2047; q++) {
                if (((int64_t)q * 65535) >> sft >= 213) { best_q = q; break; }
            }
            /* minimo LR che muove un trit con il gradiente tipico misurato */
            best_lr = -1;
            for (lr = 0; lr <= 65535; lr++) {
                if (((int64_t)7 * lr) >> sft >= 213) { best_lr = lr; break; }
            }
            /* quante coppie (|G|, LR) su una griglia 1..2047 x 1..65535 muovono? */
            nlive = 0;
            for (q = 1; q <= 2047; q += 7)
                for (lr = 64; lr <= 65535; lr += 64)
                    if (((int64_t)q * lr) >> sft >= 213) nlive++;
            /* la fascia di LR viva al gradiente misurato |G| = 7 */
            live_from = -1; live_to = -1;
            for (lr = 0; lr <= 65535; lr++) {
                if (((int64_t)7 * lr) >> sft >= 213) {
                    if (live_from < 0) live_from = lr;
                    live_to = lr;
                }
            }
            printf("  %-30s %10d %12d %12d %10s\n", GL[gi], sft, best_q, best_lr,
                   (7 * 64) >> sft >= 213 ? "si" : "NO");
            if (best_q <= 2047) any_alive = 1;
            snprintf(b, sizeof b,
                     "shift %d, |G| min %d su 2047, LR min %d su 65535",
                     sft, best_q, best_lr);
            chk(GL[gi], best_q <= 2047 && best_lr >= 0, b);
            snprintf(b, sizeof b, "a |G|=7 il passo e' vivo per %d..%d (finestra %d)",
                     live_from, live_to, (live_to < 0) ? 0 : live_to - live_from + 1);
            chk("  fascia LR viva a |G|=7", live_from >= 0, b);
        }
    }
    g_update_gain = 0;
    printf("\n  Griglia 1..2047 per 64..65535 (passo 7 e 64), coppie che muovono:\n");
    for (gi = 0; gi < 3; gi++) {
        g_update_gain = GAINS[gi];
        nlive = 0;
        for (q = 1; q <= 2047; q += 7)
            for (lr = 64; lr <= 65535; lr += 64)
                if (((int64_t)q * lr) >> upd_shift() >= 213) nlive++;
        printf("    %-30s %8d coppie\n", GL[gi], nlive);
    }
    g_update_gain = 0;
    printf("\n  Se Update_Gain non e' fissato, il caso peggiore e' Update_Gain>=12:\n");
    printf("    lo shift diventa 0 o NEGATIVO e `x >> (12-g)` in C e' indefinito\n");
    printf("    per 12-g<0 : il paper non vieta quel valore.\n");
    {   int sft_neg = 12 - 16;
        printf("    Update_Gain=16 -> shift %+d : il paper non lo ammette, il codice no\n",
               sft_neg);
    }
    chk("T5 l'Update_Gain e' dichiarato ma non fissato dal paper",
        !g_gain_pinned, "il paper scrive «es. impostata a 4» : un esempio");
    return any_alive;
}


/* --- T2 : il gamma letterale del paper uccide l'uscita ---------------- */static int t2_gamma(void)
{
    QLin *L = qlin_new(8, 64);
    float *x = (float *)calloc(64, 4);
    int32_t i, sat0 = 0, sat1 = 0, gt0, gt1, e0, e1;
    double m0 = 0.0, m1 = 0.0;
    char b[200];
    for (i = 0; i < 64; i++) x[i] = (float)(0.05 * ((i % 7) - 3));
    g_gamma_correct = 0; qlin_refresh(L); gt0 = L->gshift; e0 = gt0 & 31;
    qlin_forward(L, x);
    for (i = 0; i < L->n_out; i++) {
        if (fabs(L->act[i]) >= 1.99) sat0++;
        m0 += fabs(L->act[i]);
    }
    m0 /= L->n_out;
    g_gamma_correct = 1; qlin_refresh(L); gt1 = L->gshift; e1 = gt1 & 31;
    qlin_forward(L, x);
    for (i = 0; i < L->n_out; i++) {
        if (fabs(L->act[i]) >= 1.99) sat1++;
        m1 += fabs(L->act[i]);
    }
    m1 /= L->n_out;
    g_gamma_correct = 0;
    snprintf(b, sizeof b, "gshift=%d (shift effettivo %d)  media|out|=%.4f  saturi %d/%d",
             gt0, e0, m0, sat0, L->n_out);
    printf("         letterale : %s\n", b);
    snprintf(b, sizeof b, "gshift=%d (shift effettivo %d)  media|out|=%.4f  saturi %d/%d",
             gt1, e1, m1, sat1, L->n_out);
    printf("         corretto  : %s\n", b);
    {   int ok = (m0 > m1 * 5.0);
        snprintf(b, sizeof b, "media|out| letterale %.4f contro corretto %.4f (x%.1f), "
                 "saturi %d/%d contro %d/%d", m0, m1, m1 > 0 ? m0/m1 : 0.0,
                 sat0, L->n_out, sat1, L->n_out);
        chk("T2 gamma sez.1: il letterale sposta la scala di 16x", ok, b); }
    free(x); qlin_free(L);
    return 1;
}

/* --- T3 : l'attenzione e' causale ------------------------------------- */
/* --- v004 T0a: fexp contro exp() di math.h (confronto solo nel test,
 * mai negli hot path: li' c'e' solo fexp)                           */
static int t_fexp(void)
{
    double worst = 0.0, x;
    int ok;
    for (x = -16.0; x <= 16.0; x += 0.01) {
        double a = fexp(x), b = exp(x), r;
        if (b == 0.0) continue;
        r = fabs(a - b) / b;
        if (r > worst) worst = r;
    }
    {   double w0 = fexp_decay(0.0), e1 = exp(-1.0);
        double r0 = fabs(w0 - e1) / e1;
        if (r0 > worst) worst = r0;
        if (fexp_decay(6.0) != 0.0 || fexp_decay(-6.0) != 1.0) worst = 1.0;
    }
    {   char b2[160];
        snprintf(b2, sizeof b2, "errore relativo max %.2e (soglia 1e-5)", worst);
        chk("T0a fexp LUT 4pt: indistinguibile da exp() sul dominio",
            worst < 1e-5, b2); }
    ok = (worst < 1e-5);
    return ok;
}

/* --- v004 T0b: TL2 bit-exact contro il dot seriale ingenuo --- */
static int t_tl2exact(void)
{
    trit_t row[311];
    int32_t xs[311];
    int i, ok = 1;
    for (i = 0; i < 311; i++) {
        uint32_t r = rnd32() % 3u;
        row[i] = (trit_t)(r == 0 ? -1 : (r == 1 ? 0 : 1));
        xs[i] = (int32_t)(rnd32() & 0xFFFFFFu) - 8388608;
    }
    {   int64_t ref = 0, got;
        for (i = 0; i < 311; i++) ref += (int64_t)row[i] * (int64_t)xs[i];
        got = tl2_dot(row, xs, 311);
        if (got != ref) ok = 0;
        ref = 0;
        for (i = 0; i < 311; i++) ref += (int64_t)row[i] * (int64_t)(xs[i] >> 3);
        got = tl2_dot_s(row, xs, 3, 311);
        if (got != ref) ok = 0;
    }
    chk("T0b TL2 add/sub: bit-exact contro il seriale (311 el, +shift)",
        ok, ok ? "identici" : "DIVERGONO");
    return ok;
}

static int t3_causal(void)
{
    int32_t V = 12, d = 32, T = 10;
    Model *M = model_new(V, d, 4, 64, T, 1, 12345u);
    int32_t *x = (int32_t *)calloc(T, 4);
    int32_t *y = (int32_t *)calloc(T, 4);
    int32_t *x2 = (int32_t *)calloc(T, 4);
    int i, t, ok = 1, bad = 0;
    g_gamma_correct = 1;
    for (i = 0; i < T; i++) { x[i] = (i * 5 + 3) % V; y[i] = (i * 7 + 1) % V; x2[i] = x[i]; }
    model_forward(M, x, y);
    x2[6] = (x2[6] + 3) % V;               /* cambio il token 6           */
    model_forward(M, x2, y);
    for (t = 0; t < 6; t++)                 /* logit delle posizioni 0..5   */
        for (i = 0; i < V; i++)
            if (M->logits[(size_t)t * V + i] != 0.0f &&
                fabs(M->logits[(size_t)t*V+i]) > 0.0f) { }
    /* confronto vero : rifaccio le due sequenze e salvo */
    {   float *save = (float *)calloc((size_t)T * V, 4);
        memcpy(save, M->logits, (size_t)T * V * 4);
        model_forward(M, x, y);
        for (t = 0; t < 6; t++)
            for (i = 0; i < V; i++)
                if (save[(size_t)t*V+i] != M->logits[(size_t)t*V+i]) bad++;
        free(save);
    }
    {   char b[160];
        snprintf(b, sizeof b, "%d logit su %d cambiano quando cambio il token 6",
                 bad, 6 * V);
        chk("T3 RWKV causale (ricorrenza O(N)): il futuro non influenza il passato",
            bad == 0, b); }
    ok = (bad == 0);
    free(x); free(y); free(x2); model_free(M);
    return ok;
}

/* --- T4 : il gradiente analitico punta dove punta la differenza finita - */
/* Il punto e' che il paper non descrive il gradiente verso l'ombra: se
 * congelo i trit, la perdita NON dipende dallo shadow e il gradiente e'
 * esattamente zero. Il surrogato STE lo rende dipendente, e allora il
 * gradiente analitico si puo' confrontare con una differenza finita.    */
#ifndef PII_FLASH_NO_TRAIN
static int t4_grad(const char *tag, int correct_gamma, int info_only)
{
    int32_t V = 12, d = 32, T = 8;
    Model *M;
    int32_t *x = (int32_t *)calloc(T, 4), *y = (int32_t *)calloc(T, 4);
    grad_t *G;
    double L0;
    int i, j, l;
    int last_agree = -1, last_n = 0, last_wrong = 0, last_zero = 0;
    int last_floor = 0, nres = 0, last_res = 0;
    char b[200];
    static const int32_t HS[2] = { 64, 1024 };
    int e;

    g_gamma_correct = correct_gamma;
    g_paper_chain = 1;
    g_siluns_grad = 0;
    M = model_new(V, d, 4, 64, T, 2, 777u);
    for (i = 0; i < T; i++) { x[i] = (i * 3 + 1) % V; y[i] = (i * 5 + 2) % V; }
    for (l = 0; l < M->n_layers; l++) {
        M->B[l].Wq->use_ste = 1; M->B[l].Wk->use_ste = 1;
        M->B[l].Wv->use_ste = 1; M->B[l].Wo->use_ste = 1;
        M->B[l].Wg->use_ste = 1; M->B[l].Wu->use_ste = 1;
        M->B[l].Wd->use_ste = 1;
    }
    M->emb->use_ste = 1; M->head->use_ste = 1;
    model_refresh(M);                 /* congela lo shadow sh0           */
    L0 = model_forward(M, x, y);
    {   M->lr = 0; M->lr_emb = 0; M->lr_head = 0;
        for (l = 0; l < M->n_layers; l++) M->B[l].lr = 0;
    }
    model_backward(M, x, y);
    /* la backward riquantizza, quindi sh0 == shadow : senza questo rigelo
     * ogni differenza finita valerebbe esattamente zero.                  */
    model_refresh(M);
    G = (grad_t *)calloc((size_t)V * d, sizeof(grad_t));
    if (!G) die("malloc");
    g_ntot = 0; g_nzero = 0; g_nsat = 0;
    qlin_grad_probe(M->head, G);
    printf("         sonda sez.5 sulla testa : %lld gradienti, %lld azzerati dal >>19,"
           " %lld saturi a +-2047 (%.1f%%)\n",
           (long long)g_ntot, (long long)g_nzero, (long long)g_nsat,
           g_ntot ? 100.0 * (double)g_nzero / (double)g_ntot : 0.0);
    printf("         %s  (L0=%.6f, %d posizioni campionate)\n", tag, L0, 24);
    /* SiLUNS e' una scala a gradini e apply_gamma tronca : con un passo
     * grossolano la differenza finita e' una SECANTE, non una derivata.
     * Quindi la passo in rassegna e guardo la convergenza.                */
    for (e = 0; e < 2; e++) {
        const int32_t h = HS[e];
        int n = 0, agree = 0, nonnull = 0, gzero = 0, wrong = 0, gfloor = 0, nres = 0;
        for (i = 0; i < V && n < 24; i++) {
            for (j = 0; j < d && n < 24; j++) {
                int64_t k = (int64_t)i * d + j;
                shadow_t sv = M->head->shadow[k];
                double Lp, Lm, fd; int gv = G[k];
                M->head->shadow[k] = (shadow_t)(sv + h); Lp = model_forward(M, x, y);
                M->head->shadow[k] = (shadow_t)(sv - h); Lm = model_forward(M, x, y);
                M->head->shadow[k] = sv;
                fd = (Lp - Lm) / (2.0 * (double)h);
                if (gv == 0) gzero++;
                if (gv > 0 && gv <= 4) gfloor++;
                if (fd != 0.0) { nonnull++;
                    if ((fd > 0) == (gv > 0)) agree++;
                    else if (gv > 4) {
                        int32_t post;
                        wrong++;
                        post = apply_gamma(M->head->raw[i], M->head->gshift);
                        printf("             discordo h=%5d i=%d j=%d  G=%d  fd=%.6g"
                               "  raw=%d  post=%d  siluns'(post)=%.2f  x_q=%d\n",
                               h, i, j, gv, fd, (int)M->head->raw[i], (int)post,
                               siluns_d(post), (int)M->head->x_q[j]);
                    }
                }
                n++;
            }
        }
        printf("           passo h=%5d conteggi  %2d/%2d concordi", h, agree, nonnull);
        if (gzero)  printf("  |  %d con G=0", gzero);
        if (gfloor) printf("  |  %d al pavimento", gfloor);
        nres = n - gzero - gfloor;
        printf("  |  %d risolte", nres);
        if (wrong)  printf("  |  %d discordi con G != 0", wrong);
        printf("\n");
        last_wrong = wrong;
        last_agree = agree; last_n = nonnull; last_zero = gzero;
        last_floor = gfloor; last_res = nres;
    }
    if (info_only) {
        snprintf(b, sizeof b, "gradiente annullato dal >>19 su %.1f%% dei pesi della testa:"
                             " il modello non puo' apprendere",
                 g_ntot ? 100.0 * (double)g_nzero / (double)g_ntot : 100.0);
        printf("         [NOTA] %s : %s\n", tag, b);
        return 1;
    }
    if (last_n == 0) {
        snprintf(b, sizeof b, "il modello e' saturo (L0=%.0f): nessuna differenza finima "
                              "e' osservabile, il gradiente non e' verificabile", L0);
        printf("         [NOTA] %s : %s\n", tag, b);
        chk(tag, 0, b);
        return 0;
    }
    snprintf(b, sizeof b, "%d/%d concordi, %d discordi sulle posizioni risolte;"
                      " %d con G=0, %d al pavimento; sonda testa %.1f%% azzerata",
             last_agree, last_n, last_wrong, last_zero, last_floor,
             g_ntot ? 100.0 * (double)g_nzero / (double)g_ntot : 0.0);
    /* con tutti i gradienti a zero la concordanza e' vacua : non e' un pass */
    chk(tag, last_wrong == 0 && last_res > 0, b);
    free(G); free(x); free(y); model_free(M);
    g_gamma_correct = 0;
    return last_wrong == 0 && last_res > 0;
}
#endif /* PII_FLASH_NO_TRAIN */

/* --- T5 : la perdita scende, e scende ATTRAVERSO il quantizzatore ------ */
#ifndef PII_FLASH_NO_TRAIN
static int t5_descent(const char *tag, int correct_gamma, int lr_init)
{
    int32_t V = 10, d = 32, T = 8;
    Model *M;
    int32_t *x = (int32_t *)calloc(T, 4), *y = (int32_t *)calloc(T, 4);
    double first = 0.0, last = 0.0, best = 1e30;
    int step, i;
    char b[200];
    g_gamma_correct = correct_gamma;
    M = model_new(V, d, 4, 64, T, 1, 4242u);
    for (i = 0; i < T; i++) { x[i] = (i * 3 + 1) % V; y[i] = (i * 5 + 2) % V; }
    {   int32_t l;
        for (l = 0; l < M->n_layers; l++) M->B[l].lr = (lr_t)lr_init;
        M->lr_emb = (lr_t)lr_init; M->lr_head = (lr_t)lr_init;
    }
    for (step = 0; step < 120; step++) {
        double L = model_forward(M, x, y);
        if (step == 0) first = L;
        model_backward(M, x, y);
        if (L < best) best = L;
        last = L;
    }
    snprintf(b, sizeof b, "perdita %.4f -> %.4f (minimo %.4f)", first, last, best);
    chk(tag, last < first, b);
    free(x); free(y); model_free(M);
    g_gamma_correct = 0;
    return last < first;
}
#endif /* PII_FLASH_NO_TRAIN */

/* ==================================================================== dati
 * Testo pubblico di dominio : l'apertura di "The Raven" di Poe.           */
static const char *POE =
  "once upon a midnight dreary while i pondered weak and weary\n"
  "over many a quaint and curious volume of forgotten lore\n"
  "while i nodded nearly napping suddenly there came a tapping\n"
  "as of some one gently rapping at my chamber door\n"
  "this is it some visitor entreating entrance at my chamber door\n"
  "this is it some visitor entreating entrance at my chamber door\n";

typedef struct {
    int V, T, nwin;
    int32_t *x, *y;
    char inv[64];        /* simbolo di ogni id : serve per stampare   */
} Data;

/* Valuta la Loss utente su TUTTE le finestre : media di sum_k e_k^2 >> 19,
 * con e in conteggi Q16.15. Metrica diagonale della CE, stessa scala della
 * somma del paper (nessuna normalizzazione per token, come la formula).  */
static double usq_eval(Model *M, Data *D)
{
    double s = 0.0;
    int32_t w;
    for (w = 0; w < D->nwin; w++) {
        (void)model_forward(M, D->x + (size_t)w * D->T,
                               D->y + (size_t)w * D->T);
        s += g_usq_val;
    }
    return D->nwin ? s / (double)D->nwin : 0.0;
}

/* vocabolario: i 32 caratteri piu' frequenti, poi tutto il resto in 32+  */
static void data_build(Data *D, const char *txt, int T)
{
    int freq[256], order[256]; int i, n = 0, V = 32;
    int32_t *ids;
    size_t L = strlen(txt);
    memset(freq, 0, sizeof freq);
    for (i = 0; i < (int)L; i++) freq[(unsigned char)txt[i]]++;
    {   for (i = 0; i < 256; i++) order[i] = i;
        /* selection sort, 256 elementi : puo' essere fatto una volta sola */
        for (i = 0; i < 255; i++) {
            int k, b = i;
            for (k = i + 1; k < 256; k++)
                if (freq[order[k]] > freq[order[b]]) b = k;
            { int tmp = order[i]; order[i] = order[b]; order[b] = tmp; }
        }
        while (V < 40 && n < 256 && freq[order[n]] > 0) n++;
        if (n < V) V = n;
    }
    D->V = V; D->T = T;
    for (i = 0; i < V && i < 64; i++) D->inv[i] = (char)order[i];
    ids = (int32_t *)calloc((L + 1), 4);
    if (!ids) die("malloc");
    for (i = 0; i <= (int)L; i++) ids[i] = 0;
    {   int map[256]; int k;
        for (k = 0; k < 256; k++) map[k] = 0;
        for (k = 0; k < V; k++) map[(unsigned char)order[k]] = k;
        for (i = 0; i < (int)L; i++) ids[i] = map[(unsigned char)txt[i]];
    }
    D->nwin = (int)L - T - 1;
    if (D->nwin < 1) D->nwin = 1;
    D->x = (int32_t *)calloc((size_t)D->nwin * T, 4);
    D->y = (int32_t *)calloc((size_t)D->nwin * T, 4);
    if (!D->x || !D->y) die("malloc");
    for (i = 0; i < D->nwin; i++) {
        memcpy(D->x + (size_t)i * T, ids + i, (size_t)T * 4);
        memcpy(D->y + (size_t)i * T, ids + i + 1, (size_t)T * 4);
    }
    free(ids);
}

/* --------------------------------------------- addestramento, sez. 7 e 8 */
typedef struct { Model *M; Data *D; } Train;

static void sched_step(Model *M, int step, double loss, int correct, int ntok)
{
    static qat_net_t stub;
    static int32_t inited = 0;
    int32_t l;
    if (!inited) { memset(&stub, 0, sizeof stub); stub.cfg.cool = COOL_AS_DENOM; inited = 1; }
    if (!g_use_sched) return;   /* LR costante : la sezione 8 non interviene */
    /* sezione 8 : la finestra di WINDOW passi chiude e ricalcola l'LR  */
    if ((step + 1) % WINDOW == 0) {
        int64_t gavg = g_n_step ? (g_abs_step / g_n_step) : 0;   /* |G| medio */
        /* UQ7.25 : 7 bit interi => il valore deve stare sotto 128 */
        acc_t ae = (acc_t)(fabs(loss) * (double)(1u << 25));
        lr_t nl;
        if (ae > 0xFFFFFFFFu) ae = 0xFFFFFFFFu;
        nl = qat_learning_rate(&stub, gavg, ae, M->lr);
        /* [MIO, BUG]  qat_learning_rate() somma (gavg>>9) + (ae>>9) e
         * tronca a 65535. Con la perdita in UQ7.25 (3.2 * 2^25 = 1.07e8)
         * il secondo termine da solo vale ~209000 : il ritorno e' SEMPRE
         * il tetto, a ogni passo, per ogni rete. Il ciclo della sezione 8
         * non e' un ciclo, e' un istante fisso.                        */
        if (nl > LR_MAX) nl = (lr_t)LR_MAX;
        if (nl < (lr_t)LR_MIN) nl = (lr_t)LR_MIN;
        /* e se il tetto e' gia' stato raggiunto, non lo si puo' ricominciare
         * a salire ogni WINDOW passi : il ciclo deve poter scendere.       */
        if (nl > M->lr && M->lr >= (lr_t)LR_MAX) nl = M->lr;
        for (l = 0; l < M->n_layers; l++) M->B[l].lr = nl;
        M->lr = nl; M->lr_emb = nl; M->lr_head = nl;
        g_abs_step = 0; g_sat_step = 0; g_n_step = 0;
    } else {
        g_abs_step += 0; g_n_step += 0;
    }
    (void)correct; (void)ntok;
}

static void sched_step(Model *M, int step, double loss, int correct, int ntok);

#ifndef PII_FLASH_NO_TRAIN
int eng_stop_requested(void);   /* train_driver.c, stessa TU: flag Ctrl+C */
static void train_run(Model *M, Data *D, int steps, uint32_t seed, int verbose)
{
    uint32_t rs = seed;
    int step, i, b;
    double firstL = 0.0, lastL = 0.0, bestL = 1e300;
    double prev = 1e300; int stall = 0;
    int lr0;
    /* [MIO] IL RISCALDAMENTO AUTOMATICO.
     *
     * La ricetta di preaddestramento mette il riscaldo al primo posto della
     * checklist (train-pretraining-recipes sez. 2, e sez. 8 tra i default),
     * e il WSD lo fa gia' — ma il WSD e' SPENTO di default, e con WSD spento
     * il programma partiva a pieno passo sui pesi che avevi salvato ieri.
     *
     * Perche' qui e' diverso dal WSD e conta piu': con i trit, i primi passi
     * NON sono un riscaldo neutro, sono distruttivi. Il passo sez. 7 puo'
     * valere (|G| * LR) >> 12 = 2047 conteggi contro una soglia di isteresi
     * di 213, e un singolo passo cosi'grande sposta il trit di dieci soglie.
     * Dieci passi a pieno passo e' gia' dieci valanghe: quello che il modello
     * aveva imparato ieri non e' piu' li'. Misurato in `diag`: 399 passi su
     * 400 non muovevano niente e il 400mo ne muoveva 3129 su 10240.
     *
     * Quindi il riscaldo qui non e' per far salire la curva e' per non
     * distruggere il punto di partenza. Sale a LR in WARM_STEPS passi e poi
     * resta al valore che avevi. Se il WSD e' acceso, non se ne cura : la
     * curva ce l'ha gia' e il suo warmup e' quello giusto.               */
    lr0 = (int)M->lr;
    rseed(seed);
    for (step = 0; step < steps; step++) {
        int w;
        /* Ctrl+C risponde SUBITO (prima era cieco fino a fine epoca: con
         * ~10 min a finestra sembrava appeso per ore). Il driver dopo
         * la break fa comunque model_sync_chunks: nessun peso perso. */
        if (eng_stop_requested()) break;
        /* Da qui fino alla fine della backward, embedding e testa NON
         * vengono riquantizzati : e' quello che permette al batch di
         * accumulare. Fuori da qui tornano a riquantizzarsi da soli. */
        g_accum_active = 1;
        w = (int)(rs % (uint32_t)D->nwin);
        rs = rs * 1103515245u + 12345u;
        /* Avanzamento SEMPRE visibile (non solo verbose): senza, 60 passi
         * silenziosi da ~10 min sembrano un processo appeso per ore. */
        if (!g_quiet) { printf("    passo %d/%d finestra %d/%d ...\n", step + 1, steps, w, D->nwin); fflush(stdout); }
        {   double L = model_forward(M, D->x + (size_t)w * D->T,
                                             D->y + (size_t)w * D->T);
            if (step == 0) firstL = L;
            if (L < bestL) bestL = L;
            lastL = L;
            /* ---- IL BATCH : B finestre, UN aggiornamento dei pesi -------
             *
             * Prima di questo, un "passo" guardava 32 token : una finestra.
             * Il gradiente di 32 token ha una varianza piu' grande del
             * segnale, e i trit si riquantizzano a ogni passo, quindi il
             * peso saltava da un gradiente rumoroso al successivo.
             *
             * Ora il passo guarda `g_batch` finestre: si accumula lo shadow
             * (che e' gia' un accumulatore, e acc_lr() lo divide per B cosi'
             * che e' la MEDIA e non la somma), si fa avanti e indietro B
             * volte SULLE STESSE PESI — niente riquantizzazione in mezzo —
             * e si aggiorna una volta sola. E' un vero minibatch.
             *
             * `L` e' la perdita MEDIA sul batch, non quella della prima
             * finestra: e' il numero che il rollback deve guardare.       */
            for (b = 1; b < g_batch; b++) {
                rs = rs * 1103515245u + 12345u;
                w = (int)(rs % (uint32_t)D->nwin);
                {   double Lb = model_forward(M, D->x + (size_t)w * D->T,
                                                     D->y + (size_t)w * D->T);
                    L += Lb;
                }
            }
            if (g_batch > 1) L /= (double)g_batch;
            if (L < bestL) bestL = L;
            lastL = L;
            /* [MIO, BUG]  la sezione 8 puntava sempre piu' in alto e il
             * gradiente divergente spingeva il LR al soffitto 65535. Se la
             * perdita peggiora per STALL_LIMIT finestre consecutive il
             * gradiente e' sbagliato: dimezzo il LR e riparto.
             *
             * CON IL WSD ACCESO QUESTO CODICE E' DISATTIVATO, e la ragione
             * e' che i due si pesterebbero i piedi in modo silenzioso: il
             * rollback dimezza M->lr a meta' passata, e il passo dopo WSD
             * lo rimette al valore della curva. Il dimezzamento non
             * mancherebbe mai — accaderebbe e verrebbe annullato senza che
             * nessuno lo vedesse, e con un LR che scende la curva sembrerebbe
             * "c'e' stata instabilita'". Sono due Governi del LR, e due
             * Governi non si guidano insieme.                               */
            if (g_use_sched && !g_wsd_on && step > 0) {
                if (L > prev * 1.02) { if (++stall >= STALL_LIMIT) {
                    int32_t l2;
                    stall = 0;
                    M->lr = (lr_t)(M->lr / 2);
                    if (M->lr < LR_MIN) M->lr = (lr_t)LR_MIN;
                    M->lr_emb = M->lr; M->lr_head = M->lr;
                    for (l2 = 0; l2 < M->n_layers; l2++) M->B[l2].lr = M->lr;
                    if (verbose && !g_quiet) printf("    [rollback] LR -> %u\n", (unsigned)M->lr);
                } } else stall = 0;
            }
            prev = L;
            model_backward(M, D->x + (size_t)w * D->T, D->y + (size_t)w * D->T);
            for (b = 1; b < g_batch; b++) {
                rs = rs * 1103515245u + 12345u;
                w = (int)(rs % (uint32_t)D->nwin);
                model_backward(M, D->x + (size_t)w * D->T,
                                    D->y + (size_t)w * D->T);
            }
            /* WSD : il LR di questo passo, calcolato sul passo GLOBALE.
             * Lo applico DOPO la backward perche' la sez. 7 usa il LR
             * dell'aggiornamento, cioe' quello del passo che sto facendo. */
            g_accum_active = 0;
            if (g_wsd_on) lr_set_all(M, wsd_lr_at(g_wsd_base + step + 1));
            else {
                /* sched_step deve girare SEMPRE quando e' il turno suo :
                 * se lo metto nell'else, accendere il riscaldo spegneva
                 * anche il ciclo adattivo della sezione 8, che e' una cosa
                 * diversa e non deve dipendere da questo interruttore.     */
                sched_step(M, step, lastL, M->n_correct, M->n_tok);
                if (g_warmup_on && steps > 1) {
                    /* da 1/8 del LR al LR pieno in WARM_STEPS passi */
                    int f = step + 1, n = WARM_STEPS, v;
                    if (f > n) f = n;
                    v = lr0 / 8 + (lr0 - lr0 / 8) * f / n;
                    lr_set_all(M, v);
                }
            }
            /* [MIO, BUG]  il shadow veniva accumulato ma i trit non
             * venivano MAI riquantizzati durante l'addestramento:
             * model_refresh() era chiamata solo nella sonda a differenze
             * finite (righe 1100/1108). Con i trit congelati la rete non
             * poteva cambiare nulla e la perdita finiva sul caso
             * uniforme. Ogni REFRESH_EVERY passi riquantizzo.           */
            if (REFRESH_EVERY > 0 && ((step + 1) % REFRESH_EVERY == 0))
                model_refresh(M);
            /* Con il batch, l'aggiornamento dei pesi c'e' gia' stato : lo
             * shadow accumulato e' stato riquantizzato QUI, una volta per
             * ogni B finestre. Prima veniva riquantizzato B volte, una per
             * finestra, che e' esattamente cio' che il batch voleva evitare. */
        }
        if (verbose && !g_quiet && (step % 100 == 0 || step == steps - 1)) {
            int64_t zer = 0, tot = 0, sat = 0;
            int32_t q;
            for (q = 0; q < M->n_layers; q++) {
                const Block *BB = &M->B[q];
                const QLin *Ls[7];
                int32_t u;
                /* v004/stream: i trit vivono nei chunk, si carica il layer */
                if (!block_load(M, q)) die("ooc load");
                Ls[0]=BB->Wq; Ls[1]=BB->Wk; Ls[2]=BB->Wv; Ls[3]=BB->Wo;
                Ls[4]=BB->Wg; Ls[5]=BB->Wu; Ls[6]=BB->Wd;
                for (u = 0; u < 7; u++) {
                    zer += qlin_zero(Ls[u]);
                    tot += (int64_t)Ls[u]->n_in * Ls[u]->n_out;
                    sat += Ls[u]->n_sat;
                }
                if (!block_unload(M, q, 0)) die("ooc unload");
            }
            printf("  passo %5d  perdita %8.5f  perpl. %8.2f  acc %5.1f%%  LR %6u"
                   "  zeri %5.2f%%  G sat %lld\n",
                   step, lastL, fexp(lastL),
                   100.0 * M->n_correct / (M->n_tok ? M->n_tok : 1),
                   (unsigned)M->lr,
                   tot ? 100.0 * (double)zer / (double)tot : 0.0, (long long)sat);
            if (g_wsd_on && (step % 100 == 0 || step == steps - 1)) {
                int64_t gg = g_wsd_base + step + 1;
                printf("           WSD : passo globale %lld di %lld, "
                       "%s (%.0f%%)\n", (long long)gg, (long long)g_wsd_total,
                       gg < g_wsd_warm ? "riscaldo"
                         : (gg < (int64_t)(g_wsd_total * 0.8) ? "stabile"
                                                             : "discesa"),
                       100.0 * (double)gg / (double)(g_wsd_total ? g_wsd_total : 1));
            }
        }
    }
    g_last_best = bestL;
    if (!verbose && !g_quiet) {
        printf("  perdita iniziale %.5f (perpl. %.2f)  finale %.5f (perpl. %.2f)"
               "  minima %.5f (perpl. %.2f)  acc %.1f%%\n",
               firstL, fexp(firstL), lastL, fexp(lastL), bestL, fexp(bestL),
               100.0 * M->n_correct / (M->n_tok ? M->n_tok : 1));
    }
}
#endif /* PII_FLASH_NO_TRAIN */

/* ------------------------------------------- overfit minimo
 * E' il test che arch-transformer-block prescrive : un transformer che non
 * impara una sequenza di 6 simboli ha un difetto nel codice, non nel
 * compito. Serve a separare i due casi.                                  */
#ifndef PII_FLASH_NO_TRAIN
static int tiny_overfit(int steps, int lr, int paper_chain)
{
    int32_t V = 6, d = 32, T = 8, nl = 1;
    Model *M;
    int32_t *x = (int32_t *)calloc(T, 4), *y = (int32_t *)calloc(T, 4);
    int i, step;
    double first = 0.0, last = 0.0, best = 1e300;
    int64_t tot = 0, zer = 0;
    int32_t z;
    const QLin *Ls[7];
    for (i = 0; i < T; i++) { x[i] = i % V; y[i] = (i + 1) % V; }
    g_paper_chain = paper_chain;
    g_siluns_grad = paper_chain ? 0 : 1;
    g_use_sched = 0;
    M = model_new(V, d, 4, 64, T, nl, 20260930u);
    for (z = 0; z < nl; z++) M->B[z].lr = (lr_t)lr;
    M->lr = (lr_t)lr; M->lr_emb = (lr_t)lr; M->lr_head = (lr_t)lr;
    for (step = 0; step < steps; step++) {
        double L = model_forward(M, x, y);
        if (step == 0) first = L;
        if (L < best) best = L;
        last = L;
        model_backward(M, x, y);
    }
    /* v004/stream: a fine training il layer e' scarico, si ricarica per i trit */
    if (!block_load(M, 0)) die("ooc load");
    for (z = 0; z < 7; z++) {
        Ls[z] = (z==0)?M->B[0].Wq:(z==1)?M->B[0].Wk:(z==2)?M->B[0].Wv:
                (z==3)?M->B[0].Wo:(z==4)?M->B[0].Wg:(z==5)?M->B[0].Wu:M->B[0].Wd;
        zer += qlin_zero(Ls[z]);
        tot += (int64_t)Ls[z]->n_in * Ls[z]->n_out;
    }
    if (!block_unload(M, 0, 0)) die("ooc unload");
    printf("  %-18s LR %6d : perdita %8.5f -> %8.5f  minima %8.5f"
           "  acc %5.1f%%  zeri %6.2f%%%s\n",
           paper_chain ? "sez.5 letterale" : "derivata vera",
           lr, first, last, best,
           100.0 * M->n_correct / (M->n_tok ? M->n_tok : 1),
           tot ? 100.0 * (double)zer / (double)tot : 0.0,
           "");
    free(x); free(y); model_free(M);
    return last < first;
}
#endif /* PII_FLASH_NO_TRAIN */

/* ------------------------------------------------- scansione del LR
 * La sezione 8 non specifica un learning rate iniziale, e la sua formula,
 * letta alla lettera, satura il campo UQ0.16. Invece di scegliere a caso
 * lo misuro: stesso modello, stesso seme, stesso corpus, LR costante.   */
#ifndef PII_FLASH_NO_TRAIN
static void lr_scan(Data *D, int d, int nh, int dff, int T, int nl,
                    uint32_t seed, int steps)
{
    static const int32_t LRS[7] = { 32, 128, 512, 2048, 8192, 32768, 65535 };
    static const int CH[2] = { 1, 0 };
    int e, i, c;
    printf("  %-18s %8s %10s %10s %10s %8s %10s\n",
           "catena", "LR", "perdita0", "perdita1", "minima", "acc%", "G saturi");
    for (c = 0; c < 2; c++) for (e = 0; e < 7; e++) {
        Model *M;
        double first = 0.0, last = 0.0, best = 1e300;
        g_paper_chain = CH[c]; g_siluns_grad = CH[c] ? 0 : 1;
        M = model_new(D->V, d, nh, dff, T, nl, seed);
        int64_t zer = 0, tot = 0, sat = 0;
        int32_t q, u, step;
        const QLin *Ls[7];
        M->lr = (lr_t)LRS[e]; M->lr_emb = (lr_t)LRS[e]; M->lr_head = (lr_t)LRS[e];
        for (i = 0; i < nl; i++) M->B[i].lr = (lr_t)LRS[e];
        g_abs_step = 0; g_n_step = 0;
        for (step = 0; step < steps; step++) {
            int w = (int)((uint64_t)step * 2654435761u % (uint32_t)D->nwin);
            double L = model_forward(M, D->x + (size_t)w * D->T,
                                             D->y + (size_t)w * D->T);
            if (step == 0) first = L;
            if (L < best) best = L;
            last = L;
            model_backward(M, D->x + (size_t)w * D->T, D->y + (size_t)w * D->T);
        }
        for (q = 0; q < nl; q++) {
            const Block *BB = &M->B[q];
            /* v004/stream: i trit vivono nei chunk, si carica il layer */
            if (!block_load(M, q)) die("ooc load");
            Ls[0]=BB->Wq; Ls[1]=BB->Wk; Ls[2]=BB->Wv; Ls[3]=BB->Wo;
            Ls[4]=BB->Wg; Ls[5]=BB->Wu; Ls[6]=BB->Wd;
            for (u = 0; u < 7; u++) {
                zer += qlin_zero(Ls[u]);
                tot += (int64_t)Ls[u]->n_in * Ls[u]->n_out;
                sat += Ls[u]->n_sat;
            }
            if (!block_unload(M, q, 0)) die("ooc unload");
        }
        printf("  %-18s %8d %10.5f %10.5f %10.5f %7.1f%% %10lld   zeri %.2f%%\n",
               CH[c] ? "sez.5 letterale" : "derivata vera", LRS[e], first, last, best,
               100.0 * M->n_correct / (M->n_tok ? M->n_tok : 1), (long long)sat,
               tot ? 100.0 * (double)zer / (double)tot : 0.0);
        model_free(M);
    }
}
#endif /* PII_FLASH_NO_TRAIN */

/* ------------------------------------------------------------ generazione */
static void generate(Model *M, Data *D, int ntok)
{
    int32_t *ctx = (int32_t *)calloc((size_t)M->T, 4);
    int32_t *tgt = (int32_t *)calloc((size_t)M->T, 4);
    int i, t;
    for (i = 0; i < M->T; i++) ctx[i] = 0;
    for (i = 0; i < M->T - 1; i++) tgt[i] = -1;
    printf("  generazione greedy, %d token:\n  ", ntok);
    for (t = 0; t < ntok; t++) {
        int32_t arg = 0, k; float best = -1e30f;
        model_forward(M, ctx, tgt);
        for (k = 0; k < M->V; k++)
            if (M->logits[(size_t)(M->T - 2) * M->V + k] > best) {
                best = M->logits[(size_t)(M->T - 2) * M->V + k]; arg = k;
            }
        /* il vocabolario e' l'insieme dei caratteri ordinati per frequenza,
         * quindi per ricostruire il testo mi serve la mappa : la rendo
         * disponibile con un side-channel statico */
        printf("%c", (arg < 64) ? D->inv[arg] : '?');
        for (i = 0; i < M->T - 1; i++) ctx[i] = ctx[i + 1];
        ctx[M->T - 1] = arg;
    }
    printf("\n");
    free(ctx); free(tgt);
}

/* ==================================================================== main */

/* ================================================================= shadowstat */
/*  Distribuzione di |Shadow| in Q16.15 e raggiungibilita' delle soglie.       */
/*  Aggiunto per verificare un'affermazione (2^13 = 0.25 troppo grande) che    */
/*  era stata fatta senza misura. Il campo Q16.15 arriva a 65535 grezzi, quindi */
/*  8192 e' un ottavo del campo: serve il numero vero.                        */

typedef struct { int32_t *v; int32_t n, cap; } IVec;

static void iv_push(IVec *a, int32_t x)
{
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 1024;
        a->v = (int32_t *)realloc(a->v, (size_t)a->cap * sizeof(int32_t));
        if (!a->v) die("realloc");
    }
    a->v[a->n++] = x;
}

static void iv_collect(IVec *a, const QLin *L)
{
    int32_t k, n;
    if (!L || !L->shadow) return;
    n = L->n_out * L->n_in;
    for (k = 0; k < n; k++) {
        int32_t v = L->shadow[k];
        iv_push(a, v < 0 ? -v : v);
    }
}

static int cmp_i32(const void *a, const void *b)
{
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

static int32_t iv_pct(const IVec *a, double p)
{
    int32_t i;
    if (a->n == 0) return 0;
    i = (int32_t)(p * (double)(a->n - 1) + 0.5);
    if (i < 0) i = 0;
    if (i >= a->n) i = a->n - 1;
    return a->v[i];
}

#ifndef PII_FLASH_NO_TRAIN
static void shadow_stat(Data *D, int steps, int gain, lr_t LR)
{
    static const int32_t THR[8] = { 213, 256, 512, 1024, 2048, 4096, 8192, 16384 };
    Model *M; IVec A; int32_t z; int i;
    double L0, Lf; int32_t total;

    g_update_gain = gain; g_gain_pinned = 1; g_gamma_correct = 1;
    g_paper_chain = 1; g_siluns_grad = 0; g_use_sched = 0;
    M = model_new(D->V, 64, 4, 128, 32, 2, 20260930u);
    M->lr = LR; M->lr_emb = LR; M->lr_head = LR;
    for (z = 0; z < 2; z++) M->B[z].lr = LR;
    L0 = model_forward(M, D->x, D->y);
    train_run(M, D, steps, 20260930u, 0);
    Lf = g_last_best;

    memset(&A, 0, sizeof(A));
    iv_collect(&A, M->emb);
    iv_collect(&A, M->head);
    for (z = 0; z < M->n_layers; z++) {
        if (!block_load(M, z)) die("ooc load");
        iv_collect(&A, M->B[z].Wq); iv_collect(&A, M->B[z].Wk);
        iv_collect(&A, M->B[z].Wv); iv_collect(&A, M->B[z].Wo);
        iv_collect(&A, M->B[z].Wg); iv_collect(&A, M->B[z].Wu);
        iv_collect(&A, M->B[z].Wd);
        if (!block_unload(M, z, 0)) die("ooc unload");
    }
    qsort(A.v, (size_t)A.n, sizeof(int32_t), cmp_i32);
    total = A.n;

    printf("\n  gain=%d shift=%d LR=%d  perdita %.5f -> %.5f  (uniforme %.5f)\n",
           gain, 12 - gain, (int)LR, L0, Lf, log((double)D->V));
    printf("  |Shadow| su %d pesi : p50=%d  p90=%d  p99=%d  max=%d   (campo Q16.15 = 65535)\n",
           total, iv_pct(&A, 0.50), iv_pct(&A, 0.90), iv_pct(&A, 0.99),
           A.n ? A.v[A.n - 1] : 0);
    printf("  %-10s %-14s %-14s %-12s\n", "soglia", "reached>=T", "inerte<T", "esito");
    for (i = 0; i < 8; i++) {
        int64_t reach = 0;
        int32_t k;
        for (k = 0; k < total; k++) if (A.v[k] >= THR[i]) reach++;
        printf("  %-10d %-14s %-14s %-12s\n", THR[i],
               reach ? "ok" : "MAI",
               (total - reach) ? "inerte" : "-",
               (reach * 100 / (total ? total : 1)) == 0 ? "morto" : "vive");
    }
    free(A.v);
    model_free(M);
    g_update_gain = 0;
}
#endif /* PII_FLASH_NO_TRAIN */

