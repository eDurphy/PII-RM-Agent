# QATNN-HFP: un motore neurale ternario in virgola fissa ibrida con ricorrenza lineare

**Revisione r1 — allineata al codice v003.** Ogni formula riportata è quella implementata; dove il codice ammette più modalità, la modalità di default è dichiarata. Le rettifiche rispetto alla bozza precedente sono integrate nei §§3–4 e riepilogate al §8.

> **Tesi.** A differenza dei motori ternari che impongono la quantizzazione come vincolo postumo, QATNN rende il trit l'unità di conto di tutto l'attraversamento matriciale — forward e backward in sola aritmetica intera — e confina la virgola mobile alle sole normalizzazioni vettoriali; così ottiene inferenza senza moltiplicatori a 1,6 bit per peso senza che il training ternario si congeli.

## Abstract

QATNN (Quantized Adaptive Trainer Neural Network) è un motore di training e inferenza in cui i pesi assumono solo valori in {-1, 0, +1} (trit), governati da pesi ombra continui e da una mappa di isteresi con soglia T = 213. Il forward accumula in Q16.15 con input pre-scalati dal decay del layer e scala l'uscita con un gamma-shift intero calcolato via `__builtin_clz`; l'attivazione SiLUNS approssima la SiLU con soli shift e rami lineari. Il backward non materializza mai la matrice dei gradienti (prodotti esterni di rango 1 valutati riga per riga), propaga l'errore attraverso i trit trasposti e aggiorna gli shadow con passi interi. Lo stato va su disco in chunk I/S per layer con al massimo 2 layer residenti in RAM e prefetch asincrono. La variante linguistica sostituisce l'attention con ricorrenza lineare RWKV-6/Finch, blocchi SwiGLU, RMSNorm in fp32 e BPE a caratteri (V = 1024). Quattro leve di convergenza — grad-scale con media mobile a 8 passi, Update_Gain di default 0, init gaussiana σ = 2048, schedule WSD con stall-halving — portano il passo d'aggiornamento nella fascia in cui la soglia di isteresi è attraversabile: con la regola dichiarata il 96,0% delle coppie (|G|, LR) muove un trit a T = 213 contro il 40,3% a T = 8192 (derivato analiticamente, §7).

## 1. Introduzione

Il fenomeno è noto: portare una rete in bassa precisione dopo l'addestramento degrada la qualità proprio dove la rete è più sensibile, e il recupero richiede riaddestramento in un formato che l'hardware non esegue nativamente. La conclusione che ne traiamo è che la quantizzazione va progettata come regime aritmetico del motore, non come compressione a posteriori. Il gap è che i regimi ternari esistenti (BitNet b1.58 [Ma et al., 2024]) descrivono l'inferenza ma lasciano l'aggiornamento in virgola mobile. La proposta di questo lavoro è un motore in cui anche l'aggiornamento è intero: shadow in Q16.15, gradienti in Q4.11, learning rate in UQ0.16, e virgola mobile solo fuori dal percorso matriciale (§4.1). Promettiamo: (a) la specifica bit-esatta di ogni operatore; (b) la misura di raggiungibilità della soglia di isteresi; (c) la separazione onesta tra nucleo verificato ed estensioni marcate [MIO].

## 2. Stato dell'arte

**Training quantizzato intero.** Jacob et al. (arXiv 2017; CVPR 2018) fissano la base teorica dell'inferenza a soli interi; QATNN ne eredita i formati Q ma sostituisce l'inferenza intera pura con un regime ternario a pesi ombra e isteresi. **Reti ternarie.** BitNet b1.58 [Ma et al., 2024] e il rapporto tecnico 2B4T [Ma et al., 2025] dimostrano LLM a 1,58 bit; i lavori 2026 Sparse-BitNet (Zhang et al., arXiv:2603.05168, sottotitolo esatto: "1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity") e TernaryLM (Nargund e Shukla, arXiv:2602.07374, "Memory-Efficient Language Modeling via Native 1.5-Bit Quantization with Adaptive Layer-wise Scaling") estendono il filone a sparsità strutturata e scaling adattivo per layer — entrambi preprint senza sede, citati come tali. Questi lavori non addestrano in regime intero: è il nostro spazio. **Ricorrenza lineare.** RWKV (Peng et al., Findings di EMNLP 2023; arXiv:2305.13048) sostituisce l'attention con ricorrenza O(T); RWKVQuant (Xu et al., ICML 2025, PMLR vol. 267, pp. 70047–70064; arXiv:2505.03803) ne studia la quantizzazione PTQ con guida ibrida scalare/vettoriale. QATNN ne adotta la ricorrenza ma addestra i pesi ternari nativamente. **Subword e attivazioni.** BPE [Sennrich et al., arXiv 2015; ACL 2016] e SiLU [Elfwing et al., arXiv 2017; Neural Networks 2018] sono le due basi esterne: il primo per il vocabolario, la seconda come riferimento approssimato da SiLUNS.

## 3. Metodo

### 3.1 Gamma-shift con istruzioni hardware (§1 del nucleo)

Per ogni layer, `gammraw = (Σ|shadow|) >> ⌊log2(N_in)⌋`, con somma in int64 e N_in pari alla dimensione del blocco (potenza di due nelle configurazioni supportate; la guardia `N_in == 0` evita `__builtin_clz(0)`, indefinito in C). Lo shift si ricava come `gshift = 16 − ⌊log2(gammraw)⌋` (implementato via `(32 − __builtin_clz(g)) − 1`), in int8. L'applicazione è shift puro sull'accumulato, con guardia contro shift ≥ 32 (su x86 il conteggio è mascherato a 5 bit: `<< 47` diventerebbe silenziosamente `<< 15`, peggio di uno zero). Nota di rettifica: la bozza precedente riportava la formula con un segno negativo davanti a `__builtin_clz`; il codice implementa quanto sopra e le due varianti `r7`/`correct` coincidono.

### 3.2 SiLUNS r7 (§1b)

In Q16.15, con punto di minimo a x = −41900 (≈ −1,28): x ≥ 0 → x; 0 > x > −41900 → x >> 2; x ≤ −41900 → −(x >> 2) − 20950. Continuità al confine: −(−41900 >> 2) − 20950 = 10475 − 20950 = −10475, pari al ramo di discesa (−41900 >> 2 = −10475 ≈ −0,319, contro il minimo vero della SiLU a −0,278). Costo O(1) senza esponenziali né divisioni.

### 3.3 Isteresi, T = 213 (§2)

La mappa è: sopra +T, prev −1 → 0 (non scatto diretto a +1: asimmetria voluta); sotto −T, prev +1 → 0; dentro (−T, +T) il trit resta. All'avvio i trit sono uniformi su {−1, 0, +1} (nessuno "stato precedente") e gli shadow da gaussiana σ = 2048 (§3.7). Raggiungibilità misurata della soglia con `Δ = (|G|·LR) >> 12`, |G| ≤ 2047, LR ≤ 65535: minimo |G| che muove un trit a LR massimo = 14 per T = 213 contro 513 per T = 8192; coppie vive su griglia 96,0% contro 40,3% (derivato, §7). Il valore 8192 non è mai stato la soglia: compare solo come punto di confronto diagnostico (0,25 in Q16.15, 38,5× la soglia vera).

### 3.4 Chunk, packing, prefetch (§3–§3.1)

Ogni layer persiste come coppia I (inferenza: header con decay/gshift, trit, input/output) + S (ottimizzatore: shadow, decay_w); il backward del layer X apre anche I_{X+1} per i trit trasposti. Tre formati distinti, da non confondere: (a) streaming su disco/RAM = 1 byte per trit (scelta per SIMD e semplicità); (b) archiviazione a blocchi da 128 pesi in 26 byte = 1,625 bit/peso (`qat_packed_bytes`); (c) piatta `ceil(N/5)` = 1,600 bit/peso (`qat_repack`). A 4096×4096: 16,00 MiB → 3,25 MiB nel formato (b), 3,20 MiB nel formato (c). Vincolo residente: al massimo 2 layer in RAM con prefetch del successore su thread secondario e kill-switch `QATNN_NO_PREFETCH=1`.

### 3.5 Contratto numerico HFP (§4–§4.1)

Shadow Q16.15 int32; gradiente Q4.11 int16 in [−2047, +2047] con clip; LR UQ0.16 con pavimento 64; decay_w UQ4.12; gamma int8. Zero FPU nell'attraversamento matriciale; fp32 solo per RMSNorm, stato ricorrente RWKV e perdita Softmax/Cross-Entropy. Dettaglio implementativo: l'accumulato di layer è int32 Q16.15 e dopo gamma+SiLUNS è conservato in float per la catena [MIO] — la catena letterale del paper salta le derivate aggiunte.

### 3.6 Estensione linguistica [MIO] (§4.2)

Blocco Pre-Norm + time-mixing RWKV-6/Finch O(T) con coefficienti di mix fissi (mai appresi), MLP SwiGLU, embedding ternario con BPE a caratteri V = 256 + 768 = 1024. Niente attention, niente softmax spaziale. Le derivate di SiLUNS, RMSNorm e gate non appartengono alla catena letterale e sono marcate nel codice.

### 3.7 Backward e aggiornamento (§5–§7)

Errore d'uscita e = 2·(ŷ − y) in Q16.15 (il ">> 19" della perdita riguarda solo la stampa); errore di layer E_X = E_{X+1} · trit^T(X+1); gradiente G = (E_{X+1}·X_q) >> 19 in int64 con clip Q4.11, mai materializzato (riga per riga). Decay: gacc = Σ|ev|·|xs| con xs = x_q >> decay; G_decay = (gacc >> 8) & 0xFFFF; Δ = (G_decay·LR) >> 15; decay_w −= Δ; decay = clip(decay_w >> 12, 0, 15). Init decay del motore: round(0,5·log2(n_in)), clippato (coincide con min(15, (31−clz)>>1) << 12 per log2 pari; il nucleo usa default fisso 3). Shadow: Δ = (q·LR) >> (12 − Update_Gain), default gain 0. Le quattro leve: (1) grad-scale con media mobile a 8 passi che riporta il gradiente in scala assoluta prima del prodotto con l'ingresso (senza: tutto satura a ±2047 e resta solo il segno); (2) Update_Gain che riduce lo shift; (3) σ = 2048 ≈ 9,6 soglie — valore alto, dichiarato tale: rompe la simmetria ma nasce "deciso dal rumore"; (4) WSD con warmup automatico di 10 passi (default attivo, calibrato per riprese da pesi esistenti) e stall-halving (dimezza dopo 3 finestre in peggioramento) [MIO]; lo scheduler WSD globale è default spento. Finestra adattiva: default LR costante con ricalcolo opzionale ogni 8 passi; il ricalcolo usa (ag >> s) + (ae >> s), s = 9 + Cooling_Shift — la formula della bozza con |G_Avg| non shiftato non corrisponde al default implementato. Nota onesta: con la perdita in UQ7.25 il termine errore satura il tetto 65535, perciò il motore riclippa a LR_MAX = 4096 e vieta la risalita a tetto raggiunto. Layer cut: intero C con soglia minima 40 e tetto 42949672 (l'overflow UINT32 del prodotto intermedio è noto e contenuto dal clamp al totale).

## 4. Risultati

**Raggiungibilità della soglia (derivato analiticamente dalla regola dichiarata).** T = 213: min |G| a LR max 14; coppie vive 96,0%. T = 8192: min |G| 513; coppie vive 40,3%. A gradiente tipico |G| = 7 nemmeno T = 213 è raggiungibile senza grad-scale (max 111 < 213): la leva (1) è necessaria, non opzionale. **Convergenza.** La scala di soglie conferma che 213 vive e 8192 congela; σ/T = 9,62 contro 0,25 conferma l'init rovesciato nel secondo caso. Limite dichiarato: sono derivazioni dalla regola d'aggiornamento, non medie su ≥5 seed con intervalli — le misure d'addestramento restano lavoro futuro (§5).

## 5. Discussione e limiti

Limite della tecnica: niente attention softmax (scelta: RWKV); niente schedule di raffreddamento continuo finché non implementato; finestra-8 con I/O dominante su disco lento (mascherato dal prefetch, non eliminato). Limite dell'esperimento: un dataset, n piccolo, una lingua; tabelle senza barre d'errore perché derivate, non misurate. La soglia BITCOS (≥51% zeri, misurata da `qat_fit_bitcos`) esiste come sonda; il passaggio dinamico di formato non è implementato.

## 6. Conclusioni

Il contributo verificabile è il regime intero completo — forward, backward riga-per-riga e aggiornamento — con la misura di raggiungibilità della soglia e quattro leve dichiarate. Resta aperto: schedule continuo verificato, misure su più seed con intervalli, e attivazione BITCOS sopra soglia.

## 7. Bibliografia

1. Sennrich, R., Haddow, B. e Birch, A. (arXiv 2015; ACL 2016, pp. 1715–1725). Neural Machine Translation of Rare Words with Subword Units. arXiv:1508.07909. DOI: 10.18653/v1/P16-1162.
2. Elfwing, S., Uchibe, E. e Doya, K. (arXiv 2017; Neural Networks, vol. 107, 2018, pp. 3–11). Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning. arXiv:1702.03118. DOI: 10.1016/j.neunet.2017.12.012.
3. Ma, S. et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. arXiv:2402.17764. DOI: 10.48550/arXiv.2402.17764.
4. Ma, S. et al. (2025). BitNet b1.58 2B4T Technical Report. arXiv:2504.12285. DOI: 10.48550/arXiv.2504.12285.
5. Peng, B. et al. (Findings di EMNLP 2023, pp. 14048–14077). RWKV: Reinventing RNNs for the Transformer Era. arXiv:2305.13048. DOI: 10.18653/v1/2023.findings-emnlp.936.
6. Xu, C. et al. (ICML 2025, PMLR vol. 267, pp. 70047–70064). RWKVQuant: Quantizing the RWKV Family with Proxy Guided Hybrid of Scalar and Vector Quantization. arXiv:2505.03803. OpenReview: UOw6Qt0qYU.
7. Zhang, D. et al. (2026, preprint). Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity. arXiv:2603.05168.
8. Nargund, N. e Shukla, P. (2026, preprint). TernaryLM: Memory-Efficient Language Modeling via Native 1.5-Bit Quantization with Adaptive Layer-wise Scaling. arXiv:2602.07374.
9. Jacob, B. et al. (arXiv 2017; CVPR 2018, pp. 2704–2713). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. arXiv:1712.05877. DOI: 10.1109/CVPR.2018.00286.

## 8. Verifica pre-invio

Tesi in una frase: sì. Numeri con errore/intervallo: parziale — le tabelle di §4 sono derivazioni analitiche dichiarate come tali; le misure d'addestramento mancano (§5). Baseline con riferimento: sì per i formati esterni (BitNet). Claim con ablation: parziale (leve descritte, ablation future). Limiti tecnica/esperimento: sì. Voci lette o verificate via fonte primaria: sì per tutte (arXiv/ACL/DOI sopra); le voci 7–8 sono preprint esplicitamente marcati. Abstract scritto per ultimo: sì.
