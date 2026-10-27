/* PII-AGENT :: pii_train.h — batch training frontend for pii_trainer.exe.
 *
 *   pii_train_on_path(path, epochs)
 *     Accepts either a single document file or a folder; collects every
 *     trainable document (.txt/.md/.pdf/.csv/.json/.log), extracts PDF text
 *     via pii_pdf_extract_text() into data/train_stage/, then trains the
 *     resident engine on the explicit file list (bypasses ../datasets/).
 *
 * Returns 0 on success, nonzero on error / nothing trainable.
 */
#ifndef PII_TRAIN_H
#define PII_TRAIN_H

/* Raccoglie i documenti allenabili per `path` (file o cartella): i PDF
 * sono estratti in testo in data/train_stage/, i testi usati in place.
 * Scrive fino a `max` percorsi ordinati in `out`, ritorna il numero (0 = ok
 * ma niente da allenare, -1 = errore). Non tocca il motore. */
int pii_collect_documents(const char *path, char out[][560], int max);

#endif
