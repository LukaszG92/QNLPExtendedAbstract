# QNLP Benchmark Scripts (TweetEval Sentiment)

Questa cartella contiene **1 script per tecnica** (1)…(8), eseguibili in locale.
Ogni script:
- scarica/carica `cardiffnlp/tweet_eval` (config `sentiment`) via `datasets`
- prepara l’input (TF-IDF/SVD, hashing, light compression, char n-grams, ecc.)
- costruisce un QNode PennyLane per la tecnica
- encoda train/val/test
- allena LogisticRegression e calcola **accuracy + macro-F1** su **val** e **test**
- stima `q`, `L`, `depth`, `#2q`, `feature_dim`, `encoding_time/sample`, backend, shots
- salva/aggiorna `runs/results_checklist_all_methods.csv` e `.json` (idempotente per `technique`)

## Dipendenze
In un venv:
- `pip install pennylane datasets scikit-learn pandas numpy`

## Esecuzione (esempi)
Dalla cartella `qnlp_benchmark_scripts/`:

```bash
python tech01_angle_linear.py --outdir runs
python tech02_angle_full_entanglement.py --outdir runs
python tech03_angle_reuploading.py --outdir runs --L 3
python tech04_rich_readout.py --outdir runs
python tech05_amplitude_hashing.py --outdir runs --q_amp 6
python tech06_light_compression.py --outdir runs --k_base 64 --q_amp 6
python tech07_block_entanglement.py --outdir runs --block_size 2
python tech08_morphte_char.py --outdir runs --char_ng_min 3 --char_ng_max 5
```

Per esecuzione analitica (deterministico): `--shots 0`.
Per sampling NISQ-like: `--shots 1024`.

## Note runtime
Questi script possono essere **molto lenti**: l’encoding quantistico viene fatto sample-by-sample.
Ogni tecnica è separata per evitare di perdere risultati se una tecnica crasha.

## Output
- `runs/results_checklist_all_methods.csv`
- `runs/results_checklist_all_methods.json`
- caching preprocessing in `cache/` (riusato tra script)
