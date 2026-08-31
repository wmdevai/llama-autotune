# llama-autotune

Benchmark e ottimizzazione automatica delle configurazioni di `llama.cpp`.

Analizza hardware e modello GGUF, poi valuta configurazioni candidate con
benchmark reali (`llama-bench`) in tre fasi: baseline euristica (**A**),
ricerca locale (**B**) e raffinamento bayesiano (**C**).

> Fork di [Najafu/llama-autotune](https://github.com/Najafu/llama-autotune).

## Installazione

Richiede Python **3.12+** e una build di `llama.cpp` con `llama-bench`
(e `llama-server` per `launch`).

```bash
git clone https://github.com/wmdevai/llama-autotune.git
cd llama-autotune
uv sync --dev
```

Indica dove si trovano i binari di `llama.cpp`:

```bash
export LLAMA_CPP_DIR="$HOME/.local/src/llama.cpp/build/bin"
```

## Guida rapida

```bash
.venv/bin/llama-autotune inspect model.gguf    # hardware + metadati
.venv/bin/llama-autotune search model.gguf     # trova la config migliore
.venv/bin/llama-autotune launch model.gguf     # avvia llama-server
```

## Comandi

- `inspect model.gguf` — hardware e metadati del modello.
- `benchmark model.gguf [--batch-size N --n-gpu-layers N ...]` — benchmark singolo.
- `search model.gguf [--objective O --resume]` — ricerca in tre stadi.
- `launch model.gguf [--profile p.json]` — avvia `llama-server`.
- `export/import profile.json` — profili di avvio.
- `web` — Web UI in italiano su `http://127.0.0.1:8766` (`--browser brave` la apre nel browser e ferma il server alla chiusura).

### Obiettivi di `search`

`balanced` (default) · `balanced_context` · `max_context` ·
`max_generation_tps` · `max_prompt_tps` · `min_latency` · `max_efficiency`.

Budget: **12** valutazioni in Stage B, **20** in Stage C. Con `--resume`
riusa i benchmark già salvati per lo stesso modello.

## Note sul funzionamento

- Il `ctx_size` iniziale arriva fino a **24576**; con `balanced_context` e
  `max_context` si estende al context length del modello.
- La stima VRAM considera **pesi + KV-cache** (GQA/MQA-aware): entrambi devono
  entrare in GPU. La ricerca prova da sola la quantizzazione della KV-cache
  (`q8_0`/`q4_0`) per contesti più ampi.
- `n_gpu_layers` ≥ numero di layer = **full offload**.

## Sviluppo

```bash
.venv/bin/pytest -q                    # 326 passed, 4 skipped — copertura 87%
.venv/bin/ruff check .                 # lint
.venv/bin/pyright src/llama_autotune   # type checking
```

I test smoke usano `llama-bench` reale e richiedono un modello piccolo:

```bash
LLAMA_AUTOTUNE_SMOKE_MODEL=/path/modello.gguf .venv/bin/pytest tests/test_smoke.py
```

Cronologia dettagliata: [CHANGELOG.md](CHANGELOG.md).
