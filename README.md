# llama-autotune

Benchmark e ottimizzazione automatica delle configurazioni di `llama.cpp`.

`llama-autotune` analizza l'hardware e i metadati di un modello GGUF, genera
configurazioni candidate e le valuta con benchmark reali (`llama-bench`) per
trovare la migliore rispetto all'obiettivo scelto. L'ottimizzazione avviene in
tre fasi: baseline euristica (**Stage A**), esplorazione dei parametri
principali (**Stage B**) e raffinamento bayesiano (**Stage C**).

> Fork di [Najafu/llama-autotune](https://github.com/Najafu/llama-autotune),
> personalizzato e mantenuto su
> [wmdevai/llama-autotune](https://github.com/wmdevai/llama-autotune).

## Caratteristiche

- **Rilevamento hardware** — CPU, RAM, GPU, vendor, backend e VRAM
  (nvidia-smi, rocm-smi, WMI, system_profiler);
- **Ispezione GGUF** — architettura, parametri, quantizzazione, layer, teste
  (inclusa GQA/MQA), context length e modelli MoE;
- **Benchmark reali** — esegue `llama-bench` e ne interpreta l'output;
- **Ottimizzazione a tre stadi** — baseline euristica, ricerca locale e
  raffinamento bayesiano con Optuna (TPE);
- **Obiettivi multipli** — velocità di generazione/prompt, latenza, contesto,
  efficienza e profili bilanciati;
- **Vincoli e stima VRAM** — stima di pesi + KV-cache (consapevole di GQA/MQA e
  della quantizzazione della cache) con filtri di plausibilità pre-benchmark;
- **Calibrazione VRAM** — misura reale dell'overhead per quantizzazione, con
  feedback automatico nella ricerca;
- **Full GPU offload** — `n_gpu_layers` ≥ numero di layer è trattato come
  offload completo;
- **Cache e resume** — persistenza SQLite dei benchmark e ripresa (`--resume`)
  senza ripetere le valutazioni già fatte;
- **Profili di avvio** — export/import di configurazioni JSON;
- **Web UI** — interfaccia web in italiano con setup guidato, calibrazione,
  presets, archivio e guida.

## Installazione

### Prerequisiti

- Python **3.12** o superiore;
- una build funzionante di `llama.cpp` con il binario `llama-bench`
  (e `llama-server` per il comando `launch`);
- [`uv`](https://docs.astral.sh/uv/) per gestire ambiente e dipendenze
  (in alternativa un `venv` + `pip`).

### Dal sorgente

```bash
git clone https://github.com/wmdevai/llama-autotune.git
cd llama-autotune
uv sync --dev
```

`uv sync` crea la directory `.venv` con tutte le dipendenze e i tool di
sviluppo (`pytest`, `ruff`, `pyright`). I comandi si eseguono da lì:

```bash
.venv/bin/llama-autotune --version
```

### Configurare `llama.cpp`

I binari vengono cercati in questo ordine:

1. directory indicata da `LLAMA_CPP_DIR`;
2. directory vicine al pacchetto installato;
3. `PATH` di sistema.

In pratica, su Linux:

```bash
export LLAMA_CPP_DIR="$HOME/.local/src/llama.cpp/build/bin"
```

Verifica che i binari siano presenti:

```bash
ls -l "$LLAMA_CPP_DIR/llama-bench" "$LLAMA_CPP_DIR/llama-server"
```

Per renderla permanente in Bash:

```bash
echo 'export LLAMA_CPP_DIR="$HOME/.local/src/llama.cpp/build/bin"' >> ~/.bashrc
source ~/.bashrc
```

## Guida rapida

```bash
cd ~/.local/src/llama-autotune

export LLAMA_CPP_DIR="$HOME/.local/src/llama.cpp/build/bin"

# 1. hardware + metadati del modello
.venv/bin/llama-autotune inspect ~/Modelli/llama.cpp/model.gguf

# 2. ricerca della configurazione migliore
.venv/bin/llama-autotune search ~/Modelli/llama.cpp/model.gguf

# 3. avvio di llama-server con la configurazione trovata
.venv/bin/llama-autotune launch ~/Modelli/llama.cpp/model.gguf
```

## Comandi

### `inspect`

Mostra hardware rilevato e metadati del modello GGUF.

```bash
.venv/bin/llama-autotune inspect model.gguf
```

Per i modelli MoE sono mostrati anche i parametri attivi.

### `benchmark`

Esegue un singolo benchmark con `llama-bench`.

```bash
.venv/bin/llama-autotune benchmark model.gguf
```

Con parametri personalizzati:

```bash
.venv/bin/llama-autotune benchmark model.gguf \
  --batch-size 4096 \
  --ubatch-size 1024 \
  --n-gpu-layers 999 \
  --flash-attn
```

### `search`

Esegue la ricerca automatica in tre stadi.

```bash
.venv/bin/llama-autotune search model.gguf \
  --objective balanced_context
```

Obiettivi disponibili:

| Obiettivo | Descrizione |
|---|---|
| `balanced` | velocità di generazione e prompt bilanciate |
| `balanced_context` | velocità di generazione e contesto più ampio |
| `max_context` | contesto massimo caricabile |
| `max_generation_tps` | massima velocità di generazione |
| `max_prompt_tps` | massima velocità di processamento del prompt |
| `min_latency` | latenza minima |
| `max_efficiency` | massima efficienza (tps / memoria) |

Opzioni utili:

```bash
.venv/bin/llama-autotune search model.gguf \
  --trials-b 12 \
  --trials-c 20 \
  --resume          # riusa i benchmark già salvati per questo modello
```

### `launch`

Avvia `llama-server` con la configurazione disponibile.

```bash
.venv/bin/llama-autotune launch model.gguf --host 127.0.0.1 --port 8080
```

### `export` / `import`

Esporta o importa un profilo di avvio JSON.

```bash
.venv/bin/llama-autotune export profile.json \
  --model model.gguf --hardware "RTX 5060 Ti"

.venv/bin/llama-autotune import profile.json
```

### `web`

Avvia la Web UI integrata.

```bash
.venv/bin/llama-autotune web
```

L'interfaccia è disponibile su `http://127.0.0.1:8766` e offre i tab:

- **Pannello** — hardware rilevato e stato di `llama.cpp`;
- **Setup Guidato** — flusso in 5 passi (modello → calibrazione → priorità →
  ottimizzazione → risultato);
- **Ispezione Modello** — metadati GGUF;
- **Benchmark** — singolo benchmark con parametri scelti;
- **Calibrazione** — misura della VRAM reale e overhead per quantizzazione;
- **Ottimizzazione** — ricerca automatica della configurazione;
- **Presets** — visualizza/applica il `presets.ini` attuale e quello consigliato;
- **Archivio** — file generati e slot KV-cache, con eliminazione;
- **Guida** — spiegazione dei parametri e risoluzione dei problemi.

Host e porta sono configurabili con `--host` e `--port`.

## Concetti chiave

### Le tre fasi dell'ottimizzazione

- **Stage A — Baseline**: genera una configurazione iniziale dalle euristiche
  (modello, CPU, GPU, VRAM, backend) e la valuta. Se fallisce prova configurazioni
  di fallback (thread e modalità CPU).
- **Stage B — Ricerca locale**: esplora uno alla volta i parametri principali
  (`threads`, `batch_size`, `ubatch_size`, `n_gpu_layers`, `ctx_size`,
  `cache_type_k/v`) con un budget predefinito di **12 valutazioni**.
- **Stage C — Raffinamento bayesiano**: usa il sampler TPE di Optuna attorno
  alla miglior configurazione, con un budget di **20 valutazioni valide**.
  Duplicati, configurazioni implausibili e benchmark falliti vengono scartati
  senza consumare budget.

### Context size

La configurazione iniziale usa un `ctx_size` fino a **24576**, compatibilmente
col context length del modello. Con gli obiettivi `balanced_context` e
`max_context` il range si estende fino al context length del modello.

> **Nota su `llama-bench`**: la build locale di `llama-bench` non espone un
> flag `--ctx-size`, quindi durante i benchmark il contesto non viene impostato
> esplicitamente (la KV-cache è allocata per `n_prompt + n_gen`).
> `llama-autotune` rileva automaticamente se il binario supporta `--ctx-size`
> (probe di `--help`, con cache) e in tal caso lo propaga ai benchmark. La
> validazione a fine ricerca usa comunque un carico di prompt proporzionale al
> contesto trovato.

### Batch size e ubatch size

Lo spazio di ricerca include `batch_size` fino a **8192** e `ubatch_size` fino
a **1024**. Il valore migliore dipende da modello, GPU, VRAM e contesto: un
valore più alto non è necessariamente migliore, ed è il benchmark reale a
decidere.

### GPU offload e VRAM

La stima della VRAM tiene conto di:

- dimensione del modello e overhead del backend/quantizzazione;
- frazione del modello caricata sulla GPU (`n_gpu_layers`);
- memoria della KV-cache (GQA/MQA-aware, con `cache_type_k/v`).

Per `n_gpu_layers`: `0` = nessun offload, valori minori del numero di layer =
offload parziale, valori ≥ numero di layer = **full offload**.

Sia i **pesi** sia la **KV-cache** devono entrare nella VRAM fisica:
`llama-server` pre-alloca l'intera KV-cache al caricamento. Per contesti ampi
si può quantizzare la cache (`cache-type-k/v = q8_0` o `q4_0`), dimezzandone la
memoria: la ricerca la prova automaticamente. Le configurazioni che superano i
limiti vengono escluse prima del benchmark.

## Sviluppo

```bash
cd ~/.local/src/llama-autotune
.venv/bin/pytest -q                    # test
.venv/bin/ruff check .                 # lint
.venv/bin/pyright src/llama_autotune   # type checking
```

Stato verificato più recente:

```text
326 passed, 4 skipped — copertura 87%
```

I test di integrazione "smoke" usano `llama-bench` reale e richiedono un
modello GGUF piccolo (50 MB – 3 GB), da indicare esplicitamente:

```bash
LLAMA_AUTOTUNE_SMOKE_MODEL=/percorso/al/modello.gguf \
  .venv/bin/pytest tests/test_smoke.py
```

Senza il modello i test smoke vengono saltati automaticamente. Per la
copertura:

```bash
.venv/bin/pytest -q --cov=llama_autotune --cov-report=term-missing
```

## Note

`llama-autotune` esegue benchmark reali tramite `llama-bench`: i risultati
dipendono dal sistema, dal modello GGUF, dalla quantizzazione, dal backend,
dalla memoria e dai parametri di esecuzione. La configurazione "migliore" è
quindi relativa alla macchina e al modello usati durante la ricerca. La
cronologia dettagliata delle modifiche è in [CHANGELOG.md](CHANGELOG.md).
