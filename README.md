# llama-autotune

`llama-autotune` è uno strumento di benchmark e ottimizzazione per `llama.cpp`.

Il progetto analizza l'hardware disponibile e le caratteristiche di un modello GGUF, genera configurazioni candidate e le valuta tramite benchmark reali di `llama.cpp` per individuare una configurazione adatta all'obiettivo di ottimizzazione selezionato.

L'ottimizzazione è organizzata in tre fasi:

- **Stage A**: generazione e valutazione di una configurazione iniziale basata su euristiche;
- **Stage B**: esplorazione controllata dei parametri principali entro un budget definito di trial;
- **Stage C**: raffinamento locale delle configurazioni migliori.

## Caratteristiche

- **Rilevamento hardware**: CPU, RAM, GPU, backend e VRAM disponibili;
- **Ispezione GGUF**: lettura di architettura, parametri, quantizzazione, layer, context length e caratteristiche MoE;
- **Benchmark con llama.cpp**: utilizzo di `llama-bench` per misurare le prestazioni delle configurazioni;
- **Ottimizzazione in tre fasi**: baseline euristica, Stage B e raffinamento Stage C;
- **Obiettivi multipli**: velocità di generazione, velocità del prompt, latenza, contesto, efficienza e profilo bilanciato;
- **Gestione dei vincoli**: stima della VRAM e controlli di plausibilità prima dei benchmark;
- **Full GPU offload**: valori di `n_gpu_layers` pari o superiori al numero di layer del modello sono trattati come offload completo;
- **Benchmark adattati al contesto**: il carico del prompt può essere adattato al valore di `ctx_size`;
- **Database SQLite**: persistenza dei benchmark e dei profili di avvio;
- **Profili di avvio**: esportazione e importazione di configurazioni JSON;
- **Web UI integrata**: interfaccia web avviabile dalla CLI.

## Installazione

### Prerequisiti

Per utilizzare `llama-autotune` sono necessari:

- Python 3.12 o superiore;
- una build funzionante di `llama.cpp`;
- il binario `llama-bench` per eseguire i benchmark;
- il binario `llama-server` per utilizzare il comando `launch`.

### Installazione dal sorgente

Clona il repository:

```bash
git clone https://github.com/Najafu/llama-autotune.git
cd llama-autotune
```

Il progetto utilizza un ambiente virtuale Python. Una volta disponibile la directory `.venv`, il comando può essere eseguito direttamente da lì:

```bash
.venv/bin/llama-autotune --version
```

Nella configurazione locale, se `llama-autotune` non è disponibile globalmente nel `PATH`, utilizzare:

```bash
.venv/bin/llama-autotune
```

### Configurazione di llama.cpp

`llama-autotune` utilizza i binari di `llama.cpp`:

- `llama-bench` per eseguire i benchmark;
- `llama-server` per il comando `launch`.

La ricerca dei binari avviene nell'ordine seguente:

1. directory indicata dalla variabile d'ambiente `LLAMA_CPP_DIR`;
2. directory vicine al pacchetto installato;
3. `PATH` di sistema.

Su Linux è possibile indicare esplicitamente la directory dei binari:

```bash
export LLAMA_CPP_DIR=/percorso/alla/llama.cpp/build/bin
```

Ad esempio, se `llama.cpp` si trova in:

```text
~/.local/src/llama.cpp
```

la configurazione sarà:

```bash
export LLAMA_CPP_DIR="$HOME/.local/src/llama.cpp/build/bin"
```

Per verificare che i binari siano presenti:

```bash
echo "$LLAMA_CPP_DIR"

ls -l \
  "$LLAMA_CPP_DIR/llama-bench" \
  "$LLAMA_CPP_DIR/llama-server"
```

Se vuoi rendere permanente la variabile per Bash:

```bash
echo 'export LLAMA_CPP_DIR="$HOME/.local/src/llama.cpp/build/bin"' >> ~/.bashrc
source ~/.bashrc
```

### Verifica dell'installazione

Dalla directory del progetto:

```bash
cd ~/.local/src/llama-autotune
```

Verifica la versione:

```bash
.venv/bin/llama-autotune --version
```

Visualizza l'aiuto:

```bash
.venv/bin/llama-autotune --help
```

Verifica il comando Web UI:

```bash
.venv/bin/llama-autotune web --help
```

Esegui l'ispezione di un modello:

```bash
.venv/bin/llama-autotune inspect /percorso/al/modello.gguf
```

## Utilizzo

### inspect

Mostra le informazioni sull'hardware e i metadati del modello GGUF.

```bash
.venv/bin/llama-autotune inspect model.gguf
```

Per i modelli MoE vengono mostrate anche le informazioni sui parametri attivi.

### benchmark

Esegue un benchmark utilizzando `llama-bench`.

Esempio:

```bash
.venv/bin/llama-autotune benchmark model.gguf
```

È possibile specificare parametri personalizzati:

```bash
.venv/bin/llama-autotune benchmark model.gguf \
  --batch-size 4096 \
  --ubatch-size 1024 \
  --n-gpu-layers 999 \
  --flash-attn
```

Le opzioni disponibili possono essere visualizzate con:

```bash
.venv/bin/llama-autotune benchmark --help
```

## Ottimizzazione

Il comando `search` esegue la ricerca automatica di una configurazione.

Esempio:

```bash
.venv/bin/llama-autotune search model.gguf
```

È possibile specificare un obiettivo:

```bash
.venv/bin/llama-autotune search model.gguf \
  --objective max_generation_tps
```

Gli obiettivi disponibili dipendono dalle opzioni definite dal progetto e possono essere visualizzati con:

```bash
.venv/bin/llama-autotune search --help
```

Per ripartire da una ricerca precedente senza ripetere i benchmark già
eseguiti sullo stesso modello, usare `--resume`:

```bash
.venv/bin/llama-autotune search model.gguf --resume
```

## Le tre fasi dell'ottimizzazione

### Stage A — Configurazione iniziale

Stage A genera una configurazione iniziale utilizzando:

- caratteristiche del modello;
- CPU;
- GPU;
- VRAM disponibile;
- RAM disponibile;
- informazioni sul backend.

Questa configurazione costituisce la baseline iniziale per le fasi successive.

### Stage B — Esplorazione dei parametri principali

Stage B esplora in modo controllato i parametri principali della configurazione.

Nella configurazione attuale il budget predefinito è di **12 valutazioni**.

Tra i parametri esplorati rientrano:

- `batch_size`;
- `ubatch_size`;
- `n_gpu_layers`;
- `ctx_size`.

Le configurazioni vengono sottoposte ai controlli di plausibilità e alla stima della VRAM prima dell'esecuzione del benchmark.

### Stage C — Raffinamento locale

Stage C esegue un raffinamento locale delle configurazioni migliori ottenute nelle fasi precedenti.

Nella configurazione attuale il budget predefinito è di **20 valutazioni valide**.

Durante questa fase:

- vengono generate configurazioni candidate nell'area delle configurazioni migliori;
- le configurazioni duplicate possono essere recuperate dalla cache;
- i cache hit non vengono considerati nuove valutazioni complete;
- i trial non validi o duplicati vengono gestiti separatamente dal conteggio delle valutazioni valide.

## Context size

La configurazione iniziale utilizza un valore di `ctx_size` fino a **24576**, compatibilmente con il context length dichiarato dal modello.

Il valore può essere modificato durante la ricerca e viene considerato anche nella stima della memoria necessaria.

## Batch size e ubatch size

Lo spazio di ricerca include:

- `batch_size` fino a **8192**;
- `ubatch_size` fino a **1024**.

La configurazione migliore dipende dal modello, dalla GPU, dalla VRAM disponibile e dal contesto utilizzato.

Un valore elevato non garantisce automaticamente prestazioni migliori: la configurazione viene verificata tramite benchmark reali.

## GPU offload e VRAM

La stima della VRAM tiene conto della quantità di modello effettivamente trasferita sulla GPU.

Per `n_gpu_layers`:

- `None` o `0` indicano nessun offload GPU;
- valori inferiori al numero di layer rappresentano un offload parziale;
- valori pari o superiori al numero di layer del modello vengono trattati come **full GPU offload**.

La stima considera:

- dimensione del modello;
- overhead del backend;
- frazione del modello caricata sulla GPU;
- memoria necessaria per la KV cache.

Le configurazioni che superano i limiti di memoria o risultano non plausibili possono essere escluse prima del benchmark.

## Profili di avvio

È possibile esportare e importare configurazioni.

### Export

```bash
.venv/bin/llama-autotune export profile.json \
  --model model.gguf \
  --hardware "Nome GPU"
```

### Import

```bash
.venv/bin/llama-autotune import profile.json
```

## launch

Il comando `launch` avvia `llama-server` utilizzando la configurazione disponibile.

Esempio:

```bash
.venv/bin/llama-autotune launch model.gguf
```

Le opzioni disponibili possono essere visualizzate con:

```bash
.venv/bin/llama-autotune launch --help
```

## Web UI

`llama-autotune` include una Web UI integrata.

Per avviarla:

```bash
cd ~/.local/src/llama-autotune

.venv/bin/llama-autotune web
```

Per impostazione predefinita l'interfaccia viene avviata su:

```text
http://127.0.0.1:8766
```

È possibile modificare host e porta:

```bash
.venv/bin/llama-autotune web \
  --host 127.0.0.1 \
  --port 8766
```

Per visualizzare tutte le opzioni:

```bash
.venv/bin/llama-autotune web --help
```

## Esempio completo

Supponendo che:

- il progetto si trovi in `~/.local/src/llama-autotune`;
- `llama.cpp` si trovi in `~/.local/src/llama.cpp`;
- il modello sia disponibile localmente;

è possibile procedere così:

```bash
cd ~/.local/src/llama-autotune

export LLAMA_CPP_DIR="$HOME/.local/src/llama.cpp/build/bin"

.venv/bin/llama-autotune inspect \
  "$HOME/Modelli/llama.cpp/model.gguf"

.venv/bin/llama-autotune search \
  "$HOME/Modelli/llama.cpp/model.gguf"
```

Al termine della ricerca è possibile avviare il modello:

```bash
.venv/bin/llama-autotune launch \
  "$HOME/Modelli/llama.cpp/model.gguf"
```

Oppure avviare la Web UI:

```bash
.venv/bin/llama-autotune web
```

## Sviluppo

Per eseguire i test dal repository:

```bash
cd ~/.local/src/llama-autotune

.venv/bin/pytest -q
```

Lo stato verificato più recente del progetto è:

```text
157 passed, 4 skipped
```

Per misurare la copertura dei test:

```bash
.venv/bin/pytest -q --cov=llama_autotune --cov-report=term-missing
```

## Roadmap

Piano di lavoro in tre fasi: **P0** (igiene del repository), **P1** (correttezza
della stima VRAM e robustezza della detection hardware) e **P2** (qualità della
ricerca, UX e testing).

### P0 — Igiene del repository

- [x] Rimuovere i file `.bak` / `.backup.*` dentro `src/llama_autotune/` (e
      `tests/`), versioni di lavoro di `optimizer.py`, `cli.py` e `web.py`.
- [x] Eliminare i file accidentali nella root: `ed -n 430,780p \` (redirect
      errato) e i log vuoti `benchmark*.log`.
- [x] Rafforzare `.gitignore`: `*.bak`, `*.backup.*`, `*.log`, `logs/`,
      `results/`.

### P1 — Correttezza della stima VRAM

- [x] GQA/MQA non considerati: leggere `attention.head_count_kv` dal GGUF e
      usarlo nella stima della KV-cache (oggi si usa `head_count`, che per
      modelli come Llama 3 70B o Qwen sovrastima la VRAM).
- [x] Formula KV-cache troppo rozza: `2 * n_layers * n_heads * ctx * 2 * 2`
      hardcoda 2 byte/elemento e ignora `cache_type_k/v` (q8_0/q4_0/f16);
      inoltre non considera che la KV-cache può essere offloadata solo sui
      layer su GPU.
- [x] Overhead per MoE: la stima moltiplica l'intero file per la frazione di
      layer; per MoE il peso è concentrato negli expert, stimare per blocchi
      (attention/shared vs expert).

### P1 — Robustezza della detection hardware

- [x] VRAM AMD su Linux non rilevata: `rocm-smi --showproductinfo` cattura solo
      il nome; usare `--showmeminfo vram`.
- [x] VRAM macOS assente e Windows via WMI inaffidabile (AdapterRAM overflowa
      su GPU > 4 GB); usare `system_profiler`/`ioreg` e `dxdiag`/nvapi.
- [x] `_verify_gpu_backend` fragile: l'euristica `"none" in stdout` può dare
      falsi positivi; parsare l'output JSON di `--list-devices` se disponibile.
- [x] Fallback se `llama-bench` manca: errore esplicito che suggerisca di
      impostare `LLAMA_CPP_DIR`.

### P2 — Qualità della ricerca

- [x] Hardcode discutibili: `max_layers = min(n_layers, 200)` e step
      `max_layers // 4`; range di `n_gpu_layers` dipendente dalla VRAM. Step
      batch/ubatch fissi (128/64) → usare step logaritmici o categoriali.
- [x] `config_from_params` inefficiente: `model_validate` per ogni chiave in un
      loop; costruire il dict una volta o usare `setattr` dopo una singola
      validazione.
- [x] `optimizer.py` da 1392 righe: separare gli stadi in moduli o classi per
      testabilità.
- [x] Ripresa dei run (`--resume`): ripartire da un run interrotto sfruttando la
      cache SQLite senza ri-benchmarkare.

### P2 — UX e testing

- [x] Web UI: progresso live (SSE/WebSocket) invece del solo polling, e tab con
      i valori rilevati di hardware e modello.
- [x] Test: metrica di coverage (`pytest-cov`) e test d'integrazione "smoke"
      che esegua `llama-bench` su un modello piccolo quando presente.

### P3 — Qualità dello Stage C e igiene runtime

- [x] Early-stopping di Stage C errato: l'euristica del "miglioramento del 2%
      tra trial consecutivi" avrebbe fermato lo stage dopo ~2 trial (aggiunta
      dopo l'ultimo run reale e mai esercitata); sostituita con "nessun nuovo
      best globale per N trial completati".
- [x] Stage C non realmente bayesiano: `enqueue_trial` con candidati
      equispaziati bypassava del tutto il sampler TPE; ora TPE campiona davvero
      lo spazio categoriale e i duplicati/impossibili vengono potati
      dall'obiettivo.
- [x] Log fuorviante: "Best is trial X" di Optuna si riferiva al solo studio
      locale, non al best globale; Optuna è silenziato durante lo stage e il
      best globale è loggato esplicitamente.
- [x] Leak di connessioni SQLite: le sessioni in `cli.py` non venivano mai
      chiuse; introdotto il context manager `session_scope` in `database.py`.
- [x] Copertura: test per i lettori GGUF (`model_inspector` 38% → 96%), per
      `candidates.grid_values` / `sample_param`, per i detector hardware
      (`hardware` 57% → 91%), i comandi CLI (`cli` 48% → 84%), il benchmark
      (`benchmark` 54% → 94%) e gli endpoint web (`web` 59% → 84%); totale
      61% → 82%.

## Note

`llama-autotune` esegue benchmark reali tramite `llama-bench`. I risultati dipendono quindi dalla configurazione effettiva del sistema, dal modello GGUF utilizzato, dalla quantizzazione, dal backend, dalla memoria disponibile e dai parametri di esecuzione.

La configurazione trovata come migliore rappresenta il risultato della ricerca effettuata sul sistema e sul modello utilizzati durante l'ottimizzazione.