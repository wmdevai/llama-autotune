document.addEventListener("DOMContentLoaded", () => {
    setupNavigation();
    setupDashboard();
    setupModels();
    setupInspector();
    setupBenchmark();
    setupCalibrate();
    setupOptimize();
    setupPresets();
    setupStorage();
});


function $(id) {
    return document.getElementById(id);
}


function setHidden(element, hidden) {
    if (!element) {
        return;
    }

    element.classList.toggle("hidden", hidden);
}


function getErrorMessage(error) {
    if (error instanceof Error) {
        return error.message;
    }

    return String(error);
}


async function apiFetch(url, options = {}) {
    const response = await fetch(url, options);

    let data = null;

    try {
        data = await response.json();
    } catch {
        // Il chiamante riceverà un errore utile qui sotto.
    }

    if (!response.ok) {
        const message =
            data?.detail ||
            `Richiesta fallita con HTTP ${response.status}`;

        throw new Error(message);
    }

    return data;
}


/* =========================================================
   NAVIGAZIONE
   ========================================================= */

function setupNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    const tabs = document.querySelectorAll(".tab");

    navItems.forEach((button) => {
        button.addEventListener("click", () => {
            const target = button.dataset.tab;

            navItems.forEach((item) => {
                item.classList.remove("active");
            });

            tabs.forEach((tab) => {
                tab.classList.remove("active");
            });

            button.classList.add("active");

            const targetTab = $(target);

            if (targetTab) {
                targetTab.classList.add("active");
            }
        });
    });
}


/* =========================================================
   PANNELLO
   ========================================================= */

function setupDashboard() {
    $("refresh-dashboard").addEventListener(
        "click",
        loadDashboard
    );

    loadDashboard();
}


async function loadDashboard() {
    try {
        const data = await apiFetch("/api/dashboard");

        const hw = data.hardware;
        const llama = data.llama_cpp;

        $("cpu-name").textContent =
            hw.cpu_name || "-";

        $("cpu-cores").textContent =
            `${hw.physical_cores ?? "-"} fisici / ` +
            `${hw.logical_cores ?? "-"} logici`;

        $("ram-gb").textContent =
            hw.ram_used_gb != null
                ? `${hw.ram_used_gb} GB`
                : (hw.ram_gb != null ? `${hw.ram_gb} GB` : "-");

        $("ram-total").textContent =
            hw.ram_total_gb != null
                ? `di ${hw.ram_total_gb} GB totali`
                : "";

        $("gpu-name").textContent =
            hw.gpu_models?.length
                ? hw.gpu_models.join(", ")
                : "Nessuna GPU rilevata";

        $("gpu-vram").textContent =
            hw.vram_total_gb != null && hw.vram_total_gb > 0
                ? `${hw.vram_used_gb} GB usati / ${hw.vram_total_gb} GB totali ` +
                  `(${hw.vram_free_gb} liberi)`
                : (hw.vram_per_gpu?.length
                    ? `${hw.vram_per_gpu.join(" / ")} GB VRAM`
                    : "");

        $("gpu-sensors").textContent =
            hw.gpu_temperature_c != null && hw.gpu_utilization_pct != null
                ? `${hw.gpu_temperature_c}°C · ${hw.gpu_utilization_pct}%`
                : (hw.gpu_temperature_c != null
                    ? `${hw.gpu_temperature_c}°C`
                    : "");

        $("backend").textContent =
            hw.backend || "-";

        $("llama-server").textContent =
            llama.llama_server || "-";

        $("llama-bench").textContent =
            llama.llama_bench || "-";

        $("llama-version").textContent =
            llama.version || "-";

        const sys = data.system || {};

        $("models-count").textContent =
            sys.models_count ?? "-";

        $("models-size").textContent =
            sys.models_size_gb != null
                ? `${sys.models_size_gb} GB`
                : "";

        $("calibrations-count").textContent =
            sys.calibrations_count ?? "-";

        $("tool-version").textContent =
            sys.version || "-";

        $("disk-free").textContent =
            sys.disk_free_gb != null
                ? `${sys.disk_free_gb} GB disco libero`
                : "";

    } catch (error) {
        console.error("Impossibile caricare il pannello:", error);

        $("cpu-name").textContent =
            "Impossibile caricare il pannello";
    }
}


/* =========================================================
   SELEZIONE MODELLO
   ========================================================= */

function setupModels() {
    $("model-select").addEventListener(
        "change",
        handleModelSelection
    );

    $("refresh-models").addEventListener(
        "click",
        () => loadModels(true)
    );

    loadModels();
}


async function loadModels(showStatus = false) {
    const select = $("model-select");
    const refreshButton = $("refresh-models");

    const previousValue = select.value;

    refreshButton.disabled = true;
    refreshButton.textContent = "Aggiornamento...";

    select.innerHTML = "";

    const loadingOption = document.createElement("option");

    loadingOption.value = "";
    loadingOption.textContent = "Caricamento modelli...";

    select.appendChild(loadingOption);

    try {
        const data = await apiFetch("/api/models");

        select.innerHTML = "";

        const placeholder = document.createElement("option");

        placeholder.value = "";
        placeholder.textContent =
            data.models.length
                ? "Seleziona un modello..."
                : "Nessun modello GGUF trovato";

        select.appendChild(placeholder);

        for (const model of data.models) {
            const option = document.createElement("option");

            option.value = model.path;
            option.textContent = model.relative_path;

            select.appendChild(option);
        }

        const currentPaths = [
            previousValue,
            $("inspect-model-path").value.trim(),
            $("benchmark-model-path").value.trim(),
            $("calibrate-model-path").value.trim(),
            $("optimize-model-path").value.trim(),
        ].filter(Boolean);

        for (const path of currentPaths) {
            const exists = Array.from(select.options).some(
                (option) => option.value === path
            );

            if (exists) {
                select.value = path;
                syncModelPath(path);
                break;
            }
        }

        if (showStatus) {
            console.info(
                `Caricati ${data.models.length} modello/i GGUF da ${data.directory}`
            );
        }

    } catch (error) {
        console.error("Impossibile caricare i modelli:", error);

        select.innerHTML = "";

        const option = document.createElement("option");

        option.value = "";
        option.textContent = "Impossibile caricare i modelli";

        select.appendChild(option);

    } finally {
        refreshButton.disabled = false;
        refreshButton.textContent = "Aggiorna Modelli";
    }
}


function handleModelSelection(event) {
    const modelPath = event.target.value;

    if (!modelPath) {
        return;
    }

    syncModelPath(modelPath);
}


function syncModelPath(modelPath) {
    if (!modelPath) {
        return;
    }

    $("inspect-model-path").value = modelPath;
    $("benchmark-model-path").value = modelPath;
    $("calibrate-model-path").value = modelPath;
    $("optimize-model-path").value = modelPath;
}


function getSelectedModelPath() {
    const selectValue = $("model-select").value.trim();

    if (selectValue) {
        return selectValue;
    }

    const inspectorValue =
        $("inspect-model-path").value.trim();

    if (inspectorValue) {
        return inspectorValue;
    }

    const benchmarkValue =
        $("benchmark-model-path").value.trim();

    if (benchmarkValue) {
        return benchmarkValue;
    }

    const calibrateValue =
        $("calibrate-model-path").value.trim();

    if (calibrateValue) {
        return calibrateValue;
    }

    return $("optimize-model-path").value.trim();
}


/* =========================================================
   ISPEZIONE MODELLO
   ========================================================= */

function setupInspector() {
    $("inspect-button").addEventListener(
        "click",
        inspectModel
    );
}


async function inspectModel() {
    const modelPath =
        $("inspect-model-path").value.trim() ||
        getSelectedModelPath();

    const button = $("inspect-button");
    const errorBox = $("inspect-error");
    const results = $("inspect-results");

    setHidden(errorBox, true);
    setHidden(results, true);

    if (!modelPath) {
        errorBox.textContent =
            "Seleziona un modello o inserisci un percorso.";

        setHidden(errorBox, false);

        return;
    }

    syncModelPath(modelPath);

    button.disabled = true;
    button.textContent = "Ispezione in corso...";

    try {
        const data = await apiFetch(
            "/api/inspect",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                },

                body: JSON.stringify({
                    model_path: modelPath,
                }),
            }
        );

        $("inspect-path").textContent =
            data.path ?? "-";

        $("inspect-architecture").textContent =
            data.architecture ?? "-";

        $("inspect-parameters").textContent =
            formatParameters(data.parameters);

        $("inspect-quantization").textContent =
            data.quantization ?? "-";

        $("inspect-layers").textContent =
            data.n_layers ?? "-";

        $("inspect-context").textContent =
            formatNumber(data.training_context);

        $("inspect-moe").textContent =
            data.is_moe ? "Sì" : "No";

        $("inspect-size").textContent =
            data.file_size_gb != null
                ? `${data.file_size_gb.toFixed(2)} GB`
                : "-";

        setHidden(results, false);

    } catch (error) {
        errorBox.textContent =
            getErrorMessage(error);

        setHidden(errorBox, false);

    } finally {
        button.disabled = false;
        button.textContent = "Ispeziona Modello";
    }
}


/* =========================================================
   BENCHMARK
   ========================================================= */

function setupBenchmark() {
    $("benchmark-button").addEventListener(
        "click",
        runBenchmark
    );
}


async function runBenchmark() {
    const modelPath =
        $("benchmark-model-path").value.trim() ||
        getSelectedModelPath();

    const button = $("benchmark-button");

    const statusBox = $("benchmark-status");
    const errorBox = $("benchmark-error");
    const resultsBox = $("benchmark-results");

    setHidden(statusBox, true);
    setHidden(errorBox, true);
    setHidden(resultsBox, true);

    if (!modelPath) {
        errorBox.textContent =
            "Seleziona un modello o inserisci un percorso.";

        setHidden(errorBox, false);

        return;
    }

    syncModelPath(modelPath);

    const payload = {
        model_path: modelPath,

        threads: optionalInteger(
            $("benchmark-threads").value
        ),

        n_gpu_layers: optionalInteger(
            $("benchmark-gpu-layers").value
        ),

        batch_size: optionalInteger(
            $("benchmark-batch-size").value
        ),

        ubatch_size: optionalInteger(
            $("benchmark-ubatch-size").value
        ),

        flash_attn:
            $("benchmark-flash-attn").checked,

        cache_type_k: optionalString(
            $("benchmark-cache-k").value
        ),

        cache_type_v: optionalString(
            $("benchmark-cache-v").value
        ),

        repetitions:
            requiredInteger(
                $("benchmark-repetitions").value,
                1
            ),
    };

    button.disabled = true;
    button.textContent = "Benchmark in corso...";

    statusBox.textContent =
        "Benchmark in corso. Potrebbe richiedere qualche istante.";

    setHidden(statusBox, false);

    try {
        const data = await apiFetch(
            "/api/benchmark",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                },

                body: JSON.stringify(payload),
            }
        );

        const result = data.result;

        $("result-prompt-tps").textContent =
            formatTPS(result.prompt_tps);

        $("result-generation-tps").textContent =
            formatTPS(result.generation_tps);

        $("result-startup-time").textContent =
            result.startup_time != null
                ? `${Number(result.startup_time).toFixed(2)} s`
                : "-";

        $("result-memory").textContent =
            result.memory_usage != null
                ? `${Number(result.memory_usage).toFixed(1)} MB`
                : "-";

        $("result-vram").textContent =
            result.vram_usage != null && result.vram_usage > 0
                ? `${Number(result.vram_usage).toFixed(1)} MB`
                : "-";

        $("result-success").textContent =
            result.success ? "Riuscito" : "Fallito";

        $("benchmark-raw-output").textContent =
            result.raw_output || "";

        statusBox.textContent =
            result.success
                ? "Benchmark completato con successo."
                : "Benchmark completato con un errore.";

        setHidden(resultsBox, false);

    } catch (error) {
        errorBox.textContent =
            getErrorMessage(error);

        setHidden(errorBox, false);

    } finally {
        button.disabled = false;
        button.textContent = "Avvia Benchmark";
    }
}


/* =========================================================
   CALIBRAZIONE
   ========================================================= */

function setupCalibrate() {
    $("calibrate-button").addEventListener(
        "click",
        runCalibrate
    );

    loadCalibrations();
}


async function runCalibrate() {
    const modelPath =
        $("calibrate-model-path").value.trim() ||
        getSelectedModelPath();

    const button = $("calibrate-button");

    const statusBox = $("calibrate-status");
    const errorBox = $("calibrate-error");
    const resultsBox = $("calibrate-results");

    setHidden(statusBox, true);
    setHidden(errorBox, true);
    setHidden(resultsBox, true);

    if (!modelPath) {
        errorBox.textContent =
            "Seleziona un modello o inserisci un percorso.";

        setHidden(errorBox, false);

        return;
    }

    syncModelPath(modelPath);

    button.disabled = true;
    button.textContent = "Calibrazione in corso...";

    statusBox.textContent =
        "Calibrazione in corso. Verrà eseguito un benchmark controllato.";

    setHidden(statusBox, false);

    try {
        const data = await apiFetch(
            "/api/calibrate",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                },

                body: JSON.stringify({
                    model_path: modelPath,
                }),
            }
        );

        $("calib-quantization").textContent =
            data.quantization ?? "-";

        $("calib-file-size").textContent =
            data.file_size_mb != null
                ? `${Number(data.file_size_mb).toFixed(0)} MB`
                : "-";

        $("calib-vram").textContent =
            data.measured_vram_mb != null
                ? `${Number(data.measured_vram_mb).toFixed(0)} MB`
                : "-";

        $("calib-ratio").textContent =
            data.measured_ratio != null
                ? Number(data.measured_ratio).toFixed(3)
                : "-";

        $("calib-overhead").textContent =
            data.overhead != null
                ? Number(data.overhead).toFixed(2)
                : "-";

        statusBox.textContent =
            "Calibrazione completata e salvata.";

        setHidden(resultsBox, false);

        loadCalibrations();

    } catch (error) {
        errorBox.textContent =
            getErrorMessage(error);

        setHidden(errorBox, false);

    } finally {
        button.disabled = false;
        button.textContent = "Calibra Modello";
    }
}


async function loadCalibrations() {
    const list = $("calibrations-list");

    if (!list) {
        return;
    }

    try {
        const data = await apiFetch("/api/calibrations");
        const calibrations = data.calibrations || {};

        const quants = Object.keys(calibrations);

        if (!quants.length) {
            list.innerHTML =
                '<p class="muted">Nessuna calibrazione salvata.</p>';

            return;
        }

        const table = document.createElement("table");
        table.className = "calibrations-table";

        const thead = document.createElement("thead");
        thead.innerHTML = `
            <tr>
                <th>Quantizzazione</th>
                <th>Overhead</th>
                <th>Rapporto VRAM/file</th>
                <th>VRAM misurata</th>
                <th>Modello</th>
            </tr>
        `;
        table.appendChild(thead);

        const tbody = document.createElement("tbody");

        for (const quant of quants.sort()) {
            const record = calibrations[quant];

            const row = document.createElement("tr");
            row.innerHTML = `
                <td><strong>${escapeHtml(quant)}</strong></td>
                <td>${Number(record.overhead).toFixed(2)}</td>
                <td>${Number(record.measured_ratio).toFixed(3)}</td>
                <td>${Number(record.measured_vram_mb).toFixed(0)} MB</td>
                <td><code>${escapeHtml(record.model_path || "")}</code></td>
            `;

            tbody.appendChild(row);
        }

        table.appendChild(tbody);

        list.innerHTML = "";
        list.appendChild(table);

    } catch (error) {
        console.error("Impossibile caricare le calibrazioni:", error);

        list.innerHTML =
            '<p class="muted">Impossibile caricare le calibrazioni.</p>';
    }
}


function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = String(value ?? "");
    return div.innerHTML;
}


/* =========================================================
   PRESETS
   ========================================================= */

function setupPresets() {
    $("presets-apply-button").addEventListener(
        "click",
        applyPresets
    );

    loadPresets();
}


async function loadPresets() {
    const errorBox = $("presets-error");

    setHidden(errorBox, true);

    try {
        const data = await apiFetch("/api/presets");

        $("presets-current-path").textContent =
            data.current.path || "-";

        $("presets-recommended-path").textContent =
            data.recommended.path || "-";

        $("presets-current").textContent =
            data.current.content || "(vuoto)";

        $("presets-recommended").textContent =
            data.recommended.content ||
            "(nessun consigliato — esegui prima un'ottimizzazione)";

    } catch (error) {
        errorBox.textContent =
            getErrorMessage(error);

        setHidden(errorBox, false);
    }
}


async function applyPresets() {
    const button = $("presets-apply-button");
    const statusBox = $("presets-status");
    const errorBox = $("presets-error");

    setHidden(statusBox, true);
    setHidden(errorBox, true);

    if (
        !confirm(
            "Sovrascrivere il presets.ini attuale con il consigliato? " +
            "Verrà creato un backup."
        )
    ) {
        return;
    }

    button.disabled = true;
    button.textContent = "Applicazione...";

    try {
        const data = await apiFetch(
            "/api/presets/apply",
            { method: "POST" }
        );

        statusBox.textContent =
            "Applicato con successo. " +
            (data.backup
                ? `Backup creato: ${data.backup}`
                : "(nessun file attuale da salvare)");

        setHidden(statusBox, false);

        loadPresets();

    } catch (error) {
        errorBox.textContent =
            getErrorMessage(error);

        setHidden(errorBox, false);

    } finally {
        button.disabled = false;
        button.textContent = "Applica consigliato (con backup)";
    }
}


/* =========================================================
   ARCHIVIO
   ========================================================= */

function setupStorage() {
    $("storage-delete-all-slots").addEventListener(
        "click",
        () => deleteStorageItem("slots", "tutti gli slot")
    );

    loadStorage();
}


async function loadStorage() {
    const errorBox = $("storage-error");

    setHidden(errorBox, true);

    try {
        const data = await apiFetch("/api/storage");

        $("storage-slots-dir").textContent =
            data.slots_dir || "-";

        const itemsBox = $("storage-items");

        if (!data.items.length) {
            itemsBox.innerHTML =
                '<p class="muted">Nessun file generato.</p>';
        } else {
            itemsBox.innerHTML = "";

            for (const item of data.items) {
                itemsBox.appendChild(
                    buildStorageRow(
                        item.name,
                        item.size_human,
                        item.key
                    )
                );
            }
        }

        const slotsBox = $("storage-slots");

        if (!data.slots.length) {
            slotsBox.innerHTML =
                '<p class="muted">Nessuno slot.</p>';
        } else {
            slotsBox.innerHTML = "";

            for (const slot of data.slots) {
                slotsBox.appendChild(
                    buildStorageRow(
                        slot.name,
                        slot.size_human,
                        `slot:${slot.name}`
                    )
                );
            }

            const totalRow = document.createElement("div");
            totalRow.className = "storage-row";

            totalRow.innerHTML =
                '<span class="name"><strong>Totale slot</strong></span>' +
                `<span class="size">${data.slots_total_human}</span>`;

            slotsBox.appendChild(totalRow);
        }

    } catch (error) {
        errorBox.textContent =
            getErrorMessage(error);

        setHidden(errorBox, false);
    }
}


function buildStorageRow(name, size, key) {
    const row = document.createElement("div");
    row.className = "storage-row";

    const nameSpan = document.createElement("span");
    nameSpan.className = "name";
    nameSpan.textContent = name;

    const sizeSpan = document.createElement("span");
    sizeSpan.className = "size";
    sizeSpan.textContent = size;

    const button = document.createElement("button");
    button.className = "secondary-button";
    button.textContent = "Elimina";
    button.addEventListener(
        "click",
        () => deleteStorageItem(key, name)
    );

    row.appendChild(nameSpan);
    row.appendChild(sizeSpan);
    row.appendChild(button);

    return row;
}


async function deleteStorageItem(key, label) {
    const statusBox = $("storage-status");
    const errorBox = $("storage-error");

    setHidden(statusBox, true);
    setHidden(errorBox, true);

    if (!confirm(`Eliminare "${label || key}"?`)) {
        return;
    }

    try {
        const data = await apiFetch(
            "/api/storage/delete",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                },

                body: JSON.stringify({ key }),
            }
        );

        statusBox.textContent =
            `Eliminato. Spazio liberato: ${data.freed_human}.`;

        setHidden(statusBox, false);

        loadStorage();

    } catch (error) {
        errorBox.textContent =
            getErrorMessage(error);

        setHidden(errorBox, false);
    }
}


/* =========================================================
   OTTIMIZZAZIONE
   ========================================================= */

function setupOptimize() {
    $("optimize-button").addEventListener(
        "click",
        runOptimize
    );
}


async function runOptimize() {
    const modelPath =
        $("optimize-model-path").value.trim() ||
        getSelectedModelPath();

    const button = $("optimize-button");

    const statusBox = $("optimize-status");
    const errorBox = $("optimize-error");
    const resultsBox = $("optimize-results");
    const logBox = $("optimize-log");

    setHidden(statusBox, true);
    setHidden(errorBox, true);
    setHidden(resultsBox, true);

    if (logBox) {
        logBox.textContent = "";
        setHidden(logBox, false);
    }

    if (!modelPath) {
        errorBox.textContent =
            "Seleziona un modello dall'Ispezione o inserisci un percorso.";

        setHidden(errorBox, false);

        return;
    }

    syncModelPath(modelPath);

    const payload = {
        model_path: modelPath,

        objective:
            $("optimize-objective").value,

        trials_b:
            requiredInteger(
                $("optimize-trials-b").value,
                12
            ),

        trials_c:
            requiredInteger(
                $("optimize-trials-c").value,
                20
            ),
    };

    button.disabled = true;
    button.textContent = "Ottimizzazione...";

    statusBox.textContent =
        "Ottimizzazione in corso. Verranno eseguiti più benchmark reali.";

    setHidden(statusBox, false);

    try {
        const job = await apiFetch(
            "/api/optimize",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                },

                body: JSON.stringify(payload),
            }
        );

        await streamOptimizeEvents(
            job.job_id,
            statusBox,
            errorBox,
            resultsBox,
            logBox
        );

    } catch (error) {
        errorBox.textContent =
            getErrorMessage(error);

        setHidden(errorBox, false);

    } finally {
        button.disabled = false;
        button.textContent = "Avvia Ottimizzazione";
    }
}


function streamOptimizeEvents(
    jobId,
    statusBox,
    errorBox,
    resultsBox,
    logBox
) {
    return new Promise((resolve) => {
        const source = new EventSource(
            `/api/optimize/events/${jobId}`
        );

        source.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === "log") {
                if (logBox) {
                    logBox.textContent += `${data.message}\n`;
                    logBox.scrollTop = logBox.scrollHeight;
                }

                if (statusBox) {
                    statusBox.textContent = data.message;
                }

                return;
            }

            if (data.type === "done") {
                source.close();

                if (data.error) {
                    if (errorBox) {
                        errorBox.textContent = data.error;
                        setHidden(errorBox, false);
                    }
                } else if (data.result) {
                    renderOptimizeResult(data.result);

                    if (statusBox) {
                        statusBox.textContent =
                            "Ottimizzazione completata con successo.";
                    }

                    setHidden(resultsBox, false);
                }

                resolve();
            }
        };

        source.onerror = () => {
            source.close();
            resolve();
        };
    });
}


function renderOptimizeResult(data) {
    const baseline = data.baseline_result;
    const result = data.final_result;

    $("optimize-best-score").textContent =
        data.best_score != null
            ? Number(data.best_score).toFixed(4)
            : "-";

    $("optimize-total-evaluations").textContent =
        data.total_evaluations ?? "-";

    $("optimize-baseline-prompt-tps").textContent =
        formatTPS(baseline?.prompt_tps);

    $("optimize-baseline-generation-tps").textContent =
        formatTPS(baseline?.generation_tps);

    $("optimize-prompt-tps").textContent =
        formatTPS(result?.prompt_tps);

    $("optimize-generation-tps").textContent =
        formatTPS(result?.generation_tps);

    const improvement =
        data.best_score != null
            ? (Number(data.best_score) - 1) * 100
            : null;

    $("optimize-improvement").textContent =
        improvement != null
            ? `${improvement >= 0 ? "+" : ""}${improvement.toFixed(2)}%`
            : "-";

    $("optimize-best-config").textContent =
        JSON.stringify(
            data.best_config,
            null,
            2
        );
}


/* =========================================================
   HELPER
   ========================================================= */

function optionalInteger(value) {
    const trimmed = String(value ?? "").trim();

    if (!trimmed) {
        return null;
    }

    const number = Number.parseInt(trimmed, 10);

    return Number.isNaN(number)
        ? null
        : number;
}


function requiredInteger(value, fallback) {
    const number =
        optionalInteger(value);

    return number != null && number >= 1
        ? number
        : fallback;
}


function optionalString(value) {
    const trimmed =
        String(value ?? "").trim();

    return trimmed || null;
}


function formatNumber(value) {
    if (value == null) {
        return "-";
    }

    return Number(value).toLocaleString();
}


function formatParameters(value) {
    if (value == null) {
        return "-";
    }

    const number = Number(value);

    if (Number.isNaN(number)) {
        return String(value);
    }

    if (number >= 1_000_000_000) {
        return `${(number / 1_000_000_000).toFixed(2)}B`;
    }

    if (number >= 1_000_000) {
        return `${(number / 1_000_000).toFixed(2)}M`;
    }

    return formatNumber(number);
}


function formatTPS(value) {
    if (value == null) {
        return "-";
    }

    const number = Number(value);

    return Number.isNaN(number)
        ? "-"
        : number.toFixed(2);
}
