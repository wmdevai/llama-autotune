document.addEventListener("DOMContentLoaded", () => {
    setupNavigation();
    setupDashboard();
    setupModels();
    setupInspector();
    setupBenchmark();
    setupOptimize();
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
        // The caller will receive a useful error below.
    }

    if (!response.ok) {
        const message =
            data?.detail ||
            `Request failed with HTTP ${response.status}`;

        throw new Error(message);
    }

    return data;
}


/* =========================================================
   NAVIGATION
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
   DASHBOARD
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
            `${hw.physical_cores ?? "-"} physical / ` +
            `${hw.logical_cores ?? "-"} logical cores`;

        $("ram-gb").textContent =
            hw.ram_gb != null
                ? `${hw.ram_gb} GB`
                : "-";

        $("gpu-name").textContent =
            hw.gpu_models?.length
                ? hw.gpu_models.join(", ")
                : "No GPU detected";

        $("gpu-vram").textContent =
            hw.vram_per_gpu?.length
                ? `${hw.vram_per_gpu.join(" / ")} GB VRAM`
                : "";

        $("backend").textContent =
            hw.backend || "-";

        $("llama-server").textContent =
            llama.llama_server || "-";

        $("llama-bench").textContent =
            llama.llama_bench || "-";

        $("llama-version").textContent =
            llama.version || "-";

    } catch (error) {
        console.error("Unable to load dashboard:", error);

        $("cpu-name").textContent =
            "Unable to load dashboard";
    }
}


/* =========================================================
   MODEL SELECTOR
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
    refreshButton.textContent = "Refreshing...";

    select.innerHTML = "";

    const loadingOption = document.createElement("option");

    loadingOption.value = "";
    loadingOption.textContent = "Loading models...";

    select.appendChild(loadingOption);

    try {
        const data = await apiFetch("/api/models");

        select.innerHTML = "";

        const placeholder = document.createElement("option");

        placeholder.value = "";
        placeholder.textContent =
            data.models.length
                ? "Select a model..."
                : "No GGUF models found";

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
                `Loaded ${data.models.length} GGUF model(s) from ${data.directory}`
            );
        }

    } catch (error) {
        console.error("Unable to load models:", error);

        select.innerHTML = "";

        const option = document.createElement("option");

        option.value = "";
        option.textContent = "Unable to load models";

        select.appendChild(option);

    } finally {
        refreshButton.disabled = false;
        refreshButton.textContent = "Refresh Models";
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

    return $("optimize-model-path").value.trim();
}


/* =========================================================
   MODEL INSPECTOR
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
            "Select a model or enter a model path.";

        setHidden(errorBox, false);

        return;
    }

    syncModelPath(modelPath);

    button.disabled = true;
    button.textContent = "Inspecting...";

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
            data.is_moe ? "Yes" : "No";

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
        button.textContent = "Inspect Model";
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
            "Select a model or enter a model path.";

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
    button.textContent = "Running Benchmark...";

    statusBox.textContent =
        "Benchmark in progress. This may take a moment.";

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

        $("result-success").textContent =
            result.success ? "Yes" : "No";

        $("benchmark-raw-output").textContent =
            result.raw_output || "";

        statusBox.textContent =
            result.success
                ? "Benchmark completed successfully."
                : "Benchmark completed with an error.";

        setHidden(resultsBox, false);

    } catch (error) {
        errorBox.textContent =
            getErrorMessage(error);

        setHidden(errorBox, false);

    } finally {
        button.disabled = false;
        button.textContent = "Run Benchmark";
    }
}


/* =========================================================
   AUTO OPTIMIZE
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
            "Select a model from Model Inspector or enter a model path.";

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
    button.textContent = "Optimizing...";

    statusBox.textContent =
        "Optimization in progress. Multiple real benchmarks will be executed.";

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
        button.textContent = "Start Optimization";
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
                            "Optimization completed successfully.";
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
   HELPERS
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
