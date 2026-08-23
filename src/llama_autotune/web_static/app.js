document.addEventListener("DOMContentLoaded", () => {
    setupNavigation();
    setupDashboard();
    setupInspector();
    setupBenchmark();

    loadDashboard();
});


function setupNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    const tabs = document.querySelectorAll(".tab");

    navItems.forEach((item) => {
        item.addEventListener("click", () => {
            const tabName = item.dataset.tab;

            navItems.forEach((nav) => {
                nav.classList.remove("active");
            });

            tabs.forEach((tab) => {
                tab.classList.remove("active");
            });

            item.classList.add("active");

            const target = document.getElementById(tabName);

            if (target) {
                target.classList.add("active");
            }
        });
    });
}


function setupDashboard() {
    const refreshButton = document.getElementById(
        "refresh-dashboard"
    );

    if (refreshButton) {
        refreshButton.addEventListener(
            "click",
            loadDashboard
        );
    }
}


async function loadDashboard() {
    const loading = document.getElementById(
        "dashboard-loading"
    );

    const content = document.getElementById(
        "dashboard-content"
    );

    loading.textContent = "Detecting hardware...";
    loading.classList.remove("hidden");
    content.classList.add("hidden");

    try {
        const response = await fetch(
            "/api/dashboard"
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Unable to load dashboard."
            );
        }

        const hardware = data.hardware;
        const llama = data.llama_cpp;

        setText(
            "cpu-name",
            hardware.cpu_name || "-"
        );

        setText(
            "physical-cores",
            hardware.physical_cores ?? "-"
        );

        setText(
            "logical-cores",
            hardware.logical_cores ?? "-"
        );

        setText(
            "ram",
            hardware.ram_gb
                ? `${hardware.ram_gb} GB`
                : "-"
        );

        setText(
            "gpu-name",
            hardware.gpu_models &&
            hardware.gpu_models.length
                ? hardware.gpu_models.join(", ")
                : "No GPU detected"
        );

        const vram = hardware.vram_per_gpu &&
            hardware.vram_per_gpu.length
            ? hardware.vram_per_gpu
                .map((value) => `${value} GB`)
                .join(", ")
            : "VRAM unavailable";

        setText(
            "gpu-details",
            `${hardware.gpu_count || 0} GPU(s) • ${vram}`
        );

        setText(
            "backend",
            hardware.backend || "-"
        );

        setText(
            "llama-version",
            llama.version || "Unknown"
        );

        setText(
            "llama-server",
            llama.llama_server || "Not found"
        );

        setText(
            "llama-bench",
            llama.llama_bench || "Not found"
        );

        loading.classList.add("hidden");
        content.classList.remove("hidden");

    } catch (error) {
        loading.textContent =
            `Dashboard error: ${error.message}`;
    }
}


function setupInspector() {
    const button = document.getElementById(
        "inspect-button"
    );

    if (!button) {
        return;
    }

    button.addEventListener(
        "click",
        inspectModel
    );

    const input = document.getElementById(
        "inspect-model-path"
    );

    input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            inspectModel();
        }
    });
}


async function inspectModel() {
    const input = document.getElementById(
        "inspect-model-path"
    );

    const button = document.getElementById(
        "inspect-button"
    );

    const errorBox = document.getElementById(
        "inspect-error"
    );

    const resultBox = document.getElementById(
        "model-result"
    );

    const modelPath = input.value.trim();

    errorBox.classList.add("hidden");
    resultBox.classList.add("hidden");

    if (!modelPath) {
        errorBox.textContent =
            "Please enter a GGUF model path.";

        errorBox.classList.remove("hidden");
        return;
    }

    const originalText = button.textContent;

    button.disabled = true;
    button.textContent = "Inspecting...";

    try {
        const response = await fetch(
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

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Unable to inspect model."
            );
        }

        setText(
            "model-path",
            data.path || modelPath
        );

        setText(
            "architecture",
            data.architecture || "-"
        );

        setText(
            "parameters",
            formatNumber(data.parameters)
        );

        setText(
            "quantization",
            data.quantization || "-"
        );

        setText(
            "layers",
            data.n_layers ?? "-"
        );

        setText(
            "context-length",
            data.training_context ?? "-"
        );

        setText(
            "file-size",
            data.file_size_gb !== undefined
                ? `${data.file_size_gb} GB`
                : "-"
        );

        resultBox.classList.remove("hidden");

        const benchmarkInput = document.getElementById(
            "benchmark-model-path"
        );

        if (benchmarkInput &&
            !benchmarkInput.value.trim()) {
            benchmarkInput.value = modelPath;
        }

    } catch (error) {
        errorBox.textContent = error.message;
        errorBox.classList.remove("hidden");

    } finally {
        button.disabled = false;
        button.textContent = originalText;
    }
}


function setupBenchmark() {
    const button = document.getElementById(
        "run-benchmark"
    );

    if (button) {
        button.addEventListener(
            "click",
            runBenchmark
        );
    }
}


async function runBenchmark() {
    const button = document.getElementById(
        "run-benchmark"
    );

    const statusBox = document.getElementById(
        "benchmark-status"
    );

    const errorBox = document.getElementById(
        "benchmark-error"
    );

    const resultBox = document.getElementById(
        "benchmark-result"
    );

    statusBox.classList.add("hidden");
    errorBox.classList.add("hidden");
    resultBox.classList.add("hidden");

    const modelPath = getValue(
        "benchmark-model-path"
    );

    if (!modelPath) {
        errorBox.textContent =
            "Please enter a GGUF model path.";

        errorBox.classList.remove("hidden");
        return;
    }

    const repetitions = getNumber(
        "repetitions",
        1
    );

    if (repetitions < 1) {
        errorBox.textContent =
            "Repetitions must be at least 1.";

        errorBox.classList.remove("hidden");
        return;
    }

    const payload = {
        model_path: modelPath,
        threads: getOptionalNumber("threads"),
        n_gpu_layers: getOptionalNumber("gpu-layers"),
        batch_size: getOptionalNumber("batch-size"),
        ubatch_size: getOptionalNumber("ubatch-size"),
        flash_attn: document.getElementById(
            "flash-attn"
        ).checked,
        cache_type_k: getOptionalValue(
            "cache-type-k"
        ),
        cache_type_v: getOptionalValue(
            "cache-type-v"
        ),
        repetitions: repetitions,
    };

    const originalText = button.textContent;

    button.disabled = true;
    button.textContent = "Benchmark running...";

    statusBox.textContent =
        "Running llama-bench. Please wait...";

    statusBox.classList.remove("hidden");

    try {
        const response = await fetch(
            "/api/benchmark",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Benchmark failed."
            );
        }

        const result = data.result;

        setText(
            "prompt-tps",
            formatMetric(result.prompt_tps)
        );

        setText(
            "generation-tps",
            formatMetric(result.generation_tps)
        );

        setText(
            "startup-time",
            result.startup_time !== undefined
                ? `${Number(result.startup_time).toFixed(2)} s`
                : "-"
        );

        setText(
            "memory-usage",
            result.memory_usage !== undefined
                ? `${Number(result.memory_usage).toFixed(1)} MB`
                : "-"
        );

        setText(
            "benchmark-success",
            result.success ? "Yes" : "No"
        );

        setText(
            "raw-output",
            JSON.stringify(
                {
                    config: data.config,
                    result: result,
                },
                null,
                2
            )
        );

        statusBox.classList.add("hidden");
        resultBox.classList.remove("hidden");

        if (!result.success) {
            errorBox.textContent =
                result.error ||
                "Benchmark completed unsuccessfully.";

            errorBox.classList.remove("hidden");
        }

    } catch (error) {
        statusBox.classList.add("hidden");

        errorBox.textContent = error.message;
        errorBox.classList.remove("hidden");

    } finally {
        button.disabled = false;
        button.textContent = originalText;
    }
}


function setText(id, value) {
    const element = document.getElementById(id);

    if (element) {
        element.textContent = value;
    }
}


function getValue(id) {
    const element = document.getElementById(id);

    return element
        ? element.value.trim()
        : "";
}


function getOptionalValue(id) {
    const value = getValue(id);

    return value || null;
}


function getNumber(id, fallback) {
    const element = document.getElementById(id);

    if (!element || element.value === "") {
        return fallback;
    }

    const value = Number(element.value);

    return Number.isFinite(value)
        ? value
        : fallback;
}


function getOptionalNumber(id) {
    const element = document.getElementById(id);

    if (!element || element.value.trim() === "") {
        return null;
    }

    const value = Number(element.value);

    return Number.isFinite(value)
        ? value
        : null;
}


function formatNumber(value) {
    if (value === undefined ||
        value === null) {
        return "-";
    }

    return Number(value).toLocaleString();
}


function formatMetric(value) {
    if (value === undefined ||
        value === null) {
        return "-";
    }

    return Number(value).toFixed(2);
}
