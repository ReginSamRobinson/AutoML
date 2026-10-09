// Setup page: dataset management and launching a run.
(function () {
    const config = JSON.parse(document.getElementById("setupData").textContent);
    const PREFERRED_TARGETS = ["target", "label", "labels", "class", "outcome", "y"];

    const els = {
        form: document.getElementById("runForm"),
        dataset: document.getElementById("datasetSelect"),
        datasetSummary: document.getElementById("datasetSummary"),
        target: document.getElementById("targetSelect"),
        metrics: document.getElementById("metricsInput"),
        maxTime: document.getElementById("maxTime"),
        maxRam: document.getElementById("maxRam"),
        minPerf: document.getElementById("minPerf"),
        cvFolds: document.getElementById("cvFolds"),
        preferred: document.getElementById("preferredMetrics"),
        interpretable: document.getElementById("interpretable"),
        runButton: document.getElementById("runButton"),
        runStatus: document.getElementById("runStatus"),
        reset: document.getElementById("resetDefaults"),
        useSample: document.getElementById("useSample"),
        rows: document.getElementById("datasetRows"),
        notice: document.getElementById("datasetNotice"),
        fileInput: document.getElementById("fileInput"),
        uploadLabel: document.getElementById("uploadLabel"),
        card: document.getElementById("datasetsCard"),
    };

    let selectedDataset = "";

    // ---------- helpers ----------

    function el(tag, props, children) {
        const node = document.createElement(tag);
        Object.assign(node, props || {});
        (children || []).forEach((child) => node.append(child));
        return node;
    }

    function option(value, label) {
        return el("option", { value: value, textContent: label });
    }

    function formatBytes(bytes) {
        if (!Number.isFinite(bytes)) return "—";
        if (bytes < 1024) return bytes + " B";
        const units = ["KB", "MB", "GB"];
        let value = bytes / 1024;
        let unit = 0;
        while (value >= 1024 && unit < units.length - 1) {
            value /= 1024;
            unit += 1;
        }
        return value.toFixed(1) + " " + units[unit];
    }

    function formatDate(seconds) {
        if (!Number.isFinite(seconds)) return "—";
        return new Date(seconds * 1000).toLocaleString(undefined, {
            year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
        });
    }

    function parseList(value) {
        return value.split(",").map((item) => item.trim()).filter(Boolean);
    }

    function numberOrNull(input) {
        return input.value.trim() === "" ? null : Number(input.value);
    }

    function showNotice(message, kind) {
        els.notice.textContent = message;
        els.notice.className = "notice" + (kind ? " is-" + kind : "");
        els.notice.hidden = false;
    }

    async function requestJson(url, options) {
        const response = await fetch(url, options);
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || payload.error) {
            throw new Error(payload.error || "Request failed (" + response.status + ")");
        }
        return payload;
    }

    // ---------- datasets ----------

    function renderRows(datasets) {
        els.rows.replaceChildren();
        if (!datasets.length) {
            els.rows.append(el("tr", {}, [el("td", { colSpan: 4, className: "empty", textContent: "No datasets yet. Upload a file to get started." })]));
            return;
        }
        datasets.forEach((item) => {
            const nameCell = el("td", {}, [item.name]);
            if (item.name === config.sample) {
                nameCell.append(el("span", { className: "tag", textContent: "sample" }));
            }
            const actions = el("div", { className: "row-actions" }, [
                el("button", { type: "button", className: "btn btn-ghost btn-sm", textContent: "Use", onclick: () => selectDataset(item.name) }),
            ]);
            if (item.name !== config.sample) {
                actions.append(el("button", {
                    type: "button", className: "btn btn-ghost btn-sm btn-danger", textContent: "Delete",
                    onclick: () => deleteDataset(item.name),
                }));
            }
            els.rows.append(el("tr", {}, [
                nameCell,
                el("td", { className: "num", textContent: formatBytes(item.size) }),
                el("td", { className: "muted", textContent: formatDate(item.modified) }),
                el("td", {}, [actions]),
            ]));
        });
    }

    function renderOptions(datasets) {
        els.dataset.replaceChildren();
        if (!datasets.length) {
            els.dataset.append(option("", "No datasets"));
            els.dataset.disabled = true;
            setTargets([]);
            return;
        }
        datasets.forEach((item) => els.dataset.append(option(item.name, item.name)));
        els.dataset.disabled = false;
    }

    async function refreshDatasets(preferred) {
        try {
            const payload = await requestJson("/api/datasets");
            const datasets = payload.datasets || [];
            renderRows(datasets);
            renderOptions(datasets);
            const names = datasets.map((item) => item.name);
            const next = [preferred, selectedDataset, config.sample, names[0]].find((name) => name && names.includes(name));
            if (next) selectDataset(next);
        } catch (error) {
            showNotice(error.message, "error");
        }
    }

    function setTargets(columns) {
        els.target.replaceChildren();
        if (!columns.length) {
            els.target.append(option("", "Select a dataset first"));
            els.target.disabled = true;
            return;
        }
        columns.forEach((column) => els.target.append(option(column, column)));
        const preferred = columns.find((column) => PREFERRED_TARGETS.includes(String(column).trim().toLowerCase()));
        els.target.value = preferred || columns[columns.length - 1];
        els.target.disabled = false;
    }

    async function selectDataset(name) {
        selectedDataset = name;
        els.dataset.value = name;
        els.datasetSummary.textContent = "Reading columns…";
        setTargets([]);
        try {
            const payload = await requestJson("/api/datasets/" + encodeURIComponent(name));
            if (selectedDataset !== name) return; // a newer selection won
            const preview = payload.preview || {};
            const [rows, cols] = preview.shape || [0, 0];
            const missing = Object.values(preview.missing_values || {}).reduce((sum, n) => sum + n, 0);
            els.datasetSummary.textContent =
                rows.toLocaleString() + " rows · " + cols + " columns · " +
                (missing ? missing.toLocaleString() + " missing values" : "no missing values");
            setTargets(preview.columns || []);
        } catch (error) {
            els.datasetSummary.textContent = error.message;
        }
    }

    async function deleteDataset(name) {
        if (!window.confirm('Delete "' + name + '"? This removes the file from the uploads folder.')) return;
        try {
            const payload = await requestJson("/api/datasets/" + encodeURIComponent(name), { method: "DELETE" });
            if (selectedDataset === name) selectedDataset = "";
            showNotice(payload.message || "Deleted " + name, "success");
            refreshDatasets();
        } catch (error) {
            showNotice(error.message, "error");
        }
    }

    async function uploadFile(file) {
        if (!file) return;
        const body = new FormData();
        body.append("file", file);
        els.uploadLabel.textContent = "Uploading…";
        showNotice("Uploading " + file.name + "…");
        try {
            const payload = await requestJson("/api/upload", { method: "POST", body: body });
            showNotice(payload.message, "success");
            await refreshDatasets(payload.filename);
        } catch (error) {
            showNotice(error.message, "error");
        } finally {
            els.uploadLabel.textContent = "Upload file";
            els.fileInput.value = "";
        }
    }

    // ---------- run ----------

    function fillConstraints(settings) {
        const c = settings.constraints || {};
        els.metrics.value = (settings.metrics || []).join(", ");
        els.cvFolds.value = settings.cv_folds ?? "";
        els.maxTime.value = c.max_training_time ?? "";
        els.maxRam.value = c.max_ram_mb ?? "";
        els.minPerf.value = c.min_performance ?? "";
        els.preferred.value = (c.preferred_metrics || []).join(", ");
        els.interpretable.checked = Boolean(c.interpretability_required);
    }

    function setRunning(running, message, isError) {
        els.runButton.disabled = running;
        els.runButton.replaceChildren();
        if (running) els.runButton.append(el("span", { className: "spinner" }));
        els.runButton.append(running ? "Running…" : "Run search");
        els.runStatus.textContent = message;
        els.runStatus.classList.toggle("is-error", Boolean(isError));
    }

    async function startRun(event) {
        event.preventDefault();
        if (!els.dataset.value || !els.target.value) {
            setRunning(false, "Choose a dataset and a target column first.", true);
            return;
        }
        const payload = {
            dataset: els.dataset.value,
            target: els.target.value,
            metrics: parseList(els.metrics.value),
            cv_folds: numberOrNull(els.cvFolds),
            constraints: {
                max_training_time: numberOrNull(els.maxTime),
                max_ram_mb: numberOrNull(els.maxRam),
                min_performance: numberOrNull(els.minPerf),
                preferred_metrics: parseList(els.preferred.value),
                interpretability_required: els.interpretable.checked,
            },
        };

        const started = Date.now();
        const tick = () => setRunning(true, "Evaluating candidates on " + payload.dataset + " · " + Math.round((Date.now() - started) / 1000) + " s");
        tick();
        const timer = setInterval(tick, 1000);
        try {
            const result = await requestJson("/api/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            clearInterval(timer);
            setRunning(true, "Done. Opening results…");
            window.location.assign(result.redirect || "/?view=dashboard");
        } catch (error) {
            clearInterval(timer);
            setRunning(false, error.message, true);
        }
    }

    // ---------- wiring ----------

    els.form.addEventListener("submit", startRun);
    els.dataset.addEventListener("change", () => selectDataset(els.dataset.value));
    els.reset.addEventListener("click", () => {
        fillConstraints(config.defaults);
        els.runStatus.classList.remove("is-error");
        els.runStatus.textContent = "Defaults restored. They are saved when the next run starts.";
    });
    if (els.useSample) {
        els.useSample.addEventListener("click", () => selectDataset(config.sample));
    }
    els.fileInput.addEventListener("change", () => uploadFile(els.fileInput.files[0]));

    ["dragenter", "dragover"].forEach((type) =>
        els.card.addEventListener(type, (event) => {
            event.preventDefault();
            els.card.classList.add("is-dragging");
        })
    );
    ["dragleave", "drop"].forEach((type) =>
        els.card.addEventListener(type, (event) => {
            event.preventDefault();
            if (type === "dragleave" && els.card.contains(event.relatedTarget)) return;
            els.card.classList.remove("is-dragging");
        })
    );
    els.card.addEventListener("drop", (event) => uploadFile(event.dataTransfer.files[0]));

    refreshDatasets();
})();
