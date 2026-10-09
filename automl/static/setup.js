// Setup page: dataset management, live run manifest and launching a run.
(function () {
    const config = JSON.parse(document.getElementById("setupData").textContent);
    const PREFERRED_TARGETS = ["target", "label", "labels", "class", "outcome", "y"];

    const els = {
        form: document.getElementById("runForm"),
        dataset: document.getElementById("datasetSelect"),
        datasetSummary: document.getElementById("datasetSummary"),
        stats: document.getElementById("datasetStats"),
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
        manifest: document.getElementById("manifest"),
        manifestState: document.getElementById("manifestState"),
        families: document.querySelectorAll("#families li"),
        chips: document.querySelectorAll(".chip[data-metric]"),
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

    // ---------- run manifest ----------

    function setSpec(key, value) {
        const node = els.manifest.querySelector('[data-m="' + key + '"]');
        if (!node) return;
        const text = value === null || value === undefined || value === "" ? "—" : String(value);
        if (node.textContent !== text) {
            node.textContent = text;
            node.title = text;
            node.classList.remove("is-flash");
            void node.offsetWidth; // restart the highlight animation
            node.classList.add("is-flash");
        }
        node.classList.toggle("is-empty", text === "—" || text === "none");
    }

    function capText(input, unit) {
        const value = input.value.trim();
        return value === "" ? "none" : value + " " + unit;
    }

    function updateManifest() {
        const metrics = parseList(els.metrics.value);
        const primary = parseList(els.preferred.value)[0] || metrics[0] || "";
        setSpec("dataset", els.dataset.disabled ? "" : els.dataset.value);
        setSpec("target", els.target.disabled ? "" : els.target.value);
        setSpec("primary", primary);
        setSpec("metrics", metrics.length ? metrics.length + " · " + metrics.join(", ") : "");
        setSpec("time", capText(els.maxTime, "s"));
        setSpec("ram", capText(els.maxRam, "MB"));
        setSpec("perf", els.minPerf.value.trim() || "none");
        setSpec("folds", els.cvFolds.value.trim() ? els.cvFolds.value.trim() + "-fold" : "");
        setSpec("families", els.interpretable.checked ? "interpretable only" : "all");

        const active = new Set(metrics.map((m) => m.toLowerCase()));
        els.chips.forEach((chip) => {
            chip.setAttribute("aria-pressed", String(active.has(chip.dataset.metric)));
            chip.classList.toggle("is-primary", chip.dataset.metric === primary.toLowerCase());
        });
        els.families.forEach((item) => {
            item.classList.toggle("is-off", els.interpretable.checked && item.dataset.interp !== "true");
        });
    }

    function toggleMetric(metric) {
        const list = parseList(els.metrics.value);
        const index = list.findIndex((m) => m.toLowerCase() === metric);
        if (index >= 0) list.splice(index, 1);
        else list.push(metric);
        els.metrics.value = list.join(", ");
        updateManifest();
    }

    // ---------- datasets ----------

    function extensionOf(name) {
        const dot = name.lastIndexOf(".");
        return dot > 0 ? name.slice(dot + 1) : "file";
    }

    function markActiveRow() {
        Array.from(els.rows.rows).forEach((row) => {
            row.classList.toggle("is-active", Boolean(row.dataset.name) && row.dataset.name === selectedDataset);
        });
    }

    function renderRows(datasets) {
        els.rows.replaceChildren();
        if (!datasets.length) {
            els.rows.append(el("tr", {}, [el("td", { colSpan: 4, className: "empty", textContent: "No datasets yet. Upload a file to get started." })]));
            return;
        }
        datasets.forEach((item) => {
            const name = el("span", { className: "file-name" }, [
                el("span", { className: "ext", textContent: extensionOf(item.name) }),
                item.name,
            ]);
            if (item.name === config.sample) {
                name.append(el("span", { className: "tag", textContent: "sample" }));
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
            const row = el("tr", {}, [
                el("td", {}, [name]),
                el("td", { className: "num", textContent: formatBytes(item.size) }),
                el("td", { className: "muted", textContent: formatDate(item.modified) }),
                el("td", {}, [actions]),
            ]);
            row.dataset.name = item.name;
            els.rows.append(row);
        });
        markActiveRow();
    }

    function renderOptions(datasets) {
        els.dataset.replaceChildren();
        if (!datasets.length) {
            els.dataset.append(option("", "No datasets"));
            els.dataset.disabled = true;
            setTargets([]);
            setStats(null);
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
            else updateManifest();
        } catch (error) {
            showNotice(error.message, "error");
        }
    }

    function setTargets(columns) {
        els.target.replaceChildren();
        if (!columns.length) {
            els.target.append(option("", "Select a dataset first"));
            els.target.disabled = true;
            updateManifest();
            return;
        }
        columns.forEach((column) => els.target.append(option(column, column)));
        const preferred = columns.find((column) => PREFERRED_TARGETS.includes(String(column).trim().toLowerCase()));
        els.target.value = preferred || columns[columns.length - 1];
        els.target.disabled = false;
        updateManifest();
    }

    function setStats(stats, loading) {
        els.stats.classList.toggle("is-loading", Boolean(loading));
        const values = stats || {};
        ["rows", "cols", "missing"].forEach((key) => {
            const node = els.stats.querySelector('[data-stat="' + key + '"]');
            const value = values[key];
            node.textContent = Number.isFinite(value) ? value.toLocaleString() : "—";
            node.classList.toggle("is-warn", key === "missing" && value > 0);
        });
    }

    async function selectDataset(name) {
        selectedDataset = name;
        els.dataset.value = name;
        els.datasetSummary.textContent = "Reading columns…";
        setStats(null, true);
        setTargets([]);
        markActiveRow();
        try {
            const payload = await requestJson("/api/datasets/" + encodeURIComponent(name));
            if (selectedDataset !== name) return; // a newer selection won
            const preview = payload.preview || {};
            const [rows, cols] = preview.shape || [0, 0];
            const missing = Object.values(preview.missing_values || {}).reduce((sum, n) => sum + n, 0);
            setStats({ rows: rows, cols: cols, missing: missing });
            els.datasetSummary.textContent = missing
                ? "Missing values are imputed during training."
                : "Complete table with no missing values.";
            setTargets(preview.columns || []);
        } catch (error) {
            setStats(null);
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
        updateManifest();
    }

    function setRunning(running, message, isError) {
        els.runButton.disabled = running;
        els.runButton.replaceChildren();
        if (running) els.runButton.append(el("span", { className: "spinner" }));
        els.runButton.append(running ? "Searching…" : "Run search");
        els.runStatus.textContent = message;
        els.runStatus.classList.toggle("is-error", Boolean(isError));
        els.manifest.classList.toggle("is-running", running);
        els.manifest.classList.toggle("is-error", Boolean(isError));
        els.manifestState.textContent = running ? "running" : isError ? "blocked" : "ready";
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
    els.form.addEventListener("input", updateManifest);
    els.form.addEventListener("change", updateManifest);
    els.dataset.addEventListener("change", () => selectDataset(els.dataset.value));
    els.chips.forEach((chip) => chip.addEventListener("click", () => toggleMetric(chip.dataset.metric)));
    els.form.querySelectorAll(".step-btn").forEach((button) =>
        button.addEventListener("click", () => {
            const current = Number(els.cvFolds.value) || 5;
            els.cvFolds.value = Math.min(20, Math.max(2, current + Number(button.dataset.step)));
            updateManifest();
        })
    );
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

    updateManifest();
    refreshDatasets();
})();
