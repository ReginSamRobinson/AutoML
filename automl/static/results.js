// Results page: trade-off and resource charts, leaderboard sorting and filtering.
(function () {
    const data = JSON.parse(document.getElementById("resultsData").textContent);
    const primary = data.primary;
    const points = data.points;
    const SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x"];

    // Model types in rank order; each keeps the same marker shape everywhere.
    const families = [];
    points.forEach((p) => {
        if (!families.includes(p.label)) families.push(p.label);
    });
    const symbolFor = (label) => SYMBOLS[families.indexOf(label) % SYMBOLS.length];

    function token(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    function theme() {
        return {
            accent: token("--chart-accent"),
            muted: token("--chart-muted"),
            grid: token("--chart-grid"),
            text: token("--chart-text"),
            strong: token("--text"),
            surface: token("--surface"),
            border: token("--border-strong"),
            font: token("--font"),
            display: token("--display"),
            mono: token("--mono"),
        };
    }

    function axis(t, extra) {
        return Object.assign({
            gridcolor: t.grid,
            linecolor: t.border,
            zeroline: false,
            tickfont: { family: t.mono, size: 11, color: t.text },
            title: { font: { size: 12, color: t.text } },
            automargin: true,
        }, extra || {});
    }

    function baseLayout(t) {
        return {
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            font: { family: t.font, size: 12, color: t.text },
            hoverlabel: { bgcolor: t.surface, bordercolor: t.border, font: { family: t.font, color: t.strong, size: 12 } },
            showlegend: false,
        };
    }

    const config = { displayModeBar: false, responsive: true };
    const fmt = (value, digits) => (value === null || value === undefined ? "—" : Number(value).toFixed(digits));

    function hoverText(p) {
        return "<b>" + p.label + "</b> · rank " + p.rank +
            "<br>" + primary.label + ": " + fmt(p.metrics[primary.key], 4) +
            "<br>Time: " + fmt(p.time, 2) + " s · Memory: " + fmt(p.ram, 1) + " MB" +
            (p.violations.length ? "<br>Broke: " + p.violations.map((v) => v.label).join(", ") : "") +
            "<br><span style='font-family:monospace'>" + p.params + "</span>";
    }

    function drawTradeoff(t) {
        const target = document.getElementById("tradeoffChart");
        if (!target) return;
        const usable = points.filter((p) => p.time !== null && p.metrics[primary.key] !== null);
        const traces = families.map((label) => {
            const group = usable.filter((p) => p.label === label && !p.selected);
            return {
                type: "scatter",
                mode: "markers",
                name: label,
                x: group.map((p) => p.time),
                y: group.map((p) => p.metrics[primary.key]),
                text: group.map(hoverText),
                hovertemplate: "%{text}<extra></extra>",
                marker: {
                    size: 9,
                    color: t.muted,
                    symbol: group.map((p) => symbolFor(label) + (p.violations.length ? "-open" : "")),
                    line: { width: 1.5, color: t.muted },
                },
            };
        });

        const selected = usable.find((p) => p.selected);
        const annotations = [];
        if (selected) {
            traces.push({
                type: "scatter",
                mode: "markers",
                name: "Selected",
                showlegend: false,
                hoverinfo: "skip",
                x: [selected.time],
                y: [selected.metrics[primary.key]],
                marker: { size: 30, color: t.accent, opacity: 0.18, line: { width: 0 } },
            });
            traces.push({
                type: "scatter",
                mode: "markers",
                name: "Selected",
                x: [selected.time],
                y: [selected.metrics[primary.key]],
                text: [hoverText(selected)],
                hovertemplate: "%{text}<extra></extra>",
                marker: { size: 13, color: t.accent, symbol: symbolFor(selected.label), line: { width: 2, color: t.surface } },
            });
            annotations.push({
                x: selected.time, y: selected.metrics[primary.key],
                xref: "x", yref: "y",
                text: "<b>" + selected.label + "</b>", showarrow: false,
                xanchor: "left", xshift: 16,
                font: { family: t.display, size: 13, color: t.strong },
            });
        }

        const times = usable.map((p) => p.time).filter((v) => v > 0);
        const logX = times.length > 1 && Math.max(...times) / Math.min(...times) > 20;

        const layout = Object.assign(baseLayout(t), {
            margin: { l: 56, r: 16, t: 36, b: 44 },
            showlegend: true,
            legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { size: 12, color: t.text } },
            xaxis: axis(t, { type: logX ? "log" : "linear", title: { text: "Training time, all folds (s)" } }),
            yaxis: axis(t, { title: { text: primary.label + (primary.higher_is_better ? " (higher is better)" : " (lower is better)") } }),
            annotations: annotations,
        });
        Plotly.react(target, traces, layout, config);
    }

    function drawResources(t) {
        const target = document.getElementById("resourceChart");
        if (!target) return;
        const best = points.filter((p) => p.best_of_model);
        const labels = best.map((p) => p.label);
        const colors = best.map((p) => (p.selected ? t.accent : t.muted));
        const panels = [
            { key: "time", title: "Time (s)", digits: 2 },
            { key: "ram", title: "Memory (MB)", digits: 1 },
            { key: "cpu", title: "CPU (%)", digits: 0 },
        ];
        const width = 0.29;
        const gap = (1 - width * panels.length) / (panels.length - 1);

        const layout = Object.assign(baseLayout(t), {
            margin: { l: 8, r: 16, t: 36, b: 28 },
            bargap: 0.55,
            yaxis: axis(t, { autorange: "reversed", gridcolor: "rgba(0,0,0,0)", ticks: "", tickfont: { family: t.font, size: 12, color: t.strong } }),
            annotations: [],
        });

        const traces = panels.map((panel, i) => {
            const suffix = i === 0 ? "" : String(i + 1);
            const values = best.map((p) => p[panel.key]);
            const max = Math.max(0, ...values.filter((v) => v !== null));
            const start = i * (width + gap);
            layout["xaxis" + suffix] = axis(t, {
                domain: [start, start + width],
                range: [0, (max || 1) * 1.35],
                showticklabels: false,
                showgrid: false,
                showline: false,
            });
            layout.annotations.push({
                text: panel.title, xref: "x" + suffix + " domain", yref: "paper",
                x: 0, y: 1.06, xanchor: "left", yanchor: "bottom", showarrow: false,
                font: { size: 12, color: t.text },
            });
            return {
                type: "bar",
                orientation: "h",
                x: values,
                y: labels,
                xaxis: "x" + suffix,
                yaxis: "y",
                marker: { color: colors, cornerradius: 4 },
                text: values.map((v) => fmt(v, panel.digits)),
                textposition: "outside",
                cliponaxis: false,
                textfont: { family: token("--mono"), size: 11, color: t.strong },
                hovertemplate: "%{y}: %{text} " + panel.title.replace(/^.*\((.*)\)$/, "$1") + "<extra></extra>",
            };
        });
        Plotly.react(target, traces, layout, config);
    }

    function drawCharts() {
        if (typeof Plotly === "undefined") return;
        const t = theme();
        drawTradeoff(t);
        drawResources(t);
    }

    // ---------- leaderboard ----------

    const table = document.getElementById("leaderboard");
    const bestOnly = document.getElementById("bestOnly");
    const count = document.getElementById("boardCount");

    function applyFilter() {
        if (!table) return;
        const rows = Array.from(table.tBodies[0].rows);
        let shown = 0;
        rows.forEach((row) => {
            const hide = bestOnly.checked && row.dataset.best !== "true";
            row.hidden = hide;
            if (!hide) shown += 1;
        });
        count.textContent = shown === rows.length
            ? rows.length + " candidates"
            : "Showing " + shown + " of " + rows.length + " candidates";
    }

    function sortBy(button) {
        const th = button.closest("th");
        const key = button.dataset.key;
        const numeric = button.dataset.type === "num";
        const ascending = th.getAttribute("aria-sort") !== "ascending";
        table.querySelectorAll("th[aria-sort]").forEach((cell) => cell.removeAttribute("aria-sort"));
        th.setAttribute("aria-sort", ascending ? "ascending" : "descending");

        const body = table.tBodies[0];
        const rows = Array.from(body.rows);
        rows.sort((a, b) => {
            const va = a.getAttribute("data-" + key);
            const vb = b.getAttribute("data-" + key);
            if (va === "" && vb === "") return 0;
            if (va === "") return 1; // missing values always last
            if (vb === "") return -1;
            const diff = numeric ? Number(va) - Number(vb) : va.localeCompare(vb);
            return ascending ? diff : -diff;
        });
        rows.forEach((row) => body.append(row));
    }

    if (table) {
        table.querySelectorAll(".sort-btn").forEach((button) => button.addEventListener("click", () => sortBy(button)));
        bestOnly.addEventListener("change", applyFilter);
        applyFilter();
    }

    drawCharts();
    document.addEventListener("themechange", drawCharts);
})();
