// Theme switch: system, light or dark, remembered across visits.
(function () {
    const KEY = "automl-theme";
    const buttons = Array.from(document.querySelectorAll("[data-theme-set]"));

    function read() {
        try {
            return localStorage.getItem(KEY) || "system";
        } catch (e) {
            return "system";
        }
    }

    function apply(theme) {
        if (theme === "system") {
            document.documentElement.removeAttribute("data-theme");
        } else {
            document.documentElement.setAttribute("data-theme", theme);
        }
        buttons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.themeSet === theme)));
        document.dispatchEvent(new CustomEvent("themechange"));
    }

    apply(read());

    buttons.forEach((button) =>
        button.addEventListener("click", function () {
            const theme = button.dataset.themeSet;
            try {
                localStorage.setItem(KEY, theme);
            } catch (e) {}
            apply(theme);
        })
    );

    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
        document.dispatchEvent(new CustomEvent("themechange"));
    });
})();
