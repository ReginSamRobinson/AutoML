// Theme switch: a single light/dark toggle, remembered across visits.
// Until the user picks one, the page follows the system setting.
(function () {
    const KEY = "automl-theme";
    const root = document.documentElement;
    const buttons = Array.from(document.querySelectorAll("[data-theme-toggle]"));
    const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

    function saved() {
        try {
            const value = localStorage.getItem(KEY);
            return value === "light" || value === "dark" ? value : null;
        } catch (e) {
            return null;
        }
    }

    function apply(theme) {
        root.setAttribute("data-theme", theme);
        const label = theme === "dark" ? "Switch to light theme" : "Switch to dark theme";
        buttons.forEach((button) => {
            button.setAttribute("aria-label", label);
            button.title = label;
        });
        document.dispatchEvent(new CustomEvent("themechange"));
    }

    apply(saved() || (systemDark.matches ? "dark" : "light"));

    buttons.forEach((button) =>
        button.addEventListener("click", function () {
            const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
            try {
                localStorage.setItem(KEY, next);
            } catch (e) {}
            apply(next);
        })
    );

    systemDark.addEventListener("change", function (event) {
        if (!saved()) apply(event.matches ? "dark" : "light");
    });
})();
