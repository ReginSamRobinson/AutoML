// Cycles the colour theme between system, light and dark and remembers the choice.
(function () {
    const KEY = "automl-theme";
    const order = ["system", "light", "dark"];
    const button = document.getElementById("themeToggle");

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
        if (button) {
            button.textContent = "Theme: " + theme;
        }
        document.dispatchEvent(new CustomEvent("themechange"));
    }

    let current = read();
    apply(current);

    if (button) {
        button.addEventListener("click", function () {
            current = order[(order.indexOf(current) + 1) % order.length];
            try {
                localStorage.setItem(KEY, current);
            } catch (e) {}
            apply(current);
        });
    }

    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
        document.dispatchEvent(new CustomEvent("themechange"));
    });
})();
