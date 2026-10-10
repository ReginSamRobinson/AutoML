// Landing page: nav border once scrolled, and sections fading in as they enter view.
(function () {
    const nav = document.getElementById("lpNav");
    const onScroll = () => nav.classList.toggle("is-scrolled", window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });

    const items = document.querySelectorAll(".lp-in");
    if (!("IntersectionObserver" in window) || !items.length) return;

    // Content stays visible unless this script can reveal it again.
    document.documentElement.classList.add("js-reveal");
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                entry.target.classList.add("is-in");
                observer.unobserve(entry.target);
            });
        },
        { rootMargin: "0px 0px -8% 0px", threshold: 0.12 }
    );
    items.forEach((item) => observer.observe(item));
})();
