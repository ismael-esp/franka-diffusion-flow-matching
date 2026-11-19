/* Load sequence for the report pages (web only). Bars grow from their baseline, then the key numbers count up, once.
   Skipped entirely with ?static=1 (used when the page is captured for the PowerPoint) or prefers-reduced-motion. */
(function () {
  const q = new URLSearchParams(location.search);
  const still = q.has("static") || matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (q.has("static")) document.documentElement.setAttribute("data-static", "");
  if (still) return;

  // 1. bars: every filled rect inside a chart that is not a background track or frame
  document.querySelectorAll(".chart svg").forEach((svg, k) => {
    let i = 0;
    svg.querySelectorAll("rect").forEach(r => {
      const f = (r.getAttribute("fill") || "").trim();
      if (!f || f === "none" || f.includes("--line") || f.startsWith("rgba")) return;
      const w = +r.getAttribute("width"), h = +r.getAttribute("height");
      if (w < 2 || h < 2) return;
      r.classList.add(w >= h ? "g-x" : "g-y");
      r.style.setProperty("--d", (0.3 + k * 0.05 + i * 0.04).toFixed(2) + "s");
      i++;
    });
  });

  // 2. key numbers: <span class="cu">42.1</span> counts from 0 to its own text, keeping its decimals
  const reduce = false;
  document.querySelectorAll(".cu").forEach((el, k) => {
    const final = el.textContent, v = parseFloat(final.replace(",", ""));
    if (!isFinite(v)) return;
    const dec = (final.split(".")[1] || "").length, t0 = performance.now() + 450 + k * 70, dur = 1100;
    el.textContent = (0).toFixed(dec);
    (function tick(now) {
      const p = Math.min(1, Math.max(0, (now - t0) / dur)), e = 1 - Math.pow(1 - p, 3);
      el.textContent = (v * e).toFixed(dec);
      if (p < 1) requestAnimationFrame(tick); else el.textContent = final;
    })(performance.now());
  });
})();
