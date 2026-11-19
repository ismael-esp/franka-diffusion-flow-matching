/* Charts for the project site: plain SVG built from the numbers in report/RESULTS.md (copied below, nothing re-derived
   except the one-standard-error bar sqrt(p(1-p)/n) that RESULTS.md defines). Charts redraw at their real pixel width so
   text stays 12-14 px on a phone. No dependencies. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, text) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }
  function se(p, n) { return Math.sqrt((p / 100) * (1 - p / 100) / n) * 100; }
  function f1(v) { return v.toFixed(1); }

  /* ---------- data (report/RESULTS.md) ---------- */
  var MAIN = [
    { g: "Stacking, demos only", sub: "10 steps, seeds 1001-1003", rows: [
      { l: "Diffusion", v: 10.2, n: 600, c: "dp", dots: [8.0, 9.0, 13.5] },
      { l: "Flow matching", v: 7.3, n: 600, c: "fm", dots: [6.0, 10.0, 6.0] }] },
    { g: "Drawer", sub: "10 steps, seeds 1001-1003", rows: [
      { l: "Diffusion", v: 92.2, n: 600, c: "dp", se: 1.1 },
      { l: "Flow matching", v: 92.3, n: 600, se: 1.1, c: "fm", dots: [95.0, 90.5, 91.5] }] },
    { g: "Lift", sub: "10 steps, seeds 1001-1003", rows: [
      { l: "Diffusion", v: 62.8, n: 600, c: "dp", dots: [63.0, 66.0, 59.5] },
      { l: "Flow matching", v: 78.2, n: 600, c: "fm", dots: [77.0, 78.0, 79.5] }] },
    { g: "Pouring", sub: "5 steps, seed 201, 200 episodes", rows: [
      { l: "Diffusion", v: 27.5, n: 200, c: "dp" },
      { l: "Flow matching", v: 57.0, n: 200, c: "fm" }] }
  ];
  var STEPS = [
    { t: "Lift", sub: "seed 1001", x: [1, 5, 10], s: [
      { n: "Diffusion", c: "dp", p: { 5: 60.5, 10: 63.0 }, nc: { 1: "n/c" } },
      { n: "Flow matching", c: "fm", p: { 1: 66.0, 5: 81.0, 10: 77.0 } }] },
    { t: "Drawer", sub: "seed 1001, flow only", x: [1, 5, 10], s: [
      { n: "Flow matching", c: "fm", p: { 1: 75.5, 5: 93.0, 10: 95.0 } }] },
    { t: "Stacking", sub: "seed 1001, demos only, flow only", x: [1, 5, 10], s: [
      { n: "Flow matching", c: "fm", p: { 1: 0.0, 5: 10.0, 10: 6.0 } }] },
    { t: "Pouring", sub: "seed 201", x: [5, 10, 50], s: [
      { n: "Diffusion", c: "dp", p: { 5: 27.5, 50: 51.5 } },
      { n: "Flow matching", c: "fm", p: { 5: 57.0, 10: 50.0, 50: 41.5 } }] }
  ];
  var DAGGER = [
    { l: "Scripted teacher (100 episodes)", v: 94.0, se: 2.4, c: "t" },
    { l: "Diffusion, demos only", v: 10.9, se: 1.1, c: "dp" },
    { l: "Flow, demos only", v: 9.3, se: 1.0, c: "fm" },
    { l: "DAgger round 2 (diffusion, fine-tune chain)", v: 42.1, se: 1.7, c: "dp", b: true },
    { l: "DAgger round 3 (diffusion, fine-tune chain)", v: 23.1, se: 1.5, c: "dp" },
    { l: "Diffusion, from scratch on demos + rounds 1-3", v: 2.0, se: 0.5, c: "dp" },
    { l: "Flow, from scratch on demos + rounds 1-3", v: 2.8, se: 0.6, c: "fm" },
    { l: "Diffusion, from scratch + round 4", v: 2.1, se: 0.5, c: "dp" },
    { l: "Flow, from scratch + round 4", v: 2.0, se: 0.5, c: "fm" }
  ];
  var OUTCOMES = [ /* success / none stacked / one stacked, of 800 episodes */
    { l: "Diffusion, demos only", a: 87, b: 491, c: 222 },
    { l: "Flow, demos only", a: 74, b: 540, c: 186 },
    { l: "DAgger round 2", a: 337, b: 169, c: 294 },
    { l: "DAgger round 3", a: 185, b: 174, c: 441 },
    { l: "Diffusion, from scratch, rounds 1-3", a: 16, b: 432, c: 352 },
    { l: "Flow, from scratch, rounds 1-3", a: 22, b: 347, c: 431 }
  ];
  var MSE = [
    { l: "Demos only", v: 0.0060, c: "t" },
    { l: "+ round 1", v: 0.0109, c: "t" },
    { l: "+ rounds 1-2", v: 0.0276, c: "t" },
    { l: "+ rounds 1-3 (fine-tune chain)", v: 0.0472, c: "t" },
    { l: "From scratch, rounds 1-3, flow", v: 0.0404, c: "fm" },
    { l: "From scratch, rounds 1-3, diffusion", v: 0.0560, c: "dp" }
  ];
  var DRAWER = [
    { l: "PPO teacher", v: 97.0, se: 1.2, c: "t" },
    { l: "Diffusion student of the PPO teacher", v: 8.7, se: 1.2, c: "neg" },
    { l: "Scripted IK teacher (GPU eval)", v: 90.5, se: 2.1, c: "t" },
    { l: "Diffusion student of the scripted teacher", v: 92.2, se: 1.1, c: "dp" },
    { l: "Flow-matching student, scripted teacher", v: 92.3, se: 1.1, c: "fm" }
  ];

  // squeeze any label that is wider than the chart (phones) instead of letting it run off the edge
  function fit(host, W) {
    host.querySelectorAll("text.lab, text.gl").forEach(function (t) {
      var l = t.getComputedTextLength ? t.getComputedTextLength() : 0;
      if (l > W - 4) { t.setAttribute("textLength", W - 4); t.setAttribute("lengthAdjust", "spacingAndGlyphs"); }
    });
  }

  /* ---------- generic horizontal bars ---------- */
  function width(host) { return Math.max(260, Math.floor(host.getBoundingClientRect().width)); }

  function hbars(host, o) {
    // o.groups: [{g, sub, rows:[{l,v,se|n,c,dots,b}]}], o.max, o.unit, o.dec, o.aria
    var W = width(host), R = o.dec === 4 ? 70 : 118, LBL = 20, BAR = 15, ROW = LBL + BAR + 8, GH = 30, T = 4, B = 24;
    var plotW = W - R, H = T, y = T, items = [];
    o.groups.forEach(function (g) {
      if (g.g) { items.push({ h: g, y: y }); y += g.sub ? GH + 16 : GH; }
      g.rows.forEach(function (r) { items.push({ r: r, y: y }); y += ROW; });
      y += g.g ? 8 : 0;
    });
    H = y + B;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": o.aria });
    function x(v) { return plotW * v / o.max; }
    var ticks = o.ticks || [0, o.max / 2, o.max];
    ticks.forEach(function (t) {
      svg.appendChild(el("line", { x1: x(t), x2: x(t), y1: T, y2: H - B, class: "grid" }));
      svg.appendChild(el("text", { x: Math.min(x(t), plotW - 1), y: H - 6, class: "tick", "text-anchor": t === 0 ? "start" : (t === o.max ? "end" : "middle") }, (o.dec === 4 ? t.toFixed(2) : t) + (o.unit || "")));
    });
    items.forEach(function (it) {
      if (it.h) {
        svg.appendChild(el("text", { x: 0, y: it.y + 18, class: "gl" }, it.h.g));
        if (it.h.sub) svg.appendChild(el("text", { x: 0, y: it.y + 35, class: "tick" }, it.h.sub));
        return;
      }
      var r = it.r, by = it.y + LBL, col = r.c;
      svg.appendChild(el("text", { x: 0, y: it.y + 14, class: "lab", "font-weight": r.b ? 700 : 400 }, r.l));
      svg.appendChild(el("rect", { x: 0, y: by, width: Math.max(1.5, x(r.v)), height: BAR, class: "bar-" + col }));
      var s = r.se != null ? r.se : (r.n ? se(r.v, r.n) : null);
      if (s != null) {
        var cy = by + BAR / 2, x0 = x(Math.max(0, r.v - s)), x1 = x(Math.min(o.max, r.v + s));
        svg.appendChild(el("path", { d: "M" + x0 + " " + cy + "H" + x1 + "M" + x0 + " " + (cy - 4) + "V" + (cy + 4) + "M" + x1 + " " + (cy - 4) + "V" + (cy + 4), class: "se" }));
      }
      (r.dots || []).forEach(function (d) {
        svg.appendChild(el("circle", { cx: x(d), cy: by + BAR / 2, r: 3.2, class: "seed" }));
      });
      var txt = (o.dec === 4 ? r.v.toFixed(4) : f1(r.v) + (o.unit || ""));
      var vx = Math.max(x(r.v), s != null ? x(Math.min(o.max, r.v + s)) : 0, (r.dots || []).length ? x(Math.max.apply(null, r.dots)) + 4 : 0) + 9;
      var vt = el("text", { x: vx, y: by + BAR - 2.5, class: "val " + (col === "dp" ? "t-dp" : col === "fm" ? "t-fm" : "") }, txt);
      svg.appendChild(vt);
      if (r.se != null && o.showSE) {
        var span = el("tspan", { class: "tick", "font-weight": 400, dx: 5 }, "±" + f1(s));
        vt.appendChild(span);
      }
    });
    host.textContent = "";
    host.appendChild(svg);
    fit(host, W);
  }

  /* ---------- stacked outcome bars ---------- */
  function stacked(host, o) {
    var W = width(host), LBL = 20, BAR = 24, ROW = LBL + BAR + 10, T = 2, H = T + ROW * o.rows.length;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": o.aria });
    o.rows.forEach(function (r, i) {
      var y = T + i * ROW, tot = r.a + r.b + r.c, x = 0;
      svg.appendChild(el("text", { x: 0, y: y + 14, class: "lab" }, r.l));
      [["a", "seg-ok", "in"], ["c", "seg-mid", "in-mid"], ["b", "seg-none", "in-soft"]].forEach(function (s) {
        var w = W * r[s[0]] / tot;
        svg.appendChild(el("rect", { x: x, y: y + LBL, width: w, height: BAR, class: s[1] }));
        if (w > 30) svg.appendChild(el("text", { x: x + w / 2, y: y + LBL + BAR / 2 + 4.5, "text-anchor": "middle", class: s[2] }, r[s[0]]));
        x += w;
      });
    });
    host.textContent = "";
    host.appendChild(svg);
    fit(host, W);
  }

  /* ---------- small line chart: success vs sampling steps ---------- */
  function lines(host, o) {
    var W = width(host), H = 190, L = 34, Rr = 14, T = 16, B = 38, pw = W - L - Rr, ph = H - T - B;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": o.aria });
    function px(i) { return L + (o.x.length === 1 ? pw / 2 : pw * (i + .5) / o.x.length); }
    function py(v) { return T + ph * (1 - v / 100); }
    [0, 50, 100].forEach(function (t) {
      svg.appendChild(el("line", { x1: L, x2: W - Rr, y1: py(t), y2: py(t), class: "grid" }));
      svg.appendChild(el("text", { x: L - 6, y: py(t) + 4, class: "tick", "text-anchor": "end" }, t));
    });
    o.x.forEach(function (xv, i) {
      svg.appendChild(el("text", { x: px(i), y: H - 18, class: "tick", "text-anchor": "middle" }, xv));
    });
    svg.appendChild(el("text", { x: L + pw / 2, y: H - 3, class: "tick", "text-anchor": "middle" }, "sampling steps, success %"));
    o.s.forEach(function (s) {
      // points that are measured but not comparable: dashed ring on the axis, no line, labelled n/c
      Object.keys(s.nc || {}).forEach(function (xv) {
        var i = o.x.indexOf(+xv); if (i < 0) return;
        svg.appendChild(el("circle", { cx: px(i), cy: py(0), r: 5, fill: "none", stroke: "var(--ref-t)", "stroke-width": 1.6, "stroke-dasharray": "3 2" }));
        svg.appendChild(el("text", { x: px(i), y: py(0) - 9, "text-anchor": "middle", class: "tick", "font-weight": 700 }, s.nc[xv]));
      });
      var pts = [];
      o.x.forEach(function (xv, i) { if (s.p[xv] != null) pts.push([px(i), py(s.p[xv]), s.p[xv], xv]); });
      if (pts.length > 1) svg.appendChild(el("polyline", { points: pts.map(function (p) { return p[0] + "," + p[1]; }).join(" "), fill: "none", "stroke-width": 2.6, class: "ln-" + s.c, "stroke-linejoin": "round" }));
      pts.forEach(function (p) {
        svg.appendChild(el("circle", { cx: p[0], cy: p[1], r: 4.6, class: "pt-" + s.c }));
        // higher series label above, lower below; the lone series above
        var other = o.s.filter(function (q) { return q !== s && q.p[p[3]] != null; })[0];
        var below = other && other.p[p[3]] > p[2] && p[2] > 12;
        var ty = below ? p[1] + 18 : p[1] - 9;
        svg.appendChild(el("text", { x: p[0], y: ty, "text-anchor": "middle", class: "val t-" + s.c }, f1(p[2])));
      });
    });
    host.textContent = "";
    host.appendChild(svg);
  }

  /* ---------- registry ---------- */
  var charts = {
    main: function (h) {
      hbars(h, { groups: MAIN.map(function (g) { return { g: g.g, sub: g.sub, rows: g.rows }; }), max: 100, unit: "%", showSE: true,
        aria: "Success rate, diffusion against flow matching: stacking demos only 10.2 against 7.3, drawer 92.2 against 92.3, lift 62.8 against 78.2, pouring at 5 steps 27.5 against 57.0 percent." });
    },
    dagger: function (h) {
      hbars(h, { groups: [{ rows: DAGGER }], max: 100, unit: "%", showSE: true,
        aria: "Stacking success on held-out seeds 1002 to 1005: scripted teacher 94.0, diffusion demos only 10.9, flow demos only 9.3, DAgger round 2 42.1, round 3 23.1, all from-scratch runs about 2 percent." });
    },
    drawer: function (h) {
      hbars(h, { groups: [{ rows: DRAWER }], max: 100, unit: "%", showSE: true,
        aria: "Drawer success: PPO teacher 97.0, its diffusion student 8.7, scripted teacher 90.5, diffusion student 92.2, flow-matching student 92.3 percent." });
    },
    outcomes: function (h) {
      stacked(h, { rows: OUTCOMES, aria: "Stacking outcomes over 800 episodes: success, one cube stacked, none stacked, for six models." });
    },
    mse: function (h) {
      hbars(h, { groups: [{ rows: MSE }], max: 0.06, ticks: [0, 0.02, 0.04, 0.06], dec: 4,
        aria: "Best validation action error as DAgger data is added: 0.0060, 0.0109, 0.0276, 0.0472; from scratch on rounds 1 to 3: flow 0.0404, diffusion 0.0560." });
    },
    steps: function (h) {
      var d = STEPS[+h.getAttribute("data-i")];
      lines(h, { x: d.x, s: d.s, aria: d.t + ": success against sampling steps. " + d.s.map(function (s) {
        return s.n + " " + (s.nc ? "1 step not comparable, " : "") + d.x.filter(function (x) { return s.p[x] != null; }).map(function (x) { return x + " steps " + s.p[x] + "%"; }).join(", ");
      }).join("; ") + "." });
    }
  };

  function drawAll() {
    document.querySelectorAll("[data-chart]").forEach(function (h) {
      var w = Math.round(h.getBoundingClientRect().width);
      if (w < 10) return;
      if (h.__w && Math.abs(h.__w - w) < 3) return;
      h.__w = w;
      charts[h.getAttribute("data-chart")](h);
    });
  }
  window.addEventListener("DOMContentLoaded", function () {
    drawAll();
    if ("ResizeObserver" in window) {
      var t, ro = new ResizeObserver(function () { clearTimeout(t); t = setTimeout(drawAll, 80); });
      document.querySelectorAll("[data-chart]").forEach(function (h) { ro.observe(h); });
    } else window.addEventListener("resize", drawAll);
  });
})();
