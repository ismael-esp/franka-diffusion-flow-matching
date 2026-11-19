/* Site behaviour: videos (visible-only playback, play/pause button, reduced motion) and the section nav highlight. */
(function () {
  "use strict";
  var reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  var ICON_PLAY = '<svg class="i-play" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
  var ICON_PAUSE = '<svg class="i-pause" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>';

  function fmt(t) { t = Math.max(0, Math.round(t || 0)); return Math.floor(t / 60) + ":" + ("0" + (t % 60)).slice(-2); }

  document.querySelectorAll("video").forEach(function (v) {
    var media = v.parentNode, label = v.getAttribute("data-label") || "video";
    var auto = v.hasAttribute("data-autoplay") && !reduce;
    var ctl = v.hasAttribute("data-ctl"), big = v.hasAttribute("data-big");
    var userPaused = false, byObserver = false;
    v.muted = true;
    v.setAttribute("playsinline", "");
    v.controls = false;   // the HTML has `controls` as a no-JS fallback; here we use our own so nothing covers the legends
    // a pause the visitor makes is remembered; one made by the observer is not
    v.addEventListener("pause", function () { if (!byObserver && !v.ended) userPaused = true; byObserver = false; });
    v.addEventListener("play", function () { userPaused = false; });

    var btn = null, sync = function () {};
    function mk(cls) {
      var b = document.createElement("button");
      b.className = cls; b.type = "button"; b.innerHTML = ICON_PAUSE + ICON_PLAY; return b;
    }
    if (ctl) {
      // control bar under the video: play/pause, seek, time
      var bar = document.createElement("div"); bar.className = "vbar"; if (big) bar.hidden = true;
      btn = mk("vplay");
      var seek = document.createElement("input"); seek.type = "range"; seek.className = "vseek"; seek.min = 0; seek.max = 1000; seek.value = 0; seek.setAttribute("aria-label", "Seek: " + label);
      var tm = document.createElement("span"); tm.className = "vtime";
      var D = parseFloat(v.getAttribute("data-dur")) || 0;
      var upd = function () { var d = isFinite(v.duration) ? v.duration : D; seek.value = d ? Math.round(1000 * v.currentTime / d) : 0; tm.textContent = fmt(v.currentTime) + " / " + fmt(d); };
      upd();
      bar.appendChild(btn); bar.appendChild(seek); bar.appendChild(tm);
      var fig = media.parentNode; fig.insertBefore(bar, media.nextSibling);
      v.addEventListener("timeupdate", upd); v.addEventListener("loadedmetadata", upd);
      seek.addEventListener("input", function () {
        if (v.readyState === 0) { v.preload = "metadata"; v.load(); }
        var d = isFinite(v.duration) ? v.duration : D; if (d) { try { v.currentTime = d * seek.value / 1000; } catch (e) {} }
      });
      if (big) v.addEventListener("play", function () { bar.hidden = false; });
    } else if (v.hasAttribute("data-autoplay") || v.hasAttribute("data-loopbtn")) {
      btn = mk("vbtn"); media.appendChild(btn);
    }
    if (btn) {
      sync = function () {
        var playing = !v.paused;
        btn.setAttribute("aria-pressed", playing ? "true" : "false");
        btn.setAttribute("aria-label", (playing ? "Pause: " : "Play: ") + label);
      };
      btn.addEventListener("click", function () { if (v.paused) { v.play().catch(function () {}); } else { v.pause(); } });
      v.addEventListener("play", sync); v.addEventListener("pause", sync);
      sync();
    }

    if (big) {
      // poster with a big play button; the control bar appears after the first play
      var bg = document.createElement("button");
      bg.className = "vbig"; bg.type = "button"; bg.setAttribute("aria-label", "Play: " + label);
      bg.innerHTML = '<span><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg></span>';
      media.appendChild(bg);
      bg.addEventListener("click", function () { bg.hidden = true; v.play().catch(function () {}); if (btn) btn.focus(); });
      v.addEventListener("play", function () { bg.hidden = true; });
    }

    if (v.hasAttribute("data-autoplay") && "IntersectionObserver" in window) {
      new IntersectionObserver(function (es) {
        es.forEach(function (e) {
          if (e.isIntersecting) {
            if (auto && !userPaused && v.paused) v.play().catch(function () {});
          } else if (!v.paused) { byObserver = true; v.pause(); }
        });
      }, { threshold: 0.35 }).observe(v);
    } else if (big && "IntersectionObserver" in window) {
      new IntersectionObserver(function (es) {
        es.forEach(function (e) { if (!e.isIntersecting && !v.paused) v.pause(); });
      }, { threshold: 0.05 }).observe(v);
    }
  });

  /* ---- nav highlight ---- */
  var links = Array.prototype.slice.call(document.querySelectorAll(".site-nav a[href^='#']"));
  var map = {};
  links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
  var secs = Object.keys(map).map(function (id) { return document.getElementById(id); }).filter(Boolean);
  function setCurrent(id) {
    links.forEach(function (a) { a.removeAttribute("aria-current"); });
    if (!id) { var n0 = document.querySelector(".site-nav"); if (n0) n0.scrollLeft = 0; }
    if (map[id]) {
      map[id].setAttribute("aria-current", "true");
      var nav = map[id].closest(".site-nav");
      if (nav && nav.scrollWidth > nav.clientWidth) {
        var a = map[id], target = a.offsetLeft - (nav.clientWidth - a.offsetWidth) / 2;
        nav.scrollTo({ left: target, behavior: reduce ? "auto" : "smooth" });
      }
    }
  }
  function onScroll() {
    var y = window.scrollY + 90, cur = null;
    secs.forEach(function (s) { if (s.offsetTop <= y) cur = s.id; });
    setCurrent(cur);
  }
  var tick = false;
  window.addEventListener("scroll", function () { if (!tick) { tick = true; requestAnimationFrame(function () { tick = false; onScroll(); }); } }, { passive: true });
  window.addEventListener("load", onScroll);
})();
