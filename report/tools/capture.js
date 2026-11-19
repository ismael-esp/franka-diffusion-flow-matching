/* Run inside a report page loaded with ?static=1 (eval(await (await fetch('/tools/capture.js')).text())).
   Posts every chart SVG (computed styles inlined, class removed) and layout.json (positions + styles of all text and boxes)
   to tools/recv.py on :8766 under <page>/. build_onepager.py reads them. */
(async () => {
  const page = location.pathname.split('/').pop().replace('.html', '') || 'index';
  const post = (name, body) => fetch('http://127.0.0.1:8766/', { method: 'POST', body: JSON.stringify({ name: page + '/' + name, svg: body }) });
  window.scrollTo(0, 0);
  const props = ['fill','stroke','stroke-width','stroke-opacity','fill-opacity','opacity','font-family','font-size','font-weight','text-anchor','dominant-baseline','stroke-dasharray','stroke-linecap','stroke-linejoin','letter-spacing'];
  const svgs = [...document.querySelectorAll('svg')], names = [];
  for (const src of svgs) {
    const name = (src.closest('.chart') || {}).id || ('svg' + names.length); names.push(name);
    const clone = src.cloneNode(true);
    const a = [src, ...src.querySelectorAll('*')], b = [clone, ...clone.querySelectorAll('*')];
    a.forEach((e, k) => { const cs = getComputedStyle(e); let st = ''; props.forEach(p => { const v = cs.getPropertyValue(p); if (v) st += p + ':' + v + ';'; }); b[k].setAttribute('style', st); b[k].removeAttribute('class'); });
    const r = src.getBoundingClientRect(); clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg'); clone.setAttribute('width', r.width); clone.setAttribute('height', r.height);
    await post(name + '.svg', new XMLSerializer().serializeToString(clone));
  }
  await post('order.json', JSON.stringify(names));
  const rgb2hex = c => { const m = c.match(/[\d.]+/g); return m ? m.slice(0, 3).map(x => (+x | 0).toString(16).padStart(2, '0')).join('') : '000000'; };
  const R = e => { const r = e.getBoundingClientRect(); return [Math.round((r.left + scrollX) * 10) / 10, Math.round((r.top + scrollY) * 10) / 10, Math.round(r.width * 10) / 10, Math.round(r.height * 10) / 10]; };
  const lineBreaks = el => {   // the browser's own wrapping of a headline, as one string per line
    const words = [], tw = document.createTreeWalker(el, NodeFilter.SHOW_TEXT); let t;
    while (t = tw.nextNode()) { const re = /\S+/g; let m; while (m = re.exec(t.textContent)) { const rg = document.createRange(); rg.setStart(t, m.index); rg.setEnd(t, m.index + m[0].length); words.push({ w: m[0], top: Math.round(rg.getBoundingClientRect().top) }); } }
    const lines = []; let last = null; words.forEach(x => { if (last === null || Math.abs(x.top - last) > 5) { lines.push([]); last = x.top; } lines[lines.length - 1].push(x.w); });
    return lines.map(l => l.join(' '));
  };
  const runOf = (c, cs) => c.nodeType === 3
    ? { t: c.textContent, fs: parseFloat(cs.fontSize), fw: cs.fontWeight, col: rgb2hex(cs.color), ff: cs.fontFamily.split(',')[0].replace(/"/g, ''), blk: false }
    : (c.nodeType === 1 ? (s => ({ t: c.innerText, fs: parseFloat(s.fontSize), fw: s.fontWeight, col: rgb2hex(s.color), ff: s.fontFamily.split(',')[0].replace(/"/g, ''), blk: s.display === 'block' }))(getComputedStyle(c)) : null);
  const texts = [], walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT); let n;
  while (n = walker.nextNode()) {
    if (n.closest('svg') || ['SCRIPT', 'STYLE', 'VIDEO', 'IMG'].includes(n.tagName)) continue;
    const own = [...n.childNodes].some(c => c.nodeType === 3 && c.textContent.trim()); if (!own) continue;
    const cs = getComputedStyle(n), runs = [...n.childNodes].map(c => runOf(c, cs)).filter(Boolean);
    let rr = R(n);   // a legend label starts after its swatch (<i>): use the text node's own left edge
    if (n.firstElementChild && n.firstElementChild.tagName === 'I' && n.firstChild === n.firstElementChild) {
      const tn = [...n.childNodes].find(c => c.nodeType === 3 && c.textContent.trim()), rg = document.createRange(); rg.selectNodeContents(tn);
      const l = rg.getBoundingClientRect().left + scrollX; if (l > rr[0] + 2) { rr = [l, rr[1], rr[2] - (l - rr[0]), rr[3]]; }
    }
    texts.push({ r: rr, tag: n.tagName, cls: n.className, runs, bg: cs.backgroundColor, ta: cs.textAlign, lh: parseFloat(cs.lineHeight) || 0, lines: n.tagName === 'H1' ? lineBreaks(n) : null,
      parentOwn: !!n.parentElement && [...n.parentElement.childNodes].some(c => c.nodeType === 3 && c.textContent.trim()) });
  }
  const boxes = []; document.querySelectorAll('header,.panel,video,img,svg,.f,.legend i,.findings,footer').forEach(e => { const cs = getComputedStyle(e);
    boxes.push({ r: R(e), tag: e.tagName, cls: (e.className && e.className.baseVal !== undefined) ? e.className.baseVal : e.className, src: e.getAttribute('src') || '', bg: cs.backgroundColor,
      bt: cs.borderTopColor + ' ' + cs.borderTopWidth, bb: cs.borderBottomColor + ' ' + cs.borderBottomWidth, bl: cs.borderLeftColor + ' ' + cs.borderLeftWidth, radius: cs.borderRadius }); });
  const bad = [...document.querySelectorAll('body *')].filter(e => { const cs = getComputedStyle(e); return !e.closest('svg') && (cs.opacity === '0' || (cs.transform !== 'none' && !e.closest('video'))); }).length;
  await post('layout.json', JSON.stringify({ page: [document.documentElement.scrollWidth, document.documentElement.scrollHeight], bodybg: getComputedStyle(document.body).backgroundColor, texts, boxes, hidden_or_transformed: bad }));
  return { page, svgs: names, videos: document.querySelectorAll('video').length, hidden_or_transformed: bad };
})()
