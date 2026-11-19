#!/usr/bin/env python3
"""PowerPoint that reproduces the HTML report pages 1:1, one slide per page (index.html, page2.html).

Inputs (captured from the live page at 1600 px width): layout.json (position + style of every text element and box)
and the page's chart SVGs. 1 page px = 0.01 in, so the slide is 16 x 12.4 in. Text is native and editable, videos are
embedded with poster frames, images are the originals, charts are the page's own SVGs rasterised at 4x.
Usage: build_onepager.py <out.pptx> <layout.json> <svg dir> [<layout.json> <svg dir> ...]
Every page lists its chart names (in DOM order) in <svg dir>/order.json, written by tools/capture.js. The slide height fits the
tallest page; shorter pages sit at the top."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt

R = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
PAGES = [(json.load(open(sys.argv[i])), Path(sys.argv[i + 1])) for i in range(2, len(sys.argv), 2)]
MEDIA, POST = R / "media", R / "export_build" / "posters"
PX = 9144                                            # EMU per page px (0.01 in)
PAGE_H = int(max(max(b['r'][1] + b['r'][3] for b in L['boxes']) for L, _ in PAGES) + 22)   # content bottom + padding
E = lambda v: Emu(int(round(v * PX)))
PT = lambda px: Pt(round(px * 0.72, 1))              # 1 px = 0.01 in = 0.72 pt
FONT = {"Archivo": "Arial Narrow", "IBM Plex Sans": "Calibri"}   # the two faces of shared.css; anything else falls back to Calibri
PAD = 8                                              # px margin added around each chart render (no clipped labels)


def css_rgb(c):
    m = re.findall(r"[\d.]+", c)
    return (RGBColor(*[int(float(v)) for v in m[:3]]), float(m[3]) if len(m) > 3 else 1.0) if m else (None, 0.0)


prs = Presentation()
prs.slide_width, prs.slide_height = E(1600), E(PAGE_H)
s = None                                             # current slide, set per page below


def rect(x, y, w, h, fill=None, line=None, line_w=1.0, radius_px=0, name=None, alpha=None):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius_px else MSO_SHAPE.RECTANGLE, E(x), E(y), E(w), E(h))
    if radius_px:
        shp.adjustments[0] = min(0.5, radius_px / max(1e-6, min(w, h)))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
        if alpha is not None and alpha < 1:          # python-pptx has no API for fill alpha: set it in the XML
            from pptx.oxml.ns import qn
            clr = shp.fill._xPr.find(qn("a:solidFill"))[0]
            a = clr.makeelement(qn("a:alpha"), {"val": str(int(alpha * 100000))}); clr.append(a)
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line; shp.line.width = Pt(line_w * 0.72)
    shp.shadow.inherit = False
    if name:
        shp.name = name
    shp.text_frame.text = ""
    return shp


def hline(x, y, w, color, px):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, E(x), E(y), E(x + w), E(y))
    ln.line.color.rgb = color; ln.line.width = Pt(px * 0.72)
    return ln


for page, (L, SVGDIR) in enumerate(PAGES):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = css_rgb(L["bodybg"])[0]
    order = json.load(open(SVGDIR / "order.json"))
    # ---------------------------------------------------------------- boxes: panels, chips, rules, swatches
    for b in L["boxes"]:
        x, y, w, h = b["r"]; cls = b["cls"] if isinstance(b["cls"], str) else ""
        fill, a = css_rgb(b["bg"])
        if b["tag"] in ("ARTICLE", "SECTION") and "panel" in cls:
            rect(x, y, w, h, fill, RGBColor(0xD3, 0xDB, 0xE2), 1, float(re.findall(r"[\d.]+", b.get("radius", "4"))[0]), name="Panel")
        elif b["tag"] == "HEADER":
            hline(x, y + h - 1, w, RGBColor(0x12, 0x20, 0x2B), 2)
        elif b["tag"] == "DIV" and cls.startswith("f"):    # findings: coloured top rule
            hline(x, y + 1.5, w, css_rgb(b["bt"].rsplit(" ", 1)[0])[0], 3)
        elif b["tag"] == "FOOTER":
            hline(x, y, w, RGBColor(0xD3, 0xDB, 0xE2), 1)
        elif b["tag"] == "I":                             # legend swatches
            if cls == "h":
                rect(x, y, w, h, RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xB8, 0x47, 0x3F), 1, name="Swatch DAgger")
            elif cls == "ex":
                rect(x, y, w, h, RGBColor(0x12, 0x20, 0x2B), None, name="Swatch executed", alpha=0.10)
            elif cls == "heat":                           # inferno gradient: four stops
                for k, c in enumerate([(0x1B, 0x0C, 0x41), (0xB5, 0x36, 0x7A), (0xFB, 0xB3, 0x2F), (0xFC, 0xFF, 0xA4)]):
                    rect(x + k * w / 4, y, w / 4, h, RGBColor(*c), None, name="Swatch attention")
            elif cls == "kp":
                shp = s.shapes.add_shape(MSO_SHAPE.OVAL, E(x), E(y), E(w), E(h)); shp.name = "Swatch keypoint"
                shp.fill.solid(); shp.fill.fore_color.rgb = RGBColor(0x50, 0xE6, 0xFF); shp.line.color.rgb = RGBColor(0x12, 0x20, 0x2B)
            else:
                rect(x, y, w, h, fill, None, name="Swatch")

    # ---------------------------------------------------------------- media
    for b in L["boxes"]:
        x, y, w, h = b["r"]
        if b["tag"] == "VIDEO":
            stem = Path(b["src"]).stem
            mv = s.shapes.add_movie(str(MEDIA / f"{stem}.mp4"), E(x), E(y), E(w), E(h), poster_frame_image=str(POST / f"{stem}.png"), mime_type="video/mp4")
            mv.name = f"Video {stem}"
        elif b["tag"] == "IMG":
            p = s.shapes.add_picture(str(MEDIA / Path(b["src"]).name), E(x), E(y), E(w), E(h)); p.name = Path(b["src"]).stem
    svgs = [b for b in L["boxes"] if b["tag"] == "svg"]
    for name, b in zip(order, svgs):
        x, y, w, h = b["r"]
        p = s.shapes.add_picture(str(SVGDIR / f"{name}.png"), E(x - PAD), E(y - PAD), E(w + 2 * PAD), E(h + 2 * PAD)); p.name = f"Chart {name}"

    # ---------------------------------------------------------------- text (native, editable)
    for t in L["texts"]:
        if t["parentOwn"]:                                # nested run: already part of its parent's runs
            continue
        x, y, w, h = t["r"]
        fill, a = css_rgb(t["bg"])
        if a > 0:                                         # overlay tags on videos/images (SCENE, GRASP 1.4 S, ...)
            rect(x, y, w, h, fill, None, radius_px=2, name="Tag", alpha=a)
        runs = [dict(r) for r in t["runs"] if r["t"]]
        cls = t["cls"] if isinstance(t["cls"], str) else ""
        for i in range(1, len(runs)):                     # spacing the page gets from CSS margins (01 Title, +14 pts 42 -> 56 %)
            r0, r1 = runs[i - 1], runs[i]
            if not r1.get("blk") and min(r0["fs"], r1["fs"]) / max(r0["fs"], r1["fs"]) < 0.7 and not r0["t"].endswith(" ") and not r1["t"].startswith(" "):
                r1["t"] = " " + r1["t"]
        if t["tag"] == "H1" and t.get("lines") and len(t["lines"]) > 1:   # the browser's own line breaks, one paragraph per line
            paras = [[{**runs[0], "t": ln}] for ln in t["lines"]]
        else:
            paras = [[]]                                  # a child that is display:block (a caption under a number) starts a new paragraph
            for r in runs:
                if r.get("blk") and paras[-1]:
                    paras.append([])
                paras[-1].append(r)
        single = t["lh"] and h < 1.6 * t["lh"]
        pad = 0 if a == 0 else 4
        tb = s.shapes.add_textbox(E(x + pad), E(y), E(w - pad + (60 if single else 4)), E(h))
        tb.name = f"Text {t['runs'][0]['t'][:24]}" if t["runs"] else "Text"
        tf = tb.text_frame; tf.word_wrap = not single; tf.auto_size = None
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE if a > 0 else MSO_ANCHOR.TOP
        for pi, pr in enumerate(paras):
            para = tf.paragraphs[0] if pi == 0 else tf.add_paragraph()
            para.alignment = PP_ALIGN.CENTER if t["ta"] == "center" else PP_ALIGN.LEFT
            para.line_spacing = 0.95 if t["tag"] == "H1" else (0.85 if "n" in cls.split() else 1.0)   # CSS line-height 1 on the big numbers
            if pi and not t["tag"] == "H1":
                para.space_before = Pt(3)
            for r in pr:
                txt = re.sub(r"\s+", " ", r["t"])
                if not txt.strip() and not txt:
                    continue
                run = para.add_run(); run.text = txt
                f = run.font; f.size = PT(r["fs"]); f.bold = int(float(r["fw"])) >= 600
                f.name = FONT.get(r["ff"], "Calibri"); f.color.rgb = RGBColor.from_string(r["col"].upper())
                if txt == "k" and t["runs"] and "DDIM samples" in t["runs"][0]["t"]:   # the page's <sub>k</sub>
                    run.font._rPr.set("baseline", "-25000")

prs.save(OUT)
print("saved", OUT, OUT.stat().st_size // 1024, "KB;", len(prs.slides), "slides")
