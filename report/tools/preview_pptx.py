"""Rough PIL preview of a deck built by build_onepager.py (no LibreOffice here): draws every shape at its real position,
rounded rectangles, rules, pictures (video = its poster), and text boxes wrapped to their box width with Liberation Sans
(Arial Narrow -> Liberation Sans Narrow; Calibri -> Liberation Sans at 0.92 size, since Calibri is narrower).
Usage: preview_pptx.py deck.pptx out_prefix [scale]"""
import sys, io
from pptx import Presentation
from pptx.util import Emu
from PIL import Image, ImageDraw, ImageFont
from pptx.enum.shapes import MSO_SHAPE_TYPE
D = "/usr/share/fonts/truetype/liberation"
def font(name, px, bold):
    narrow = name == "Arial Narrow"; px = px if narrow else px * 0.92
    f = f"{D}/LiberationSansNarrow-{'Bold' if bold else 'Regular'}.ttf" if narrow else f"{D}/LiberationSans-{'Bold' if bold else 'Regular'}.ttf"
    return ImageFont.truetype(f, max(5, int(round(px))))
prs = Presentation(sys.argv[1]); S = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
PX = 9144
for si, slide in enumerate(prs.slides):
    W, H = int(prs.slide_width / PX * S), int(prs.slide_height / PX * S)
    bgc = slide.background.fill.fore_color.rgb; im = Image.new("RGB", (W, H), tuple(bgc)); d = ImageDraw.Draw(im, "RGBA")
    for sh in slide.shapes:
        x, y, w, h = [v / PX * S for v in (sh.left, sh.top, sh.width, sh.height)]
        if sh.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE:
            fill = sh.fill.fore_color.rgb if sh.fill.type == 1 else None
            line = sh.line.color.rgb if sh.line.fill.type == 1 else None
            r = 0
            try: r = sh.adjustments[0] * min(w, h)
            except Exception: pass
            d.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=tuple(fill) if fill else None, outline=tuple(line) if line else None)
        elif sh.shape_type == MSO_SHAPE_TYPE.LINE or sh.name.startswith("Connector"):
            d.line([x, y, x + w, y + h], fill=tuple(sh.line.color.rgb), width=max(1, int(sh.line.width / PX * S)))
        elif sh.shape_type in (MSO_SHAPE_TYPE.PICTURE, MSO_SHAPE_TYPE.MEDIA):
            try:
                img = Image.open(io.BytesIO(sh.image.blob)) if sh.shape_type == MSO_SHAPE_TYPE.PICTURE else Image.open(f"export_build/posters/{sh.name[6:]}.png")
                im.paste(img.convert("RGB").resize((max(1, int(w)), max(1, int(h)))), (int(x), int(y)))
            except Exception as e:
                d.rectangle([x, y, x + w, y + h], outline=(255, 0, 0))
        elif sh.has_text_frame and sh.text_frame.text.strip():
            cy = y
            for p in sh.text_frame.paragraphs:
                words = []
                for r in p.runs:
                    sz = r.font.size.pt / 0.72 * S if r.font.size else 12 * S
                    f = font(r.font.name or "Calibri", sz, bool(r.font.bold)); col = tuple(r.font.color.rgb) if r.font.color and r.font.color.type else (0, 0, 0)
                    for wd in r.text.split(" "):
                        if wd: words.append((wd, f, col, sz))
                if not words: cy += 10 * S; continue
                ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.0; lh = max(wsz for *_, wsz in words) * 1.2 * ls; cx = x; line = []; 
                def flush():
                    global cy
                    xx = x
                    for wd, f, col, sz in line:
                        d.text((xx, cy), wd, font=f, fill=col); xx += d.textlength(wd + " ", font=f)
                    cy += lh
                for wd in words:
                    wl = d.textlength(wd[0] + " ", font=wd[1])
                    if line and cx + wl - d.textlength(" ", font=wd[1]) > x + w + 2 and sh.text_frame.word_wrap:
                        flush(); line = []; cx = x
                    line.append(wd); cx += wl
                flush()
    out = f"{sys.argv[2]}{si+1}.png"; im.save(out); print(out, im.size)
