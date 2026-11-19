"""Rasterise the captured chart SVGs (4x, 8 px of padding so no label is clipped), white panel background.
Usage: make_pngs.py <capture dir>/<page>   (PYTHONPATH=tools/pylib)"""
import glob, re, sys, cairosvg
PAD = 8
for f in glob.glob(sys.argv[1] + "/*.svg"):
    s = open(f).read()
    w = float(re.search(r'<svg[^>]* width="([\d.]+)"', s).group(1)); h = float(re.search(r'<svg[^>]* height="([\d.]+)"', s).group(1))
    m = re.search(r'viewBox="([^"]+)"', s); vx, vy, vw, vh = map(float, m.group(1).split()); pv = PAD * vw / w
    s = s[:m.start()] + f'viewBox="{vx-pv} {vy-pv} {vw+2*pv} {vh+2*pv}"' + s[m.end():]
    s = re.sub(r'(<svg[^>]* )width="[\d.]+"', lambda k: k.group(1) + f'width="{w+2*PAD}"', s, 1)
    s = re.sub(r'(<svg[^>]* )height="[\d.]+"', lambda k: k.group(1) + f'height="{h+2*PAD}"', s, 1)
    cairosvg.svg2png(bytestring=s.encode(), write_to=f[:-4] + ".png", scale=4, background_color="#ffffff")
print("ok", sys.argv[1])
