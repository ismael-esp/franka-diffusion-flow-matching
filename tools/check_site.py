#!/usr/bin/env python3
"""Checks the GitHub Pages site in docs/: every relative href/src/poster exists, every #anchor has a target,
no root-absolute paths, no file over 100 MB, lists assets that nothing references. Usage: python3 tools/check_site.py"""
import os, sys, re
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
SITE = "https://ismael-esp.github.io/franka-diffusion-flow-matching/"

class P(HTMLParser):
    def __init__(s):
        super().__init__(); s.refs = []; s.ids = set(); s.metas = []
    def handle_starttag(s, tag, a):
        a = dict(a)
        if "id" in a: s.ids.add(a["id"])
        for k in ("href", "src", "poster"):
            if k in a: s.refs.append((tag, k, a[k]))
        if tag == "meta" and a.get("content", "").startswith("http") and a.get("property", a.get("name", "")).endswith("image"):
            s.metas.append(a["content"])

errors, external, used = [], set(), set()
pages = [f for f in os.listdir(DOCS) if f.endswith(".html")]
for page in pages:
    p = P(); p.feed(open(os.path.join(DOCS, page), encoding="utf-8").read())
    for tag, k, v in p.refs:
        if v.startswith(("mailto:", "data:", "javascript:")): continue
        if v.startswith("#"):
            if v[1:] and v[1:] not in p.ids: errors.append("%s: missing anchor %s" % (page, v))
            continue
        u = urlparse(v)
        if u.scheme in ("http", "https"):
            external.add(v); continue
        if v.startswith("/"): errors.append("%s: root-absolute path %s" % (page, v)); continue
        path = os.path.normpath(os.path.join(DOCS, unquote(u.path)))
        if not os.path.exists(path): errors.append("%s: missing file %s (%s %s)" % (page, v, tag, k))
        else: used.add(os.path.relpath(path, DOCS))
    for m in p.metas:  # social-card URLs must be absolute; check they map to a local file
        if m.startswith(SITE):
            rel = m[len(SITE):]
            if not os.path.exists(os.path.join(DOCS, rel)): errors.append("%s: og/twitter image missing locally: %s" % (page, rel))
            else: used.add(rel)
        else: errors.append("%s: image meta not under %s: %s" % (page, SITE, m))
# css url(...) and js-referenced assets
for root, _, files in os.walk(DOCS):
    for f in files:
        if f.endswith(".css"):
            txt = open(os.path.join(root, f)).read()
            for m in re.findall(r"url\(['\"]?([^)'\"]+)", txt):
                if m.startswith(("data:", "http")): continue
                path = os.path.normpath(os.path.join(root, m))
                if not os.path.exists(path): errors.append("%s: css url missing %s" % (f, m))
                else: used.add(os.path.relpath(path, DOCS))
total = 0; unused = []
for root, _, files in os.walk(DOCS):
    for f in files:
        full = os.path.join(root, f); rel = os.path.relpath(full, DOCS); sz = os.path.getsize(full); total += sz
        if sz > 100 * 1024 * 1024: errors.append("over 100 MB: %s" % rel)
        if rel not in used and not f.endswith(".html") and f != ".nojekyll" and not rel.endswith((".js", ".css")): unused.append(rel)
print("pages:", pages); print("total size: %.1f MB" % (total / 1e6))
print("external links (%d):" % len(external)); [print("  ", e) for e in sorted(external)]
print("unreferenced assets:", unused or "none")
print("ERRORS:" if errors else "OK: every local reference resolves"); [print("  ", e) for e in errors]
sys.exit(1 if errors else 0)
