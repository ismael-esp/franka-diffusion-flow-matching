"""Receives the captured SVGs and layout.json that tools/capture.js POSTs from the page: writes OUT/<page>/<name>."""
import http.server, json, sys, os
OUT = sys.argv[1]
class H(http.server.BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Access-Control-Allow-Headers", "*"); self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()
    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        parts = [p for p in d["name"].split("/") if p not in ("", ".", "..")]
        path = os.path.join(OUT, *parts); os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(d["svg"])
        self.send_response(200); self._cors(); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", 8766), H).serve_forever()
