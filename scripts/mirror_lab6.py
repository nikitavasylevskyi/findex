"""Lab 6: a local HTML mirror of the corpus, with failures on purpose.

Serves data/peps and data/docs as linked HTML pages, so the crawler can be
measured without touching anybody else's server. Every response waits
``--latency`` seconds, like a remote site would. Some pages fail in a
reproducible way:

- every 37th PEP answers 429 with ``Retry-After: 1`` the first time;
- every 41st PEP answers 503 twice, then 200;
- PEP 666 never answers in time (sleeps 30 s);
- /loop/a and /loop/b redirect to each other;
- /private/ is disallowed in robots.txt; /peps/pep-9999.html is a dead link (404).

    uv run python scripts/mirror_lab6.py data --port 8765 --latency 0.05
"""

from __future__ import annotations

import argparse
import html
import re
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PEP_REF = re.compile(r"\bPEP\s+(\d{1,4})\b")


class Mirror:
    def __init__(self, root: Path, latency: float) -> None:
        self.latency = latency
        self.peps = {int(p.stem.removeprefix("pep-")): p for p in sorted((root / "peps").glob("pep-*.rst"))}
        self.docs = {p.stem: p for p in sorted((root / "docs").glob("*.rst"))}
        self.hits: Counter[str] = Counter()
        self.lock = threading.Lock()

    def hit(self, path: str) -> int:
        with self.lock:
            self.hits[path] += 1
            return self.hits[path]

    def index_page(self) -> str:
        links = ['<li><a href="/loop/a">loop</a></li>', '<li><a href="/private/secret.html">private</a></li>',
                 '<li><a href="/peps/pep-9999.html">PEP 9999</a></li>', '<li><a href="/#top">top</a></li>',
                 '<li><a href="/peps/pep-0666.html">PEP 666</a></li>']
        links += [f'<li><a href="/peps/pep-{n:04d}.html">PEP {n}</a></li>' for n in self.peps]
        links += [f'<li><a href="/docs/{name}.html">{html.escape(name)}</a></li>' for name in self.docs]
        return page("findex mirror", "<ul>" + "".join(links) + "</ul>")

    def text_page(self, title: str, text: str) -> str:
        refs = sorted({int(m.group(1)) for m in PEP_REF.finditer(text)} & self.peps.keys())
        nav = " ".join(f'<a href="/peps/pep-{n:04d}.html">PEP {n}</a>' for n in refs[:30])
        return page(title, f'<p><a href="/">index</a> {nav}</p><pre>{html.escape(text)}</pre>')


def page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset=utf-8><title>{html.escape(title)}</title></head>"
            f"<body><h1>{html.escape(title)}</h1>{body}</body></html>")


def make_handler(mirror: Mirror) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

        def send(self, status: int, body: str = "", ctype: str = "text/html; charset=utf-8",
                 headers: dict[str, str] | None = None) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            time.sleep(mirror.latency)
            path = self.path.split("?")[0]
            if path == "/robots.txt":
                return self.send(200, "User-agent: *\nDisallow: /private/\n", "text/plain")
            if path == "/":
                return self.send(200, mirror.index_page())
            if path == "/loop/a":
                return self.send(302, headers={"Location": "/loop/b"})
            if path == "/loop/b":
                return self.send(302, headers={"Location": "/loop/a"})
            if m := re.fullmatch(r"/peps/pep-(\d{4})\.html", path):
                n = int(m.group(1))
                if n not in mirror.peps:
                    return self.send(404, page("not found", ""))
                attempt = mirror.hit(path)
                if n == 666:
                    time.sleep(30)
                if n and n % 37 == 0 and attempt == 1:
                    return self.send(429, page("slow down", ""), headers={"Retry-After": "1"})
                if n and n % 41 == 0 and attempt <= 2:
                    return self.send(503, page("unavailable", ""))
                text = mirror.peps[n].read_text(encoding="utf-8", errors="replace")
                return self.send(200, mirror.text_page(f"PEP {n}", text))
            if (m := re.fullmatch(r"/docs/([\w.\-]+)\.html", path)) and m.group(1) in mirror.docs:
                text = mirror.docs[m.group(1)].read_text(encoding="utf-8", errors="replace")
                return self.send(200, mirror.text_page(m.group(1), text))
            return self.send(404, page("not found", ""))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="data")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--latency", type=float, default=0.05)
    args = parser.parse_args()
    mirror = Mirror(Path(args.root), args.latency)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(mirror))
    server.daemon_threads = True
    print(f"mirror of {len(mirror.peps)} PEPs and {len(mirror.docs)} docs on http://127.0.0.1:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
