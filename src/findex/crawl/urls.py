"""One spelling per address, so that the seen set really prevents double downloads."""

from __future__ import annotations

import posixpath
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

DEFAULT_PORTS = {"http": 80, "https": 443}


def canonicalize(url: str, base: str | None = None) -> str | None:
    """Absolute, normalised http(s) URL, or None for anything we do not crawl.

    - relative links are resolved against ``base``;
    - scheme and host go to lower case, the default port is dropped;
    - the ``#fragment`` is dropped (same document);
    - ``.`` / ``..`` segments and repeated slashes are collapsed, an empty path becomes ``/``;
    - a trailing ``/index.html`` is the same page as the directory, it becomes ``/``;
    - query parameters are sorted.
    """
    if base is not None:
        url = urljoin(base, url.strip())
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS or not parts.hostname:
        return None
    host = parts.hostname.lower()
    if port is not None and port != DEFAULT_PORTS[scheme]:
        host = f"{host}:{port}"
    path = parts.path or "/"
    while "//" in path:
        path = path.replace("//", "/")
    trailing = path.endswith("/")
    path = posixpath.normpath(path)
    if path in (".", "//"):
        path = "/"
    if trailing and not path.endswith("/"):
        path += "/"
    if path.endswith("/index.html"):
        path = path[: -len("index.html")]
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, host, path, query, ""))


def host_of(url: str) -> str:
    return urlsplit(url).netloc
