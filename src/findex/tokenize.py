"""Streaming tokenizer.

Rules (they are also checked in tests/test_tokenize.py):

1. Text is normalised with Unicode NFKC and then ``casefold()`` — so ``ﬁle`` and
   ``FILE`` both become ``file``, ``Straße`` becomes ``strasse``.
2. A token is a run of letters and digits (any script). Underscore joins parts,
   because Python identifiers such as ``__init__`` or ``sys_path`` are searched
   as a whole; leading/trailing underscores are dropped (``__init__`` -> ``init``).
3. Apostrophes inside a word are removed and the parts are joined:
   ``don't`` -> ``dont``, ``Python's`` -> ``pythons``. Both ASCII ``'`` and
   the typographic ``’`` are handled.
4. Hyphens split words: ``read-only`` -> ``read``, ``only``.
5. Numbers are kept as tokens (``3``, ``13``, ``572``), so that ``PEP 572`` or
   ``Python 3.13`` can be searched. A decimal point splits: ``3.13`` -> ``3``, ``13``.
6. Single letters are kept except ``s`` and ``t`` left over from apostrophes.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Iterator

_APOSTROPHES = re.compile(r"(?<=\w)['’](?=\w)")
_TOKEN = re.compile(r"[^\W_]+(?:_+[^\W_]+)*")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def tokenize(text: str) -> Iterator[str]:
    """Yield normalised tokens of one text lazily (re.finditer does not build a list)."""
    text = _APOSTROPHES.sub("", normalize(text))
    for m in _TOKEN.finditer(text):
        yield m.group()


def tokenize_many(texts: Iterable[str]) -> Iterator[str]:
    for text in texts:
        yield from tokenize(text)
