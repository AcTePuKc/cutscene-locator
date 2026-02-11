"""Text normalization utilities used for matching stages."""

from __future__ import annotations

import re
import string

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    """Return a normalized form of text for matching.

    Rules:
    - lowercase
    - trim surrounding whitespace
    - collapse repeated whitespace
    - strip punctuation
    """

    lowered = text.lower()
    trimmed = lowered.strip()
    collapsed = _WHITESPACE_RE.sub(" ", trimmed)
    no_punct = collapsed.translate(_PUNCT_TRANSLATION)
    return _WHITESPACE_RE.sub(" ", no_punct).strip()
