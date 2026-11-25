"""Streaming clause splitter for low-latency TTS synthesis."""

from __future__ import annotations

STRONG_TERMINATORS = frozenset("。！？!?；;\n…")
SOFT_SEPARATORS = frozenset("，,、")


class SentenceChunker:
    """Incrementally split streamed LLM text deltas into speakable clauses."""

    def __init__(self, min_clause_chars: int = 12, max_clause_chars: int = 120) -> None:
        if min_clause_chars < 1:
            raise ValueError("min_clause_chars must be >= 1")
        if max_clause_chars < min_clause_chars:
            raise ValueError("max_clause_chars must be >= min_clause_chars")
        self.min_clause_chars = min_clause_chars
        self.max_clause_chars = max_clause_chars
        self._buffer = ""

    def reset(self) -> None:
        """Clear any buffered characters."""
        self._buffer = ""

    def _pop_clean(self) -> str | None:
        clause = self._buffer.strip()
        self._buffer = ""
        return clause or None

    def _split_overflow(self) -> str | None:
        text = self._buffer
        if len(text) < self.max_clause_chars:
            return None
        half = self.max_clause_chars // 2
        split_at = self.max_clause_chars
        # Prefer splitting at whitespace in the latter half to avoid bisecting Latin words.
        for idx in range(self.max_clause_chars - 1, half - 1, -1):
            if text[idx].isspace():
                split_at = idx + 1
                break
        head = text[:split_at].strip()
        self._buffer = text[split_at:].lstrip()
        return head or None

    def feed(self, delta: str) -> list[str]:
        """Consume a streamed text delta and emit zero or more complete clauses."""
        if not delta:
            return []
        clauses: list[str] = []
        for ch in delta:
            if ch == "\r":
                continue
            # Treat ASCII period followed by whitespace as an English sentence boundary,
            # while avoiding splits inside decimal numbers (e.g. "3.14") or abbreviations like "1.5".
            if (
                ch.isspace()
                and self._buffer.endswith(".")
                and len(self._buffer) >= 2
                and not self._buffer[-2].isdigit()
            ):
                emitted = self._pop_clean()
                if emitted:
                    clauses.append(emitted)
                continue

            self._buffer += ch
            if ch in STRONG_TERMINATORS:
                emitted = self._pop_clean()
                if emitted:
                    clauses.append(emitted)
            elif ch in SOFT_SEPARATORS and len(self._buffer.strip()) >= self.min_clause_chars:
                emitted = self._pop_clean()
                if emitted:
                    clauses.append(emitted)
            elif len(self._buffer) >= self.max_clause_chars:
                emitted = self._split_overflow()
                if emitted:
                    clauses.append(emitted)
        return clauses

    def flush(self) -> list[str]:
        """Flush any remaining buffered text at the end of a stream."""
        emitted = self._pop_clean()
        return [emitted] if emitted else []
