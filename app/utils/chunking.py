import re


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries.

    Sentences are grouped together until adding the next one would exceed
    chunk_size. When a boundary must be forced mid-sentence, the chunk is
    split on whitespace instead. Consecutive chunks share up to `overlap`
    characters of context from the tail of the previous chunk.
    """
    text = text.strip()
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    sentences = _split_sentences(text)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not current:
            candidate = sentence
        else:
            candidate = f"{current} {sentence}"

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = _tail(current, overlap)
            candidate = f"{current} {sentence}".strip()

        if len(sentence) > chunk_size:
            # A single sentence is longer than chunk_size on its own, so it
            # must be force-split on whitespace.
            for piece in _split_long_sentence(sentence, chunk_size, overlap):
                chunks.append(piece)
            current = ""
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _tail(text: str, overlap: int) -> str:
    if overlap == 0 or len(text) <= overlap:
        return ""
    return text[-overlap:].strip()


def _split_long_sentence(sentence: str, chunk_size: int, overlap: int) -> list[str]:
    words = sentence.split()
    pieces: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                pieces.append(current)
                current = f"{_tail(current, overlap)} {word}".strip()
            else:
                current = word

    if current:
        pieces.append(current)

    return pieces
