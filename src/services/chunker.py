from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int  # position in document
    char_start: int
    char_end: int


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(text=chunk_text, index=index, char_start=start, char_end=end)
            )
            index += 1

        if end == len(text):  # ← ADD THIS
            break

        start = end - overlap

    return chunks
