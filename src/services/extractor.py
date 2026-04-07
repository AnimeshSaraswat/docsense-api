from pathlib import Path

import pypdf


SUPPORTED_TYPES = {"application/pdf", "text/plain"}


def extract_text(file_path: Path, content_type: str) -> str:
    if content_type == "application/pdf":
        return _extract_pdf(file_path)
    if content_type == "text/plain":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported content type: {content_type}")


def _extract_pdf(file_path: Path) -> str:
    reader = pypdf.PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()
