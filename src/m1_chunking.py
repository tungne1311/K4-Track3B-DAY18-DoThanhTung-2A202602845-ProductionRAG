"""M1: semantic, hierarchical and Markdown-aware chunking."""

from __future__ import annotations

import glob
import hashlib
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    from pypdf import PdfReader

    return "\n\n".join(
        page.extract_text() or "" for page in PdfReader(path).pages
    ).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(path, encoding="utf-8") as handle:
            docs.append(
                {"text": handle.read(), "metadata": {"source": os.path.basename(path)}}
            )
    for path in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(path)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(path)}})
        else:
            print(
                f"Skipped scanned PDF without OCR: {os.path.basename(path)}", flush=True
            )
    return docs


def chunk_basic(
    text: str, chunk_size: int = 500, metadata: dict | None = None
) -> list[Chunk]:
    """Original paragraph baseline; a long paragraph can exceed chunk_size."""
    chunks, current = [], ""
    for paragraph in (p.strip() for p in text.split("\n\n") if p.strip()):
        if len(current) + len(paragraph) > chunk_size and current:
            chunks.append(
                Chunk(current.strip(), {**(metadata or {}), "chunk_index": len(chunks)})
            )
            current = ""
        current += paragraph + "\n\n"
    if current.strip():
        chunks.append(
            Chunk(current.strip(), {**(metadata or {}), "chunk_index": len(chunks)})
        )
    return chunks


@lru_cache(maxsize=1)
def _semantic_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def chunk_semantic(
    text: str, threshold: float = SEMANTIC_THRESHOLD, metadata: dict | None = None
) -> list[Chunk]:
    import numpy as np

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", text) if s.strip()]
    if not sentences:
        return []
    vectors = _semantic_encoder().encode(sentences, normalize_embeddings=True)
    groups = [[sentences[0]]]
    for index in range(1, len(sentences)):
        if float(np.dot(vectors[index - 1], vectors[index])) < threshold:
            groups.append([])
        groups[-1].append(sentences[index])
    return [
        Chunk(
            "\n".join(group),
            {**(metadata or {}), "strategy": "semantic", "chunk_index": index},
        )
        for index, group in enumerate(groups)
    ]


def _bounded_parts(text: str, size: int) -> list[str]:
    """Prefer paragraph/line/word boundaries, but enforce the character limit."""
    parts = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= size:
            parts.append(remaining)
            break
        end = size
        for separator in ("\n\n", "\n", " "):
            boundary = remaining.rfind(separator, 0, size + 1)
            if boundary >= size // 2:
                end = boundary
                break
        part = remaining[:end].strip()
        if part:
            parts.append(part)
        remaining = remaining[end:].strip()
    return parts


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    if not 0 < child_size < parent_size:
        raise ValueError("Require 0 < child_size < parent_size")
    metadata = dict(metadata or {})
    identity = hashlib.sha256(
        (str(metadata.get("source", "")) + "\0" + text).encode()
    ).hexdigest()[:16]
    parents, children = [], []
    for index, part in enumerate(_bounded_parts(text, parent_size)):
        pid = f"{identity}:parent:{index}"
        parents.append(
            Chunk(
                part,
                {
                    **metadata,
                    "strategy": "hierarchical",
                    "chunk_type": "parent",
                    "parent_id": pid,
                },
            )
        )
        # Also split short parents so their children are strictly smaller (except a single character).
        limit = min(child_size, max(1, len(part) - 1))
        for child_index, child in enumerate(_bounded_parts(part, limit)):
            children.append(
                Chunk(
                    child,
                    {
                        **metadata,
                        "strategy": "hierarchical",
                        "chunk_type": "child",
                        "chunk_index": child_index,
                        "chunk_id": f"{pid}:child:{child_index}",
                    },
                    pid,
                )
            )
    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    chunks, lines, header, fence = [], [], "", None

    def flush():
        content = "\n".join(lines).strip()
        if content:
            chunks.append(
                Chunk(
                    content,
                    {
                        **(metadata or {}),
                        "section": header,
                        "strategy": "structure",
                        "chunk_index": len(chunks),
                    },
                )
            )

    for line in text.splitlines():
        fence_match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
        if fence is None and re.match(r"^#{1,6}\s+\S", line):
            flush()
            lines = []
            header = line.strip()
        lines.append(line)
    flush()
    return chunks


def compare_strategies(documents: list[dict]) -> dict:
    def stats(chunks):
        sizes = [len(chunk.text) for chunk in chunks]
        return {
            "count": len(sizes),
            "avg_len": round(sum(sizes) / len(sizes)) if sizes else 0,
            "min_len": min(sizes, default=0),
            "max_len": max(sizes, default=0),
        }

    text = "\n\n".join(doc["text"] for doc in documents)
    parents, children = chunk_hierarchical(text, metadata={"source": "all"})
    result = {
        "basic": stats(chunk_basic(text)),
        "semantic": stats(chunk_semantic(text)),
        "hierarchical": {**stats(children), "parents": len(parents)},
        "structure": stats(chunk_structure_aware(text)),
    }
    for name, values in result.items():
        print(name, values, flush=True)
    return result
