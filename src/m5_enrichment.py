"""M5: enrichment with one API request per chunk and offline fallbacks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import ENRICHMENT_API_KEY, ENRICHMENT_BASE_URL, ENRICHMENT_MODEL


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _shorten(summary: str, text: str) -> str:
    if not text:
        return ""
    if len(summary) < len(text):
        return summary.strip()
    words = text[: max(0, len(text) // 2)].rsplit(" ", 1)[0]
    return words or text[: max(0, len(text) - 1)]


def _fallback(text: str, source: str = "") -> dict:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    title = next(
        (
            line.lstrip("# ").strip()
            for line in text.splitlines()
            if line.startswith("#")
        ),
        source or "chính sách nội bộ",
    )
    return {
        "summary": _shorten(" ".join(sentences[:2]), text),
        "questions": [
            f"Nội dung quy định về {s.rstrip('.!?')} là gì?" for s in sentences[:3]
        ],
        "context": f"Trích từ tài liệu {source or title}, chủ đề: {title}.",
        "metadata": {
            "topic": title,
            "entities": [],
            "category": "policy",
            "language": "vi",
        },
    }


def _enrich_single_call(text: str, source: str) -> dict:
    fallback = _fallback(text, source)
    if not ENRICHMENT_API_KEY or ENRICHMENT_API_KEY == "sk-...":
        fallback["metadata"]["enrichment_status"] = "offline"
        return fallback
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=ENRICHMENT_API_KEY,
            base_url=ENRICHMENT_BASE_URL or None,
            timeout=60,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=ENRICHMENT_MODEL,
            temperature=0,
            max_tokens=450,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": 'Chỉ dựa vào đoạn văn, trả JSON: {"summary": "tóm tắt NGẮN HƠN đoạn gốc", '
                    '"questions": ["3 câu hỏi đoạn văn trả lời được, kết thúc bằng ?"], '
                    '"context": "một câu định vị đoạn trong tài liệu, không thêm sự kiện", '
                    '"metadata": {"topic": "...", "entities": [], "category": "policy|hr|it|finance", '
                    '"language": "vi"}}. Không đổi số liệu hay phủ định.',
                },
                {"role": "user", "content": f"Tài liệu: {source}\nĐoạn văn:\n{text}"},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        if not isinstance(data, dict):
            raise TypeError("Enrichment response must be a JSON object")
        summary = data.get("summary")
        questions = data.get("questions")
        context = data.get("context")
        metadata = data.get("metadata")
        return {
            "summary": _shorten(summary, text)
            if isinstance(summary, str)
            else fallback["summary"],
            "questions": [
                q.strip().rstrip("?") + "?"
                for q in questions
                if isinstance(q, str) and q.strip()
            ][:3]
            if isinstance(questions, list)
            else fallback["questions"],
            "context": context if isinstance(context, str) else fallback["context"],
            "metadata": {
                **(metadata if isinstance(metadata, dict) else fallback["metadata"]),
                "enrichment_status": "api",
            },
        }
    except Exception as error:  # noqa: BLE001 -- recover at external-service boundary
        print(f"Enrichment API failed: {type(error).__name__}: {error}", flush=True)
        fallback["metadata"]["enrichment_status"] = "failed"
        return fallback


def summarize_chunk(text: str) -> str:
    return _enrich_single_call(text, "")["summary"]


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    if n_questions <= 0:
        return []
    return _enrich_single_call(text, "")["questions"][:n_questions]


def contextual_prepend(text: str, document_title: str = "") -> str:
    context = _enrich_single_call(text, document_title)["context"]
    return f"{context}\n\n{text}" if context else text


def extract_metadata(text: str) -> dict:
    return _enrich_single_call(text, "")["metadata"]


def enrich_chunks(
    chunks: list[dict], methods: list[str] | None = None
) -> list[EnrichedChunk]:
    methods = ["combined"] if methods is None else methods
    allowed = {"combined", "summary", "hyqa", "contextual", "metadata"}
    if not set(methods) <= allowed:
        raise ValueError("Unknown enrichment method")
    enriched = []
    for index, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")
        if "combined" in methods:
            data = _enrich_single_call(text, source)
            summary, questions = data["summary"], data["questions"]
            enriched_text = f"{data['context']}\n\n{text}"
            auto_metadata = data["metadata"]
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = (
                contextual_prepend(text, source) if "contextual" in methods else text
            )
            auto_metadata = extract_metadata(text) if "metadata" in methods else {}
        enriched.append(
            EnrichedChunk(
                text,
                enriched_text,
                summary,
                questions,
                {**auto_metadata, **chunk.get("metadata", {})},
                "+".join(methods),
            )
        )
        if (index + 1) % 10 == 0 or index + 1 == len(chunks):
            print(f"Enriched {index + 1}/{len(chunks)} chunks", flush=True)
    return enriched
