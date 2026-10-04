"""M3: CrossEncoder reranking and measured latency."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


@lru_cache(maxsize=1)
def _cached_model(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=1024)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = _cached_model(self.model_name)
        return self._model

    def rerank(
        self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[RerankResult]:
        import numpy as np

        if not documents or top_k <= 0:
            return []
        scores = np.asarray(
            self._load_model().predict(
                [(query, doc["text"]) for doc in documents], batch_size=4
            )
        ).reshape(-1)
        ordered = sorted(
            zip(scores, documents), key=lambda pair: float(pair[0]), reverse=True
        )
        return [
            RerankResult(
                doc["text"],
                float(doc.get("score", 0.0)),
                float(score),
                dict(doc.get("metadata", {})),
                rank,
            )
            for rank, (score, doc) in enumerate(ordered[:top_k])
        ]


def benchmark_reranker(
    reranker, query: str, documents: list[dict], n_runs: int = 5
) -> dict:
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }
