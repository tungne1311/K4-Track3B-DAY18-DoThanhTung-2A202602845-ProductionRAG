"""M2: Vietnamese BM25, bge-m3/Qdrant and reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def segment_vietnamese(text: str) -> str:
    from underthesea import word_tokenize

    return word_tokenize(text, format="text").replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks)
        self.corpus_tokens = [
            segment_vietnamese(c["text"].lower()).split() for c in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens) if any(self.corpus_tokens) else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        if self.bm25 is None or top_k <= 0 or not query.strip():
            return []
        scores = self.bm25.get_scores(segment_vietnamese(query.lower()).split())
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            SearchResult(
                self.documents[i]["text"],
                float(scores[i]),
                dict(self.documents[i].get("metadata", {})),
                "bm25",
            )
            for i in indices
            if scores[i] > 0
        ][:top_k]


@lru_cache(maxsize=1)
def get_dense_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient

        self.backend = "server"
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
            self.client.get_collections()
        except Exception as error:  # noqa: BLE001 -- recover at external-service boundary
            print(
                f"Qdrant server unavailable; using in-memory backend: {error}",
                flush=True,
            )
            self.client = QdrantClient(":memory:")
            self.backend = "memory"
        self._encoder = None
        self._collections = set()

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = get_dense_encoder()
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        # Replace only this lab collection, leaving all other collections untouched.
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        self._collections.add(collection)
        if not chunks:
            return
        vectors = self._get_encoder().encode(
            [c["text"] for c in chunks],
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        for start in range(0, len(chunks), 64):
            points = [
                PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    payload={
                        "text": chunks[i]["text"],
                        "metadata": chunks[i].get("metadata", {}),
                    },
                )
                for i in range(start, min(start + 64, len(chunks)))
            ]
            self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(
        self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME
    ) -> list[SearchResult]:
        if top_k <= 0 or not query.strip():
            return []
        if collection not in self._collections and not self.client.collection_exists(
            collection
        ):
            return []
        vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
        response = self.client.query_points(
            collection_name=collection, query=vector, limit=top_k, with_payload=True
        )
        return [
            SearchResult(
                point.payload["text"],
                float(point.score),
                dict(point.payload.get("metadata", {})),
                "dense",
            )
            for point in response.points
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]], k: int = 60, top_k: int = HYBRID_TOP_K
) -> list[SearchResult]:
    if k < 0:
        raise ValueError("k must be non-negative")
    merged = {}
    for results in results_list:
        seen = set()
        for rank, result in enumerate(results):
            identity = result.metadata.get("chunk_id") or result.text
            if identity in seen:
                continue
            seen.add(identity)
            entry = merged.setdefault(identity, {"result": result, "score": 0.0})
            entry["score"] += 1.0 / (k + rank + 1)
    ranked = sorted(merged.values(), key=lambda entry: entry["score"], reverse=True)
    return [
        SearchResult(
            entry["result"].text,
            entry["score"],
            dict(entry["result"].metadata),
            "hybrid",
        )
        for entry in ranked[: max(0, top_k)]
    ]


class HybridSearch:
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        return reciprocal_rank_fusion(
            [
                self.bm25.search(query, BM25_TOP_K),
                self.dense.search(query, DENSE_TOP_K),
            ],
            top_k=top_k,
        )
