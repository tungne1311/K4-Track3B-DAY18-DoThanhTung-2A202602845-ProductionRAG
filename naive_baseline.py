"""Basic paragraph/dense-only baseline, evaluated with the same judge and prompt."""
import json
import sys
import time
from pathlib import Path

from config import LLM_MODEL, NAIVE_COLLECTION
from src.m1_chunking import chunk_basic, load_documents
from src.m2_search import DenseSearch
from src.m4_eval import METRICS, evaluate_ragas, load_test_set, save_report
from src.pipeline import generate_answer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("BASIC RAG BASELINE", flush=True)
    telemetry = {"latency_ms": {}, "query_latency_ms": [], "generation_model": LLM_MODEL}
    start = time.perf_counter()
    docs = load_documents()
    chunks = [{"text": c.text, "metadata": c.metadata}
              for doc in docs for c in chunk_basic(doc["text"], metadata=doc["metadata"])]
    telemetry.update({"documents": len(docs), "chunks": len(chunks)})
    telemetry["latency_ms"]["chunk"] = (time.perf_counter() - start) * 1000
    print(f"{len(chunks)} basic paragraph chunks", flush=True)
    start = time.perf_counter()
    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)
    telemetry["latency_ms"]["index"] = (time.perf_counter() - start) * 1000
    telemetry["qdrant_backend"] = search.backend
    questions, answers, contexts, truths = [], [], [], []
    test_set = load_test_set()
    for index, item in enumerate(test_set):
        start = time.perf_counter()
        found = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        context = [r.text for r in found]
        search_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        answer = generate_answer(item["question"], context, telemetry)
        telemetry["query_latency_ms"].append({
            "question": item["question"], "search": search_ms,
            "LLM": (time.perf_counter() - start) * 1000})
        questions.append(item["question"])
        answers.append(answer)
        contexts.append(context)
        truths.append(item["ground_truth"])
        print(f"[{index + 1}/{len(test_set)}] {item['question']}\n  {answer}", flush=True)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/naive_answers.json").write_text(json.dumps(
        {"questions": questions, "answers": answers, "contexts": contexts,
         "ground_truths": truths, "telemetry": telemetry}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    start = time.perf_counter()
    results = evaluate_ragas(questions, answers, contexts, truths)
    telemetry["latency_ms"]["eval"] = (time.perf_counter() - start) * 1000
    save_report(results, [], "reports/naive_baseline_report.json", extra={"telemetry": telemetry})
    for metric in METRICS:
        print(f"{metric}: {results[metric]:.4f}", flush=True)
    return results


if __name__ == "__main__":
    main()
