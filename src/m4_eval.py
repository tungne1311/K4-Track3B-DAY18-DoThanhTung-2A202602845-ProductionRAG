"""M4: genuine RAGAS scores, explicit evaluation status and Diagnostic Tree."""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass

from config import TEST_SET_PATH

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    if len({len(questions), len(answers), len(contexts), len(ground_truths)}) != 1:
        raise ValueError("Evaluation inputs must have equal lengths")
    result = {name: 0.0 for name in METRICS}
    result["per_question"] = []
    try:
        from config import (
            EVAL_MODEL,
            EVAL_RPM,
            EVAL_SUPPORTS_N,
            LLM_MODEL,
            OPENAI_API_KEY,
            OPENAI_BASE_URL,
        )

        if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-...":
            raise RuntimeError("OPENAI_API_KEY is missing; RAGAS was not measured")
        if not questions:
            result["evaluation"] = {"status": "empty", "engine": "ragas"}
            return result
        from datasets import Dataset
        from langchain_core.embeddings import Embeddings
        from langchain_core.rate_limiters import InMemoryRateLimiter
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig

        from src.m2_search import get_dense_encoder

        class LocalEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return (
                    get_dense_encoder()
                    .encode(texts, batch_size=8, normalize_embeddings=True)
                    .tolist()
                )

            def embed_query(self, text):
                return self.embed_documents([text])[0]

        class SingleCompletionLLM(LangchainLLMWrapper):
            """Send n separate requests instead of one request with n>1."""

            def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
                results = [
                    super(SingleCompletionLLM, self).generate_text(
                        prompt, 1, temperature or self.get_temperature(n), stop, callbacks
                    )
                    for _ in range(n)
                ]
                results[0].generations = [[r.generations[0][0] for r in results]]
                return results[0]

            async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
                results = [
                    await super(SingleCompletionLLM, self).agenerate_text(
                        prompt, 1, temperature or self.get_temperature(n), stop, callbacks
                    )
                    for _ in range(n)
                ]
                results[0].generations = [[r.generations[0][0] for r in results]]
                return results[0]

        def make_evaluator():
            # Each evaluate() creates a new event loop; give retries a fresh async client.
            llm = ChatOpenAI(
                model=EVAL_MODEL or LLM_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL or None,
                temperature=0,
                max_tokens=1024,
                timeout=180,
                max_retries=1,
                rate_limiter=InMemoryRateLimiter(
                    requests_per_second=EVAL_RPM / 60, max_bucket_size=1
                )
                if EVAL_RPM
                else None,
            )
            return llm if EVAL_SUPPORTS_N else SingleCompletionLLM(llm)

        evaluator = make_evaluator()
        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        scores = evaluate(
            dataset,
            metrics=copy.deepcopy(
                [faithfulness, answer_relevancy, context_precision, context_recall]
            ),
            llm=evaluator,
            embeddings=LocalEmbeddings(),
            run_config=RunConfig(timeout=600, max_retries=5, max_wait=90, max_workers=1),
            raise_exceptions=False,
        )
        frame = scores.to_pandas()
        retried = []
        metric_objects = dict(
            zip(
                METRICS,
                [faithfulness, answer_relevancy, context_precision, context_recall],
            )
        )
        # Retry missing scores once; never replace a successfully measured score.
        for name in METRICS:
            missing = [
                int(i)
                for i in frame.index
                if not math.isfinite(float(frame.at[i, name]))
            ]
            if not missing:
                continue
            print(f"Retrying {len(missing)} missing {name} scores", flush=True)
            subset = Dataset.from_dict(
                {
                    "question": [questions[i] for i in missing],
                    "answer": [answers[i] for i in missing],
                    "contexts": [contexts[i] for i in missing],
                    "ground_truth": [ground_truths[i] for i in missing],
                }
            )
            retry = evaluate(
                subset,
                metrics=[copy.deepcopy(metric_objects[name])],
                llm=make_evaluator(),
                embeddings=LocalEmbeddings(),
                run_config=RunConfig(timeout=600, max_retries=5, max_wait=90, max_workers=1),
                raise_exceptions=False,
            ).to_pandas()
            for offset, index in enumerate(missing):
                retried.append({"question_index": index, "metric": name})
                value = float(retry.iloc[offset][name])
                if math.isfinite(value):
                    frame.at[index, name] = value
        invalid = []
        for index, row in frame.iterrows():
            values = {}
            for name in METRICS:
                value = float(row[name])
                if not math.isfinite(value):
                    invalid.append({"question_index": int(index), "metric": name})
                    value = 0.0
                values[name] = value
            result["per_question"].append(
                EvalResult(
                    questions[index],
                    answers[index],
                    contexts[index],
                    ground_truths[index],
                    **values,
                )
            )
        for name in METRICS:
            values = [getattr(item, name) for item in result["per_question"]]
            result[name] = sum(values) / len(values) if values else 0.0
        result["evaluation"] = {
            "status": "partial" if invalid else "ok",
            "engine": "ragas",
            "judge_model": EVAL_MODEL or LLM_MODEL,
            "embedding_model": "BAAI/bge-m3",
            "invalid_scores": invalid,
            "retried_cells": retried,
        }
    except Exception as error:  # noqa: BLE001 -- recover at external-service boundary
        print(f"RAGAS evaluation failed: {type(error).__name__}: {error}", flush=True)
        result["per_question"] = [
            EvalResult(q, a, list(c), gt, 0.0, 0.0, 0.0, 0.0)
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
        result["evaluation"] = {
            "status": "failed",
            "engine": "ragas",
            "error": f"{type(error).__name__}: {error}",
            "note": "Zero values are failure sentinels, not measured scores.",
        }
    return result


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    tree = {
        "faithfulness": (
            "Unsupported claims in answer",
            "Require source evidence for every claim; reduce temperature.",
        ),
        "answer_relevancy": (
            "Answer does not directly address the question",
            "Answer all subquestions concisely; clarify ambiguous intent.",
        ),
        "context_precision": (
            "Irrelevant or redundant retrieved context",
            "Rerank more candidates; deduplicate parents and filter obsolete versions.",
        ),
        "context_recall": (
            "Relevant evidence missing from retrieved context",
            "Return parent text; increase candidate coverage; retrieve multiple documents.",
        ),
    }
    ordered = sorted(
        eval_results, key=lambda item: sum(getattr(item, m) for m in METRICS) / 4
    )
    failures = []
    for item in ordered[: max(0, bottom_n)]:
        worst = min(METRICS, key=lambda name: getattr(item, name))
        diagnosis, fix = tree[worst]
        failures.append(
            {
                "question": item.question,
                "answer": item.answer,
                "ground_truth": item.ground_truth,
                "contexts": item.contexts,
                "worst_metric": worst,
                "score": getattr(item, worst),
                "average_score": sum(getattr(item, name) for name in METRICS) / 4,
                "metrics": {name: getattr(item, name) for name in METRICS},
                "diagnosis": diagnosis,
                "suggested_fix": fix,
                "error_tree": (
                    "Output incorrect -> context sufficient -> prompt/generation"
                    if worst in ("faithfulness", "answer_relevancy")
                    else "Output incomplete -> context insufficient -> retrieval/chunking"
                ),
            }
        )
    return failures


def save_report(
    results: dict,
    failures: list[dict],
    path: str = "reports/ragas_report.json",
    extra: dict | None = None,
):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    report = {
        "aggregate": {name: results.get(name, 0.0) for name in METRICS},
        "num_questions": len(results.get("per_question", [])),
        "per_question": [asdict(item) for item in results.get("per_question", [])],
        "evaluation": results.get("evaluation", {}),
        "failures": failures,
        **(extra or {}),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report saved to {path}", flush=True)
