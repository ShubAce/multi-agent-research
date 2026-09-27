"""
eval/ragas_eval.py

Runs RAGAS evaluation against the live agent pipeline (planner → RAG →
synthesiser → fact-check → critic) with web search disabled, so scores reflect
the paper knowledge base and stay reproducible.

Usage:
    poetry run python eval/ragas_eval.py              # full eval
    poetry run python eval/ragas_eval.py --ci         # fail if below thresholds
    make eval

Outputs:
    eval/reports/ragas_results_{timestamp}.csv
    eval/reports/latest.json    ← always overwritten, used by CI
"""

from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add backend root to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.agents.graph import build_graph

setup_logging()
log = get_logger(__name__)

# ── CI thresholds — PR fails if any score drops below these ──────────────────
THRESHOLDS = {
    "faithfulness":      0.70,
    "answer_relevancy":  0.75,
    "context_recall":    0.65,
    "context_precision": 0.65,
}

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_qa_pairs() -> list[dict]:
    qa_path = Path(__file__).parent / "data" / "qa_pairs.json"
    with open(qa_path) as f:
        return json.load(f)


def _initial_state(question: str) -> dict:
    return {
        "query": question, "session_id": "", "use_web_search": False, "messages": [],
        "conversation_history": "", "sub_tasks": [], "routing": {}, "web_results": [],
        "rag_results": [], "source_contexts": [], "web_fallback": False, "final_answer": None,
        "citations": [], "agents_used": [], "confidence_score": None, "fact_check": None,
        "critic_score": None, "critic_feedback": None, "retry_count": 0, "error": None,
        "iteration_count": 0,
    }


def run_rag_pipeline(question: str, graph) -> tuple[str, list[str]]:
    """Run one question through the agent graph → (answer, retrieved passages)."""
    state = graph.invoke(_initial_state(question))
    answer = state.get("final_answer") or ""
    contexts = [p["text"] for r in state.get("rag_results", []) for p in r.get("passages", [])]
    return answer, contexts


def build_ragas_dataset(qa_pairs: list[dict], graph) -> Dataset:
    """Build the HuggingFace Dataset that RAGAS expects."""
    questions, answers, contexts, ground_truths = [], [], [], []

    for pair in qa_pairs:
        log.info("evaluating_question", q=pair["question"][:60])
        answer, ctx = run_rag_pipeline(pair["question"], graph)
        questions.append(pair["question"])
        answers.append(answer)
        contexts.append(ctx if ctx else ["No context retrieved."])
        ground_truths.append(pair["ground_truth"])

    return Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": ground_truths,
    })


def main(ci_mode: bool = False) -> int:
    """
    Run evaluation.
    Returns 0 on success, 1 if CI thresholds are not met.
    """
    log.info("ragas_eval_starting", ci_mode=ci_mode)
    settings = get_settings()

    graph = build_graph()

    # Load eval set
    qa_pairs = load_qa_pairs()
    log.info("eval_set_loaded", count=len(qa_pairs))

    # Build RAGAS dataset
    dataset = build_ragas_dataset(qa_pairs, graph)

    # Run evaluation
    log.info("running_ragas_evaluation")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from app.agents.llm import get_llm

    llm = get_llm()
    evaluator_llm = LangchainLLMWrapper(llm)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    scores = {
        "faithfulness":      float(results["faithfulness"]),
        "answer_relevancy":  float(results["answer_relevancy"]),
        "context_recall":    float(results["context_recall"]),
        "context_precision": float(results["context_precision"]),
    }

    # Save CSV report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORTS_DIR / f"ragas_results_{timestamp}.csv"
    results.to_pandas().to_csv(csv_path, index=False)

    # Save latest.json (overwrite) for CI to read
    latest = {"timestamp": timestamp, "scores": scores, "thresholds": THRESHOLDS}
    with open(REPORTS_DIR / "latest.json", "w") as f:
        json.dump(latest, f, indent=2)

    # Pretty-print results
    print("\n" + "=" * 55)
    print("  RAGAS Evaluation Results")
    print("=" * 55)
    for metric, score in scores.items():
        threshold = THRESHOLDS[metric]
        status = "[PASS]" if score >= threshold else "[FAIL]"
        print(f"  {status:<6}  {metric:<22}  {score:.3f}  (min: {threshold})")
    print("=" * 55)
    print(f"  Report saved: {csv_path.name}")
    print()

    # CI mode: exit 1 if any threshold not met
    if ci_mode:
        failed = [m for m, s in scores.items() if s < THRESHOLDS[m]]
        if failed:
            print(f"[FAIL] CI FAILED — metrics below threshold: {', '.join(failed)}")
            return 1
        print("[PASS] All CI thresholds met.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Exit with code 1 if any RAGAS score is below its threshold.",
    )
    args = parser.parse_args()
    sys.exit(main(ci_mode=args.ci))
