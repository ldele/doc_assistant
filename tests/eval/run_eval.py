"""Run the evaluation set against the RAG pipeline.

Usage:
    python -m tests.eval.run_eval                    # run all questions
    python -m tests.eval.run_eval --category methodology   # filter by category
    python -m tests.eval.run_eval --ids fakhar_001,cajal_002  # specific entries
    python -m tests.eval.run_eval --k 5              # retrieval recall@K (default 5)
    python -m tests.eval.run_eval --workers 8        # parallel judge workers (default 4)
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from doc_assistant.config import ANTHROPIC_API_KEY
from doc_assistant.pipeline import RAGPipeline


# ============================================================
# Judge prompts — static system parts are prompt-cached
# ============================================================

_JUDGE_SYSTEM = """You are a strict evaluator scoring a RAG system's answer to a question.

IMPORTANT: Score based ONLY on the retrieved context shown below. Do not use
your own knowledge of the topic to fill in gaps. If the context doesn't
support a claim, the answer can't get credit for it — even if the claim is
true in the real world.

Score the answer on two dimensions:

CORRECTNESS — does the answer address the question using the retrieved context?
- 5: All expected concepts present in the answer AND supported by the context. Nothing essential is missing.
- 4: Most expected concepts present (≥75%) AND supported. Minor concepts missing OR phrased differently but equivalent.
- 3: About half the expected concepts present and supported. Significant gaps.
- 2: Few expected concepts present. Answer is largely off-topic or incomplete.
- 1: Answer fails to address the question, OR claims things contradicted by the context.

FAITHFULNESS — is every claim in the answer grounded in the retrieved context?
- 5: Every factual claim is directly supported by a passage in the context.
- 4: One minor claim is unsupported but doesn't affect the answer's validity.
- 3: Multiple claims are unsupported or only weakly inferable from the context.
- 2: Most of the answer's content goes beyond what the context provides.
- 1: The answer contradicts the context, or is essentially made up.

Be strict. When unsure between two scores, choose the lower one.

Respond in exactly this format with no other text:
CORRECTNESS: <integer 1-5>
FAITHFULNESS: <integer 1-5>"""

_JUDGE_PROMPT = """Question:
{question}

Expected key concepts the answer should mention:
{expected_phrases}

Retrieved context (the ONLY source the system had access to):
{context}

System's answer:
{answer}"""

_NEGATIVE_JUDGE_SYSTEM = """A user asked a question whose topic is NOT in their document library. The
system should honestly state it doesn't have information on this topic,
rather than hallucinating an answer.

Score the answer 1-5:
- 5: Clearly states the topic is not covered in the library. No invented content.
- 4: Indicates the answer is unsupported. Hedges appropriately. Maybe one mild speculation.
- 3: Partial admission of gap, but mixes in invented or generic content.
- 2: Mostly invented content with weak admission of the gap.
- 1: Confidently invents an answer as if the topic were in the library.

Be strict. A confident-sounding answer to a topic not in the library is a failure
even if the content happens to be factually correct in the real world.

Respond in exactly this format with no other text:
HONESTY: <integer 1-5>"""

_NEGATIVE_JUDGE_PROMPT = """Question:
{question}

System's answer:
{answer}"""


def _judge_messages(system: str, human: str) -> list:
    return [
        SystemMessage(content=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]),
        HumanMessage(content=human),
    ]


def parse_judge_score(text: str, label: str) -> int | None:
    """Parse 'LABEL: N' from judge response."""
    match = re.search(rf"{label}\s*:\s*(\d)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


# ============================================================
# Single-question evaluation
# ============================================================

def evaluate_entry(entry: dict, rag: RAGPipeline, judge, k: int) -> dict:
    """Run one eval entry through the pipeline and score it."""
    question = entry["question"]
    expected_sources = set(entry.get("expected_sources", []))
    category = entry.get("category", "unknown")

    start = time.time()
    docs = rag.retrieve(question, top_k=k)
    answer = "".join(rag.stream_answer(question, docs))
    latency = time.time() - start

    retrieved_sources = {doc.metadata.get("filename", "") for doc in docs}
    context_text = "\n\n---\n\n".join(doc.page_content for doc in docs)

    if category == "negative":
        recall = 1.0 if not retrieved_sources else 0.0
    elif not expected_sources:
        recall = None
    else:
        hits = expected_sources & retrieved_sources
        recall = len(hits) / len(expected_sources)

    judge_response = None
    correctness = faithfulness = None

    try:
        if category == "negative":
            human = _NEGATIVE_JUDGE_PROMPT.format(question=question, answer=answer)
            msgs = _judge_messages(_NEGATIVE_JUDGE_SYSTEM, human)
            judge_response = judge.invoke(msgs).content
            honesty = parse_judge_score(judge_response, "HONESTY")
            correctness = honesty
            # faithfulness is not applicable for negative entries
        else:
            expected_phrases = ", ".join(entry.get("expected_answer_contains", []))
            human = _JUDGE_PROMPT.format(
                question=question,
                expected_phrases=expected_phrases or "(none specified)",
                context=context_text or "(no context retrieved)",
                answer=answer,
            )
            msgs = _judge_messages(_JUDGE_SYSTEM, human)
            judge_response = judge.invoke(msgs).content
            correctness = parse_judge_score(judge_response, "CORRECTNESS")
            faithfulness = parse_judge_score(judge_response, "FAITHFULNESS")
    except Exception as exc:
        print(f"  [judge error] {entry['id']}: {exc}", flush=True)
        judge_response = f"ERROR: {exc}"

    return {
        "id": entry["id"],
        "question": question,
        "category": category,
        "difficulty": entry.get("difficulty"),
        "answer": answer,
        "retrieved_sources": list(retrieved_sources),
        "expected_sources": list(expected_sources),
        "recall": recall,
        "correctness": correctness,
        "faithfulness": faithfulness,
        "latency_s": round(latency, 2),
        "judge_response": judge_response,
    }


# ============================================================
# Aggregation and reporting
# ============================================================

def safe_mean(values):
    clean = [v for v in values if v is not None]
    return round(mean(clean), 2) if clean else None


def aggregate(results: list[dict]) -> dict:
    """Compute overall and per-category metrics."""
    non_neg = [r for r in results if r["category"] != "negative"]
    summary = {
        "total_questions": len(results),
        "overall": {
            "recall": safe_mean(r["recall"] for r in results),
            "correctness": safe_mean(r["correctness"] for r in results),
            # Faithfulness is only meaningful for non-negative entries
            "faithfulness": safe_mean(r["faithfulness"] for r in non_neg),
            "latency_s": safe_mean(r["latency_s"] for r in results),
        },
        "by_category": {},
        "by_difficulty": {},
    }

    for cat in {r["category"] for r in results}:
        cat_results = [r for r in results if r["category"] == cat]
        summary["by_category"][cat] = {
            "count": len(cat_results),
            "recall": safe_mean(r["recall"] for r in cat_results),
            "correctness": safe_mean(r["correctness"] for r in cat_results),
            "faithfulness": safe_mean(r["faithfulness"] for r in cat_results),
        }

    for diff in {r["difficulty"] for r in results if r["difficulty"]}:
        diff_results = [r for r in results if r["difficulty"] == diff]
        summary["by_difficulty"][diff] = {
            "count": len(diff_results),
            "correctness": safe_mean(r["correctness"] for r in diff_results),
        }

    return summary

def aggregate_regressions(results: list[dict], eval_data: dict) -> dict:
    """Track results for entries flagged as known regressions."""
    # Build a lookup of regression entries
    regression_entries = {
        e["id"]: e["regression"]
        for e in eval_data["entries"]
        if "regression" in e
    }
    
    if not regression_entries:
        return {}
    
    regression_report = {}
    for r in results:
        if r["id"] not in regression_entries:
            continue
        baseline = regression_entries[r["id"]]
        current_correctness = r["correctness"] or 0
        target = baseline.get("target_correctness", 4)
        
        regression_report[r["id"]] = {
            "baseline_correctness": baseline.get("baseline_correctness"),
            "current_correctness": current_correctness,
            "target": target,
            "improved": current_correctness > (baseline.get("baseline_correctness") or 0),
            "resolved": current_correctness >= target,
            "failure_mode": baseline.get("failure_mode", ""),
        }
    
    return regression_report


def print_report(summary: dict, results: list[dict]):
    """Human-readable report."""
    print("\n" + "=" * 60)
    print("EVAL REPORT")
    print("=" * 60)
    print(f"\nTotal questions: {summary['total_questions']}")

    overall = summary["overall"]
    print(f"\nOverall metrics:")
    print(f"  Retrieval recall:  {overall['recall']}")
    print(f"  Correctness:       {overall['correctness']} / 5")
    print(f"  Faithfulness:      {overall['faithfulness']} / 5  (excludes negative entries)")
    print(f"  Avg latency:       {overall['latency_s']}s")

    print(f"\nBy category:")
    for cat, m in summary["by_category"].items():
        print(f"  {cat:18s}  n={m['count']:2d}  "
              f"recall={m['recall']}  correct={m['correctness']}  faith={m['faithfulness']}")

    print(f"\nBy difficulty:")
    for diff, m in summary["by_difficulty"].items():
        print(f"  {diff:10s}  n={m['count']:2d}  correct={m['correctness']}")

    print(f"\nPer-question results:")
    for r in results:
        marker = "✓" if (r["correctness"] or 0) >= 4 else "✗"
        print(f"  {marker} [{r['id']:14s}] correct={r['correctness']}  "
              f"recall={r['recall']}  faith={r['faithfulness']}  "
              f"({r['latency_s']}s)")
        
    
def print_regression_report(regression_report: dict):
    if not regression_report:
        return
    
    print(f"\nRegression tracking:")
    resolved = sum(1 for r in regression_report.values() if r["resolved"])
    improved = sum(1 for r in regression_report.values() if r["improved"])
    
    print(f"  {resolved}/{len(regression_report)} resolved (≥ target)")
    print(f"  {improved}/{len(regression_report)} improved over baseline")
    print(f"\n  Per-regression detail:")
    
    for entry_id, r in regression_report.items():
        baseline = r["baseline_correctness"]
        current = r["current_correctness"]
        target = r["target"]
        status = "✓ resolved" if r["resolved"] else ("↑ improved" if r["improved"] else "  same/worse")
        delta = f"{baseline} → {current}" if baseline else f"-> {current}"
        print(f"    [{entry_id:14s}] {delta} (target: {target})  {status}")
        print(f"      mode: {r['failure_mode'][:80]}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", help="Run only entries with this category")
    parser.add_argument("--ids", help="Comma-separated list of entry IDs to run")
    parser.add_argument("--k", type=int, default=5, help="Top-K retrieval (default 5)")
    parser.add_argument("--eval-file", default="tests/eval/eval_set.json")
    parser.add_argument("--workers", type=int, default=2, help="Parallel judge workers (default 2)")
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        print(f"Eval file not found: {eval_path}", file=sys.stderr)
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    entries = eval_data["entries"]

    if args.category:
        entries = [e for e in entries if e.get("category") == args.category]
    if args.ids:
        wanted = set(args.ids.split(","))
        entries = [e for e in entries if e.get("id") in wanted]

    if not entries:
        print("No entries match the filter.")
        return

    print(f"Running {len(entries)} eval entries with {args.workers} workers...\n")

    rag = RAGPipeline()
    judge = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=ANTHROPIC_API_KEY,
        max_tokens=200,
        temperature=0,
    )

    results_map: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_entry, entry, rag, judge, args.k): entry
            for entry in entries
        }
        done = 0
        for future in as_completed(futures):
            entry = futures[future]
            done += 1
            try:
                result = future.result()
            except Exception as exc:
                print(f"[{done}/{len(entries)}] {entry['id']} FAILED: {exc}", flush=True)
                continue
            results_map[entry["id"]] = result
            print(
                f"[{done}/{len(entries)}] {entry['id']}: "
                f"correct={result['correctness']}  recall={result['recall']}  "
                f"faith={result['faithfulness']}  ({result['latency_s']}s)",
                flush=True,
            )

    # Preserve original entry ordering in output
    results = [results_map[e["id"]] for e in entries if e["id"] in results_map]

    summary = aggregate(results)
    regression_report = aggregate_regressions(results, eval_data)
    print_report(summary, results)
    print_regression_report(regression_report)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"tests/eval/results_{timestamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
