"""Sweep TOP_K values to find the optimal setting.

Runs the eval suite once per K value, collects metrics, prints a comparison.
"""
import os
import subprocess
import sys
import json
from pathlib import Path


K_VALUES = [8, 10, 12, 15]
RESULTS_DIR = Path("tests/eval")


def run_one(k: int):
    """Run the eval with a specific K value. Returns the result file path."""
    print(f"\n{'='*60}")
    print(f"Running TOP_K = {k}")
    print(f"{'='*60}")
    
    env = os.environ.copy()
    env["USE_PARENT_CHILD"] = "true"
    env["USE_MULTI_QUERY"] = "false"
    env["TOP_K"] = str(k)
    
    # Snapshot existing result files so we can identify the new one
    before = set(RESULTS_DIR.glob("results_*.json"))
    
    result = subprocess.run(
        [sys.executable, "-m", "tests.eval.run_eval"],
        env=env,
        capture_output=True,
        text=True,
    )
    print(result.stdout[-2000:])  # last 2KB of output
    if result.returncode != 0:
        print(f"FAILED: {result.stderr}", file=sys.stderr)
        return None
    
    after = set(RESULTS_DIR.glob("results_*.json"))
    new_files = after - before
    if not new_files:
        print(f"WARNING: no new result file detected for K={k}", file=sys.stderr)
        return None
    return new_files.pop()


def load_summary(result_path: Path) -> dict:
    with open(result_path) as f:
        data = json.load(f)
    return data["summary"]


def main():
    print(f"Sweeping TOP_K over {K_VALUES}")
    print(f"This will take roughly {len(K_VALUES) * 4} minutes\n")
    
    sweep_results = {}
    for k in K_VALUES:
        result_path = run_one(k)
        if result_path is None:
            print(f"Skipping K={k} due to error")
            continue
        sweep_results[k] = load_summary(result_path)
    
    # Print comparison table
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'K':>4} | {'Correctness':>12} | {'Faithfulness':>12} | {'Recall':>8} | {'Latency':>8}")
    print(f"{'-'*4} | {'-'*12} | {'-'*12} | {'-'*8} | {'-'*8}")
    for k in K_VALUES:
        if k not in sweep_results:
            continue
        s = sweep_results[k]["overall"]
        print(f"{k:>4} | "
              f"{s['correctness']:>12} | "
              f"{s['faithfulness']:>12} | "
              f"{s['recall']:>8} | "
              f"{s['latency_s']:>7}s")
    
    print()
    
    # Save the full sweep result
    sweep_path = RESULTS_DIR / "topk_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2, default=str)
    print(f"Full results saved to {sweep_path}")


if __name__ == "__main__":
    main()