import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path regardless of execution entrypoint
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.controller import solve


@dataclass
class TaskEvalResult:
    task_id: str
    difficulty: str
    success: bool
    first_try_pass: bool
    attempts_count: int
    total_tokens: int
    total_time: float
    initial_error: str | None
    final_status: str


@dataclass
class DifficultyStats:
    total: int
    first_try_passes: int
    final_passes: int
    total_attempts: int
    total_tokens: int
    total_time: float


@dataclass
class ErrorTypeStats:
    encountered: int
    fixed: int
    unfixed: int


@dataclass
class EvalSummary:
    total_tasks: int
    first_try_passes: int
    final_passes: int
    total_repaired: int
    repair_opportunities: int
    avg_attempts: float
    avg_tokens: float
    avg_time: float
    by_difficulty: dict[str, DifficultyStats]
    error_stats: dict[str, ErrorTypeStats]


def load_tasks(tasks_path: Path) -> list[dict]:
    """Load benchmark task definitions from JSON file."""
    with open(tasks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cache(cache_path: Path) -> dict[str, TaskEvalResult]:
    """Load existing cached evaluation results from disk."""
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            return {k: TaskEvalResult(**v) for k, v in raw.items()}
    except Exception:
        return {}


def save_cache(cache: dict[str, TaskEvalResult], cache_path: Path) -> None:
    """Save updated evaluation results cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        serialized = {k: asdict(v) for k, v in cache.items()}
        json.dump(serialized, f, indent=2)


def compute_summary(results: list[TaskEvalResult]) -> EvalSummary:
    """Calculate aggregated evaluation metrics, difficulty breakdowns, and repair stats."""
    total = len(results)
    if total == 0:
        return EvalSummary(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, {}, {})

    first_try = sum(1 for r in results if r.first_try_pass)
    final_pass = sum(1 for r in results if r.success)
    repaired = sum(1 for r in results if not r.first_try_pass and r.success)
    opportunities = sum(1 for r in results if not r.first_try_pass)

    diff_map: dict[str, DifficultyStats] = defaultdict(lambda: DifficultyStats(0, 0, 0, 0, 0, 0.0))
    for r in results:
        ds = diff_map[r.difficulty]
        ds.total += 1
        if r.first_try_pass:
            ds.first_try_passes += 1
        if r.success:
            ds.final_passes += 1
        ds.total_attempts += r.attempts_count
        ds.total_tokens += r.total_tokens
        ds.total_time += r.total_time

    error_map: dict[str, ErrorTypeStats] = defaultdict(lambda: ErrorTypeStats(0, 0, 0))
    for r in results:
        if r.initial_error:
            es = error_map[r.initial_error]
            es.encountered += 1
            if r.success:
                es.fixed += 1
            else:
                es.unfixed += 1

    return EvalSummary(
        total_tasks=total,
        first_try_passes=first_try,
        final_passes=final_pass,
        total_repaired=repaired,
        repair_opportunities=opportunities,
        avg_attempts=round(sum(r.attempts_count for r in results) / total, 2),
        avg_tokens=round(sum(r.total_tokens for r in results) / total, 1),
        avg_time=round(sum(r.total_time for r in results) / total, 2),
        by_difficulty=dict(diff_map),
        error_stats=dict(error_map),
    )


def generate_markdown_report(summary: EvalSummary, results: list[TaskEvalResult]) -> str:
    """Generate a clean GitHub-flavored markdown report containing all benchmark metrics."""
    first_try_pct = (summary.first_try_passes / summary.total_tasks * 100) if summary.total_tasks else 0
    final_pct = (summary.final_passes / summary.total_tasks * 100) if summary.total_tasks else 0
    repair_pct = (summary.total_repaired / summary.repair_opportunities * 100) if summary.repair_opportunities else 0

    lines = [
        "# SandboxAgent Benchmark Evaluation Results",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Total Tasks Evaluated** | {summary.total_tasks} |",
        f"| **Pass Rate (0 repairs / 1st try)** | {first_try_pct:.1f}% ({summary.first_try_passes}/{summary.total_tasks}) |",
        f"| **Pass Rate (<= 3 repairs)** | {final_pct:.1f}% ({summary.final_passes}/{summary.total_tasks}) |",
        f"| **Autonomous Repairs Succeeded** | {summary.total_repaired} / {summary.repair_opportunities} ({repair_pct:.1f}%) |",
        f"| **Average Attempts per Task** | {summary.avg_attempts:.2f} |",
        f"| **Average Tokens per Task** | {summary.avg_tokens:,.0f} |",
        f"| **Average Time per Task** | {summary.avg_time:.2f}s |",
        "",
        "## 2. Breakdown by Difficulty",
        "",
        "| Difficulty | Tasks | 0-Repair Pass | Final Pass | Repair Boost | Avg Attempts | Avg Tokens | Avg Time |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for diff in ["easy", "medium", "hard"]:
        if diff in summary.by_difficulty:
            ds = summary.by_difficulty[diff]
            p0 = (ds.first_try_passes / ds.total * 100) if ds.total else 0
            pf = (ds.final_passes / ds.total * 100) if ds.total else 0
            boost = pf - p0
            avg_att = (ds.total_attempts / ds.total) if ds.total else 0
            avg_tok = (ds.total_tokens / ds.total) if ds.total else 0
            avg_t = (ds.total_time / ds.total) if ds.total else 0
            lines.append(
                f"| **{diff.capitalize()}** | {ds.total} | {p0:.1f}% ({ds.first_try_passes}/{ds.total}) | "
                f"{pf:.1f}% ({ds.final_passes}/{ds.total}) | +{boost:.1f}% | {avg_att:.2f} | {avg_tok:,.0f} | {avg_t:.2f}s |"
            )

    lines.extend([
        "",
        "## 3. Breakdown of Initial Errors (Fixed vs. Unfixed)",
        "",
        "| Initial Error Type | Total Encountered | Repaired & Passed | Unresolved | Fix Rate |",
        "|---|---|---|---|---|",
    ])

    if summary.error_stats:
        for err_type, es in sorted(summary.error_stats.items()):
            rate = (es.fixed / es.encountered * 100) if es.encountered else 0
            lines.append(
                f"| `{err_type}` | {es.encountered} | {es.fixed} | {es.unfixed} | {rate:.1f}% |"
            )
    else:
        lines.append("| *None* | 0 | 0 | 0 | N/A |")

    lines.extend([
        "",
        "## 4. Per-Task Detailed Breakdown",
        "",
        "| Task ID | Difficulty | 1st Try | Final | Attempts | Initial Error | Final Status | Tokens | Time |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for r in results:
        p0_icon = "PASS" if r.first_try_pass else "FAIL"
        pf_icon = "PASS" if r.success else "FAIL"
        init_err = f"`{r.initial_error}`" if r.initial_error else "-"
        lines.append(
            f"| `{r.task_id}` | {r.difficulty} | {p0_icon} | {pf_icon} | {r.attempts_count} | "
            f"{init_err} | `{r.final_status}` | {r.total_tokens:,} | {r.total_time:.2f}s |"
        )

    lines.append("")
    return "\n".join(lines)


def print_terminal_summary(summary: EvalSummary) -> None:
    """Print high-level benchmark evaluation summary to terminal."""
    first_try_pct = (summary.first_try_passes / summary.total_tasks * 100) if summary.total_tasks else 0
    final_pct = (summary.final_passes / summary.total_tasks * 100) if summary.total_tasks else 0

    print("\n=======================================================")
    print("           SANDBOXAGENT BENCHMARK RESULTS              ")
    print("=======================================================")
    print(f"Total Tasks:                {summary.total_tasks}")
    print(f"0-Repair Pass Rate (1st):   {first_try_pct:.1f}% ({summary.first_try_passes}/{summary.total_tasks})")
    print(f"<=3 Repairs Pass Rate:      {final_pct:.1f}% ({summary.final_passes}/{summary.total_tasks})")
    print(f"Self-Repaired Tasks:        {summary.total_repaired} / {summary.repair_opportunities}")
    print(f"Average Attempts:           {summary.avg_attempts:.2f}")
    print(f"Average Tokens:             {summary.avg_tokens:,.0f}")
    print(f"Average Time:               {summary.avg_time:.2f}s")
    print("-------------------------------------------------------")
    print("Difficulty Breakdown:")
    for diff in ["easy", "medium", "hard"]:
        if diff in summary.by_difficulty:
            ds = summary.by_difficulty[diff]
            p0 = (ds.first_try_passes / ds.total * 100) if ds.total else 0
            pf = (ds.final_passes / ds.total * 100) if ds.total else 0
            print(f"  [{diff.upper():<6}] 1st Try: {p0:>5.1f}% | Final: {pf:>5.1f}% | Avg Attempts: {ds.total_attempts/ds.total:.2f}")
    print("-------------------------------------------------------")
    print("Error Types Fixed vs. Unfixed:")
    for err_type, es in summary.error_stats.items():
        rate = (es.fixed / es.encountered * 100) if es.encountered else 0
        print(f"  {err_type:<15} Encountered: {es.encountered:<2} | Fixed: {es.fixed:<2} | Fix Rate: {rate:.1f}%")
    print("=======================================================\n")


def run_eval(
    tasks_path: Path,
    cache_path: Path,
    output_path: Path,
    limit: int | None = None,
    force: bool = False,
) -> EvalSummary:
    """Run full evaluation suite across tasks with persistent caching and reporting."""
    tasks = load_tasks(tasks_path)
    if limit:
        tasks = tasks[:limit]

    cache = {} if force else load_cache(cache_path)
    eval_results: list[TaskEvalResult] = []

    print(f"Starting evaluation of {len(tasks)} tasks...")
    if cache:
        print(f"Found {len(cache)} cached results.")

    for i, t in enumerate(tasks, 1):
        task_id = t["id"]
        difficulty = t.get("difficulty", "medium")
        desc = t["description"]
        tests = t["tests"]

        if task_id in cache and not force:
            cached_result = cache[task_id]
            eval_results.append(cached_result)
            status_tag = "PASS" if cached_result.success else "FAIL"
            print(f"[{i}/{len(tasks)}] [CACHED] {task_id} ({difficulty}): {status_tag} in {cached_result.attempts_count} attempt(s)")
            continue

        print(f"[{i}/{len(tasks)}] Evaluating {task_id} ({difficulty})...")
        run_res = solve(task=desc, tests=tests, max_repairs=3)

        first_try_pass = len(run_res.attempts) > 0 and run_res.attempts[0].result.status == "passed"
        initial_error = None if first_try_pass else run_res.attempts[0].result.status
        final_status = run_res.attempts[-1].result.status if run_res.attempts else "unknown"

        record = TaskEvalResult(
            task_id=task_id,
            difficulty=difficulty,
            success=run_res.success,
            first_try_pass=first_try_pass,
            attempts_count=len(run_res.attempts),
            total_tokens=run_res.total_tokens,
            total_time=run_res.total_time,
            initial_error=initial_error,
            final_status=final_status,
        )

        cache[task_id] = record
        save_cache(cache, cache_path)
        eval_results.append(record)

        status_tag = "PASS" if record.success else "FAIL"
        print(f"         Result: {status_tag} | Attempts: {record.attempts_count} | Tokens: {record.total_tokens} | Time: {record.total_time:.2f}s")

    summary = compute_summary(eval_results)
    markdown_content = generate_markdown_report(summary, eval_results)
    output_path.write_text(markdown_content, encoding="utf-8")
    print(f"\nSaved evaluation markdown report to: {output_path.resolve()}")

    print_terminal_summary(summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark evaluation for SandboxAgent")
    parser.add_argument("--tasks", default="bench/tasks.json", help="Path to tasks.json")
    parser.add_argument("--cache", default="bench/eval_cache.json", help="Path to evaluation cache")
    parser.add_argument("--output", default="results.md", help="Path for markdown summary report")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to evaluate")
    parser.add_argument("--force", action="store_true", help="Force re-run and ignore existing cache")
    args = parser.parse_args()

    run_eval(
        tasks_path=Path(args.tasks),
        cache_path=Path(args.cache),
        output_path=Path(args.output),
        limit=args.limit,
        force=args.force,
    )
