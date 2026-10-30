import argparse
from dataclasses import dataclass
import time
from typing import Callable

from agent.llm import (
    Attempt,
    generate_adversarial_tests,
    generate_code,
    generate_profiling_harness,
    get_token_usage,
    repair_code,
)
from agent.sandbox import ExecutionResult, ProfileResult, run_in_sandbox, run_profiler_in_sandbox


@dataclass
class AttemptDetail:
    code: str
    result: ExecutionResult


@dataclass
class RunResult:
    success: bool
    final_code: str
    attempts: list[AttemptDetail]
    total_tokens: int
    total_time: float
    adversarial_tests: str | None = None
    adversarial_result: ExecutionResult | None = None
    adversarial_passed: bool | None = None
    profile_result: ProfileResult | None = None
    user_hint: str | None = None


def _format_error(result: ExecutionResult) -> str:
    """Format a diagnostic error message from execution stdout and stderr."""
    if result.status == "timeout":
        return f"TimeoutError: Execution timed out after {result.duration:.1f}s."
    parts = []
    if result.stdout.strip():
        parts.append(f"Standard Output:\n{result.stdout.strip()}")
    if result.stderr.strip():
        parts.append(f"Standard Error:\n{result.stderr.strip()}")
    return "\n\n".join(parts) if parts else f"Execution failed with status: {result.status}"


def _base_error(error_text: str) -> str:
    """Extract raw error text without repetition notes."""
    return error_text.split("\n\n[CRITICAL NOTE:")[0].strip()


def _notify(on_event: Callable[[dict], None] | None, event: dict) -> None:
    """Safely deliver a progress event dictionary to the on_event callback."""
    if on_event:
        try:
            on_event(event)
        except Exception:
            pass


def solve(
    task: str,
    tests: str | None = None,
    max_repairs: int = 3,
    on_event: Callable[[dict], None] | None = None,
    language: str = "python",
    user_hint: str | None = None,
    enable_adversarial: bool = False,
    enable_profiler: bool = False,
) -> RunResult:
    """Execute autonomous generate-run-repair loop with optional adversarial testing and profiling."""
    start_time = time.perf_counter()
    tokens_before = get_token_usage().total_tokens

    attempts: list[AttemptDetail] = []
    history: list[Attempt] = []
    current_code = ""
    active_tests = tests
    success = False

    # Step 1: Initial code generation with optional steering
    _notify(on_event, {"step": "generating", "attempt": 1})
    gen_result = generate_code(task, tests=active_tests, language=language, user_hint=user_hint)
    if isinstance(gen_result, tuple):
        current_code, active_tests = gen_result
    else:
        current_code = gen_result

    # Step 2: Sandbox execution and repair loop
    for attempt_idx in range(1 + max_repairs):
        attempt_num = attempt_idx + 1

        if attempt_idx > 0:
            _notify(on_event, {"step": "repairing", "attempt": attempt_num})
            current_code = repair_code(task, history, language=language, user_hint=user_hint)

        _notify(on_event, {"step": "executing", "attempt": attempt_num, "code": current_code})
        exec_result = run_in_sandbox(current_code, active_tests, language=language)
        attempts.append(AttemptDetail(code=current_code, result=exec_result))
        _notify(on_event, {"step": "executed", "attempt": attempt_num, "result": exec_result})

        if exec_result.status == "passed":
            success = True
            break

        # Format failure and detect repeating identical errors
        raw_error = _format_error(exec_result)
        formatted_error = raw_error
        if history and _base_error(history[-1].error) == raw_error.strip():
            formatted_error = (
                f"{raw_error}\n\n"
                "[CRITICAL NOTE: This identical error occurred twice consecutively. "
                "Your previous fix did not resolve the problem. "
                "You MUST switch to an entirely different implementation strategy.]"
            )

        history.append(Attempt(code=current_code, error=formatted_error))

    # Step 3: Adversarial QA Fuzzing (if requested and base solution passed)
    adv_tests: str | None = None
    adv_result: ExecutionResult | None = None
    adv_passed: bool | None = None

    if success and enable_adversarial:
        _notify(on_event, {"step": "adversarial_testing"})
        adv_tests = generate_adversarial_tests(task, current_code, language=language)
        adv_result = run_in_sandbox(current_code, adv_tests, language=language)
        adv_passed = adv_result.status == "passed"
        _notify(on_event, {"step": "adversarial_completed", "passed": adv_passed, "result": adv_result})

        # If adversarial edge cases broke the code, attempt one targeted adversarial repair
        if not adv_passed:
            _notify(on_event, {"step": "repairing", "attempt": len(attempts) + 1, "reason": "adversarial_patch"})
            adv_error = f"[ADVERSARIAL EDGE CASE FAILURE]:\n{_format_error(adv_result)}"
            history.append(Attempt(code=current_code, error=adv_error))
            repaired_code = repair_code(task, history, language=language, user_hint=user_hint)

            recheck_result = run_in_sandbox(repaired_code, adv_tests, language=language)
            attempts.append(AttemptDetail(code=repaired_code, result=recheck_result))
            if recheck_result.status == "passed":
                current_code = repaired_code
                adv_passed = True
                adv_result = recheck_result

    # Step 4: Empirical Docker Performance Profiler (if requested)
    profile_res: ProfileResult | None = None
    if success and enable_profiler:
        _notify(on_event, {"step": "profiling"})
        harness = generate_profiling_harness(task, current_code, language=language)
        profile_res = run_profiler_in_sandbox(current_code, harness, language=language)
        _notify(on_event, {"step": "profiled", "result": profile_res})

    total_time = round(time.perf_counter() - start_time, 2)
    total_tokens = get_token_usage().total_tokens - tokens_before

    _notify(
        on_event,
        {
            "step": "completed",
            "success": success,
            "total_attempts": len(attempts),
            "total_time": total_time,
            "total_tokens": total_tokens,
            "adversarial_passed": adv_passed,
        },
    )

    return RunResult(
        success=success,
        final_code=current_code,
        attempts=attempts,
        total_tokens=total_tokens,
        total_time=total_time,
        adversarial_tests=adv_tests,
        adversarial_result=adv_result,
        adversarial_passed=adv_passed,
        profile_result=profile_res,
        user_hint=user_hint,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous coding sandbox agent")
    parser.add_argument("task", help="Natural language description of programming task")
    parser.add_argument("--tests", default=None, help="Optional tests to validate against")
    parser.add_argument("--language", default="python", choices=["python", "cpp"], help="Programming language (python or cpp)")
    parser.add_argument("--hint", default=None, help="Human steering hint/directive")
    parser.add_argument("--adversarial", action="store_true", help="Run adversarial QA fuzzer")
    parser.add_argument("--profile", action="store_true", help="Run Docker performance profiler")
    parser.add_argument("--max-repairs", type=int, default=3, help="Maximum repair attempts (default: 3)")
    args = parser.parse_args()

    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{BOLD}{CYAN}=== SandboxAgent: Starting Run [{args.language.upper()}] ==={RESET}")
    print(f"{BOLD}Task:{RESET} {args.task}")
    if args.hint:
        print(f"{BOLD}Steering Hint:{RESET} {args.hint}")
    print()

    def cli_progress(event: dict) -> None:
        """Display live step progress and outcome in the terminal."""
        step = event.get("step")
        num = event.get("attempt")
        if step == "generating":
            print(f"[{num}] Generating initial solution with LLM...")
        elif step == "repairing":
            print(f"[{num}] Repairing solution with LLM...")
        elif step == "executing":
            print(f"[{num}] Running tests in isolated Docker container...")
        elif step == "executed":
            res = event.get("result")
            if res.status == "passed":
                print(f"{GREEN}[PASS]{RESET} Attempt {num} succeeded ({res.duration:.2f}s)")
            else:
                print(f"{RED}[FAIL]{RESET} Attempt {num} status: {res.status} ({res.duration:.2f}s)")
        elif step == "adversarial_testing":
            print("[⚔️ QA] Running adversarial attacker fuzzer tests...")
        elif step == "profiling":
            print("[⚡ Profiler] Profiling Big-O complexity and memory inside Docker...")

    run_summary = solve(
        task=args.task,
        tests=args.tests,
        max_repairs=args.max_repairs,
        on_event=cli_progress,
        language=args.language,
        user_hint=args.hint,
        enable_adversarial=args.adversarial,
        enable_profiler=args.profile,
    )

    print(f"\n{BOLD}{CYAN}=== Run Summary ==={RESET}")
    status_label = f"{GREEN}SUCCESS{RESET}" if run_summary.success else f"{RED}FAILED{RESET}"
    print(f"Outcome:      {status_label}")
    print(f"Attempts:     {len(run_summary.attempts)}")
    print(f"Total Time:   {run_summary.total_time:.2f}s")
    print(f"Total Tokens: {run_summary.total_tokens}")

    if run_summary.adversarial_passed is not None:
        adv_label = f"{GREEN}PASSED{RESET}" if run_summary.adversarial_passed else f"{RED}FAILED{RESET}"
        print(f"Adversarial QA: {adv_label}")

    if run_summary.profile_result and run_summary.profile_result.status == "passed":
        print(f"Big-O Est.:   {run_summary.profile_result.estimated_complexity}")
        print(f"Peak Memory:  {run_summary.profile_result.peak_memory_kb:.1f} KB")

    print(f"\n{BOLD}Final Code:{RESET}\n{run_summary.final_code}\n")
