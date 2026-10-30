from dataclasses import dataclass
import os
import re
import time
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, InternalServerError, OpenAI, RateLimitError

load_dotenv()

# Centralized model configuration - swap provider / model name here
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4o-mini")


@dataclass
class Attempt:
    code: str
    error: str


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


_token_tracker = TokenUsage()


def get_token_usage() -> TokenUsage:
    """Return accumulated token usage across LLM calls."""
    return _token_tracker


def reset_token_usage() -> None:
    """Reset the accumulated token usage counters."""
    global _token_tracker
    _token_tracker = TokenUsage()


def _get_system_prompt(language: str = "python", role: str = "developer") -> str:
    """Return system prompt configured for Python or C++ and agent role."""
    is_cpp = language.lower() in ("cpp", "c++")
    lang_name = "C++17" if is_cpp else "Python 3.11"
    tag = "cpp" if is_cpp else "python"

    if role == "attacker":
        return (
            f"You are a ruthless Adversarial QA Engineer and Fuzzing Specialist.\n"
            f"Your mission is to write vicious edge-case unit tests in {lang_name} to break the developer's solution.\n"
            f"Focus on subtle boundary conditions, extreme inputs, empty inputs, duplicates, and off-by-one errors.\n"
            f"Return ONLY the test code inside a single ```{tag} ``` block without explanation."
        )

    if role == "profiler":
        return (
            f"You are a Performance Engineer specialized in algorithmic complexity and microbenchmarking.\n"
            f"Write a standalone benchmark script in {lang_name} that measures runtime and memory across scaling input sizes.\n"
            f"Return ONLY valid benchmark code inside a single ```{tag} ``` block."
        )

    return (
        f"You are an expert programmer.\n"
        f"Write clean, correct, and robust {lang_name} code.\n"
        f"Return ONLY valid code inside a single ```{tag} ``` block.\n"
        f"Do not include any conversational explanation or text outside the code block."
    )


def _get_client() -> OpenAI:
    """Initialize and return the OpenAI client using environment variables."""
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in .env.")
    base_url = os.getenv("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


def _call_llm(messages: list[dict[str, str]], max_retries: int = 3, initial_delay: float = 1.0) -> str:
    """Call the LLM with retry and exponential backoff, updating token usage."""
    client = _get_client()
    delay = initial_delay
    model = os.getenv("LLM_MODEL", MODEL_NAME)
    max_tokens_env = os.getenv("MAX_TOKENS")
    max_tokens = int(max_tokens_env) if max_tokens_env else (950 if "qwen" in model.lower() else None)

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.2,
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            response = client.chat.completions.create(**kwargs)
            if response.usage:
                _token_tracker.prompt_tokens += response.usage.prompt_tokens
                _token_tracker.completion_tokens += response.usage.completion_tokens
                _token_tracker.total_tokens += response.usage.total_tokens

            return response.choices[0].message.content or ""
        except RateLimitError as exc:
            err_msg = str(exc)
            if "reduce max_tokens" in err_msg.lower() or "otpm" in err_msg.lower():
                limit_match = re.search(r"Limit\s+(\d+)", err_msg, re.IGNORECASE)
                max_tokens = int(limit_match.group(1)) - 50 if limit_match else 800
            if attempt == max_retries - 1:
                raise exc
            time.sleep(delay)
            delay *= 2
        except (APIConnectionError, InternalServerError, APIError) as exc:
            if attempt == max_retries - 1:
                raise exc
            time.sleep(delay)
            delay *= 2

    return ""


def extract_code(text: str) -> str:
    """Extract code from markdown code fences or raw text robustly."""
    match = re.search(r"```(?:python|cpp|c\+\+|c)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            candidate = parts[1].strip()
            first_line = candidate.split("\n", 1)[0].lower()
            if first_line in ("python", "cpp", "c++", "c"):
                candidate = candidate[len(first_line):].lstrip()
            return candidate

    return text.strip()


def _extract_code_and_tests(text: str, language: str = "python") -> tuple[str, str]:
    """Extract both solution code and test code from LLM output."""
    blocks = re.findall(r"```(?:python|cpp|c\+\+|c)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if len(blocks) >= 2:
        return blocks[0].strip(), blocks[1].strip()

    code_block = extract_code(text)
    separators = [
        "// --- TESTS ---",
        "# --- TESTS ---",
        "// TESTS",
        "# TESTS",
        "### TESTS",
    ]
    for sep in separators:
        if sep in code_block:
            parts = code_block.split(sep, 1)
            return parts[0].strip(), parts[1].strip()

    is_cpp = language.lower() in ("cpp", "c++")
    if is_cpp:
        main_match = re.search(r"\n(?=int\s+main\s*\()", code_block)
        if main_match:
            idx = main_match.start()
            return code_block[:idx].strip(), code_block[idx:].strip()
        return code_block, '#include <cassert>\n#include "solution.hpp"\nint main() { return 0; }\n'
    else:
        test_match = re.search(r"\n(?=def test_|class Test)", code_block)
        if test_match:
            idx = test_match.start()
            return code_block[:idx].strip(), code_block[idx:].strip()
        return code_block, "def test_default():\n    pass\n"


def generate_code(
    task: str,
    tests: str | None = None,
    language: str = "python",
    user_hint: str | None = None,
) -> str | tuple[str, str]:
    """Generate solution code (and tests if none provided) for Python or C++."""
    is_cpp = language.lower() in ("cpp", "c++")
    lang_name = "C++17" if is_cpp else "Python 3.11"
    sol_file = "solution.hpp" if is_cpp else "solution.py"

    hint_clause = f"\n\n[HUMAN ARCHITECTURAL DIRECTIVE / HINT]:\n{user_hint}\nYou MUST follow this instruction." if user_hint else ""

    if tests:
        user_prompt = (
            f"Task ({lang_name}):\n{task}{hint_clause}\n\n"
            f"The solution will be validated against the following tests:\n"
            f"```{language}\n{tests}\n```\n\n"
            f"Instructions:\n"
            f"1. Write the implementation satisfying the task and passing the tests.\n"
            f"2. The code will be saved as `{sol_file}`. Ensure all function/class signatures match the tests.\n"
            f"3. Return ONLY a single ```{language} ``` code block containing the solution."
        )
    else:
        sep_comment = "// --- TESTS ---" if is_cpp else "# --- TESTS ---"
        test_template = (
            '#include <cassert>\n#include "solution.hpp"\nint main() {\n    // assertions\n    return 0;\n}'
            if is_cpp
            else "from solution import ...\ndef test_...():\n    assert ..."
        )
        user_prompt = (
            f"Task ({lang_name}):\n{task}{hint_clause}\n\n"
            f"Instructions:\n"
            f"1. Write the implementation to solve the task. Saved as `{sol_file}`.\n"
            f"2. Write comprehensive unit tests for the solution.\n"
            f"3. Return a single ```{language} ``` code block containing both the solution and tests, "
            f"separated by the exact line `{sep_comment}`:\n\n"
            f"```{language}\n// Implementation\n...\n\n{sep_comment}\n{test_template}\n```\n"
            f"Return ONLY this single code block without any commentary."
        )

    messages = [
        {"role": "system", "content": _get_system_prompt(language)},
        {"role": "user", "content": user_prompt},
    ]

    response_text = _call_llm(messages)

    if tests:
        return extract_code(response_text)
    return _extract_code_and_tests(response_text, language=language)


def repair_code(
    task: str,
    history: list[Attempt],
    language: str = "python",
    user_hint: str | None = None,
) -> str:
    """Generate a repaired code solution given the task and history of past attempts."""
    is_cpp = language.lower() in ("cpp", "c++")
    lang_name = "C++17" if is_cpp else "Python 3.11"
    sol_file = "solution.hpp" if is_cpp else "solution.py"

    formatted_attempts = []
    for i, attempt in enumerate(history, 1):
        formatted_attempts.append(
            f"=== Attempt {i} ===\n"
            f"Code:\n```{language}\n{attempt.code}\n```\n"
            f"Error Output:\n{attempt.error}\n"
        )
    history_text = "\n".join(formatted_attempts)
    hint_clause = f"\n\n[HUMAN ARCHITECTURAL DIRECTIVE / HINT]:\n{user_hint}\nYou MUST follow this instruction in the repaired code." if user_hint else ""

    user_prompt = (
        f"Task ({lang_name}):\n{task}{hint_clause}\n\n"
        f"Previous failed attempts and their execution errors:\n{history_text}\n\n"
        f"Instructions:\n"
        f"1. Carefully inspect each failed attempt and read the corresponding error output.\n"
        f"2. Identify the root cause of the failure. Do NOT make superficial tweaks or repeat earlier mistakes.\n"
        f"3. Fix the underlying root cause and handle edge cases properly.\n"
        f"4. Return ONLY a single ```{language} ``` code block containing the complete, repaired solution for `{sol_file}`."
    )

    messages = [
        {"role": "system", "content": _get_system_prompt(language)},
        {"role": "user", "content": user_prompt},
    ]

    response_text = _call_llm(messages)
    return extract_code(response_text)


def generate_adversarial_tests(task: str, code: str, language: str = "python") -> str:
    """Generate ruthless edge-case adversarial test suite designed to challenge the implementation."""
    is_cpp = language.lower() in ("cpp", "c++")
    lang_name = "C++17" if is_cpp else "Python 3.11"

    if is_cpp:
        prompt = (
            f"Task: {task}\n\n"
            f"Target C++ Implementation (in `solution.hpp`):\n```cpp\n{code}\n```\n\n"
            f"Instructions:\n"
            f"1. Write an adversarial unit test suite in test_solution.cpp.\n"
            f"2. Include extreme inputs: empty collections, single elements, maximum/minimum boundaries, negative numbers, duplicates.\n"
            f"3. Must `#include <cassert>` and `#include \"solution.hpp\"` with an `int main()` that returns 0 if all tests pass.\n"
            f"4. Return ONLY a single ```cpp ``` code block."
        )
    else:
        prompt = (
            f"Task: {task}\n\n"
            f"Target Python Implementation (in `solution.py`):\n```python\n{code}\n```\n\n"
            f"Instructions:\n"
            f"1. Write an adversarial pytest test suite in test_solution.py.\n"
            f"2. Include extreme edge cases: empty strings/lists, single elements, negative numbers, large inputs, duplicates, invalid bounds.\n"
            f"3. Must import from `solution` (e.g. `from solution import ...`).\n"
            f"4. Return ONLY a single ```python ``` code block."
        )

    messages = [
        {"role": "system", "content": _get_system_prompt(language, role="attacker")},
        {"role": "user", "content": prompt},
    ]

    response_text = _call_llm(messages)
    return extract_code(response_text)


def generate_profiling_harness(task: str, code: str, language: str = "python") -> str:
    """Generate a self-contained runtime and memory profiling script that outputs a single JSON line."""
    is_cpp = language.lower() in ("cpp", "c++")

    if is_cpp:
        prompt = (
            f"Task: {task}\n\n"
            f"Implementation in `solution.hpp`:\n```cpp\n{code}\n```\n\n"
            f"Write a standalone C++ benchmark script (`profiler.cpp`) that:\n"
            f"1. Includes `#include \"solution.hpp\"`, `#include <chrono>`, `#include <iostream>`, `#include <vector>`.\n"
            f"2. Executes the solution for 4 input scales N: e.g. 50, 200, 1000, 4000 (adjust if appropriate for problem).\n"
            f"3. Measures elapsed time in milliseconds for each scale using std::chrono.\n"
            f"4. Estimates empirical Big-O time complexity (e.g. 'O(N)', 'O(N log N)', 'O(N^2)').\n"
            f"5. Prints a SINGLE valid JSON line to standard output in exact format:\n"
            f'{{"samples": [{{"n": 50, "time_ms": 0.02, "memory_kb": 12.0}}, ...], "estimated_complexity": "O(N)", "peak_memory_kb": 48.0}}\n'
            f"Return ONLY the ```cpp ``` code block."
        )
    else:
        prompt = (
            f"Task: {task}\n\n"
            f"Implementation in `solution.py`:\n```python\n{code}\n```\n\n"
            f"Write a standalone Python profiling script (`profiler.py`) that:\n"
            f"1. Imports from `solution`, uses `time.perf_counter()` and `tracemalloc`.\n"
            f"2. Runs the solution on 4 input sizes N: e.g. 50, 200, 1000, 4000 (adjust as suitable for the task).\n"
            f"3. Records elapsed time in milliseconds (`time_ms`) and memory in kilobytes (`memory_kb`).\n"
            f"4. Estimates empirical Big-O complexity (e.g. 'O(N)', 'O(N log N)', 'O(N^2)', 'O(1)').\n"
            f"5. Prints a SINGLE valid JSON line to standard output in exact format:\n"
            f'{{"samples": [{{"n": 50, "time_ms": 0.05, "memory_kb": 8.0}}, ...], "estimated_complexity": "O(N)", "peak_memory_kb": 32.0}}\n'
            f"Return ONLY the ```python ``` code block."
        )

    messages = [
        {"role": "system", "content": _get_system_prompt(language, role="profiler")},
        {"role": "user", "content": prompt},
    ]

    response_text = _call_llm(messages)
    return extract_code(response_text)
