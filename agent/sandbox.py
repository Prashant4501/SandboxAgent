from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid

IMAGE_NAME = os.getenv("SANDBOX_IMAGE", "sandbox-agent")


@dataclass
class ExecutionResult:
    status: str
    stdout: str
    stderr: str
    exit_code: int
    duration: float


@dataclass
class ProfileResult:
    status: str
    samples: list[dict]
    estimated_complexity: str
    peak_memory_kb: float
    raw_output: str


def _truncate(text: str, max_chars: int = 2000) -> str:
    """Truncate text to the trailing max_chars characters."""
    if text and len(text) > max_chars:
        return text[-max_chars:]
    return text


def _classify_status(
    exit_code: int,
    stdout: str,
    stderr: str,
    timed_out: bool,
    language: str = "python",
) -> str:
    """Classify the sandbox execution result into a normalized status for Python or C++."""
    if timed_out:
        return "timeout"
    if exit_code == 0:
        return "passed"

    combined = f"{stdout}\n{stderr}"

    if language.lower() in ("cpp", "c++"):
        lower_combined = combined.lower()
        if "assertion" in lower_combined and "failed" in lower_combined:
            return "test_failure"
        if "error:" in combined:
            return "syntax_error"
        if exit_code in (137, 139) or "segmentation fault" in lower_combined or "aborted" in lower_combined:
            return "runtime_error"
        return "runtime_error"

    if re.search(r"\b(SyntaxError|IndentationError|TabError)\b", combined):
        return "syntax_error"

    exception_matches = set(re.findall(r"\b([A-Za-z0-9_]*(?:Error|Exception))\b", combined))
    runtime_exceptions = exception_matches - {"AssertionError", "SyntaxError", "IndentationError", "TabError"}
    if runtime_exceptions or exit_code == 137:
        return "runtime_error"

    if "AssertionError" in exception_matches or "FAILED" in combined or exit_code == 1:
        return "test_failure"

    return "runtime_error"


def run_in_sandbox(
    code: str,
    tests: str,
    timeout: int = 10,
    image: str = IMAGE_NAME,
    language: str = "python",
) -> ExecutionResult:
    """Execute code against tests inside a locked-down Docker sandbox for Python or C++."""
    container_name = f"sandbox_{uuid.uuid4().hex[:12]}"
    start_time = time.perf_counter()
    is_cpp = language.lower() in ("cpp", "c++")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir).resolve()

        if is_cpp:
            (temp_path / "solution.hpp").write_text(code, encoding="utf-8")
            (temp_path / "test_solution.cpp").write_text(tests, encoding="utf-8")
            runner_cmd = [
                "sh",
                "-c",
                "g++ -O2 -std=c++17 /app/test_solution.cpp -I/app -o /tmp/runner && /tmp/runner",
            ]
        else:
            (temp_path / "solution.py").write_text(code, encoding="utf-8")
            (temp_path / "test_solution.py").write_text(tests, encoding="utf-8")
            runner_cmd = ["pytest", "-x", "-q"]

        cmd = [
            "docker", "run",
            "--name", container_name,
            "--rm",
            "--network", "none",
            "--memory", "256m",
            "--cpus", "0.5",
            "--pids-limit", "64",
            "--read-only",
            "--tmpfs", "/tmp:rw,exec,size=64m",
            "--user", "runner",
            "-v", f"{temp_path.as_posix()}:/app:ro",
            "-w", "/app",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "PYTEST_ADDOPTS=-o cache_dir=/tmp",
            image,
            *runner_cmd,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.perf_counter() - start_time
            stdout = _truncate(proc.stdout)
            stderr = _truncate(proc.stderr)
            status = _classify_status(proc.returncode, stdout, stderr, timed_out=False, language=language)

            return ExecutionResult(
                status=status,
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                duration=round(duration, 3),
            )
        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start_time
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            return ExecutionResult(
                status="timeout",
                stdout="",
                stderr=f"Execution timed out after {timeout} seconds.",
                exit_code=124,
                duration=round(duration, 3),
            )
        except FileNotFoundError:
            return ExecutionResult(
                status="runtime_error",
                stdout="",
                stderr="Docker CLI not found. Please install Docker and ensure it is on your PATH.",
                exit_code=127,
                duration=0.0,
            )
        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


def run_profiler_in_sandbox(
    code: str,
    harness: str,
    timeout: int = 15,
    image: str = IMAGE_NAME,
    language: str = "python",
) -> ProfileResult:
    """Run an empirical runtime and memory profiler harness inside the Docker sandbox."""
    container_name = f"sandbox_prof_{uuid.uuid4().hex[:12]}"
    is_cpp = language.lower() in ("cpp", "c++")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir).resolve()

        if is_cpp:
            (temp_path / "solution.hpp").write_text(code, encoding="utf-8")
            (temp_path / "profiler.cpp").write_text(harness, encoding="utf-8")
            runner_cmd = [
                "sh",
                "-c",
                "g++ -O2 -std=c++17 /app/profiler.cpp -I/app -o /tmp/prof && /tmp/prof",
            ]
        else:
            (temp_path / "solution.py").write_text(code, encoding="utf-8")
            (temp_path / "profiler.py").write_text(harness, encoding="utf-8")
            runner_cmd = ["python", "-B", "/app/profiler.py"]

        cmd = [
            "docker", "run",
            "--name", container_name,
            "--rm",
            "--network", "none",
            "--memory", "256m",
            "--cpus", "0.5",
            "--pids-limit", "64",
            "--read-only",
            "--tmpfs", "/tmp:rw,exec,size=64m",
            "--user", "runner",
            "-v", f"{temp_path.as_posix()}:/app:ro",
            "-w", "/app",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            image,
            *runner_cmd,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            raw = f"{proc.stdout}\n{proc.stderr}".strip()
            if proc.returncode != 0:
                return ProfileResult(
                    status="error",
                    samples=[],
                    estimated_complexity="Unknown (Execution Failed)",
                    peak_memory_kb=0.0,
                    raw_output=_truncate(raw),
                )

            # Extract JSON output payload from profiler
            json_match = re.search(r"\{.*\}", proc.stdout, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                samples = data.get("samples", [])
                complexity = data.get("estimated_complexity", "O(N)")
                peak_mem = float(data.get("peak_memory_kb", 0.0))
                return ProfileResult(
                    status="passed",
                    samples=samples,
                    estimated_complexity=complexity,
                    peak_memory_kb=peak_mem,
                    raw_output=proc.stdout.strip(),
                )

            return ProfileResult(
                status="error",
                samples=[],
                estimated_complexity="Unknown (Parse Error)",
                peak_memory_kb=0.0,
                raw_output=_truncate(proc.stdout),
            )
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as exc:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            return ProfileResult(
                status="error",
                samples=[],
                estimated_complexity=f"Error: {exc}",
                peak_memory_kb=0.0,
                raw_output=str(exc),
            )
        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


if __name__ == "__main__":
    cases = [
        (
            "Good code",
            "def add(a: int, b: int) -> int:\n    return a + b\n",
            "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
            10,
        ),
        (
            "Buggy code",
            "def add(a: int, b: int) -> int:\n    return a - b\n",
            "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
            10,
        ),
        (
            "Infinite loop",
            "def spin() -> None:\n    while True:\n        pass\n",
            "from solution import spin\n\ndef test_spin():\n    spin()\n",
            2,
        ),
        (
            "Network access attempt",
            "import urllib.request\n\ndef fetch() -> str:\n    return urllib.request.urlopen('http://example.com').read().decode()\n",
            "from solution import fetch\n\ndef test_fetch():\n    assert fetch() != ''\n",
            10,
        ),
    ]

    print("Running sandbox verification cases...\n")
    for name, c, t, to in cases:
        print(f"=== Case: {name} ===")
        res = run_in_sandbox(c, t, timeout=to)
        print(f"Status:    {res.status}")
        print(f"Exit code: {res.exit_code}")
        print(f"Duration:  {res.duration:.2f}s")
        if res.stdout.strip():
            print(f"Stdout:\n{res.stdout.strip()}")
        if res.stderr.strip():
            print(f"Stderr:\n{res.stderr.strip()}")
        print("-" * 40)
