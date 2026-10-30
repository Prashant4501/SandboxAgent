import difflib
import os
import streamlit as st

from agent.controller import AttemptDetail, RunResult, solve
from agent.llm import MODEL_NAME
from agent.sandbox import IMAGE_NAME, ProfileResult

st.set_page_config(
    page_title="SandboxAgent | Autonomous Coding Agent",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .badge-pass {
        background-color: #064e3b;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    .badge-fail {
        background-color: #450a0a;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    .badge-warn {
        background-color: #451a03;
        color: #fbbf24;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    .badge-info {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PYTHON_EXAMPLES = [
    (
        "Merge Intervals",
        "Write a function merge_intervals(intervals: list[list[int]]) -> list[list[int]] that merges all overlapping intervals. Intervals that touch like [1, 4] and [4, 5] must be merged into [1, 5]. Return sorted by start time.",
    ),
    (
        "LRU Cache",
        "Implement a class LRUCache(capacity: int) with get(key: int) -> int and put(key: int, value: int) -> None. Evict least recently used key on overflow. Updating an existing key must not evict items. If capacity <= 0 raise ValueError.",
    ),
    (
        "Spiral Matrix",
        "Write a function spiral_order(matrix: list[list[int]]) -> list[int] that returns all elements of an m x n matrix in clockwise spiral order. If matrix is empty or [[]], return [].",
    ),
    (
        "Run-Length Encoding",
        "Write encode_rle(s: str) -> str and decode_rle(s: str) -> str. E.g., 'WWWBWW' -> '3W1B2W'. Multi-digit counts like '12A' must be properly parsed. Empty strings return empty strings.",
    ),
]

CPP_EXAMPLES = [
    (
        "Merge Intervals (C++)",
        "Write a function std::vector<std::vector<int>> mergeIntervals(std::vector<std::vector<int>>& intervals) that merges overlapping intervals. Touching intervals [1, 4] and [4, 5] must merge into [1, 5]. Return sorted.",
    ),
    (
        "LRU Cache (C++)",
        "Implement a class LRUCache with constructor LRUCache(int capacity), int get(int key), and void put(int key, int value). Evict least recently used on overflow. Updating existing key does not evict items.",
    ),
    (
        "Spiral Matrix (C++)",
        "Write a function std::vector<int> spiralOrder(const std::vector<std::vector<int>>& matrix) that returns all elements in clockwise spiral order. Empty matrix returns empty vector.",
    ),
    (
        "Valid Palindrome (C++)",
        "Write a function bool isPalindrome(const std::string& s) that checks if s is a palindrome considering only alphanumeric characters and ignoring cases. Empty string returns true.",
    ),
]


def compute_diff(prev_code: str, curr_code: str) -> str:
    """Compute unified diff between two code strings to highlight repairs."""
    prev_lines = prev_code.splitlines(keepends=True)
    curr_lines = curr_code.splitlines(keepends=True)
    diff = difflib.unified_diff(
        prev_lines,
        curr_lines,
        fromfile="previous_attempt",
        tofile="repaired_attempt",
        n=2,
    )
    return "".join(diff)


def status_badge(status: str) -> str:
    """Generate HTML badge for an execution status."""
    label = status.replace("_", " ").upper()
    if status == "passed":
        return f'<span class="badge-pass">✓ {label}</span>'
    elif status == "timeout":
        return f'<span class="badge-warn">⏱ {label}</span>'
    elif status == "info":
        return f'<span class="badge-info">ℹ {label}</span>'
    else:
        return f'<span class="badge-fail">✗ {label}</span>'


def render_attempt_card(idx: int, attempt: AttemptDetail, prev_code: str | None, lang: str) -> None:
    """Render an individual attempt card including diff, code, and errors."""
    attempt_num = idx + 1
    status = attempt.result.status
    badge = status_badge(status)
    title = f"Attempt {attempt_num}" if idx == 0 else f"Attempt {attempt_num} (Repair {idx})"

    with st.expander(f"{title} — {status.upper()}", expanded=True):
        header_col1, header_col2 = st.columns([1, 3])
        with header_col1:
            st.markdown(badge, unsafe_allow_html=True)
        with header_col2:
            st.caption(f"Exit code: `{attempt.result.exit_code}` | Duration: `{attempt.result.duration:.2f}s`")

        if prev_code:
            diff_text = compute_diff(prev_code, attempt.code)
            if diff_text:
                st.markdown("**Changes from previous attempt (Diff):**")
                st.code(diff_text, language="diff")

        st.markdown(f"**Generated Solution ({lang.upper()}):**")
        st.code(attempt.code, language=lang, line_numbers=True)

        if status != "passed":
            error_output = attempt.result.stdout or attempt.result.stderr or "No output captured."
            st.markdown("**Sandbox Error Output:**")
            st.code(error_output, language="text")


def main() -> None:
    """Main application routine managing state, layout, and agent execution."""
    st.title("🛡️ SandboxAgent")
    st.caption("Autonomous Coding Agent with Docker Sandbox Isolation, Adversarial QA, and Big-O Profiling.")

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_language = st.selectbox("Programming Language", ["Python", "C++"], index=0)
        lang_code = "cpp" if selected_language == "C++" else "python"

        st.markdown(f"**LLM Model:** `{os.getenv('LLM_MODEL', MODEL_NAME)}`")
        st.markdown(f"**Sandbox Image:** `{IMAGE_NAME}`")
        max_repairs = st.slider("Max Repair Attempts", min_value=1, max_value=5, value=3)

        st.divider()
        st.header("🔬 Advanced Agentic Capabilities")
        enable_adversarial = st.checkbox(
            "⚔️ Adversarial QA Fuzzer",
            value=True,
            help="Deploys an Adversarial Attacker Agent to stress-test your code against nasty boundary and edge cases.",
        )
        enable_profiler = st.checkbox(
            "⚡ Big-O & Memory Profiler",
            value=True,
            help="Runs empirical microbenchmarks inside Docker across scaling input sizes N to estimate complexity and memory.",
        )

        st.divider()
        with st.expander("🔒 Sandbox Security Limits", expanded=False):
            st.markdown(
                """
                - **Network:** `--network none` (Airgapped)
                - **Memory Cap:** `--memory 256m`
                - **CPU Limit:** `--cpus 0.5`
                - **PID Limit:** `--pids-limit 64`
                - **Filesystem:** `--read-only` root
                - **Temp Storage:** 64MB `/tmp` tmpfs (exec enabled)
                - **User:** Non-root `runner` (uid 1000)
                - **Execution:** Zero host execution
                """
            )

        with st.expander(f"🧪 Custom Tests ({selected_language}) [Optional]", expanded=False):
            placeholder_test = (
                '#include <cassert>\n#include "solution.hpp"\n\nint main() {\n    // assert(...);\n    return 0;\n}'
                if lang_code == "cpp"
                else "from solution import ...\n\ndef test_custom():\n    assert ..."
            )
            custom_tests = st.text_area(
                f"Provide custom {selected_language} tests (leave blank to let LLM auto-generate)",
                value="",
                height=180,
                placeholder=placeholder_test,
            )

    examples = CPP_EXAMPLES if lang_code == "cpp" else PYTHON_EXAMPLES

    # Track language change in session state
    if "current_lang" not in st.session_state or st.session_state["current_lang"] != lang_code:
        st.session_state["current_lang"] = lang_code
        st.session_state["task_input"] = examples[0][1]

    st.markdown(f"**Select {selected_language} Example Task:**")
    example_cols = st.columns(len(examples))
    for i, (label, desc) in enumerate(examples):
        if example_cols[i].button(label, key=f"ex_{lang_code}_{i}", use_container_width=True):
            st.session_state["task_input"] = desc
            st.rerun()

    task = st.text_area(
        f"Task Description ({selected_language}):",
        value=st.session_state["task_input"],
        height=90,
        placeholder="Describe the programming problem to solve...",
    )

    # Human-in-the-Loop Steering Input
    user_hint = st.text_input(
        "🎯 Human-in-the-Loop Steering Directive (Optional):",
        value="",
        placeholder="e.g. Use a two-pointer approach in O(1) extra space; avoid recursion to prevent stack overflow.",
        help="Inject an explicit architectural constraint or algorithmic hint directly into the agent's reasoning loop.",
    )

    run_clicked = st.button("🚀 Run Agent", type="primary", use_container_width=True)

    if run_clicked:
        if not task.strip():
            st.warning("Please provide a task description.")
            return

        status_box = st.status(f"🚀 Launching SandboxAgent ({selected_language})...", expanded=True)

        def event_handler(event: dict) -> None:
            """Handle controller progress events and update the live status indicator."""
            step = event.get("step")
            attempt_num = event.get("attempt", 1)
            if step == "generating":
                status_box.update(label=f"🤖 Generating initial {selected_language} solution...", state="running")
            elif step == "repairing":
                reason = event.get("reason", "test_failure")
                if reason == "adversarial_patch":
                    status_box.update(label="🛡️ Patching edge cases flagged by Adversarial Fuzzer...", state="running")
                else:
                    status_box.update(label=f"🔧 Repairing solution (Attempt {attempt_num})...", state="running")
            elif step == "executing":
                status_box.update(label=f"🐳 Compiling and testing Attempt {attempt_num} in Docker...", state="running")
            elif step == "executed":
                res = event.get("result")
                tag = res.status.upper() if res else "DONE"
                status_box.write(f"Attempt {attempt_num} finished with status: `{tag}`")
            elif step == "adversarial_testing":
                status_box.update(label="⚔️ Deploying Adversarial QA Agent to stress-test edge cases...", state="running")
            elif step == "adversarial_completed":
                passed = event.get("passed", False)
                if passed:
                    status_box.write("✓ Code survived all Adversarial Edge-Case Attacks!")
                else:
                    status_box.write("⚠ Adversarial Fuzzer found a breaking edge case! Initiating patch...")
            elif step == "profiling":
                status_box.update(label="⚡ Profiling empirical Big-O runtime & memory in Docker...", state="running")
            elif step == "profiled":
                prof_res: ProfileResult = event.get("result")
                if prof_res and prof_res.status == "passed":
                    status_box.write(f"✓ Profiling complete: Estimated {prof_res.estimated_complexity}")

        tests_to_pass = custom_tests.strip() if custom_tests.strip() else None
        steer_text = user_hint.strip() if user_hint.strip() else None

        with st.spinner(f"Agent running autonomous {selected_language} loop..."):
            run_result: RunResult = solve(
                task=task.strip(),
                tests=tests_to_pass,
                max_repairs=max_repairs,
                on_event=event_handler,
                language=lang_code,
                user_hint=steer_text,
                enable_adversarial=enable_adversarial,
                enable_profiler=enable_profiler,
            )

        status_box.update(
            label="✓ Execution completed successfully!" if run_result.success else "✗ Execution finished without passing all tests.",
            state="complete" if run_result.success else "error",
            expanded=False,
        )

        # 1. Timeline & Attempts
        st.subheader("📜 Execution Timeline & Repairs")
        prev_code = None
        for idx, attempt in enumerate(run_result.attempts):
            render_attempt_card(idx, attempt, prev_code, lang_code)
            prev_code = attempt.code

        # 2. Adversarial QA Report Card (Feature 1)
        if run_result.adversarial_result:
            st.divider()
            st.subheader("⚔️ Adversarial QA & Fuzzing Report")
            adv_status = "passed" if run_result.adversarial_passed else "test_failure"
            adv_badge = status_badge(adv_status)

            with st.expander(f"Adversarial Stress Test Outcome — {adv_status.upper()}", expanded=True):
                st.markdown(adv_badge, unsafe_allow_html=True)
                if run_result.adversarial_passed:
                    st.success("The implementation proved robust against all adversarial edge cases (empty inputs, negative bounds, scale tests).")
                else:
                    st.warning("Adversarial tester discovered an unhandled edge case.")

                if run_result.adversarial_tests:
                    st.markdown("**Attacker Test Suite:**")
                    st.code(run_result.adversarial_tests, language=lang_code)

        # 3. Docker Big-O & Memory Profiler Card (Feature 2)
        if run_result.profile_result and run_result.profile_result.status == "passed":
            st.divider()
            st.subheader("⚡ Docker Empirical Performance Profile")
            prof: ProfileResult = run_result.profile_result

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown(f"**Estimated Time Complexity:** `{prof.estimated_complexity}`")
            with col_p2:
                st.markdown(f"**Peak Memory Footprint:** `{prof.peak_memory_kb:.1f} KB`")

            if prof.samples:
                st.markdown("**Empirical Scaling Table ($N$ vs Runtime):**")
                table_data = []
                for s in prof.samples:
                    table_data.append({
                        "Input Size (N)": s.get("n", 0),
                        "Execution Time (ms)": s.get("time_ms", 0.0),
                        "Memory (KB)": s.get("memory_kb", 0.0),
                    })
                st.table(table_data)

                # Line chart of execution time vs N
                chart_data = {
                    "Input Size (N)": [s.get("n", 0) for s in prof.samples],
                    "Time (ms)": [s.get("time_ms", 0.0) for s in prof.samples],
                }
                st.line_chart(data=chart_data, x="Input Size (N)", y="Time (ms)")

        # 4. Final Summary
        st.divider()
        st.subheader("📊 Final Run Summary")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Outcome", "SUCCESS" if run_result.success else "FAILED")
        with m2:
            st.metric("Attempts Used", f"{len(run_result.attempts)} / {1 + max_repairs}")
        with m3:
            st.metric("Total Tokens", f"{run_result.total_tokens:,}")
        with m4:
            st.metric("Total Elapsed Time", f"{run_result.total_time:.2f}s")

        if run_result.user_hint:
            st.info(f"**Applied Human Directive:** *\"{run_result.user_hint}\"*")

        st.markdown(f"**Final Verified {selected_language} Code:**")
        st.code(run_result.final_code, language=lang_code, line_numbers=True)

        download_filename = "solution.hpp" if lang_code == "cpp" else "solution.py"
        download_mime = "text/x-c++src" if lang_code == "cpp" else "text/x-python"
        st.download_button(
            label=f"💾 Download {download_filename}",
            data=run_result.final_code,
            file_name=download_filename,
            mime=download_mime,
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
