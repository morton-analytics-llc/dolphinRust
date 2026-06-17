#!/usr/bin/env python3
"""PostToolUse guard for Rust edits.

Two jobs, both non-blocking (reminders are fed back to Claude via
additionalContext, never a hard block):
  1. Maintain formatting standards as development occurs — run rustfmt on the
     touched file.
  2. Push back on code volume and nesting depth, per the simplicity-first and
     minimal-branching rules in CLAUDE.md.
"""
import json
import os
import shutil
import subprocess
import sys

ADDED_LINES_LIMIT = 100
NESTING_DEPTH_LIMIT = 3  # warn at 4+ levels of indentation


def read_event():
    try:
        return json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return {}


def added_text(tool_input):
    if tool_input.get("content"):
        return tool_input["content"]
    if tool_input.get("new_string"):
        return tool_input["new_string"]
    edits = tool_input.get("edits")
    if isinstance(edits, list):
        return "\n".join(e.get("new_string", "") for e in edits)
    return ""


def max_nesting_depth(path):
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return 0
    depth = 0
    for line in lines:
        stripped = line.rstrip("\n")
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        depth = max(depth, indent // 4)
    return depth


def main():
    event = read_event()
    tool_input = event.get("tool_input") or {}
    path = tool_input.get("file_path", "")
    if not path.endswith(".rs"):
        return

    if shutil.which("rustfmt"):
        subprocess.run(
            ["rustfmt", "--edition", "2021", path],
            capture_output=True,
            check=False,
        )

    reminders = []

    text = added_text(tool_input)
    line_count = text.count("\n") + 1 if text else 0
    if line_count > ADDED_LINES_LIMIT:
        reminders.append(
            f"This edit wrote ~{line_count} lines. Simplicity-first (CLAUDE.md): "
            "confirm every line is required by the current phase's contract test; "
            "a helper earns its place only at 3+ call sites. Prefer the smallest "
            "change that turns the red contract test green."
        )

    depth = max_nesting_depth(path)
    if depth > NESTING_DEPTH_LIMIT:
        reminders.append(
            f"{os.path.basename(path)} nests {depth} levels deep. Flatten to <=3: "
            "early returns and guard clauses over nested if, `match` over if/else "
            "ladders, iterator chains over manual loops (CLAUDE.md idiomatic-Rust rules)."
        )

    if reminders:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": " ".join(reminders),
            }
        }))


if __name__ == "__main__":
    main()
