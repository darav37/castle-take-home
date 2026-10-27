"""Checks on this repo: no secrets, no internal notes, every doc link resolves, the code compiles.

  python scripts/check_submission.py            # fast checks
  python scripts/check_submission.py --tests    # also runs the test suite

Exits non-zero on any failure.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS = [r"sk-ant-[A-Za-z0-9_-]{8,}", r"sk-[A-Za-z0-9]{20,}", r"(?i)api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_-]{16,}['\"]"]
INTERNAL = ["Claude Code", "planning chat", "TASK_", "DARIN_TODO", "CLAUDE.md", "claude.ai/"]
BAD_PATHS = [".env", "private", "cache", "out"]
KEEP = ["README.md", "pyproject.toml", "docs/decisions.md", "docs/event_pipeline/overview.md", "docs/results/RESULTS_SUMMARY.md",
        "docs/event_pipeline/flow_map.html", "src/worsey/README.md"]
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def files():
    return [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts]


def main(tests):
    problems = []
    for rel in KEEP:
        if not (ROOT / rel).exists():
            problems.append(f"missing: {rel}")
    for name in BAD_PATHS:
        if (ROOT / name).exists():
            problems.append(f"should not ship: {name}")
    for f in files():
        rel = f.relative_to(ROOT).as_posix()
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if rel == "scripts/check_submission.py":
            continue
        for pat in SECRETS:
            if re.search(pat, text):
                problems.append(f"secret-like text in {rel}")
        for term in INTERNAL:
            if term in text:
                problems.append(f"internal term {term!r} in {rel}")
        if rel.endswith(".py"):
            try:
                compile(text, rel, "exec")
            except SyntaxError as e:
                problems.append(f"syntax error in {rel}:{e.lineno}")
        if rel.endswith(".md"):
            for target in LINK.findall(text):
                if re.match(r"[a-z]+:", target) or target.startswith("#"):
                    continue
                if not (f.parent / target).exists():
                    problems.append(f"broken link in {rel}: {target}")
    if tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT)
        if r.returncode:
            problems.append("test suite failed")
    for p in problems:
        print("FAIL", p)
    print(f"{len(problems)} problems" if problems else "all checks passed")
    return 1 if problems else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true")
    sys.exit(main(ap.parse_args().tests))
