"""Standalone Comprehensive E2E Test Suite Runner.

Usage:
    python tests/e2e/run_all.py
    python tests/e2e/run_all.py --tier 1
    python tests/e2e/run_all.py -v
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FINANCE_SRC = REPO_ROOT / "src"
FUZZYMATCH_ROOT = REPO_ROOT.parent / "fuzzymatch"

# Configure sys.path for standalone runner
if str(FINANCE_SRC) not in sys.path:
    sys.path.insert(0, str(FINANCE_SRC))

try:
    import fuzzymatch  # noqa: F401
except ImportError:
    if FUZZYMATCH_ROOT.exists() and str(FUZZYMATCH_ROOT) not in sys.path:
        sys.path.insert(0, str(FUZZYMATCH_ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Fuzzy Entity Resolution E2E Test Suite")
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run only a specific test tier (1: Features, 2: Boundaries, 3: Interactions, 4: Real-World)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose pytest output",
    )
    parser.add_argument(
        "-k",
        "--keyword",
        type=str,
        default="",
        help="Filter tests by keyword expression",
    )

    args = parser.parse_args(argv)

    e2e_dir = Path(__file__).resolve().parent
    target_paths: list[str] = []

    if args.tier == 1:
        target_paths.append(str(e2e_dir / "tier1_features"))
    elif args.tier == 2:
        target_paths.append(str(e2e_dir / "tier2_boundaries"))
    elif args.tier == 3:
        target_paths.append(str(e2e_dir / "tier3_interactions"))
    elif args.tier == 4:
        target_paths.append(str(e2e_dir / "tier4_real_world"))
    else:
        # Run all tiers
        target_paths.append(str(e2e_dir))

    pytest_args = target_paths + ["-rs"]
    if args.verbose:
        pytest_args.append("-v")
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    print("=" * 80)
    print("FUZZY RECORD LINKAGE E2E TEST RUNNER")
    print(f"Target: {'All Tiers' if not args.tier else f'Tier {args.tier}'}")
    print(f"Repository: {REPO_ROOT}")
    print("=" * 80)

    exit_code = pytest.main(pytest_args)
    print("=" * 80)
    print(f"E2E Test Run Finished with exit code: {exit_code}")
    print("=" * 80)
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
