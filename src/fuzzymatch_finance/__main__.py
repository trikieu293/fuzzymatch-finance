"""Executable module entry point for `python -m fuzzymatch_finance`."""

from __future__ import annotations

import sys

from fuzzymatch_finance.cli import main

if __name__ == "__main__":
    sys.exit(main())
