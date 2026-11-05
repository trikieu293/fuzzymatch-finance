"""Root test configuration and path resolution for fuzzymatch-finance."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure fuzzymatch is available (fallback to sibling repo if not pip-installed)
try:
    import fuzzymatch  # noqa: F401
except ImportError:
    sibling_core = Path(__file__).resolve().parent.parent.parent / "fuzzymatch"
    if sibling_core.exists() and str(sibling_core) not in sys.path:
        sys.path.insert(0, str(sibling_core))
