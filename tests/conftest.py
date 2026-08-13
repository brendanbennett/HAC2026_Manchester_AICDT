import sys
from pathlib import Path

# One package, at the repo root. This file used to insert solver/ ahead of the root to
# disambiguate two divergent copies of hac26; that duplication is gone, and with it the
# failure mode where `pytest tests` and `pytest tests solver/` imported different
# packages and passed apart while failing together.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
