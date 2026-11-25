import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for sub in (ROOT / "src", ROOT / "src" / "decision_aware_acq"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))
