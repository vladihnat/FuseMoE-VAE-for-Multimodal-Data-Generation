import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if not SRC.exists():
    raise FileNotFoundError(f"Could not find src directory at: {SRC}")

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
