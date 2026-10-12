import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parent.parent
target = here / "solutions" if os.environ.get("USE_SOLUTIONS") else here
sys.path.insert(0, str(target))
