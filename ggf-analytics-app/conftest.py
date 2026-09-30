"""Pytest bootstrap: put the project root on ``sys.path``.

Ensures ``import src...`` resolves when pytest is invoked from the project root
(or anywhere the project root is discoverable), independent of the runner's
default path insertion behavior.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
