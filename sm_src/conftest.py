"""Pytest bootstrap for Synthetic2 sm_core.

Ensures ``synthetic_myelination`` is importable regardless of the
``Path(__file__).parents[...]`` depth each individual test file assumes
(the v1 tests were written against the sm_src/ layout and use mixed
parents[1]/parents[2] offsets). Placing sm_core/ on sys.path here makes
all of them resolve ``synthetic_myelination`` consistently.
"""

import sys
from pathlib import Path

SM_CORE = Path(__file__).resolve().parent
if str(SM_CORE) not in sys.path:
    sys.path.insert(0, str(SM_CORE))
