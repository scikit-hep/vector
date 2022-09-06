from __future__ import annotations

import os
import sys

if os.environ.get("VECTOR_USE_AWKWARDV2", None):
    import awkward._v2

    sys.modules["awkward"] = awkward._v2
    sys.modules["awkward"]._v2 = awkward._v2
