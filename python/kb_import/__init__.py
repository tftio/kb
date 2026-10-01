"""kb-import: import conversation transcripts into the kb knowledge base.

The console script and the package version both live here; `cli.py` imports
`__version__` from this module, and this module imports nothing from `cli`,
so the two never form a cycle.
"""

from __future__ import annotations

import importlib.metadata

__version__ = importlib.metadata.version("kb-import")
