from __future__ import annotations

import os


# Unit tests must stay deterministic and must not spend external LLM credits
# just because a developer has a local .env file configured for smoke tests.
os.environ.setdefault("LTM_LLM_PROVIDER", "none")
