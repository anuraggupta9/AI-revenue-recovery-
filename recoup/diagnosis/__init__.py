"""Root-cause diagnosis. Standard library only.

The optional LLM tail lives in recoup.diagnosis.llm_tail and is imported lazily,
so this package keeps its zero-dependency guarantee.
"""

from recoup.diagnosis.taxonomy_map import (
    CONFIDENCE_FLOOR,
    Diagnosis,
    RootCause,
    diagnose,
)

__all__ = ["CONFIDENCE_FLOOR", "Diagnosis", "RootCause", "diagnose"]
