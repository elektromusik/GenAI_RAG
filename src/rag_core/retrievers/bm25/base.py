from dataclasses import dataclass
from typing import Any
from ...interfaces import Doc

@dataclass
class BM25Config:
    k1: float = 1.5
    b:  float = 0.75
    fields: list[str] | None = None # e.g. ["title^2", "contents"]

def to_doc(doc_id: str, text: str, score: float, **meta: Any) -> Doc:
    """Convert backend hit into a Doc object with normalized metadata."""
    m = {"score": float(score)}
    m.update(meta)
    return Doc(id=str(doc_id), text=text or "", metadata=m)

def minmax_norm(scores: dict[str, float]) -> dict[str, float]:
    """
    Normalize scores into [0,1] using min-max normalization.
    Returns {} unchanged if empty.
    """
    if not scores: 
        return scores
    vmin, vmax = min(scores.values()), max(scores.values())
    if vmax <= vmin: # avoid div by zero
        return {k: 0.0 for k in scores}
    return {k: (v - vmin) / (vmax - vmin) for k, v in scores.items()}
