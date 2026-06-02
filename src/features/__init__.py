"""Feature-engineering utilities for the QA dataset."""

from .entity_utils import extract_entities, load_entity_db, normalize_entity_name
from .feature_matrix import build_feature_frame

__all__ = [
    "build_feature_frame",
    "extract_entities",
    "load_entity_db",
    "normalize_entity_name",
]
