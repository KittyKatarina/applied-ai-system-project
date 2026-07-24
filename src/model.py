"""Inference-side wrapper around the trained taste-match model.

The model is fit by src/train_model.py and loaded here at recommendation time.
It replaces exact-match genre/mood scoring with continuous text similarity and
a regressor trained on a documented synthetic policy (see train_model.py).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import joblib

DEFAULT_MODEL_PATH = "models/taste_match_model.joblib"

# Order matters: this is the exact feature vector layout both training and
# inference must agree on.
FEATURE_NAMES = [
    "genre_similarity",
    "mood_similarity",
    "energy_closeness",
    "acoustic_bonus_flag",
]

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the trained model artifact cannot be loaded."""


@dataclass
class ScoreBreakdown:
    total: float
    contributions: Dict[str, float] = field(default_factory=dict)


class TasteMatchModel:
    """Fitted TF-IDF vectorizer + Ridge regressor for song/taste match scoring."""

    def __init__(self, vectorizer, regressor):
        self._vectorizer = vectorizer
        self._regressor = regressor

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "TasteMatchModel":
        try:
            bundle = joblib.load(path)
            model = cls(bundle["vectorizer"], bundle["regressor"])
        except Exception as exc:
            logger.error("Failed to load taste-match model from %s: %s", path, exc)
            raise ModelLoadError(f"Could not load model from {path}") from exc
        logger.info("Loaded trained taste-match model from %s", path)
        return model

    def _similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        vectors = self._vectorizer.transform([text_a, text_b])
        # Rows are L2-normalized by TfidfVectorizer, so the dot product is
        # exactly the cosine similarity.
        return float((vectors[0] @ vectors[1].T).toarray()[0, 0])

    def genre_similarity(self, a: str, b: str) -> float:
        return self._similarity(a, b)

    def mood_similarity(self, a: str, b: str) -> float:
        return self._similarity(a, b)

    def predict(self, features: List[float]) -> ScoreBreakdown:
        total = float(self._regressor.predict([features])[0])
        # Ridge is linear, so coef_i * feature_i is an exact per-feature
        # contribution to the prediction, not an approximation.
        contributions = {
            name: float(coef) * float(value)
            for name, coef, value in zip(FEATURE_NAMES, self._regressor.coef_, features)
        }
        return ScoreBreakdown(total=total, contributions=contributions)
