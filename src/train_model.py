"""Trains the taste-match model used by src/recommender.py.

Run with: python -m src.train_model

The catalog has no real listening history to learn from, so training labels
are generated from a documented "teacher policy": the same weights as the
original hand-coded formula (+2.0 genre, +1.0 mood, +1.5 energy, +0.5
acoustic), but with two deliberate improvements the model is meant to learn:

1. Genre/mood match is a continuous TF-IDF cosine similarity over character
   n-grams instead of an exact-string boolean, so e.g. "pop" and "indie pop"
   share partial credit instead of getting zero.
2. The target energy used to compute energy-closeness is clamped to [0, 1]
   before comparison, so out-of-range inputs no longer silently zero out the
   whole energy term.

Gaussian noise is added to the labels so the regressor is genuinely fit
rather than a symbolic mirror of the formula.
"""

import logging
import os
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src.logging_setup import configure_logging
from src.model import DEFAULT_MODEL_PATH, FEATURE_NAMES
from src.recommender import load_songs

logger = logging.getLogger(__name__)

N_SAMPLES = 4000
RANDOM_SEED = 42


def _collect_vocab(songs: List[dict]) -> Tuple[List[str], List[str]]:
    genres = sorted({song["genre"] for song in songs})
    moods = sorted({song["mood"] for song in songs})
    return genres, moods


def fit_vectorizer(genres: List[str], moods: List[str]) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    vectorizer.fit(genres + moods)
    return vectorizer


def _similarity(vectorizer: TfidfVectorizer, a: str, b: str) -> float:
    vectors = vectorizer.transform([a, b])
    return float((vectors[0] @ vectors[1].T).toarray()[0, 0])


def generate_training_data(
    genres: List[str],
    moods: List[str],
    vectorizer: TfidfVectorizer,
    n_samples: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    X = np.zeros((n_samples, len(FEATURE_NAMES)), dtype=float)
    y = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        user_genre = rng.choice(genres)
        song_genre = rng.choice(genres)
        user_mood = rng.choice(moods)
        song_mood = rng.choice(moods)

        # Deliberately wider than the valid [0, 1] range so label generation
        # is forced to exercise the clamping behavior.
        raw_target_energy = rng.uniform(-0.5, 1.5)
        song_energy = rng.uniform(0.0, 1.0)
        song_acousticness = rng.uniform(0.0, 1.0)
        likes_acoustic = rng.random() < 0.5

        genre_sim = _similarity(vectorizer, user_genre, song_genre)
        mood_sim = _similarity(vectorizer, user_mood, song_mood)
        clamped_energy = min(1.0, max(0.0, raw_target_energy))
        energy_closeness = max(0.0, 1 - abs(song_energy - clamped_energy) / 0.5)
        acoustic_flag = 1.0 if likes_acoustic and song_acousticness >= 0.6 else 0.0

        X[i] = [genre_sim, mood_sim, energy_closeness, acoustic_flag]
        noise = rng.normal(0, 0.05)
        y[i] = 2.0 * genre_sim + 1.0 * mood_sim + 1.5 * energy_closeness + 0.5 * acoustic_flag + noise

    return X, y


def fit_regressor(X: np.ndarray, y: np.ndarray) -> Ridge:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    regressor = Ridge(alpha=1.0)
    regressor.fit(X_train, y_train)

    val_predictions = regressor.predict(X_val)
    mae = float(np.mean(np.abs(val_predictions - y_val)))

    logger.info("Trained on %d samples, validated on %d", len(X_train), len(X_val))
    logger.info("Validation MAE: %.4f", mae)
    logger.info(
        "Learned coefficients: %s",
        dict(zip(FEATURE_NAMES, regressor.coef_.round(3).tolist())),
    )
    logger.info("Learned intercept: %.4f", regressor.intercept_)

    return regressor


def main() -> None:
    configure_logging()

    songs = load_songs("data/songs.csv")
    genres, moods = _collect_vocab(songs)
    logger.info("Collected %d genres and %d moods from catalog", len(genres), len(moods))

    vectorizer = fit_vectorizer(genres, moods)
    X, y = generate_training_data(genres, moods, vectorizer)
    regressor = fit_regressor(X, y)

    os.makedirs(os.path.dirname(DEFAULT_MODEL_PATH), exist_ok=True)
    import joblib

    joblib.dump({"vectorizer": vectorizer, "regressor": regressor}, DEFAULT_MODEL_PATH)
    logger.info("Saved trained model to %s", DEFAULT_MODEL_PATH)


if __name__ == "__main__":
    main()
