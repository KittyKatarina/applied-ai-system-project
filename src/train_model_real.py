"""Trains a second taste-match model on REAL human feedback instead of the
synthetic "teacher policy" labels train_model.py uses. Reuses the exact same
vectorizer-fitting and regressor-fitting code, just with a different (X, y).

The feedback in data/real_feedback.csv is for one fixed persona (see
REAL_USER_PROFILE below) — mark `liked` as 1 if you'd actually pick that song
for that taste, 0 if you'd skip it, then run:

    python -m src.train_model_real
"""

import csv
import logging
import os

import joblib

from src.logging_setup import configure_logging
from src.recommender import load_songs
from src.train_model import _collect_vocab, _similarity, fit_regressor, fit_vectorizer

logger = logging.getLogger(__name__)

REAL_MODEL_PATH = "models/taste_match_model_real.joblib"
REAL_FEEDBACK_PATH = "data/real_feedback.csv"

# The persona data/real_feedback.csv's "liked" column was labeled against.
REAL_USER_PROFILE = {"genre": "pop", "mood": "happy", "energy": 0.85, "likes_acoustic": False}


def load_real_feedback(songs, vectorizer):
    songs_by_id = {song["id"]: song for song in songs}
    X, y = [], []

    with open(REAL_FEEDBACK_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["liked"].strip():
                raise ValueError(
                    f"data/real_feedback.csv row for song_id={row['song_id']} has no 'liked' "
                    "value yet — fill in 1 or 0 for every row before training."
                )
            song = songs_by_id[int(row["song_id"])]

            genre_sim = _similarity(vectorizer, REAL_USER_PROFILE["genre"], song["genre"])
            mood_sim = _similarity(vectorizer, REAL_USER_PROFILE["mood"], song["mood"])
            energy_closeness = max(0.0, 1 - abs(song["energy"] - REAL_USER_PROFILE["energy"]) / 0.5)
            acoustic_flag = (
                1.0 if REAL_USER_PROFILE["likes_acoustic"] and song["acousticness"] >= 0.6 else 0.0
            )

            X.append([genre_sim, mood_sim, energy_closeness, acoustic_flag])
            y.append(float(row["liked"]))

    return X, y


def main() -> None:
    configure_logging()

    songs = load_songs("data/songs.csv")
    genres, moods = _collect_vocab(songs)
    vectorizer = fit_vectorizer(genres, moods)

    X, y = load_real_feedback(songs, vectorizer)
    logger.info("Loaded %d real listen/skip labels (%.0f%% liked)", len(y), 100 * sum(y) / len(y))

    regressor = fit_regressor(X, y)

    os.makedirs(os.path.dirname(REAL_MODEL_PATH), exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "regressor": regressor}, REAL_MODEL_PATH)
    logger.info("Saved real-feedback-trained model to %s", REAL_MODEL_PATH)


if __name__ == "__main__":
    main()
