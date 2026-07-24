"""Compares the default synthetic-trained model against the model trained on
real listen/skip feedback (see train_model_real.py), for the same persona and
catalog. Run python -m src.train_model_real first.

Run with: python -m src.demo_real_vs_synthetic
"""

from src.logging_setup import configure_logging
from src.model import TasteMatchModel
from src.recommender import load_songs, recommend_songs
from src.train_model_real import REAL_MODEL_PATH, REAL_USER_PROFILE


def _print_recommendations(label, songs, model, k=5) -> None:
    print(f"\n{label}")
    print("=" * 40)
    for rank, (song, score, explanation) in enumerate(
        recommend_songs(REAL_USER_PROFILE, songs, k=k, model=model), start=1
    ):
        print(f"{rank}. {song['title']} by {song['artist']} ({song['genre']}) - Score: {score:.2f}")
        print(f"   Because: {explanation}")


def main() -> None:
    configure_logging()

    songs = load_songs("data/songs.csv")
    synthetic_model = TasteMatchModel.load()
    real_model = TasteMatchModel.load(REAL_MODEL_PATH)

    _print_recommendations("SYNTHETIC: trained on the documented teacher policy", songs, synthetic_model)
    _print_recommendations("REAL: trained on your actual liked/skipped labels", songs, real_model)


if __name__ == "__main__":
    main()
