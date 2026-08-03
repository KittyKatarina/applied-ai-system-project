"""Tests whether the genre-skew filter bubble documented in model_card.md is a
data problem or a model problem, by running the same rock-taste profile
against the original catalog and an entirely different catalog with better
rock representation (data/songs_augmented_demo.csv, 5 rock songs among 22),
using the exact same trained model both times.

Run with: python -m src.demo_genre_balance
"""

from src.logging_setup import configure_logging
from src.recommender import load_songs, recommend_songs

ROCK_PROFILE = {"genre": "rock", "mood": "intense", "energy": 0.9}


def _print_recommendations(label, songs, k=5) -> None:
    print(f"\n{label}")
    print("=" * 40)
    for rank, (song, score, explanation) in enumerate(
        recommend_songs(ROCK_PROFILE, songs, k=k), start=1
    ):
        print(f"{rank}. {song['title']} by {song['artist']} ({song['genre']}) - Score: {score:.2f}")
        print(f"   Because: {explanation}")


def main() -> None:
    configure_logging()

    original_songs = load_songs("data/songs.csv")
    augmented_songs = load_songs("data/songs_augmented_demo.csv")

    _print_recommendations("BEFORE: original catalog (1 rock song)", original_songs)
    _print_recommendations("AFTER: augmented catalog (5 rock songs), same trained model", augmented_songs)


if __name__ == "__main__":
    main()
