"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.logging_setup import configure_logging
from src.recommender import load_songs, recommend_songs


def main() -> None:
    configure_logging()
    songs = load_songs("data/songs.csv")

    user_profiles = {
        # Distinct, "normal" taste profiles
        "High-Energy Pop": {"genre": "pop", "mood": "happy", "energy": 0.9},
        "Chill Lofi": {"genre": "lofi", "mood": "chill", "energy": 0.3},
        "Deep Intense Rock": {"genre": "rock", "mood": "intense", "energy": 0.9},

        # Adversarial / edge-case profiles
        # Genre and mood pull in opposite directions (rock genre, chill mood)
        # while energy is cranked high - conflicting signals within one profile.
        "Conflicting Signals (Rock genre, Chill mood, High energy)": {
            "genre": "rock", "mood": "chill", "energy": 0.9
        },
        # Genre/mood that don't exist anywhere in the catalog - should fall back
        # entirely to the energy-similarity term.
        "Nonexistent Genre/Mood": {
            "genre": "vaporwave", "mood": "euphoric", "energy": 0.5
        },
        # Energy value outside the expected 0-1 range.
        "Out-of-Range Energy": {"genre": "pop", "mood": "happy", "energy": 1.5},
    }

    for profile_name, user_prefs in user_profiles.items():
        recommendations = recommend_songs(user_prefs, songs, k=5)

        print(f"\nProfile: {profile_name}  ({user_prefs})")
        print("Top Recommendations")
        print("=" * 40)
        for rank, (song, score, explanation) in enumerate(recommendations, start=1):
            print(f"{rank}. {song['title']} by {song['artist']} - Score: {score:.2f}")
            print(f"   Because: {explanation}")
            print()


if __name__ == "__main__":
    main()
