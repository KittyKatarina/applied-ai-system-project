import random

from src.recommender import Song, UserProfile, Recommender, score_song, load_songs, recommend_songs
import src.recommender as recommender_module

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_partial_credit_for_similar_genre():
    user_prefs = {"genre": "pop", "mood": "focused", "energy": 0.5}
    indie_pop_song = {
        "id": 1, "title": "t", "artist": "a", "genre": "indie pop", "mood": "chill",
        "energy": 0.5, "tempo_bpm": 100, "valence": 0.5, "danceability": 0.5, "acousticness": 0.1,
    }
    jazz_song = {
        "id": 2, "title": "t", "artist": "a", "genre": "jazz", "mood": "chill",
        "energy": 0.5, "tempo_bpm": 100, "valence": 0.5, "danceability": 0.5, "acousticness": 0.1,
    }

    indie_pop_score, _ = score_song(user_prefs, indie_pop_song)
    jazz_score, _ = score_song(user_prefs, jazz_song)

    assert indie_pop_score > jazz_score


def test_energy_out_of_range_does_not_zero_score():
    user_prefs = {"genre": "nonexistent", "mood": "nonexistent", "energy": 1.5}
    high_energy_song = {
        "id": 1, "title": "t", "artist": "a", "genre": "x", "mood": "y",
        "energy": 0.95, "tempo_bpm": 100, "valence": 0.5, "danceability": 0.5, "acousticness": 0.1,
    }

    score, reasons = score_song(user_prefs, high_energy_song)

    assert score > 0
    assert reasons


def test_recommend_falls_back_without_model(monkeypatch):
    monkeypatch.setattr(recommender_module, "_get_model", lambda: None)

    rec = make_small_recommender()
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )

    results = rec.recommend(user, k=2)

    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_model_top_pick_agreement_with_formula_baseline(monkeypatch, capsys):
    """Reliability check: measures how often the trained model's #1 pick
    agrees with the original hand-coded formula's #1 pick across a battery of
    profiles, on the same (clamped) inputs. Disagreement is expected on
    profiles involving near-miss genres/moods (that's the intended fix), so
    this asserts a floor rather than requiring exact agreement."""
    songs = load_songs("data/songs.csv")
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})

    rng = random.Random(42)
    profiles = [
        {
            "genre": rng.choice(genres),
            "mood": rng.choice(moods),
            "energy": rng.uniform(-0.2, 1.2),
            "likes_acoustic": rng.random() < 0.5,
        }
        for _ in range(50)
    ]

    model_top_picks = [recommend_songs(p, songs, k=1)[0][0]["id"] for p in profiles]

    monkeypatch.setattr(recommender_module, "_get_model", lambda: None)
    formula_top_picks = [recommend_songs(p, songs, k=1)[0][0]["id"] for p in profiles]

    agreement = sum(a == b for a, b in zip(model_top_picks, formula_top_picks)) / len(profiles)
    print(f"\nModel/formula top-1 agreement across {len(profiles)} profiles: {agreement:.0%}")

    assert agreement >= 0.6
