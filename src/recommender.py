import csv
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from src.model import ModelLoadError, TasteMatchModel

logger = logging.getLogger(__name__)

_model: Optional[TasteMatchModel] = None
_model_load_attempted = False


def _get_model() -> Optional[TasteMatchModel]:
    """Loads the trained taste-match model once per process.

    Returns None (and caches that) if loading ever fails, so callers can fall
    back to the hand-coded formula without retrying on every call.
    """
    global _model, _model_load_attempted
    if not _model_load_attempted:
        _model_load_attempted = True
        try:
            _model = TasteMatchModel.load()
        except ModelLoadError:
            logger.warning("Falling back to formula-based scoring for this session")
            _model = None
    return _model


def _clamp_energy(target_energy: Optional[float]) -> Tuple[Optional[float], bool]:
    """Clamps target_energy into [0, 1]. Returns (clamped_value, was_clamped)."""
    if target_energy is None:
        return None, False
    clamped = min(1.0, max(0.0, target_energy))
    return clamped, clamped != target_energy


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

def _song_to_dict(song: Song) -> Dict:
    return {
        "id": song.id,
        "title": song.title,
        "artist": song.artist,
        "genre": song.genre,
        "mood": song.mood,
        "energy": song.energy,
        "tempo_bpm": song.tempo_bpm,
        "valence": song.valence,
        "danceability": song.danceability,
        "acousticness": song.acousticness,
    }


def _user_to_dict(user: UserProfile) -> Dict:
    return {
        "genre": user.favorite_genre,
        "mood": user.favorite_mood,
        "energy": user.target_energy,
        "likes_acoustic": user.likes_acoustic,
    }


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py

    A thin adapter over the function-based load_songs/score_song/recommend_songs
    path so scoring logic (including the trained model and its fallback) lives
    in exactly one place.
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        song_dicts = [_song_to_dict(song) for song in self.songs]
        ranked = recommend_songs(_user_to_dict(user), song_dicts, k=k)
        songs_by_id = {song.id: song for song in self.songs}
        return [songs_by_id[song_dict["id"]] for song_dict, _score, _reason in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        score, reasons = score_song(_user_to_dict(user), _song_to_dict(song))
        if not reasons:
            return f"No matching attributes (score: {score:.2f})"
        return ", ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """Loads songs from a CSV file into a list of dicts with numeric fields converted."""
    print(f"Loading songs from {csv_path}...")
    int_fields = {"id"}
    float_fields = {"energy", "tempo_bpm", "valence", "danceability", "acousticness"}

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            songs.append(row)

    for song in songs:
        print(song)

    return songs

def _score_with_formula(
    favorite_genre: Optional[str],
    favorite_mood: Optional[str],
    target_energy: Optional[float],
    likes_acoustic: bool,
    song: Dict,
) -> Tuple[float, List[str]]:
    """Original hand-coded scoring formula. Used as a fallback when the
    trained model is unavailable. target_energy is expected to already be
    clamped to [0, 1] by the caller."""
    score = 0.0
    reasons = []

    if favorite_genre is not None and song["genre"] == favorite_genre:
        score += 2.0
        reasons.append("genre match (+2.0)")

    if favorite_mood is not None and song["mood"] == favorite_mood:
        score += 1.0
        reasons.append("mood match (+1.0)")

    if target_energy is not None:
        energy_bonus = 1.5 * max(0, 1 - abs(song["energy"] - target_energy) / 0.5)
        if energy_bonus > 0:
            score += energy_bonus
            reasons.append(f"energy similarity (+{energy_bonus:.2f})")

    if likes_acoustic and song["acousticness"] >= 0.6:
        score += 0.5
        reasons.append("acoustic bonus (+0.5)")

    return score, reasons


def _score_with_model(
    model: TasteMatchModel,
    favorite_genre: Optional[str],
    favorite_mood: Optional[str],
    target_energy: Optional[float],
    likes_acoustic: bool,
    song: Dict,
) -> Tuple[float, List[str]]:
    """Scores a song using the trained taste-match model. target_energy is
    expected to already be clamped to [0, 1] by the caller."""
    genre_sim = (
        model.genre_similarity(favorite_genre, song["genre"])
        if favorite_genre is not None
        else 0.0
    )
    mood_sim = (
        model.mood_similarity(favorite_mood, song["mood"])
        if favorite_mood is not None
        else 0.0
    )
    energy_closeness = (
        max(0.0, 1 - abs(song["energy"] - target_energy) / 0.5)
        if target_energy is not None
        else 0.0
    )
    acoustic_flag = 1.0 if likes_acoustic and song["acousticness"] >= 0.6 else 0.0

    breakdown = model.predict([genre_sim, mood_sim, energy_closeness, acoustic_flag])

    reasons = []
    if breakdown.contributions.get("genre_similarity", 0) > 0.01:
        reasons.append(f"genre similarity {genre_sim:.2f} (~+{breakdown.contributions['genre_similarity']:.2f})")
    if breakdown.contributions.get("mood_similarity", 0) > 0.01:
        reasons.append(f"mood similarity {mood_sim:.2f} (~+{breakdown.contributions['mood_similarity']:.2f})")
    if breakdown.contributions.get("energy_closeness", 0) > 0.01:
        reasons.append(f"energy closeness (~+{breakdown.contributions['energy_closeness']:.2f})")
    if breakdown.contributions.get("acoustic_bonus_flag", 0) > 0.01:
        reasons.append(f"acoustic bonus (~+{breakdown.contributions['acoustic_bonus_flag']:.2f})")

    return breakdown.total, reasons


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Scores a song against user preferences and returns (score, reasons).

    Uses the trained taste-match model when available, falling back to the
    original hand-coded formula if the model can't be loaded or fails to
    score a given song.
    """
    favorite_genre = user_prefs.get("genre", user_prefs.get("favorite_genre"))
    favorite_mood = user_prefs.get("mood", user_prefs.get("favorite_mood"))
    raw_target_energy = user_prefs.get("energy", user_prefs.get("target_energy"))
    likes_acoustic = user_prefs.get("likes_acoustic", False)

    target_energy, was_clamped = _clamp_energy(raw_target_energy)
    if was_clamped:
        logger.warning(
            "target_energy %.2f out of range [0, 1]; clamped to %.2f",
            raw_target_energy,
            target_energy,
        )

    model = _get_model()
    if model is not None:
        try:
            return _score_with_model(
                model, favorite_genre, favorite_mood, target_energy, likes_acoustic, song
            )
        except Exception:
            logger.exception(
                "Model scoring failed for song id=%s; falling back to formula", song.get("id")
            )

    return _score_with_formula(
        favorite_genre, favorite_mood, target_energy, likes_acoustic, song
    )

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Scores every song against user preferences and returns the top k, ranked highest to lowest."""
    scored = [(song, *score_song(user_prefs, song)) for song in songs]
    ranked = sorted(scored, key=lambda entry: entry[1], reverse=True)
    return [(song, score, ", ".join(reasons)) for song, score, reasons in ranked[:k]]
