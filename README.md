# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

This program simulates a **content-based** music recommender: it represents each song as a set of attributes (genre, mood, energy, tempo, valence, danceability, acousticness) and represents a listener as a `UserProfile` of preferences (favorite genre, favorite mood, target energy, whether they like acoustic tracks). The `Recommender` scores every song by how closely its attributes match the user's profile, ranks the results, and returns the top `k` matches with an explanation for each pick — mirroring the "attribute-matching" half of how real platforms like Spotify recommend music (the other half, collaborative filtering based on other listeners' behavior, is out of scope here since the simulation only models a single user's taste).

---

## How The System Works

Explain your design in plain language.

Some prompts to answer:

- What features does each `Song` use in your system
  - For example: genre, mood, energy, tempo
- What information does your `UserProfile` store
- How does your `Recommender` compute a score for each song
- How do you choose which songs to recommend

You can include a simple diagram or bullet list if helpful.

Real platforms like Spotify use two approaches: comparing you to *other users* (collaborative filtering), or comparing songs to *your own past taste* (content-based filtering). This simulation only has one user to work with, so it's content-based — it just compares each song's attributes to what that one user says they like.

**`Song` features used:** `genre`, `mood`, `energy`, `tempo_bpm`, `valence`, `danceability`, and `acousticness`

**`UserProfile` stores:** `favorite_genre`, `favorite_mood`, `target_energy`, and `likes_acoustic`

**Algorithm Recipe:** each song is scored by how closely it matches the user's stated taste profile. The final recipe is:

- **+2.0 points** if the song's `genre` matches the user's `favorite_genre`
- **+1.0 point** if the song's `mood` matches the user's `favorite_mood`
- **Energy similarity bonus:** add points based on how close the song's `energy` is to the user's `target_energy`
  - Example formula: `energy_bonus = 1.5 * max(0, 1 - abs(song.energy - user.target_energy) / 0.5)`
- **Acoustic bonus:** if the user likes acoustic tracks and the song has high `acousticness`, add a small bonus such as **+0.5**

The total score is:

```text
score = genre_match + mood_match + energy_bonus + acoustic_bonus
```

After all songs are scored, the recommender sorts them from highest to lowest score and returns the top `k` results.

**Bias note:** this system may over-prioritize genre and under-value songs that are excellent matches for the user's mood or energy but happen to use a different genre. It also relies on a very simple profile, so it may miss more nuanced preferences.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

Paste a sample of your recommender's output here as a text block so a reader can see what it produces:

```
# e.g.:
# User profile: genre=indie, mood=chill, energy=low
# Recommendations:
#   1. ...
#   2. ...
#   3. ...
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



