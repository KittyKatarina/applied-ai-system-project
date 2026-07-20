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

User profile: `genre=pop, mood=happy, energy=0.8`

```
Top Recommendations
========================================
1. Sunrise City by Neon Echo - Score: 4.44
   Because: genre match (+2.0), mood match (+1.0), energy similarity (+1.44)

2. Gym Hero by Max Pulse - Score: 3.11
   Because: genre match (+2.0), energy similarity (+1.11)

3. Rooftop Lights by Indigo Parade - Score: 2.38
   Because: mood match (+1.0), energy similarity (+1.38)

4. Streetlight Rhythm by Atlas Crew - Score: 1.44
   Because: energy similarity (+1.44)

5. Firelight Parade by Brass Meridian - Score: 1.38
   Because: energy similarity (+1.38)
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried

To stress-test the scoring logic, `src/main.py` runs six user profiles against the catalog: three distinct "normal" tastes, and three adversarial/edge-case profiles designed to see whether the scoring rule breaks or produces surprising results.

### Distinct taste profiles

**High-Energy Pop** — `{genre: pop, mood: happy, energy: 0.9}`

```
1. Sunrise City by Neon Echo - Score: 4.26
   Because: genre match (+2.0), mood match (+1.0), energy similarity (+1.26)

2. Gym Hero by Max Pulse - Score: 3.41
   Because: genre match (+2.0), energy similarity (+1.41)

3. Rooftop Lights by Indigo Parade - Score: 2.08
   Because: mood match (+1.0), energy similarity (+1.08)

4. Storm Runner by Voltline - Score: 1.47
   Because: energy similarity (+1.47)

5. Neon Afterglow by Pixel Harbor - Score: 1.44
   Because: energy similarity (+1.44)
```

**Chill Lofi** — `{genre: lofi, mood: chill, energy: 0.3}`

```
1. Library Rain by Paper Lanterns - Score: 4.35
   Because: genre match (+2.0), mood match (+1.0), energy similarity (+1.35)

2. Midnight Coding by LoRoom - Score: 4.14
   Because: genre match (+2.0), mood match (+1.0), energy similarity (+1.14)

3. Focus Flow by LoRoom - Score: 3.20
   Because: genre match (+2.0), energy similarity (+1.20)

4. Spacewalk Thoughts by Orbit Bloom - Score: 2.44
   Because: mood match (+1.0), energy similarity (+1.44)

5. Golden Hour Waltz by Marina Vale - Score: 1.47
   Because: energy similarity (+1.47)
```

**Why "Library Rain" ranked #1:** with the current weights in `recommender.py` (`+2.0` genre match, `+1.0` mood match, and `energy_bonus = 1.5 * max(0, 1 - abs(song.energy - target_energy) / 0.5)`), the maximum a song can score against this profile is `2.0 + 1.0 + 1.5 = 4.5` — a perfect `lofi`/`chill` song with `energy` exactly `0.3`. "Library Rain" (`lofi`, `chill`, `energy: 0.35`) hits both categorical matches and lands only `0.05` away from the target energy, earning `1.5 * (1 - 0.05/0.5) = 1.35` and a total of `4.35`, the closest any song gets to that ceiling. "Midnight Coding" (`lofi`, `chill`, `energy: 0.42`) matches the same two categories but sits `0.12` away from the target, so its energy bonus drops to `1.14`, putting it `0.21` points behind. "Focus Flow" only matches genre (its mood is `focused`, not `chill`), so it loses the full `+1.0` mood term and falls to third despite decent energy alignment. In other words, the ranking rewards songs that satisfy *all three* signals at once, and among ties on genre+mood, the energy-similarity term (worth up to `1.5`, more than either single categorical match) acts as the tiebreaker — which is exactly why the top two spots are both genuinely chill, low-energy lofi tracks rather than songs that merely match on one attribute.

**Deep Intense Rock** — `{genre: rock, mood: intense, energy: 0.9}`

```
1. Storm Runner by Voltline - Score: 4.47
   Because: genre match (+2.0), mood match (+1.0), energy similarity (+1.47)

2. Gym Hero by Max Pulse - Score: 2.41
   Because: mood match (+1.0), energy similarity (+1.41)

3. Neon Afterglow by Pixel Harbor - Score: 1.44
   Because: energy similarity (+1.44)

4. Firelight Parade by Brass Meridian - Score: 1.32
   Because: energy similarity (+1.32)

5. Sunrise City by Neon Echo - Score: 1.26
   Because: energy similarity (+1.26)
```

### Adversarial / edge-case profiles

**Conflicting Signals** — `{genre: rock, mood: chill, energy: 0.9}` (rock genre paired with a chill mood, plus a high target energy — internally contradictory taste)

```
1. Storm Runner by Voltline - Score: 3.47
   Because: genre match (+2.0), energy similarity (+1.47)

2. Neon Afterglow by Pixel Harbor - Score: 1.44
   Because: energy similarity (+1.44)

3. Gym Hero by Max Pulse - Score: 1.41
   Because: energy similarity (+1.41)

4. Firelight Parade by Brass Meridian - Score: 1.32
   Because: energy similarity (+1.32)

5. Sunrise City by Neon Echo - Score: 1.26
   Because: energy similarity (+1.26)
```

The scorer didn't break — it just fell back to the genre and energy terms since no song is both `rock` and `chill`. The winner (Storm Runner) is a rock/intense song, which is a reasonable resolution of the conflict but shows the mood preference was effectively ignored.

**Nonexistent Genre/Mood** — `{genre: vaporwave, mood: euphoric, energy: 0.5}` (values that don't exist anywhere in the catalog)

```
1. Winter Orchard by The Cedar State - Score: 1.47
   Because: energy similarity (+1.47)

2. Velvet Skyline by Solstice Choir - Score: 1.32
   Because: energy similarity (+1.32)

3. Blue Canyon by Willow Reed - Score: 1.26
   Because: energy similarity (+1.26)

4. Midnight Coding by LoRoom - Score: 1.26
   Because: energy similarity (+1.26)

5. Focus Flow by LoRoom - Score: 1.20
   Because: energy similarity (+1.20)
```

No errors or crashes — the scorer just degrades gracefully to energy-only matching since `==` comparisons against genre/mood strings that never appear simply never award points. The recommendations end up genre-agnostic, which is the expected (if unsatisfying) behavior for a taste the catalog can't represent.

**Out-of-Range Energy** — `{genre: pop, mood: happy, energy: 1.5}` (energy above the valid `0.0-1.0` range)

```
1. Sunrise City by Neon Echo - Score: 3.00
   Because: genre match (+2.0), mood match (+1.0)

2. Gym Hero by Max Pulse - Score: 2.00
   Because: genre match (+2.0)

3. Rooftop Lights by Indigo Parade - Score: 1.00
   Because: mood match (+1.0)

4. Midnight Coding by LoRoom - Score: 0.00
   Because:

5. Storm Runner by Voltline - Score: 0.00
   Because:
```

This surfaced a real edge case: since no song can have `energy` above `1.0`, `abs(song.energy - 1.5)` is always `>= 0.5`, so `energy_bonus` clamps to `0` for every song — the energy term silently disappears from the score entirely. The ranking still "works" (it falls back to genre/mood matches), but a user who mistakenly submits an out-of-range value gets no feedback that their energy preference was ignored. This suggests `score_song` should validate/clamp `target_energy` to `[0, 1]` rather than silently zeroing out the term.

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



