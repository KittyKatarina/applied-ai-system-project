# 🎵 Music Recommender — with a Trained Taste-Match Model

> **Originally:** *Module 3 Music Recommender Simulation* — a classroom assignment (AI110) to build and explain a small content-based music recommender: represent songs and a listener's taste profile as data, design a scoring rule that turns that data into ranked recommendations, and reflect on where the system gets things right or wrong. The original version scored songs entirely with hand-picked constants (exact genre/mood match, a fixed energy-similarity formula) — no learning involved.

This repository extends that assignment with a **trained, locally-run machine learning model** that replaces the hand-coded scoring rule as the system's primary decision-maker, while keeping the original formula available as a safety-net fallback.

---

## Summary

This project is a **content-based music recommender**: given a listener's stated taste (favorite genre, favorite mood, target energy, and whether they like acoustic tracks), it scores every song in a small catalog and returns the top matches with a human-readable explanation for each pick.

What makes it more than a lookup table is `src/model.py` — a small **scikit-learn model, fit specifically for this scoring task**, that turns genre/mood into continuous text-similarity scores instead of brittle exact-string matches, and is trained to never let an out-of-range input silently zero out part of a user's preferences. That matters for two reasons: (1) it demonstrates that a "fine-tuned/specialized model" doesn't have to mean a large hosted LLM — a small, fully local, purpose-fit model can meaningfully improve a system's behavior — and (2) it fixes two real bugs the original hand-coded formula had (documented below), rather than just wrapping the same logic in a fancier interface.

Everything runs **offline** — no API keys, no network calls, no cloud inference. Training data is `pandas`/`numpy`-free synthetic data generated from a documented policy (see [Design Decisions](#design-decisions--trade-offs)), the model artifact is a few KB, and the whole pipeline runs in under a second on a laptop.

---

## Architecture Overview

See [`diagrams/architecture.mmd`](diagrams/architecture.mmd) for the full Mermaid diagram. It has three parts:

1. **Offline training** (`python -m src.train_model`, run ahead of time / whenever `data/songs.csv` changes): reads the song catalog, generates thousands of synthetic `(features → match score)` examples from a documented scoring policy, fits a TF-IDF vectorizer + Ridge regression model on them, and writes the result to `models/taste_match_model.joblib`. This artifact is committed to the repo, so nobody running the app needs to train anything themselves.
2. **Runtime recommendation flow** (`python -m src.main`): loads the song catalog and a set of user taste profiles, clamps any out-of-range `target_energy` input (logging a warning if it had to), and scores every song. Scoring tries the trained model first; if the model file is missing or errors, it falls back to the original hand-coded formula automatically, and the failure is logged rather than hidden. Ranked results with explanations are printed to the console.
3. **Verification & human-in-the-loop**: automated `pytest` tests check the scoring logic directly (including the model's fallback behavior), and a human reviews the console output across both "normal" and deliberately adversarial taste profiles, writing up findings in [`model_card.md`](model_card.md).

A shared `src/logging_setup.py` module feeds console logging into both the training and runtime paths (model loads, clamped inputs, scoring fallbacks, and errors are all logged, never silently swallowed).

---

## Setup Instructions

1. **Clone the repo and enter the project folder.**

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac/Linux
   .venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   This installs `pandas`, `pytest`, `streamlit`, `scikit-learn`, `joblib`, and `numpy`.

4. **Run the app:**

   ```bash
   python -m src.main
   ```

   The trained model artifact (`models/taste_match_model.joblib`) is already committed, so this works immediately — no training step required. It prints ranked recommendations for six user profiles (three typical tastes, three deliberately adversarial edge cases) straight to the console, with logging interleaved.

5. **(Optional) Retrain the model**, e.g. after editing `data/songs.csv`:

   ```bash
   python -m src.train_model
   ```

   This overwrites `models/taste_match_model.joblib` and logs the training sample count, validation error, and learned coefficients.

6. **Run the tests:**

   ```bash
   pytest
   ```

---

## Reproducible Execution Evidence

Everything below is copy-pasted output from actually running these exact commands in this repo (not hand-written) — see [Sample Interactions](#sample-interactions) below for annotated input/output pairs.

### Training the model

```bash
$ python -m src.train_model
```

```
2026-07-24 03:03:21,187 [INFO] __main__: Collected 15 genres and 14 moods from catalog
2026-07-24 03:03:25,396 [INFO] __main__: Trained on 3200 samples, validated on 800
2026-07-24 03:03:25,396 [INFO] __main__: Validation MAE: 0.0416
2026-07-24 03:03:25,396 [INFO] __main__: Learned coefficients: {'genre_similarity': 1.988, 'mood_similarity': 0.995, 'energy_closeness': 1.496, 'acoustic_bonus_flag': 0.497}
2026-07-24 03:03:25,396 [INFO] __main__: Learned intercept: 0.0024
2026-07-24 03:03:25,400 [INFO] __main__: Saved trained model to models/taste_match_model.joblib
```

### Running the recommender (input → output)

```bash
$ python -m src.main
```

```
2026-07-24 03:04:13,416 [INFO] src.model: Loaded trained taste-match model from models/taste_match_model.joblib

Profile: High-Energy Pop  ({'genre': 'pop', 'mood': 'happy', 'energy': 0.9})
Top Recommendations
========================================
1. Sunrise City by Neon Echo - Score: 4.24
   Because: genre similarity 1.00 (~+1.99), mood similarity 1.00 (~+0.99), energy closeness (~+1.26)

2. Gym Hero by Max Pulse - Score: 3.40
   Because: genre similarity 1.00 (~+1.99), energy closeness (~+1.41)

3. Rooftop Lights by Indigo Parade - Score: 3.32
   Because: genre similarity 0.63 (~+1.25), mood similarity 1.00 (~+0.99), energy closeness (~+1.08)
```

### Guardrail evidence: out-of-range input gets clamped, not silently dropped

Same run, later profile — note the `WARNING` lines logged *before* the recommendations print, one per song scored:

```
2026-07-24 03:04:13,525 [WARNING] src.recommender: target_energy 1.50 out of range [0, 1]; clamped to 1.00
2026-07-24 03:04:13,526 [WARNING] src.recommender: target_energy 1.50 out of range [0, 1]; clamped to 1.00
... (18 lines total, one per song) ...

Profile: Out-of-Range Energy  ({'genre': 'pop', 'mood': 'happy', 'energy': 1.5})
Top Recommendations
========================================
1. Sunrise City by Neon Echo - Score: 3.94
   Because: genre similarity 1.00 (~+1.99), mood similarity 1.00 (~+0.99), energy closeness (~+0.96)

2. Gym Hero by Max Pulse - Score: 3.28
   Because: genre similarity 1.00 (~+1.99), energy closeness (~+1.29)
```

### Guardrail evidence: model file missing → automatic fallback, no crash

Simulated by temporarily renaming `models/taste_match_model.joblib`:

```
2026-07-24 03:05:01,926 [ERROR] src.model: Failed to load taste-match model from models/taste_match_model.joblib: [Errno 2] No such file or directory: 'models/taste_match_model.joblib'
2026-07-24 03:05:01,926 [WARNING] src.recommender: Falling back to formula-based scoring for this session

Profile: High-Energy Pop  ({'genre': 'pop', 'mood': 'happy', 'energy': 0.9})
Top Recommendations
========================================
```

### Reliability check: automated test suite

```bash
$ python -m pytest -v -s
```

```
collecting ... collected 6 items

tests/test_recommender.py::test_recommend_returns_songs_sorted_by_score PASSED
tests/test_recommender.py::test_explain_recommendation_returns_non_empty_string PASSED
tests/test_recommender.py::test_partial_credit_for_similar_genre PASSED
tests/test_recommender.py::test_energy_out_of_range_does_not_zero_score PASSED
tests/test_recommender.py::test_recommend_falls_back_without_model PASSED
tests/test_recommender.py::test_model_top_pick_agreement_with_formula_baseline
Model/formula top-1 agreement across 50 profiles: 98%
PASSED

============================== 6 passed in 2.25s ==============================
```

---

## Sample Interactions

All output below is copy-pasted from an actual `python -m src.main` run.

### 1. A normal taste profile, with the model giving partial credit for a similar genre

**Input:** `{genre: "pop", mood: "happy", energy: 0.9}`

```
1. Sunrise City by Neon Echo - Score: 4.24
   Because: genre similarity 1.00 (~+1.99), mood similarity 1.00 (~+0.99), energy closeness (~+1.26)

2. Gym Hero by Max Pulse - Score: 3.40
   Because: genre similarity 1.00 (~+1.99), energy closeness (~+1.41)

3. Rooftop Lights by Indigo Parade - Score: 3.32
   Because: genre similarity 0.63 (~+1.25), mood similarity 1.00 (~+0.99), energy closeness (~+1.08)
```

"Rooftop Lights" is tagged `genre: "indie pop"`, not `"pop"`. Under the *original* exact-string-match formula this song got **zero** genre credit despite being an obviously related style. The trained model's TF-IDF similarity scores `"pop"` vs. `"indie pop"` at `0.63`, giving it a meaningful `~+1.25` — which is why it now lands at #3 instead of being buried under exact-match-only songs.

### 2. A distinct, different taste profile — the model generalizes correctly

**Input:** `{genre: "lofi", mood: "chill", energy: 0.3}`

```
1. Library Rain by Paper Lanterns - Score: 4.33
   Because: genre similarity 1.00 (~+1.99), mood similarity 1.00 (~+0.99), energy closeness (~+1.35)

2. Midnight Coding by LoRoom - Score: 4.12
   Because: genre similarity 1.00 (~+1.99), mood similarity 1.00 (~+0.99), energy closeness (~+1.14)

3. Focus Flow by LoRoom - Score: 3.19
   Because: genre similarity 1.00 (~+1.99), energy closeness (~+1.20)
```

A completely different taste (chill, low-energy lofi instead of high-energy pop) surfaces a completely different top 5, and ranks the two genuinely chill lofi tracks above one that only matches on genre — showing the model is sensitive to all three signals at once, not just genre.

### 3. An adversarial edge case — energy value outside the valid range

**Input:** `{genre: "pop", mood: "happy", energy: 1.5}` — `1.5` is outside the valid `[0, 1]` energy range.

```
1. Sunrise City by Neon Echo - Score: 3.94
   Because: genre similarity 1.00 (~+1.99), mood similarity 1.00 (~+0.99), energy closeness (~+0.96)

2. Gym Hero by Max Pulse - Score: 3.28
   Because: genre similarity 1.00 (~+1.99), energy closeness (~+1.29)
```

Console log for this run: `WARNING src.recommender: target_energy 1.50 out of range [0, 1]; clamped to 1.00`.

Under the *original* formula, any `target_energy` above `1.0` made `abs(song.energy - target_energy)` always ≥ `0.5`, silently zeroing the entire energy term for every song with no warning. Here, the input is clamped to `1.0` with a logged warning, and the energy term still contributes meaningfully (`~+0.96` for a `0.82`-energy song) instead of vanishing.

---

## Testing & Reliability

**In short: 6/6 automated tests pass, including a reliability check where the trained model's #1 pick agreed with the original scoring formula's #1 pick in 49/50 (98%) of randomly sampled taste profiles** — the one disagreement was a near-miss genre case where the model correctly gave partial credit and the formula gave none, which is the exact behavior the model was built to add.

- **Automated tests** (`pytest`, `tests/test_recommender.py`, 6 tests): ranking order, non-empty explanations, partial credit for a similar-but-not-identical genre, energy values outside `[0, 1]` no longer zeroing the score, correct behavior when the trained model is unavailable, and the model/formula agreement check described above.
- **Logging & guardrails** (`src/logging_setup.py`, `src/recommender.py`): every model load, clamped out-of-range input, and scoring fallback is logged with a level (`INFO`/`WARNING`/`ERROR`) rather than failing silently — verified by deliberately renaming the model artifact and confirming the system logged the failure and still produced correct output via the formula fallback.
- **Human review**: console output was manually compared across 6 profiles (3 typical, 3 adversarial) before and after the model was introduced, to confirm the two bugs it targets (exact-match genre bias, unclamped energy) were actually fixed rather than just relocated.

Full "what worked / what didn't / what we learned" write-up is in [`model_card.md`](model_card.md#9-testing-summary).

---

## Design Decisions & Trade-offs

**Why a small local scikit-learn model instead of an LLM/RAG pipeline.** The assignment's "fine-tuned/specialized model" requirement doesn't require a large hosted model — it asks for something trained/adjusted for a *specific task*. A hosted LLM call would need an API key and network access, adding a dependency this project doesn't need for what is fundamentally a small numeric scoring problem. A model that's a few KB, trains in under a second, and runs with zero network calls is a better fit for the actual task, and it's honest about being a genuine fit model rather than a thin wrapper around an API call.

**Why TF-IDF + Ridge regression specifically.** The two real weaknesses in the original formula were exact-string genre/mood matching (no partial credit for "pop" vs. "indie pop") and unclamped energy input. A character n-gram TF-IDF vectorizer solves the first problem directly — similar strings share substrings and get nonzero cosine similarity — without needing a hand-authored genre-similarity table that would need to be maintained forever. Ridge regression was chosen over a more expressive model (e.g. a random forest) specifically *because* it's linear: each feature's contribution to the final score can be read directly off the model's coefficients, which is what generates the "Because: ..." explanation for each recommendation. A more complex model would have made explanations much harder to produce faithfully.

**Why synthetic training data, and why that's disclosed as a limitation.** The catalog has no real listening history to learn from, so training labels come from a documented "teacher policy": the same weights as the original hand-coded formula (2.0 / 1.0 / 1.5 / 0.5), but with continuous similarity instead of boolean matches, clamped energy, and injected noise so the model is a genuine fit rather than a symbolic copy. This is disclosed plainly in `model_card.md` as a limitation — the model has learned to reproduce a human-authored rule more flexibly, not learned anything new about music taste. Training on real listening/skip data would be the natural next step.

**Why the trained artifact is committed instead of trained on first run.** `python -m src.main` should work immediately after `pip install -r requirements.txt`, with no hidden first-run latency or surprise file writes. The training script (`python -m src.train_model`) stays available and documented for anyone who wants to inspect or rerun it.

**Why the model has a fallback instead of being a hard dependency.** If the model file is ever missing, corrupted, or fails to unpickle (e.g. a scikit-learn version mismatch on someone else's machine), `src/recommender.py` catches the failure, logs it, and falls back to the original hand-coded formula rather than crashing. This trades a small amount of code complexity for the system never being fully broken by a model-loading issue — a deliberate reliability choice for a project meant to "run correctly and reproducibly."

**Why the previously-stubbed `Recommender` class was implemented as a thin adapter, not a rewrite.** The codebase had two parallel scoring implementations — a working function-based path and a stubbed object-oriented one that the test suite actually exercised. Rather than duplicate the scoring logic (and the trained-model integration) in both, the OOP `Recommender` class now just converts its dataclasses to dicts and calls the same `score_song`/`recommend_songs` functions the CLI uses. One scoring path, one source of truth, two interfaces on top of it.

---

## Limitations and Risks

- Small catalog (18 songs) — can't represent most of musical taste, and most genres have only one song, creating a filter-bubble effect for less-common tastes.
- No understanding of lyrics, artist popularity, release year, or actual listening history.
- Training labels for the trained model are synthetic (a documented teacher policy), not real listener feedback — see Design Decisions above.
- Still a single-user, content-based simulation — no collaborative filtering (comparing across multiple users), which is half of how real platforms like Spotify recommend music.

See [`model_card.md`](model_card.md) for the full bias, evaluation, and future-work writeup.

---

## Reflection

Read the full writeup in [`model_card.md`](model_card.md).

Building this showed that a recommender is really just a scoring rule wearing a friendly interface — every "recommendation" is a number, sorted — and that training a small model to produce that number doesn't change that fact, it just changes *how the number gets decided* and how gracefully it handles inputs the original rule's author didn't anticipate. Bias also doesn't require biased code: this catalog happens to have more lofi and pop songs than anything else, so those listeners get better, more varied recommendations for free, model or no model. That's made me a lot more skeptical of "neutral" algorithms in real apps — a well-fit, bug-free scoring model can still produce very uneven outcomes if it's trained on uneven data.
