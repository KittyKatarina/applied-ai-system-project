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

   Pass `--songs <path>` to score against a different catalog CSV instead of the default `data/songs.csv` (e.g. `--songs data/songs_augmented_demo.csv`, the augmented catalog used in the [genre-balance demo](#before-after-does-adding-more-catalog-songs-fix-the-genre-filter-bubble) below). Pass `--model real` to score with the real-feedback-trained model instead of the default synthetic one (see [Optional: Train on Real Feedback](#optional-train-on-real-feedback-instead-of-synthetic-labels)).

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

**Why this proves it works, not just seems to:** these tests assert on real outputs computed by `recommender.py`/`model.py` — ranking order, non-empty explanations, score comparisons — not fixed strings the code happens to print. To confirm this, I mutated the fallback formula (disabled the exact-genre-match bonus) and reran the suite: `test_model_top_pick_agreement_with_formula_baseline` failed immediately, proving it's actually exercising the scoring logic rather than passing regardless of behavior. Reverting the mutation brought the suite back to 6/6. Most tests run against the committed trained model by default; only the fallback and agreement tests force the formula path, which is what lets the agreement test double as a reliability check — it verifies the model's top pick agrees with the original hand-coded formula on 49/50 (98%) of randomly sampled profiles, catching wholesale disagreement while still tolerating the near-miss cases the model was built to improve on.

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

## Before/After: Does Adding More Catalog Songs Fix the Genre Filter Bubble?

`model_card.md` documents a real bias: 13 of 15 genres have only 1 song, so a fan of an under-represented genre (e.g. rock) gets one real match and four compromises. This tests whether adding more songs to that genre actually fixes it — **using the exact same trained model, with no retraining** — by running the same rock-taste profile against the original catalog and a completely different 22-song catalog with better rock representation (`data/songs_augmented_demo.csv`, 5 rock songs among all-new titles/artists).

```bash
$ python -m src.demo_genre_balance
```

```
BEFORE: original catalog (1 rock song)
========================================
1. Storm Runner by Voltline (rock) - Score: 4.45
   Because: genre similarity 1.00 (~+1.99), mood similarity 1.00 (~+0.99), energy closeness (~+1.47)
2. Gym Hero by Max Pulse (pop) - Score: 2.40
   Because: mood similarity 1.00 (~+0.99), energy closeness (~+1.41)
3. Neon Afterglow by Pixel Harbor (electronic) - Score: 1.59
   Because: genre similarity 0.06 (~+0.11), mood similarity 0.04 (~+0.04), energy closeness (~+1.44)
4. Firelight Parade by Brass Meridian (disco) - Score: 1.34
   Because: mood similarity 0.03 (~+0.03), energy closeness (~+1.32)
5. Sunrise City by Neon Echo (pop) - Score: 1.26
   Because: energy closeness (~+1.26)

AFTER: augmented catalog (5 rock songs), same trained model
========================================
1. Wreckage Anthem by Cinder Wolf (rock) - Score: 4.42
   Because: genre similarity 1.00 (~+1.99), mood similarity 1.00 (~+0.99), energy closeness (~+1.44)
2. Rust Belt Riot by Steel Choir (rock) - Score: 4.42
   Because: genre similarity 1.00 (~+1.99), mood similarity 1.00 (~+0.99), energy closeness (~+1.44)
3. Highway Fracture by Amber Static (rock) - Score: 3.39
   Because: genre similarity 1.00 (~+1.99), mood similarity 0.06 (~+0.06), energy closeness (~+1.35)
4. Broken Radio by The Hollow Kings (rock) - Score: 3.19
   Because: genre similarity 1.00 (~+1.99), energy closeness (~+1.20)
5. Splinter Drive by Vantage Point (rock) - Score: 3.01
   Because: genre similarity 1.00 (~+1.99), energy closeness (~+1.02)
```

**Before**, only 1 of the top 5 songs is actually rock — the rest are filler that merely shares energy or mood. **After** swapping in a completely different 22-song catalog that happens to have 5 rock songs instead of 1, all 5 top picks are rock, each correctly ranked by how well its mood and energy also match (an "intense" near-perfect match at #1–2, then progressively looser mood matches at #3–5). Nothing about `src/model.py` or `models/taste_match_model.joblib` changed between these two runs — the TF-IDF vectorizer and Ridge regressor are identical, and none of the songs are shared between the two catalogs. This confirms the genre-skew bias documented in `model_card.md` is a **data** problem, not a **model** problem: the trained model already generalizes correctly to rock songs it's never seen before, it just can't recommend rock songs that don't exist in whichever catalog it's given.

---

## Optional: Train on Real Feedback Instead of Synthetic Labels

The default model (`models/taste_match_model.joblib`) is trained on synthetic labels — a documented "teacher policy" that mimics the original formula (see Design Decisions below), since the catalog has no real listening history. This adds a second, parallel pipeline that trains on **real human liked/skipped judgments** instead, so the two can be compared directly.

**How it works, without changing any of the existing scoring code:**

1. `data/real_feedback.csv` lists all 18 catalog songs with a blank `liked` column, for one fixed persona (`REAL_USER_PROFILE` in `src/train_model_real.py`, defaults to `genre=pop, mood=happy, energy=0.85`). Fill in `1` (liked) or `0` (skipped) for each row — you're acting as that listener.
2. Train a second model on those real labels:

   ```bash
   python -m src.train_model_real
   ```

   This reuses the exact same `fit_vectorizer`/`fit_regressor` functions from `train_model.py` — only the data feeding them is different — and saves to a separate file, `models/taste_match_model_real.joblib`, so the default synthetic model is never touched.
3. Compare the two models on the same persona and catalog:

   ```bash
   python -m src.demo_real_vs_synthetic
   ```

The only change to `src/recommender.py` needed to make this comparison possible was adding an optional `model=` parameter to `score_song`/`recommend_songs` (defaulting to `None`, meaning "use the normal cached model as before") — everything else, including the fallback-to-formula guardrail, is unchanged. `src/main.py` also accepts `--model synthetic` (default) or `--model real` to run the whole 6-profile battery against either trained model.

### Actual result, using real labels

```bash
$ python -m src.train_model_real
```

```
2026-07-24 03:47:28,569 [INFO] __main__: Loaded 18 real listen/skip labels (50% liked)
2026-07-24 03:47:28,571 [INFO] src.train_model: Trained on 14 samples, validated on 4
2026-07-24 03:47:28,571 [INFO] src.train_model: Validation MAE: 0.4990
2026-07-24 03:47:28,571 [INFO] src.train_model: Learned coefficients: {'genre_similarity': -0.008, 'mood_similarity': 0.347, 'energy_closeness': -0.358, 'acoustic_bonus_flag': 0.0}
2026-07-24 03:47:28,571 [INFO] src.train_model: Learned intercept: 0.5219
2026-07-24 03:47:28,575 [INFO] __main__: Saved real-feedback-trained model to models/taste_match_model_real.joblib
```

```bash
$ python -m src.demo_real_vs_synthetic
```

```
SYNTHETIC: trained on the documented teacher policy
========================================
1. Sunrise City by Neon Echo (pop) - Score: 4.39
   Because: genre similarity 1.00 (~+1.99), mood similarity 1.00 (~+0.99), energy closeness (~+1.41)
2. Rooftop Lights by Indigo Parade (indie pop) - Score: 3.47
   Because: genre similarity 0.63 (~+1.25), mood similarity 1.00 (~+0.99), energy closeness (~+1.23)
3. Gym Hero by Max Pulse (pop) - Score: 3.25
   Because: genre similarity 1.00 (~+1.99), energy closeness (~+1.26)

REAL: trained on your actual liked/skipped labels
========================================
1. Rooftop Lights by Indigo Parade (indie pop) - Score: 0.57
   Because: mood similarity 1.00 (~+0.35)
2. Sunrise City by Neon Echo (pop) - Score: 0.52
   Because: mood similarity 1.00 (~+0.35)
3. Library Rain by Paper Lanterns (lofi) - Score: 0.52
   Because:
```

**This is genuinely revealing, not just a formality.** The synthetic model's coefficients (`genre_similarity: 1.99, mood_similarity: 0.99, energy_closeness: up to 1.5`) were designed to mirror the original formula, so of course genre/mood/energy all matter roughly the way the assignment's rubric assumed. The *real* labels tell a different story for this one real listener: `genre_similarity` came out at essentially `0` (genre didn't predict what they'd actually pick), `energy_closeness` came out *negative* (`-0.358` — being close to the stated `0.85` energy target didn't make a song more likely to be liked; several liked songs were actually low-energy), and `mood_similarity` was the only feature that meaningfully mattered (`0.347`). The real model's validation error (`MAE 0.499`, on a 0/1 target) is also far worse than the synthetic model's (`MAE 0.042`), which is expected — 18 labels for one person is nowhere near enough data to fit 4 features reliably, and it shows in the many tied scores among songs the model has no real signal to distinguish. **The takeaway `model_card.md` now documents:** the synthetic "teacher policy" encodes what the *assignment's formula assumed* mattered; it does not necessarily encode what a *real listener* actually responds to, and this experiment is direct, if small-sample, evidence of that gap.

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
