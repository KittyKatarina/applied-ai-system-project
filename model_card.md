# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**FindingVibes 1.0**

---

## 2. Intended Use  

This tool suggests songs for one listener based on their taste.

You tell it your favorite genre, favorite mood, target energy, and if you like acoustic songs.

It picks the 5 songs from the catalog that match your taste best.

It's built for a classroom project. It's a simulation, not a real product.

It assumes you can describe your taste in simple words, like "pop" or "happy."

**Don't use this for real users.** It only knows 18 songs, so it can't give real people real recommendations. Don't use it to make decisions about people's music taste, either — it's just a scoring demo.

---

## 3. How the Model Works  

Every song has traits: genre, mood, energy, tempo, valence, danceability, and how acoustic it sounds.

You describe your taste with four things: favorite genre, favorite mood, target energy, and whether you like acoustic music.

Scoring is done by **a small trained model** (`TasteMatchModel` in `src/model.py`), not just hand-picked constants:

- A **TF-IDF vectorizer** (fit on the catalog's genre/mood labels, using character n-grams) turns genre and mood into continuous similarity scores instead of exact-match booleans. This is what lets "pop" and "indie pop" share partial credit, instead of one small text difference losing you the entire +2 point bonus.
- A **Ridge regression model** takes four features — genre similarity, mood similarity, energy closeness, and an acoustic-bonus flag — and predicts a match score. It was trained (`src/train_model.py`) on 4,000 synthetic examples built from a documented "teacher policy": the same weights as the original hand-coded formula (2.0 / 1.0 / 1.5 / 0.5), but with continuous similarity instead of exact matches, small random noise, and — importantly — the target energy **clamped to `[0, 1]` before comparison**, so an out-of-range input like `1.5` no longer silently zeroes out the whole energy term.
- Because Ridge is a linear model, each feature's contribution to a song's score can be read directly off its learned coefficients — that's what produces the "Because: ..." explanation for each recommendation.

The top 5 songs become your recommendations, with a short reason attached to each one, generated from the model's own feature contributions.

If the trained model file is missing or fails to load, the system logs a warning and falls back to the original hand-coded formula, so a bad model artifact never crashes the app.

We didn't invent new rules from scratch — the model is trained to approximate the starter logic's scoring recipe, with the genre/mood exact-match and energy-clamping limitations fixed.

---

## 4. Data  

The catalog has **18 songs** in `data/songs.csv`.

Each song has 9 pieces of info: id, title, artist, genre, mood, energy, tempo, valence, danceability, and acousticness.

There are **15 different genres**, but most only show up once. Lofi has 3 songs, pop has 2, and every other genre (rock, classical, hip hop, country, disco, reggaeton, and more) has just 1.

We didn't add or remove any songs — this is the starter dataset as-is.

The dataset is missing a lot of real-world music info: no lyrics, no artist popularity, no release year, and no actual listening history from real people.

---

## 5. Strengths  

It works best when your taste lines up with a well-represented genre, like lofi or pop — you get a full top 5 of genuinely similar songs.

The #1 recommendation is almost always a strong match on genre, mood, *and* energy at the same time, not just one of the three.

The energy math correctly rewards songs that are close to your target, even when genre or mood don't match.

It doesn't crash on weird or contradictory input — it just quietly gives you its best guess.

---

## 6. Limitations and Bias 

**Genre skew creates a filter bubble.** 13 of the 15 genres in the catalog only have 1 song each. If your favorite genre is one of those, you can only ever get 1 strong genre match — the rest of your top 5 is filler that just happens to share your energy level. Lofi and pop fans get better, more varied recommendations simply because their genre has more songs in the catalog, not because the system "understands" their taste any better.

The system also treats genre as an exact text match. "Pop" and "indie pop" share zero credit, even though they're clearly similar styles. So the bubble is drawn along arbitrary label lines, not real musical similarity.

Net result: how good your recommendations are depends more on how common your taste is in the data than on how well the system understands music.

---

## 7. Evaluation  

I felt the AI was a very strong tool and could do so many things. On top of this, it's able to answer many questions that you may have, all making it very desirable when working. The AI can be flawed in the fact that it has to be watched so closely, but the drawback is worth having such a strong tool and coding assistant.

We tested 6 fake users: 3 normal profiles (**High-Energy Pop**, **Chill Lofi**, **Deep Intense Rock**) and 3 "break it on purpose" profiles (**conflicting preferences**, a **made-up genre/mood**, and an **energy value above 1.0**).

For the normal profiles, the #1 song always looked right — a real match on genre, mood, and energy.

For the broken profiles, nothing crashed. The system just quietly skipped the points it couldn't award.

**The most surprising result:** an energy value of `1.5` (above the max of `1.0`) killed the *entire* energy bonus for every song, with zero warning. Genre and mood still worked fine, so this bug was easy to miss unless you looked closely at the scores.

We also compared profiles in pairs. A few examples:

- **High-Energy Pop vs. Deep Intense Rock:** both target high energy, so both pull in the same "loud" songs — but each profile's own genre still wins the #1 spot.
- **Chill Lofi vs. Deep Intense Rock:** these sit at opposite energy extremes and picked completely different songs, which shows the system really is sensitive to energy.
- **Deep Intense Rock vs. Conflicting Signals:** same genre and energy target, but the mood flips from a match to a non-match — the #1 song stays the same, but its score drops by exactly 1 point (the size of the mood bonus).

---

## 8. Future Work  

~~Let genres get partial credit for being similar (like "pop" and "indie pop"), instead of only exact matches.~~ **Done** — the trained model's TF-IDF similarity now gives "pop" and "indie pop" partial credit instead of zero.

~~Warn or fix the input when someone enters an energy value outside `0` to `1`, instead of silently zeroing out the energy score.~~ **Done** — out-of-range `target_energy` is now clamped to `[0, 1]` (with a logged warning) before scoring, in both the model path and the formula fallback.

Add more songs, especially to genres that only have 1 song right now, so recommendations are fair no matter what genre someone likes.

**New limitation introduced by the trained model:** its training labels are synthetic — generated from a documented "teacher policy" (see Section 3) that mimics the original hand-coded formula, not from real listener feedback. So the model has learned to reproduce a human-authored scoring rule slightly more flexibly; it hasn't learned anything about music taste that wasn't already encoded in that rule. A natural next step would be training on real listening/skip data instead of a synthetic policy.

---

## 9. Testing Summary

> **In short:** 6/6 automated tests pass, including a reliability check where the trained model agreed with the original scoring formula's #1 pick in 49/50 (98%) of randomly sampled taste profiles — the 1 disagreement was a near-miss genre/mood case, which is exactly the behavior the model was built to improve on. The energy-clamping and model-load-failure guardrails were both verified by deliberately triggering them, not just by inspection.

**What worked:**

- The automated test suite (`pytest`, `tests/test_recommender.py`) has 6 tests: ranking order, non-empty explanations, partial credit for a similar-but-not-identical genre ("indie pop" vs. "pop"), energy values outside `[0, 1]` no longer zeroing the score, the system's behavior when the trained model is unavailable (simulated by monkeypatching the model loader), and a **reliability/agreement check** — 50 randomly generated taste profiles (fixed random seed for reproducibility) are scored twice, once with the trained model and once with the model forced off, comparing the #1 recommendation each time. They agreed on 49/50 (98%) profiles; the 1 disagreement involved a near-miss genre where the model correctly gave partial credit and the formula gave none — confirming the model diverges from the original rule exactly where it's supposed to, not randomly. All 6 tests pass.
- End-to-end manual verification: running `python -m src.train_model` produced sane training logs (sample count, a validation MAE around `0.04`, and learned coefficients that closely matched the original hand-picked weights — good evidence the model actually learned the intended policy instead of something unrelated). Running `python -m src.main` afterward produced correct, sensible rankings for all 6 test profiles.
- A manual "break the fallback on purpose" drill: temporarily renaming `models/taste_match_model.joblib` and rerunning `python -m src.main` confirmed the system logs a clear error, logs a fallback warning, and still returns correct rankings using the original formula — no crash, no silent failure.

**What didn't work (at first):**

- `tests/test_recommender.py` originally only exercised the OOP `Recommender` class, which was a `# TODO` stub returning placeholder values. Those tests were technically passing while validating nothing real — a stub can pass a shallow test just as easily as a correct implementation can. This was easy to miss without reading past the test file into the implementation it was calling.
- There was evidence (orphaned compiled bytecode with no matching source files) of an earlier, unfinished attempt at a full LLM/RAG integration that never got committed. That's a sign the original approach was scoped much bigger than the assignment actually needed, and it had to be abandoned rather than fixed.

**What we learned:**

- A passing test suite isn't proof of a working system if the tests were written against a stub — it's worth periodically asking "what would make this test pass even if the feature were broken?"
- Testing the failure/fallback path mattered as much as testing the happy path. The clearest evidence that the safety net actually worked wasn't a green checkmark, it was deliberately breaking the model file and watching the system recover instead of crash.
- Watching the actual console output change (old formula's scores vs. the new model's scores, for the exact same profiles) was more convincing evidence that the fixes worked than the unit tests alone — automated tests and manual end-to-end checks caught different kinds of problems.

---

## 10. Personal Reflection  

I learned that its very important to watch the AI carefully. I had an iteration where the AI was trying to import an entire LLM but my goal was nowhere near that complex. I had to rewind the changes and rewrite my prompt better. Other than that, its a very strong tool to be used very carefully. You cannot rely on it every step of the way, but you should definitely find a way to take advantage of such a powerful tool.

Building this showed me that a recommender is really just a scoring rule with a friendly interface.

The surprising part was how easily bias creeps in — not from bad code, but from an uneven dataset. The math was fair; the data wasn't.

This changed how I think about real apps like Spotify: even a completely "neutral" formula can quietly favor whatever music is most common in its training data, and the people with less-common taste get worse recommendations through no fault of the algorithm itself.

**On problem-solving specifically:** the most useful move in this whole project wasn't writing code, it was noticing when the *scope* of a solution didn't match the *size* of the problem — catching that an LLM/RAG rebuild was overkill for "give partial credit to similar genre strings" and re-scoping down to a small local model instead. The second most useful move was breaking the problem into a sequence of checkpoints I could verify independently (train the model → check its logs → run the CLI → diff the output against the old behavior → break the fallback on purpose → confirm recovery) rather than writing everything at once and hoping it worked. Bugs like the energy-clamping issue weren't found by staring at the formula — they were found by deliberately trying to break the system with adversarial inputs and reading the actual output closely. That's probably the biggest transferable lesson: most real bugs surface from testing the edges of a system, not its middle.
