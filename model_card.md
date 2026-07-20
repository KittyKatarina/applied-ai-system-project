# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **VibeFinder 1.0**  

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  

**Genre skew creates a filter bubble for niche-taste users.** Of the 18 songs in `data/songs.csv`, 13 distinct genres (72%) — including `rock`, `classical`, `hip hop`, `country`, `disco`, and `reggaeton` — have exactly one song, while `lofi` has three and `pop` has two. Because `score_song` in `src/recommender.py` awards its largest bonus (`+2.0`) only on an exact `genre` match, a user whose `favorite_genre` is one of those underrepresented genres can never get more than one strong recommendation — every other slot in their top-5 falls back to unrelated genres that merely happen to share their target energy, whereas a `lofi` or `pop` fan gets a genuinely diverse, on-genre top 5. The system therefore isn't biased toward any single genre's *content*, but it structurally favors whichever genres happen to be overrepresented in the catalog, which in a real deployment would compound over time (well-served users engage more, generating more data for their genre, further crowding out niche tastes). The strict `==` comparison also fragments closely related genres — a `pop` fan and an `indie pop` fan share no genre-match credit at all despite likely overlapping taste — so the bubble is drawn along arbitrary label boundaries rather than actual musical similarity. A fix would weight partial/adjacent genre matches and rebalance the catalog (or the scoring) so recommendation quality doesn't depend on how common a listener's favorite genre happens to be in the training data.

---

## 7. Evaluation  

I tested six user profiles through `src/main.py` (see the [README's "Experiments You Tried"](README.md#experiments-you-tried) section for full terminal output): three distinct taste profiles (**High-Energy Pop**, **Chill Lofi**, **Deep Intense Rock**) and three adversarial/edge-case profiles designed to try to break the scoring logic (**Conflicting Signals**, **Nonexistent Genre/Mood**, **Out-of-Range Energy**).

For the three normal profiles, I looked for whether the top result was a strong, believable genre+mood+energy match — in all three cases it was (e.g., "Sunrise City" for High-Energy Pop, "Library Rain" for Chill Lofi, "Storm Runner" for Deep Intense Rock), which confirmed the basic scoring recipe works as intended. For the adversarial profiles, I looked for crashes, nonsensical rankings, or silent failures rather than "correctness" in the usual sense.

**What surprised me:** the categorical preferences (genre/mood) failed *gracefully* no matter how I broke them — conflicting or nonexistent values just cost the song those points, never an error. The numeric preference (energy) did not fail gracefully — an out-of-range `energy: 1.5` silently zeroed out the entire energy term for every song with no warning, which was the one place the system quietly did the wrong thing instead of visibly degrading.

### Pairwise comparisons

- **High-Energy Pop vs. Chill Lofi** — these sit at opposite ends of the energy spectrum (target `0.9` vs. `0.3`) and picked completely disjoint top songs (Sunrise City/Gym Hero vs. Library Rain/Midnight Coding). This makes sense: the energy-similarity term is a symmetric penalty around the target, so pushing the target from one extreme to the other flips which half of the catalog scores well, with almost no overlap in the top 5.
- **High-Energy Pop vs. Deep Intense Rock** — both target high energy (`0.9`), so both pull in the same *loud* songs (Gym Hero, Storm Runner, Neon Afterglow) as filler even when they don't match genre. The difference is entirely which song reaches #1: Pop's own genre match ("Sunrise City") wins for Pop, Rock's own genre match ("Storm Runner") wins for Rock — energy sets the *pool*, genre decides the *winner*.
- **Chill Lofi vs. Deep Intense Rock** — the near-total opposite of the previous pair: low energy + relaxed mood vs. high energy + intense mood produced completely non-overlapping top 5s. This confirms the scorer is actually sensitive to energy and mood, not just returning the same "generically popular" songs regardless of input.
- **Deep Intense Rock vs. Conflicting Signals** — same genre (`rock`) and same energy target (`0.9`), only the mood flips from `intense` (matches Storm Runner) to `chill` (matches nothing rock). Both rank "Storm Runner" #1, but Deep Intense Rock's score is a full point higher (`4.47` vs. `3.47`) — exactly the size of the mood bonus it lost. This is the cleanest evidence that mood is a clean, isolated `+1.0` term that doesn't otherwise disturb the ranking logic.
- **Chill Lofi vs. Nonexistent Genre/Mood** — both target a similar low-to-mid energy (`0.3` vs `0.5`), but Chill Lofi's genre/mood actually exist in the catalog while Nonexistent Genre/Mood's don't. Chill Lofi's top picks score `4.0+` on real genre+mood+energy matches; Nonexistent Genre/Mood's top picks max out around `1.5`, entirely from energy similarity. The comparison isolates how much of a "good" score comes from categorical matches (up to `3.0`) versus energy alone (up to `1.5`) — categorical matches are worth twice as much when available.
- **High-Energy Pop vs. Out-of-Range Energy** — identical genre/mood (`pop`/`happy`), but energy target `0.9` (valid) vs. `1.5` (invalid). The top 3 songs are literally the same three tracks in the same order for both, but Out-of-Range Energy's scores are 1.0–1.5 points lower across the board because the energy term contributes nothing. This pair is what exposed the silent-zeroing bug: an invalid input doesn't change *which* songs are recommended (genre/mood still work), which makes the missing energy signal easy to miss unless you're comparing scores side by side like this.

---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  
