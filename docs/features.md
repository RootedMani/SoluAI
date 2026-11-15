# The tools around the model (v1.1)

Version 1.0 was one page: type a molecule, get a prediction and an explanation. Version 1.1 adds
tools that make the same trained model more useful in a chemistry class. **The model itself did not
change** (same file, same test scores). Nothing here retrains anything.

| Page | What it does | Where the code is |
|---|---|---|
| **Predict** | Prediction + explanation, now with a drawing editor, lab value, classic formula, everyday units, "make it more soluble" and a printable report | `explain.py`, `lab.py`, `web/static/js/predict.js` |
| **Compare** | 2 to 4 molecules side by side (salts allowed) | `web/static/js/compare.js` |
| **Trends** | Solubility along seven homologous series | `lab.series` |
| **Salts** | Solubility rules for ionic compounds + a mixing lab | `ionic.py` |
| **Quiz** | "Which one dissolves better?" against the AI | `lab.quiz` |
| **Batch** | Up to 200 molecules at once, CSV in and out | `lab.batch` |
| **Data / Model / Chemistry** | As before, plus a "kitchen test" on the Model page | `lab.household` |

Every page works in **English and Persian** (button in the top bar) and **without internet**.

---

## 1. Salts: the solubility rules (`ionic.py`)

**The problem.** The model was trained only on single organic molecules. NaCl is not a molecule: it
is a crystal of ions, and it dissolves by splitting into Na⁺ and Cl⁻. Before, SoluAI just said
"out of scope".

**The fix.** For salts SoluAI now applies the **solubility rules from the textbook**. This is a
rule engine written by hand, *not* machine learning, and every answer says so.

How it works, step by step:

1. **Read the input.** 17 cations × 12 anions = 204 salts. For each pair the code builds the
   formula by balancing charges (Al³⁺ and SO₄²⁻ → Al₂(SO₄)₃) and stores every way to write it:
   `NaCl`, `nacl`, `sodium chloride`, `سدیم کلرید`, and everyday names such as `baking soda`.
2. **Apply the first rule that fits** (function `verdict`):
   1. Group 1 metals and NH₄⁺ → soluble.
   2. Nitrates, acetates, hydrogen carbonates → soluble.
   3. Chlorides, bromides, iodides → soluble, except Ag⁺ and Pb²⁺.
   4. Fluorides → soluble, except Mg²⁺, Ca²⁺, Sr²⁺, Ba²⁺, Pb²⁺.
   5. Sulfates → soluble, except Ba²⁺, Sr²⁺, Pb²⁺ (Ca²⁺, Ag⁺ slightly).
   6. Hydroxides → insoluble, except group 1, NH₄⁺, Ba²⁺ (Ca²⁺, Sr²⁺ slightly).
   7. Carbonates, phosphates → insoluble, except group 1, NH₄⁺.
   8. Sulfides → insoluble, except group 1, NH₄⁺, Ca²⁺, Sr²⁺, Ba²⁺.
3. **Write the dissociation equation**, e.g. `Ca(OH)₂(s) ⇌ Ca²⁺(aq) + 2 OH⁻(aq)`.
4. **Check against the lab.** 50 salts have a measured value (g per 100 mL at 20 °C, from the
   Wikipedia "Solubility table", which collects handbook data). A test makes sure no rule
   contradicts a measured value we ship.

"Soluble" means more than about 1 g per 100 mL, "slightly soluble" 0.1 to 1 g, "insoluble" less
than 0.1 g. A few salts (PbCl₂ at 1.08 g, silver acetate at 1.05 g) sit right on a boundary;
textbooks disagree about them too.

**Mixing lab.** Pick two soluble salts. The code swaps the partners (double displacement), asks the
rules whether either new pair is insoluble, balances the equation by trying small whole numbers,
and writes the net ionic equation: `Ag⁺(aq) + Cl⁻(aq) → AgCl(s)`.

**Limits.** No temperature, no pH, no complex ions (AgCl does dissolve in ammonia), no common-ion
effect. Twelve pairs that react with water instead of dissolving (Al₂S₃, Fe₂(CO₃)₃…) are marked.

## 2. Three answers to the same question

Next to SoluAI's prediction the Predict page now shows:

* **the lab value**, if the molecule is one of the 8,754 the model learned from (`Predictor.measured`);
* **Delaney's equation** from 2004, the classic hand-made formula (`baselines.delaney_one`).

This lets you see in one glance whether the AI adds anything over a formula a chemist wrote by
hand. The page also says honestly that a molecule with a lab value was in the training data, so
the model has seen its answer. The fair exam is still the hidden test set on the Model page.

## 3. Everyday units

`logS = −2.3` means little to most people, so the page also shows:

* grams that dissolve in **one glass of water** (250 mL);
* **how much water you need for 1 g** ("about 3,300 L, about 22 bathtubs");
* the word a pharmacist would use (**pharmacopoeia terms**: very soluble, freely soluble, soluble,
  sparingly soluble, slightly soluble, very slightly soluble, practically insoluble), defined by
  how many mL of water dissolve 1 g (`explain.usp_term`).

## 4. Easier input

* **225 molecules work offline**, in nine categories, with autocomplete. About 90 of them also
  have a **Persian name** (`شکر`, `کافئین`, `ویتامین ث`).
* **Draw the molecule** with the JSME editor (bundled, BSD license). "Edit this molecule" opens the
  current structure in the editor, so you can add a group and predict again.

## 5. Make it more soluble (`lab.suggest`)

Drug designers do this every day: change the molecule a little so it dissolves better.
SoluAI tries 11 simple edits at every possible position (add –OH or –NH₂ to a ring, turn –CH₃ into
–CH₂OH or –COOH, swap a ring CH for N, remove a halogen, shorten a chain, turn an ester into an
acid…), predicts each new molecule, and shows the best result of each kind with the reason.

The edits are written as *reaction SMARTS* (find-and-replace patterns for molecules), e.g.
`[cH:1]>>[c:1][OH]` means "an aromatic carbon with an H gets an –OH".

**Limits.** These are ideas from a model. They ignore whether the new molecule can be made, whether
it is stable, and whether it still does its job.

## 6. Trends: homologous series (`lab.series`)

Seven families (alkanes, alcohols, carboxylic acids, amines, aldehydes, ketones, chloroalkanes)
from 1 to 12 carbons. Lines are predictions, diamonds are lab values. The slope from 4 carbons
upward gives "each extra CH₂ makes it about N times less soluble". The model's slopes (3.0 to 3.9×)
are close to the measured ones (3.5 to 5.6×), usually a little flatter.

## 7. Quiz (`lab.quiz`)

Two molecules whose **measured** solubilities differ; you and the model both answer. Four levels
(`lab.LEVELS`):

| Level | Molecules | Smallest gap |
|---|---|---|
| Easy | the familiar list | 2 log units (100×) |
| Medium | the familiar list | 1 log unit (10×) |
| Hard | any small molecule in the data | 1 log unit |
| Expert | look-alike pairs (fingerprint similarity ≥ 0.45) | 0.7 log units (5×) |

Your **opponent** is the model chosen in the picker. Every round also records every other model's
answer, so the final screen shows a leaderboard. Honest note shown to players: the machine-learning
models saw these molecules in training and win about 99% of rounds at every level; Delaney's formula
is right about 85% of the time, so it is the fair opponent.

## 7b. Choose a model (`zoo.py`)

A picker on Predict, Compare, Trends, Batch and Quiz (remembered in the browser) chooses which model
answers. The Predict and Compare pages also show **what every model predicts**, with each model's
held-out test R², and say how much the models agree. The Model page lists all models with their
scores. Models: SoluAI v2 consensus (default), SoluAI v1.0, LightGBM, robust LightGBM (Huber),
CatBoost, a similarity SVR, a neural network, and Delaney's formula. The atom colours and SHAP bars
always explain SoluAI v1.0.

## 8. Batch (`lab.batch`)

Paste a list or load a CSV, get a table, download a CSV. Each row has logS, the 90% range, g/L,
class, pharmacopoeia term, lab value, classic formula, and whether the molecule is familiar to the
model. Salts in the list are answered by the rules. Batch mode never uses the internet.

## 9. Persian and right-to-left

The language is chosen in `boot.js` before the page paints (so nothing flashes). Static text is
written twice in the HTML (`data-l="en"` / `data-l="fa"`) and CSS hides the other one. Text made by
JavaScript uses a one-line helper, `T("Predict", "پیش‌بینی")`. The explanation sentences from the
server are written in both languages in `explain.py` and `groups.py`. Numbers, formulas, SMILES and
charts always stay left-to-right.

## 10. Report and offline use

* **Save as PDF / print** uses the browser's print dialog with a print stylesheet: one clean page
  with the structure, numbers, explanation and nearest neighbours.
* Fonts (Inter, JetBrains Mono, Vazirmatn), Plotly and the editor are all in `web/static/vendor/`.
  `python run.py` starts the site and opens the browser. Only unusual names need the internet (PubChem).

## API

All of this is also a JSON API (interactive docs at `/docs`):

```
GET  /api/models                                                 every model + test scores
POST /api/predict        {"query": "caffeine", "lang": "en", "model": "v2"}   molecule or salt
POST /api/compare        {"queries": ["ethanol", "hexane"]}
POST /api/suggest        {"query": "toluene"}
POST /api/batch          {"queries": ["CCO", "NaCl", ...]}       up to 200
GET  /api/series         GET /api/quiz?n=8&level=expert&model=v1  GET /api/household
Every prediction endpoint takes "model": v2 (default), v1, v2-lgbm, v2-huber, v2-cat, v2-svr, v2-mlp, delaney.
GET  /api/salts          GET /api/salts/Ag/Cl                    POST /api/salts/mix
```
