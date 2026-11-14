# Model card: SoluAI v1.0.0

A "model card" is a short, honest fact sheet about a machine-learning model: what it is for,
what it was trained on, how good it is, and where it fails.

## Model details

| | |
|---|---|
| Name / version | SoluAI 1.0.0 (`models/soluai-v1.0.0/`) |
| Task | Regression: predict aqueous solubility **logS** (log₁₀ mol/L) at about 25 °C |
| Algorithm | LightGBM gradient-boosted trees (400 trees, 63 leaves, learning rate 0.02) |
| Input features | 14 RDKit descriptors + 1,024-bit Morgan fingerprint (radius 2) |
| Explanations | Exact SHAP values (LightGBM TreeSHAP), mapped to atoms |
| Uncertainty | 90% conformal interval (separate widths for familiar/unfamiliar molecules) + similarity-based applicability domain |
| Training data | 8,754 cleaned molecules from AqSolDB |
| Reproduce | `python -m soluai.train --config config.yaml` (seed 42, about 30 min on 4 CPU cores) |

## Intended use

* **Education:** showing how molecular structure relates to solubility, and how ML models
  are built, evaluated and explained.
* **Quick estimates** for small, neutral, drug-like or everyday organic molecules,
  when you want the right order of magnitude (e.g. "soluble or not?").

**Not intended for:** safety, environmental, medical, pharmaceutical or regulatory decisions;
formulation work; anything where an error of a factor of 10 matters.

## Data

* **Training:** AqSolDB (Sorkun et al., 2019, CC0). We removed salts and mixtures (1,098),
  compounds without carbon (91), compounds with metals or other non-organic elements (30), and
  2 unreadable SMILES. One duplicate pair that disagreed by more than 0.5 log units was dropped.
  That leaves 8,754 molecules (see `data/processed/cleaning_report.json`).
* **Second benchmark:** ESOL (Delaney, 2004), 1,116 molecules after cleaning.
  **Note:** 1,115 of these 1,116 molecules are also in AqSolDB, because AqSolDB was built partly
  from Delaney's data. ESOL is therefore used as a separate benchmark with its own splits. It is
  *not* an independent test of the AqSolDB model.
* Measured values come from many labs and methods. AqSolDB labels each value with a reliability
  group: G1 = a single source; G3/G5 = several sources that agree.

## Metrics (held-out test sets, never used for tuning)

| Dataset | Split | Delaney (2004) RMSE | **SoluAI-type model (LightGBM, desc+fp)** RMSE | MAE | R² |
|---|---|---|---|---|---|
| AqSolDB | random | 1.35 | **0.84** | 0.59 | 0.87 |
| AqSolDB | scaffold | 1.44 | **1.01** | 0.75 | 0.79 |
| ESOL | random | 1.07 | **0.61** | 0.43 | 0.90 |
| ESOL | scaffold | 1.24 | **0.83** | 0.64 | 0.84 |

All models and feature sets are in `results/metrics.md`.

**Uncertainty check (AqSolDB test sets):**

| | Random split | Scaffold split |
|---|---|---|
| 90% range actually covered | 88.6% | 86.6% (slightly too narrow) |
| Molecules flagged "outside experience" | 2.3% | 8.0% |
| RMSE familiar / unfamiliar | 0.83 / 0.98 | 1.00 / 1.10 |

The deployed model's range is **±1.49** log units for familiar molecules and **±1.59** for
unfamiliar ones. That is wide (a factor of about 30), but it is what the data honestly supports.
The similarity warning is only a **weak** signal: unfamiliar molecules have about 10–20% larger
errors, and only the very rare molecules below 0.2 similarity are much worse.

## Does it reason like a chemist?

Correlation between each feature's value and its SHAP push, over all training molecules:

| Feature | Correlation | Chemistry expects | OK? |
|---|---|---|---|
| logP | −0.92 | higher logP → less soluble | ✔ |
| Molecular weight | −0.83 | bigger → less soluble | ✔ |
| Aromatic proportion | −0.61 | flat aromatic → less soluble | ✔ |
| H-bond donors | +0.80 | more donors → more soluble | ✔ |
| Polar surface area | +0.33 | more polar → more soluble | ✔ |
| H-bond acceptors | −0.25 | more acceptors → more soluble | ✘ (probably a stand-in for size; see docs/chemistry.md §8) |

## Limitations

* **No temperature:** all data are near 25 °C.
* **No pH / ionisation:** acids and bases are predicted as the neutral molecule. Real solubility in
  acidic or basic water can be much higher.
* **No crystal information:** the model only sees the 2D structure. It cannot know the melting
  point, the crystal packing or the polymorph. This is the biggest source of error (see below).
* **Out of scope:** salts, metal complexes, inorganic substances, polymers, mixtures. The website
  refuses these with an explanation.
* **Reactive compounds:** substances that react with water (e.g. some silanes, acid chlorides) have
  measured "solubilities" that really describe a reaction product.
* **Label noise:** about 80% of the data come from a single source (G1). On test molecules
  confirmed by several sources (G3/G5), the random-split RMSE is only ≈ 0.53, compared with ≈ 0.88
  on single-source molecules. Part of the remaining "error" is noise in the measurements themselves.

## What is *not* the model (added in v1.1)

The website now has tools around the model. They do not change the model or its scores, but two of
them give answers that do **not** come from machine learning:

* **Salts** (NaCl, AgCl, CaCO₃…) are answered by a hand-written engine of 8 textbook solubility
  rules (`ionic.py`), checked against 50 measured values. Its limits: no temperature, pH,
  complex ions or common-ion effect, and borderline salts (PbCl₂, Ag₂SO₄, Ca(OH)₂) are fuzzy.
* **"Measured in the lab"** values are looked up in the training data, not predicted.

"Make it more soluble" does use the model, on molecules it invents by simple edits. Those
molecules can be outside the model's experience, may be impossible to make, and may lose the
property that made the original molecule useful. Treat them as ideas for discussion.

The quiz and the "kitchen test" use molecules from the training data, so the model's score there
is flattering. The honest numbers are the held-out test scores above.

## Failure cases (real examples from the test sets)

| Molecule | Measured | Predicted | What went wrong |
|---|---|---|---|
| Aminohydroxy-anthraquinone carbonate dye (`CCOC(=O)OCCOc1cc(O)c2c(c1N)C(=O)c1ccccc1C2=O`) | −9.2 | −3.8 | **Crystal blind spot.** Flat, hydrogen-bonded anthraquinone dyes pack extremely tightly (high melting point). The model sees polar groups and predicts "fairly soluble". Several other anthraquinone dyes fail in the same way. |
| Dechlorane plus (C₁₈H₁₂Cl₁₂) | −13.2 | −8.4 | **Extreme value.** Very few training molecules are this insoluble, and trees cannot predict beyond the range they have seen. The model gets the direction right but not the size. |
| 1,4-Bis(hydroxyethoxyethoxypropylamino)anthraquinone | +0.06 | −3.9 | The opposite case: long, flexible polyether chains make this dye unusually soluble, which the model underestimates. |
| Hexamethylcyclotrisiloxane | +0.65 | −2.9 | **Probably a data error.** Siloxanes are known to be very poorly soluble; the measured value in the dataset looks wrong. The model is probably closer to the truth. |
| tert-Butyl isobutyl ether | +0.89 | −2.5 | **Suspicious label.** An 8-carbon ether with no O–H group would be expected to be only slightly soluble, not "highly soluble". |
| Tris(isopropenyloxy)vinylsilane | +0.65 | −4.0 | **Reacts with water** (hydrolysis). The "solubility" describes the products, not the molecule. |
| Tetra(¹⁰B)boron carbide | −4.7 | −0.7 | **Should not be in the data:** an inorganic ceramic that passed our "has carbon" filter. |

Patterns:
1. Big errors usually come from **flat, polyaromatic, high-melting molecules** (dyes, anthraquinones,
   some steroids), which are predicted too soluble.
2. **Extreme values** (logS below −10) are pulled towards the middle.
3. A noticeable share of the "worst errors" are **doubtful measurements**, not model mistakes.

## Ethical and practical considerations

* Predictions are estimates. Do not use them to decide how to handle, dose or dispose of chemicals.
* The website shows a 90% range and an "outside the model's experience" warning so that users don't
  over-trust a single number.
* Name lookup sends the typed name to PubChem (a public NIH database). No other user data is stored.

## Comparison with published work

Published numbers are only roughly comparable, because papers clean the data differently (many keep
salts and inorganics, which are harder) and use different splits.

| Source | What they report | Status |
|---|---|---|
| Delaney (2004), *J. Chem. Inf. Comput. Sci.* 44:1000 | The ESOL equation, fitted on his own data | Our re-implementation with RDKit logP gives RMSE 1.07 on the ESOL random test set |
| Wu et al. (2018), MoleculeNet, *Chem. Sci.* 9:513 | ESOL random-split benchmarks; graph neural networks reported around RMSE 0.5–0.6 | ⚠ **Not checked against the paper** (the paper was not reachable while building this project). Check Table/Figure for ESOL before quoting. |
| Huang et al. (2021), Therapeutics Data Commons (TDC) | AqSolDB scaffold-split leaderboard (MAE). Top entries reported around MAE 0.75–0.8 | ⚠ **Not checked against the official leaderboard** (tdcommons.ai was not reachable). TDC keeps salts and inorganics and uses its own scaffold split, so it is **not** directly comparable to our 0.75 MAE. |

Our AqSolDB scaffold-split MAE (0.75) looks competitive. Remember, though, that we removed salts and
inorganics, which makes our task somewhat easier than TDC's.
