# Model card: SoluAI v2.0.0

A "model card" is a short, honest fact sheet about a machine-learning model: what it is for,
what it was trained on, how good it is, and where it fails.

## Model details

| | |
|---|---|
| Name / version | SoluAI 2.0.0 (`models/soluai-v2.0.0/`). The previous model is kept in `models/soluai-v1.0.0/`. |
| Task | Regression: predict aqueous solubility **logS** (log₁₀ mol/L) at about 25 °C |
| Algorithm | The Chemist–Referee algorithm ([algorithm.md](algorithm.md)). **Chemist:** LightGBM gradient-boosted trees with Huber loss (1,500 trees, 31 leaves, learning rate 0.03). **Referee:** a small LightGBM model (300 trees, 8 leaves) that predicts the chemist's expected error from 9 trust signals |
| Input features | 14 readable RDKit descriptors + 204 extended RDKit descriptors + 1,024-bit Morgan fingerprint (radius 2) = 1,242 |
| Explanations | Exact SHAP values of the chemist (LightGBM TreeSHAP), mapped to atoms; extended descriptors grouped into 6 families |
| Uncertainty | 90% normalised conformal interval (width set per molecule by the referee) + confidence label (high / medium / low) + similarity to the closest training molecule |
| Training data | 8,754 cleaned molecules from AqSolDB |
| Reproduce | `python -m soluai.train --config config.yaml` (seed 42, about 55 min on 4 CPU cores) |

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

| Dataset | Split | Delaney (2004) RMSE | v1 model RMSE | **SoluAI v2** RMSE | MAE | R² | Within ×10 |
|---|---|---|---|---|---|---|---|
| AqSolDB | random | 1.35 | 0.84 | **0.79** | 0.55 | 0.88 | 83.7% |
| AqSolDB | scaffold | 1.44 | 1.01 | **0.94** | 0.70 | 0.82 | 75.8% |
| ESOL | random | 1.07 | 0.61 | **0.58** | 0.41 | 0.91 | 92.4% |
| ESOL | scaffold | 1.24 | 0.83 | **0.82** | 0.64 | 0.84 | 78.9% |

"Within ×10" is the share of test molecules predicted within one log unit (a factor of 10) of the
lab value. On the small ESOL scaffold test, v1 and v2 are equal within noise (v2's
cross-validation score there is slightly worse, and v1 has more molecules within ×10: 81.1%).
All models and feature sets are in `results/metrics.md`.

**Confidence labels and 90% ranges (AqSolDB test sets):**

| | Random split | Scaffold split |
|---|---|---|
| High confidence: share of test molecules · within ×10 | 27% · 95.8% | 1% · 95.0% (20 molecules) |
| Medium confidence: share · within ×10 | 47% · 85.3% | 46% · 84.7% |
| Low confidence: share · within ×10 | 26% · 67.7% | 53% · 67.6% |
| 90% range actually covered | 88.7% (slightly too narrow) | 90.1% |
| Width of the range: 5th / 50th / 95th percentile | ± 0.57 / 1.03 / 2.26 | ± 0.98 / 1.42 / 2.43 |
| Errors above 2 log units that were labelled low / medium / high | 37 / 20 / 1 | 60 / 12 / 0 |

The labels mean the same thing on both exams; on new ring skeletons "high" is simply awarded
very rarely. The referee is a statistical judge, not a guarantee: about 1 in 20 high-confidence
test molecules was still off by more than a factor of 10, and coverage on the random split is one
point under the 90% target. For comparison, v1 gave nearly every molecule the same range
(± 1.3 to ± 1.5) and its "outside experience" flag was a weak signal (flagged molecules had only
10–20% larger errors).

## Does it reason like a chemist?

Correlation between each feature's value and its SHAP push, over all training molecules:

| Feature | Correlation | Chemistry expects | OK? |
|---|---|---|---|
| logP | −0.91 | higher logP → less soluble | ✔ |
| Molecular weight | −0.64 | bigger → less soluble | ✔ |
| H-bond donors | +0.78 | more donors → more soluble | ✔ |
| Polar surface area | +0.46 | more polar → more soluble | ✔ |
| H-bond acceptors | −0.11 | more acceptors → more soluble | ✘ (weak; probably a stand-in for size, see docs/chemistry.md §8) |
| Aromatic proportion | +0.24 | flat aromatic → less soluble | ✘ (see note) |

**Note on v2.** In v1 the aromatic proportion had a clear negative trend (−0.61). In v2 it does
not, and the trends for molecular weight and acceptors are weaker. The reason is that v2 has 204
more descriptors, many of which carry the same chemistry (aromatic surface area, ring counts,
complexity indices). SHAP splits the credit for one effect among all features that express it, so
a single descriptor no longer tells the whole story. The model still treats aromatic molecules as
less soluble (see the benzene-ring sentences on the Predict page, which read the atom colours);
it is this one-feature summary that became less readable. This is a real cost of the richer
feature set.

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
  confirmed by several sources (G3/G5), the random-split RMSE is only 0.49 (R² 0.95), compared with
  0.83 on single-source molecules. Part of the remaining "error" is noise in the measurements
  themselves, and it is why better models barely move the overall score
  ([algorithm.md §8](algorithm.md#8-what-we-tried-that-did-not-work)).
* **Not "95% accurate" overall.** That level is reached only for high-confidence predictions and
  against repeat-measured lab values ([algorithm.md §7](algorithm.md#7-is-it-95-accurate)).

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

Predictions are from model v2 on the held-out test sets. The last column of numbers is the
referee's 90% range and label for that prediction.

| Molecule | Measured | Predicted | Range · label | What went wrong |
|---|---|---|---|---|
| Aminohydroxy-anthraquinone carbonate dye (`CCOC(=O)OCCOc1cc(O)c2c(c1N)C(=O)c1ccccc1C2=O`) | −9.2 | −5.0 | ± 2.7 · low | **Crystal blind spot.** Flat, hydrogen-bonded anthraquinone dyes pack extremely tightly (high melting point). The model sees polar groups and predicts "fairly soluble". Several other anthraquinone dyes fail in the same way. |
| Dechlorane plus (C₁₈H₁₂Cl₁₂) | −13.2 | −8.1 | ± 2.0 · low | **Extreme value.** Very few training molecules are this insoluble, and trees cannot predict beyond the range they have seen. The model gets the direction right but not the size. |
| 1,4-Bis(hydroxyethoxyethoxypropylamino)anthraquinone | +0.06 | −3.9 | ± 3.1 · low | The opposite case: long, flexible polyether chains make this dye unusually soluble, which the model underestimates. |
| Hexamethylcyclotrisiloxane | +0.65 | −4.1 | ± 1.7 · low | **Probably a data error.** Siloxanes are known to be very poorly soluble; the measured value in the dataset looks wrong. The model is probably closer to the truth. |
| tert-Butyl isobutyl ether | +0.89 | −2.8 | ± 0.9 · medium | **Suspicious label.** An 8-carbon ether with no O–H group would be expected to be only slightly soluble, not "highly soluble". |
| Tris(isopropenyloxy)vinylsilane | +0.65 | −4.3 | ± 2.0 · low | **Reacts with water** (hydrolysis). The "solubility" describes the products, not the molecule. |
| Tetra(¹⁰B)boron carbide | −4.7 | −1.1 | ± 1.8 · low | **Should not be in the data:** an inorganic ceramic that passed our "has carbon" filter. |

Patterns:
1. Big errors usually come from **flat, polyaromatic, high-melting molecules** (dyes, anthraquinones,
   some steroids), which are predicted too soluble.
2. **Extreme values** (logS below −10) are pulled towards the middle.
3. A noticeable share of the "worst errors" are **doubtful measurements**, not model mistakes.
4. The referee labelled six of these seven "low confidence". It cannot fix a prediction, but it
   usually knows which ones to distrust. The ether is the counter-example: a confident-looking
   range around a prediction that disagrees with the (suspicious) lab value.

## Ethical and practical considerations

* Predictions are estimates. Do not use them to decide how to handle, dose or dispose of chemicals.
* The website shows a 90% range and a confidence label with every prediction so that users don't
  over-trust a single number.
* Name lookup sends the typed name to PubChem (a public NIH database). No other user data is stored.

## Comparison with published work

Published numbers are only roughly comparable, because papers clean the data differently (many keep
salts and inorganics, which are harder) and use different splits.

| Source | What they report | Status |
|---|---|---|
| Delaney (2004), *J. Chem. Inf. Comput. Sci.* 44:1000 | The ESOL equation, fitted on his own data | Our re-implementation with RDKit logP gives RMSE 1.07 on the ESOL random test set |
| Wu et al. (2018), MoleculeNet, *Chem. Sci.* 9:513 | ESOL random-split benchmarks; graph neural networks reported around RMSE 0.5–0.6 | ⚠ **Not checked against the paper** (the paper was not reachable while building this project). Check Table/Figure for ESOL before quoting. |
| Huang et al. (2021), Therapeutics Data Commons (TDC) | AqSolDB scaffold-split leaderboard (MAE). Top entries reported around MAE 0.75–0.8 | ⚠ **Not checked against the official leaderboard** (tdcommons.ai was not reachable). TDC keeps salts and inorganics and uses its own scaffold split, so it is **not** directly comparable to our 0.70 MAE. |

Our AqSolDB scaffold-split MAE (0.70 with v2, 0.75 with v1) looks competitive. Remember, though, that we removed salts and
inorganics, which makes our task somewhat easier than TDC's.
