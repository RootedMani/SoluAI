# Research notebook: how SoluAI v2 was found

These scripts are the exploratory experiments behind [docs/algorithm.md](../docs/algorithm.md).
They are **not** part of the product or of the reproducible pipeline (`python -m soluai.train`).
They are kept so that the claims in the algorithm document, including the ideas that failed, can
be checked.

They are written as quick experiments, not as polished code. All of them use the same test
splits and cross-validation folds as the real pipeline (seed 42), and every choice was judged by
**cross-validation on the training part**. The test columns were looked at afterwards.

## How to run

```bash
pip install -r requirements-dev.txt && pip install -e .
cd research
python 01_features.py            # builds research/cache/ (not in git), ~3 min
python 02_feature_sets.py        # each later script needs 01, some need 02
...
```

Each script takes 5–25 minutes on 4 CPU cores. Run them one at a time: two LightGBM jobs fighting
for the same cores are many times slower. The original output of each run is in `logs/`.

Unless noted, the model is one fixed LightGBM setting (1,500 trees, learning rate 0.03, 31 leaves,
30% of the columns per tree) so that only the thing being tested changes. Numbers are RMSE in log
units on AqSolDB: `CV` = 5-fold on the training part, `test` = the held-out 20%.

## What was asked, and what came out

### 02 · Do richer features help? Yes, a little

| Features | Random CV / test | Scaffold CV / test |
|---|---|---|
| v1: 14 descriptors + 1,024-bit Morgan | 0.834 / 0.844 | 0.947 / 1.013 |
| all 217 RDKit descriptors | 0.817 / 0.799 | 0.920 / 0.970 |
| **217 descriptors + 1,024-bit Morgan** (chosen) | **0.804 / 0.795** | **0.906 / 0.948** |
| 217 descriptors + 2,048 Morgan counts | 0.804 / 0.803 | 0.908 / 0.942 |
| 217 descriptors + MACCS keys | 0.813 / 0.801 | 0.913 / 0.947 |
| 217 descriptors + MACCS + Morgan counts | 0.805 / 0.795 | 0.902 / 0.935 |

All the richer sets are within noise of each other. We kept the one that leaves the v1 fingerprint
(and so the atom-colouring code) untouched.

### 03 · Other learners, stacking, and "correct with the neighbours": no

| Model | Random CV / test | Scaffold CV / test |
|---|---|---|
| LightGBM (from 02) | 0.804 / 0.795 | 0.906 / 0.948 |
| Support-vector regression, Tanimoto kernel | 1.107 / 1.126 | 1.654 / 1.386 |
| Support-vector regression on scaled descriptors | 0.879 / 0.853 | 1.112 / 1.094 |
| Extra-trees (500 trees) | 0.849 / 0.826 | 0.996 / 1.032 |
| Stack of all four (weights 0.67 / 0.07 / 0.18 / 0.11) | 0.795 / 0.788 | 0.898 / 0.944 |
| Stack + analogue correction | 0.823 / 0.804 | 0.902 / 0.954 |

* **Stacking** four models gains 0.01. Not worth four models, a 100+ MB file and losing exact
  SHAP explanations.
* **Analogue correction** = add to the prediction the (similarity-weighted) error the model made
  on the most similar training molecules. Intuitive, and it made things *worse* even at its best
  setting. The neighbours' errors are mostly measurement noise, which does not transfer.
  This failure is what suggested the referee: the neighbours' errors do not tell you *which way*
  the model is wrong, but they do tell you *how large* its error tends to be (script 07).

### 04 · Noise-aware training: only the robust loss helps

AqSolDB says how often a value was measured. "Reliable" = measured 2+ times and the labs agree
(groups G3, G5); "conflicting" = measured 2+ times and they disagree (G2, G4).

| Variant | Random CV / test | Scaffold CV / test |
|---|---|---|
| plain | 0.804 / 0.795 | 0.906 / 0.948 |
| reliable ×2, conflicting ×0.5 | 0.804 / 0.801 | 0.909 / 0.954 |
| reliable ×3, conflicting ×0.5 | 0.803 / 0.797 | 0.908 / 0.949 |
| reliable ×3 | 0.805 / 0.796 | 0.909 / 0.949 |
| **Huber loss (δ = 1)** (chosen) | **0.800 / 0.795** | **0.904 / 0.937** |
| self-cleaning (down-weight training values the model disagrees with), cut-off 1.5 | – / 0.803 | – / 0.953 |
| same, cut-off 2.0 | – / 0.800 | – / 0.951 |
| bigger model (3,000 trees, 63 leaves) | 0.805 / 0.789 | 0.917 / 0.948 |

Weighting by reliability does nothing: only 19% of the molecules are "reliable", and they are
already the easy ones. The same log shows the key observation of the whole project. With the
plain model on the random test set:

| Test molecules | n | RMSE | R² | within ×10 |
|---|---|---|---|---|
| all | 1,751 | 0.795 | 0.880 | 83.4% |
| reliable (G3 + G5) | 297 | 0.484 | 0.951 | 94.6% |

### 05 · Even more feature blocks: no

E-state atom types (158 numbers), Crippen atom-type counts, atom-pair counts (2,048), topological
torsions (1,024), on top of the 217 descriptors (Huber loss):

| Features | Random CV / test | Scaffold CV / test |
|---|---|---|
| descriptors + Morgan (from 04) | 0.800 / 0.795 | 0.904 / 0.937 |
| descriptors + E-state | 0.811 / 0.797 | 0.909 / 0.957 |
| descriptors + Crippen types | 0.814 / 0.795 | 0.915 / 0.960 |
| descriptors + E-state + Crippen types + Morgan | 0.802 / 0.797 | 0.903 / 0.949 |
| descriptors + E-state + Crippen types + atom pairs | 0.802 / 0.788 | 0.903 / 0.952 |
| everything, plus MACCS keys | 0.798 / 0.792 | (not run) |

Nothing moves. Together with 03 and 04 this is what "the noise in the data is the ceiling" looks
like in practice: every reasonable model lands on CV ≈ 0.80 / 0.90.

### 06 · Can a short, readable descriptor list keep the gain? No

Descriptors ranked by mean |SHAP| on the training part; the top N kept, plus the fingerprint:

| Descriptors kept | Random CV / test | Scaffold CV / test |
|---|---|---|
| the 14 of v1 (Huber loss) | 0.833 / 0.839 | 0.939 / 1.012 |
| top 15 | 0.823 / 0.832 | 0.930 / 0.996 |
| top 25 | 0.814 / 0.828 | 0.914 / 0.975 |
| top 40 | 0.803 / 0.803 | 0.900 / 0.963 |
| top 70 | 0.800 / 0.797 | 0.896 / 0.951 |
| all 217 | 0.800 / 0.795 | 0.904 / 0.937 |

The gain is spread over many descriptors, so v2 keeps all of them and groups them into families
when explaining. (The first row also shows that the Huber loss alone, without new features, does
almost nothing: 0.839 vs 0.844.)

### 07 · Can a second model predict how large the error will be? Yes

The prototype of the referee, on top of the model from 02. Test molecules sorted by the referee's
expected error:

| Most confident … of the test set | Random: RMSE / R² / within ×10 | Scaffold: RMSE / R² / within ×10 |
|---|---|---|
| 20% | 0.41 / 0.970 / 96.0% | 0.64 / 0.915 / 87.4% |
| 33% | 0.45 / 0.957 / 95.8% | 0.70 / 0.887 / 85.4% |
| 50% | 0.54 / 0.935 / 93.4% | 0.73 / 0.862 / 83.2% |
| 67% | 0.61 / 0.918 / 90.6% | 0.79 / 0.835 / 82.4% |
| 100% | 0.80 / 0.880 / 83.4% | 0.95 / 0.813 / 76.0% |

For comparison, v1's "unfamiliar" flag (closest training molecule < 30% similar) separated
RMSE 0.79 from 0.97 on the random split and 0.95 from 0.93 on the scaffold split, i.e. not at all
on the harder exam. The finished version is `soluai.uncertainty.TrustModel`; its results on the
final model are in `results/metrics.md`.

### 08 · Error by original data source

AqSolDB merges nine datasets (ID prefix A–I). Random split, model from 02:

| Source | n (test) | RMSE | within ×10 |
|---|---|---|---|
| A (the largest industrial collection) | 459 | 0.99 | 78% |
| B | 728 | 0.71 | 85% |
| C | 252 | 0.73 | 83% |
| F | 174 | 0.82 | 82% |
| D, E, G, H (small, curated) | 129 | 0.37–0.64 | 92–100% |

No source is systematically shifted against the others (mean error within ±0.2 for all sources
with more than 10 test molecules), so "which dataset did this value come from" was not used as a
model input.

### 09 · Does the two-exam referee hold up? Yes

The deployed referee learns from two cross-validations of the chemist at once (scaffold folds and
random folds), because visitors type both familiar and unfamiliar molecules. This script repeats
that on the training part of each split and checks it on the test part, using the final v2
settings.

| Test set | Referee trained on | Inside the 90% range | "High" label: share, within ×10 | "Low" label: share, within ×10 |
|---|---|---|---|---|
| random | random folds only | 89.0% | 28%, 95.4% | 26%, 67.1% |
| random | **both (as deployed)** | 89.4% | 24%, 95.8% | 25%, 67.2% |
| scaffold | scaffold folds only | 90.1% | 1%, 95.0% | 53%, 67.6% |
| scaffold | **both (as deployed)** | 90.8% | 4%, 95.2% | 58%, 68.4% |

One referee serves both situations: coverage stays at 89–91% and the labels keep their meaning.

## Ideas not tried

* A graph neural network. Published gains on AqSolDB are small, it needs PyTorch (too heavy for
  the free web server), and the explanations would be approximate.
* A melting-point model as an extra input (the general solubility equation says melting point
  matters). It needs a separate melting-point dataset.
* More or better data. This is the most promising direction: 02–06 say the model is no longer the
  bottleneck.
