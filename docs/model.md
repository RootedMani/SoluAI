# How the SoluAI model works, step by step

This document explains the whole machine-learning pipeline in simple language. The code
for each step is named in brackets.

```
 raw data ──► cleaning ──► features ──► split ──► train + tune (CV) ──► test once ──► final model ──► website
 (data/raw)   (data.py)   (features.py) (splits.py)    (train.py)        (evaluate.py)  (models/)     (web/)
```

Everything is run by **one command**:

```bash
python -m soluai.train --config config.yaml
```

All settings (random seed, split sizes, search ranges) live in `config.yaml`, so running it
again gives exactly the same numbers.

---

## Step 1: The data (`data.py`)

We learn from **measured** solubilities: real lab experiments, not computer guesses.

* **AqSolDB** (9,982 compounds) is the main dataset.
* **ESOL** (1,128 compounds) is a smaller, classic benchmark.

Molecules are written as **SMILES** strings, a way to write a molecule as text:
`CCO` is ethanol (C–C–O), and `c1ccccc1` is benzene (lowercase letters = aromatic ring).

**Cleaning.** We remove entries that are unreadable, salts or mixtures (`[Na+].[Cl-]`),
inorganic, or contain metals. We neutralise charges and write every molecule in one standard
("canonical") SMILES. Duplicates are averaged if they agree within 0.5 log units, and dropped if
they don't. The exact counts are in `data/processed/cleaning_report.json`.

## Step 2: Features: turning molecules into numbers (`features.py`)

A machine-learning model only understands numbers, so each molecule becomes a list of numbers.

### Descriptors (14 numbers a chemist can read)

| Descriptor | Meaning |
|---|---|
| MolLogP | logP, how oily the molecule is |
| MolWt | molecular weight |
| TPSA | polar surface area: the surface belonging to N and O atoms |
| NumHDonors / NumHAcceptors | hydrogen-bond donors (O–H, N–H) and acceptors |
| NumRotatableBonds | how floppy the molecule is |
| AromaticProportion | share of atoms in aromatic rings |
| FractionCSP3 | share of carbons that are "saturated" |
| RingCount, NumAromaticRings | number of rings |
| HeavyAtomCount, NumHeteroatoms, MolMR, NumValenceElectrons | more measures of size and composition |

### Morgan fingerprint (1,024 yes/no bits)

For every atom, RDKit looks at its neighbourhood: the atom itself (radius 0), its neighbours
(radius 1), and their neighbours (radius 2). Each neighbourhood is turned into a number and
switches on one of 1,024 bits. The result is a barcode of all the small fragments in the
molecule, such as "C–OH attached to a ring" or "C=O next to N". The model can then learn things
like "this fragment tends to increase solubility".

We test three feature sets: **descriptors only**, **fingerprint only**, and **both**.

## Step 3: Splitting: the exam must contain new questions (`splits.py`)

If you test a student with the exact questions they practised, you learn nothing. So we
**hide 20% of the molecules** (the *test set*) and never let the model, or us, look at them
until the very end.

* **Random split:** hide a random 20%. Test molecules usually have close relatives in
  training (1-heptanol in test, 1-hexanol and 1-octanol in training). This is the easy exam.
* **Scaffold split:** first find each molecule's **scaffold**, its ring system with the side
  chains removed (Bemis–Murcko scaffold). Then hide *whole scaffold groups*, so every test
  molecule has a ring skeleton the model has never seen. This is the hard exam, and the honest
  one if you want to predict a brand-new kind of molecule.

**Why it matters:** a model can look great on a random split just by "remembering cousins".
The scaffold split shows how well it generalises to new chemistry. We always report both.

## Step 4: Cross-validation and tuning (`train.py`)

Models have settings (*hyper-parameters*), for example how many trees and how deep. To pick
them **without touching the test set**, we use **5-fold cross-validation** on the training part:

1. cut the training molecules into 5 parts;
2. train on 4 parts and score on the 5th; repeat 5 times so every part is scored once;
3. average the 5 scores (the **CV score**).

We try a fixed number of random settings (12 for LightGBM, 6 for the slower random forest)
and keep the one with the best CV score. For the scaffold split, the folds are also
scaffold-based. Only then is the chosen model trained on the whole training part and
scored **once** on the test set.

## Step 5: The models (`baselines.py`, `models.py`)

### Baselines: what we must beat

* **Delaney's ESOL equation (2004):** a chemist's hand-made formula
  `logS = 0.16 − 0.63·logP − 0.0062·MW + 0.066·RB − 0.74·AP`.
  Nothing is learned; we use Delaney's own numbers.
* **Delaney refit:** the same 4 variables, but the computer finds the best coefficients for
  our data.
* **Linear regression:** a straight-line formula using all 14 descriptors.

### Random forest

A **decision tree** asks yes/no questions: "logP > 3.2?" → "TPSA > 40?" → … → "predict −2.7".
One tree is easy to read but makes jumpy, overconfident predictions.

A **random forest** grows hundreds of trees, each on a random sample of the molecules and
looking at a random subset of features at each question, then **averages** their predictions.
The individual trees' mistakes partly cancel, like asking many people and taking the average.

### Gradient boosting (LightGBM)

**Boosting** also uses many small trees, but builds them **one after another**:

1. start with a simple guess (the average logS);
2. look at the errors; train a small tree to predict those errors;
3. add a small fraction of that tree to the model (the *learning rate*);
4. repeat hundreds of times, each tree fixing what is still wrong.

LightGBM is a fast, popular implementation. It also computes exact **SHAP** explanations
(step 7) cheaply, which is why we deploy it on the website.

### Why no neural network?

Graph neural networks can be slightly better on some benchmarks, but they are much harder to
explain, need more computing power, and published gains on AqSolDB are small. For a project
where *understanding* matters, tree models on readable features are the better choice.

## Step 6: Measuring errors (`evaluate.py`)

| Metric | Meaning | Good value |
|---|---|---|
| **RMSE** (root mean squared error) | typical size of an error; big mistakes count extra | lower; 1.0 = "off by a factor of 10" |
| **MAE** (mean absolute error) | plain average size of the errors | lower |
| **R²** | share of the variation in logS the model explains (1 = perfect, 0 = no better than guessing the average) | higher |

Because logS is a logarithm, an RMSE of 0.7 means the prediction is typically within a factor
of 10^0.7 ≈ 5 of the true solubility. Remember that the measurements themselves often differ by
about 0.5 log units between labs, so even a perfect model could not reach RMSE = 0.

## Step 7: Explaining predictions (`explain.py`)

**SHAP** (SHapley Additive exPlanations) comes from game theory: it shares out the
"credit" for a prediction among the features fairly. For one molecule:

```
prediction = average prediction (≈ −3.0) + SHAP(logP) + SHAP(TPSA) + … + SHAP(fragment bits)
```

* On the website, the **bar chart** shows the biggest SHAP values.
* The **atom colours** share each feature's SHAP value among the atoms responsible for it:
  a fragment's atoms, each atom's own logP or polar-surface contribution, the O/N atoms that
  donate or accept hydrogen bonds, the aromatic atoms, and so on. Blue = raises solubility,
  red = lowers it.
* The **plain-language sentence** finds functional groups (–OH, –COOH, benzene ring…) with
  pattern matching and reads off whether their atoms were mostly blue or red.

## Step 8: Knowing when to be unsure (`uncertainty.py`)

* **"Outside the model's experience":** we compare the new molecule's fingerprint with every
  training molecule (Tanimoto similarity: 0 = nothing in common, 1 = identical). If even the
  closest one is below 0.30 similar, or a descriptor is outside the training range, we warn the user.
* **90% range (conformal prediction):** we collect the model's errors on molecules it had not
  seen (from cross-validation) and find the error size that 90% of them stay below. The range is
  prediction ± that size, computed separately for familiar and unfamiliar molecules. We check on
  the test set that about 90% of true values really fall inside their range.

## Step 9: The final model and versioning

After the benchmark, LightGBM with the best settings from the scaffold-split search is trained
on **all** cleaned AqSolDB molecules and saved in `models/soluai-v1.0.0/`:

* `model.joblib`: the trained model and its settings
* `training_set.csv`: the molecules it learned from (used for "closest molecules")
* `metadata.json`: version, date, settings, config fingerprint, data checksums, package versions
  and cross-validation scores

To make a new version, change `model_version` in `config.yaml` and run the training command again.

## Results

The full tables are in `results/metrics.md` and on the website's "How the model works" page.
A summary is in the [README](../README.md#results).
