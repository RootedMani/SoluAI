# The Chemist–Referee algorithm (SoluAI v2)

This document describes the prediction algorithm that SoluAI uses from version 2.0. It was designed
for this project. It is built from standard, well-understood parts (gradient boosting, nearest
neighbours, conformal prediction), put together in a way that fits this dataset. Nothing here is
magic, and every number below can be reproduced with one command.

**In one paragraph.** Two models work together. The **chemist** predicts the solubility. The
**referee** never predicts solubility: it looks at the chemist's prediction and at the most similar
molecules with a lab value, and predicts *how wrong the chemist is likely to be*. The size of the
90% range and a confidence label (high / medium / low) come from the referee. So SoluAI no longer
says "−3.2, give or take 1.5" for every molecule. It says "−3.2, and for a molecule like this I am
usually within 0.3", or "−3.2, but I have been badly wrong on molecules like this".

```
                      ┌──────────────────────────────┐
  molecule ──────────►│ CHEMIST                      │──► predicted logS
  (SMILES)            │ robust boosting, 1,242 feats │        │
      │               └──────────────────────────────┘        │
      │                                                       ▼
      │  5 most similar     ┌──────────────────────────────┐
      └─ training molecules►│ REFEREE                      │──► expected error σ
         (lab values +      │ small boosting, 9 signals    │        │
          the chemist's     └──────────────────────────────┘        ▼
          past mistakes)                              90% range = prediction ± q·σ
                                                      confidence = high / medium / low
```

Contents: [why](#1-why-a-new-algorithm) · [the chemist](#2-part-1-the-chemist) ·
[the referee](#3-part-2-the-referee) · [range and label](#4-part-3-the-range-and-the-confidence-label) ·
[pseudo-code](#5-the-whole-algorithm-in-pseudo-code) · [results](#6-results) ·
[is it 95% accurate?](#7-is-it-95-accurate) · [what did not work](#8-what-we-tried-that-did-not-work) ·
[limits](#9-limits) · [reproduce](#10-reproduce)

---

## 1. Why a new algorithm

Version 1 was a LightGBM model on 14 descriptors and a 1,024-bit fingerprint. Looking closely at
its mistakes showed two separate problems.

**Problem A: the lab values are noisy, and the model cannot beat its teacher.** AqSolDB marks how
often each molecule was measured. On the hidden test molecules:

| Test molecules (AqSolDB, random split, model v1) | Share | RMSE | R² | Within ×10 |
|---|---|---|---|---|
| Measured once (group G1) | 78% | 0.88 | 0.86 | 80% |
| Measured 2+ times, labs agree (G3, G5) | 17% | 0.53 | 0.94 | 94% |
| Measured 2+ times, labs disagree (G2, G4) | 5% | 1.03 | 0.79 | 70% |

("Within ×10" = the prediction is within one log unit, i.e. a factor of 10, of the lab value.)

The same model looks far more accurate when the lab value it is compared with is
trustworthy. A large part of the "error" on the other molecules is the measurement, not the model.
(It is not the whole story: molecules that were measured several times also tend to be simpler,
well-studied molecules.) We also split the error by the nine original data sources inside AqSolDB:
RMSE ranges from about 0.4 for small curated sources to 1.0 for the largest industrial one
([research notes](../research/README.md#08--error-by-original-data-source)).

The consequence is important: **no cleverer model can push the overall error much lower with this
data.** We tried (section 8). So the error number can only improve a little, and the real question
becomes the second problem.

**Problem B: v1 did not know when it was right.** Its 90% range was ±1.49 log units (a factor of
30) for nearly every molecule, whether the molecule was ethanol or an exotic dye. Its only warning,
"the closest training molecule is less than 30% similar", was a weak signal: flagged molecules had
errors only 10–20% larger.

Yet the errors are very uneven. Some kinds of molecules are predicted within a factor of 2 almost
every time, others are regularly wrong by a factor of 100. A model that can tell these apart is
much more useful than one with a slightly smaller average error, because the user finally knows
which numbers to trust.

So v2 does two things: a better chemist (a modest gain), and a referee (the new idea).

## 2. Part 1: the chemist

The chemist is still a LightGBM gradient-boosting model, with two changes.

**More chemistry in the input.** 1,242 numbers per molecule instead of 1,038:

| Block | Count | What it is |
|---|---|---|
| Readable descriptors | 14 | the same as v1: logP, molecular weight, polar surface area, H-bond donors… |
| Extended RDKit descriptors | 204 | everything else RDKit computes from the 2D structure (below) |
| Morgan fingerprint | 1,024 | the same as v1: yes/no bits for small fragments |

The 204 extended descriptors come in families:

| Family | Count | Examples | What they tell the model |
|---|---|---|---|
| Surface area by polarity | 57 | `SlogP_VSA2`, `PEOE_VSA6`, `SMR_VSA3` | how much of the molecule's surface is oily, polar, charged… Each one is "the surface of all atoms whose oiliness (or charge, or refractivity) lies in one range" |
| Functional-group counts | 102 | `fr_ester`, `fr_phenol`, `NumAmideBonds` | how many of each of 85 groups, and ring types |
| Shape and branching | 22 | `Chi0v`, `Kappa2`, `BertzCT` | long chain or compact ball? simple or complex skeleton? |
| Charges and electron states | 16 | `MaxPartialCharge`, `MinEStateIndex` | the most positive / negative atoms |
| Other size measures | 3 | `LabuteASA` | approximate total surface area |
| Other | 4 | `qed`, `FpDensityMorgan1` | |

Why they help: logP and polar surface area are single totals. Two molecules can have the same
totals but distribute them very differently (one very oily end and one very polar end, or
everything lukewarm). The surface-area families give the model that distribution.

A smaller, hand-picked list would be nicer to explain, so we tested it: the gain only arrives with
about 40–70 of them (section 8). We therefore keep all of them and *group them by family* when
explaining a prediction.

**A loss that forgives bad lab values (Huber loss).** v1 minimised the squared error, so one
training value that is wrong by 4 log units pulls 16 times harder than one that is wrong by 1.
v2 minimises the Huber loss with threshold δ (1.0 or 1.5 log units, chosen by cross-validation):

```
loss(error) = ½·error²                 if |error| ≤ δ      (normal behaviour)
            = δ·(|error| − ½·δ)        if |error| > δ      (grows only linearly)
```

A value that disagrees wildly with everything else the model has learned still counts, but it
cannot dominate. Section 1 showed why this matters here: a visible share of the worst "errors" are
doubtful measurements.

The settings (number of leaves, sampling, regularisation, δ) are chosen by 5-fold cross-validation
on the training part only, exactly as in v1. The search space is in `config.yaml` under
`search.soluai_v2`.

## 3. Part 2: the referee

### The idea

When the chemist is cross-validated, every training molecule gets a prediction from a model that
has not seen it. That gives 8,754 honest examples of "here is a molecule, and here is how wrong the
chemist was". The referee is a second, small model trained on exactly these examples. Its target is
the **size of the chemist's error**, not the solubility.

What should the referee look at? Not the 1,242 features again: the chemist already used them, and
the referee would mostly learn noise. Instead it gets nine **trust signals** that describe the
*situation* the chemist is in.

### The nine trust signals

For a molecule, find its 5 most similar training molecules (Tanimoto similarity of Morgan
fingerprints: 0 = nothing in common, 1 = identical). Each neighbour has a lab value and an honest
record of how wrong the chemist was on it when it had not seen it. Averages are weighted by
similarity.

| # | Signal | Question it answers |
|---|---|---|
| 1 | `max_sim` | Has the chemist seen anything like this? (similarity of the closest neighbour) |
| 2 | `mean_sim` | Is there a whole family of similar molecules, or one lonely cousin? |
| 3 | `nb_spread` | Do the neighbours' lab values agree with each other? If near-identical molecules have very different solubilities, this region is treacherous (or the data are noisy there). |
| 4 | `nb_error` | Was the chemist wrong on the neighbours? Mistakes cluster: if it failed on the cousins, it will probably fail here. |
| 5 | `nb_gap` | Does the prediction agree with the neighbours' average lab value? A large gap means the model and the nearest data tell different stories. |
| 6 | `prediction` | The predicted logS itself. Very insoluble molecules are harder to measure and to predict. |
| 7 | `logP` | Oiliness. |
| 8 | `MolWt` | Molecular weight. Big molecules are harder. |
| 9 | `heavy_atoms` | Number of non-hydrogen atoms. |

Signals 1 and 2 are the old "is it familiar?" question. Signals 3–5 are new and turned out to
matter just as much: a molecule can have very similar neighbours and still be hard, when those
neighbours disagree or were themselves predicted badly.

### Training the referee honestly

Three details keep the referee from fooling itself.

1. **Neighbours come only from other folds.** When computing the signals for a training molecule,
   its own cross-validation fold is hidden, so it sees the same kind of neighbours a brand-new
   molecule would see.
2. **Cross-fitting.** To get an honest σ for every training molecule (needed in part 3), the referee
   itself is trained 5 times, each time predicting the fold it did not see.
3. **Two exams for the final model.** For the deployed model the chemist is cross-validated twice:
   with scaffold folds ("new ring skeleton") and with random folds ("close cousins exist"). The
   referee learns from both sets of errors, because website visitors type both kinds of molecules.
   (In the benchmark tables each test uses the folds that match its split.)

The referee is deliberately tiny: 300 trees with 8 leaves each, at least 40 molecules per leaf. It
has only nine inputs and a noisy target, so a bigger model would overfit. Its output is the
**expected absolute error σ** in log units, never allowed below 0.15 (no one measures solubility
better than that).

## 4. Part 3: the range and the confidence label

**The range (conformal prediction).** For every training molecule we know the honest error and the
honest σ. Compute the ratio `error / σ` for all of them and take the value `q` that 90% of the
ratios stay below (here q ≈ 2.1). Then for a new molecule:

```
90% range = prediction ± q · σ
```

This is called a *normalised conformal interval*. It needs no assumption about the shape of the
errors. If the referee were useless (σ the same for everyone), this would fall back to v1's
fixed-width range. The better the referee, the more the widths differ between easy and hard
molecules while 90% of all molecules stay covered.

**The label.** The label only reads the width of that range:

| Label | 90% range | Meaning |
|---|---|---|
| **High confidence** | ± 0.8 or narrower | true value very probably within a factor of 6 |
| **Medium confidence** | ± 0.8 to ± 1.4 | within a factor of 6 to 25 |
| **Low confidence** | wider than ± 1.4 | rough guide only |

One safety net sits on top: if the closest training molecule is less than 30% similar, or one of
the 14 readable descriptors is outside the range seen in training, the website shows "low" whatever
the width. The referee has seen too few such molecules to be trusted there. (The tables in section 6
do not use this override; it can only make the website more cautious.)

The thresholds are fixed numbers, not "the best third of each test". That keeps the label's meaning
the same in every situation: on an exam full of unfamiliar skeletons the referee simply awards
"high" less often.

## 5. The whole algorithm in pseudo-code

```
TRAIN(molecules, lab values y):
    X          ← features(molecules)                       # 14 + 204 + 1,024 numbers each
    chemist    ← LightGBM(Huber loss), settings chosen by 5-fold CV
    oof        ← out-of-fold predictions of the chemist     # each molecule predicted unseen
    err        ← |oof − y|                                  # honest errors
    for each molecule i:
        N      ← 5 most similar molecules from the OTHER folds
        S[i]   ← 9 signals from (similarities, y[N], err[N], oof[i], logP, weight, size)
    σ_honest   ← 5-fold cross-fitted referee predictions of err from S, floored at 0.15
    q          ← 90th percentile of err / σ_honest
    referee    ← small LightGBM fitted on (S, err)
    chemist    ← refit on all molecules
    store: chemist, referee, q, and (y, err) of every training molecule

PREDICT(molecule):
    ŷ          ← chemist(features(molecule))
    N          ← 5 most similar training molecules
    σ          ← max(0.15, referee(9 signals from N and ŷ))
    return ŷ,  range ŷ ± q·σ,  label from q·σ
```

Code: `features.py` (features), `train.py` (`fit_trust`, `train_final`), `uncertainty.py`
(`trust_signals`, `TrustModel`), `predict.py` (`Predictor.domain`).

**Cost.** The model file grows from 0.8 MB to 1.9 MB. One prediction with its explanation takes
about 35 ms, so the free web server is still enough. No new dependencies.

**Explanations still work.** The chemist is one LightGBM model, so exact SHAP values are available
as before and still add up to the prediction. The extended descriptors are shared among atoms
wherever they belong to particular atoms: a surface-area descriptor colours the atoms whose surface
it measures, a functional-group count colours the atoms of that group. Shape and charge-pattern
indices describe the whole molecule and are listed next to the picture. In the bar chart the
extended descriptors appear as one bar per family.

## 6. Results

All numbers are from held-out test molecules that were used exactly once. Settings were chosen by
cross-validation on the training part. Full tables: [`results/metrics.md`](../results/metrics.md).

### 6.1 The chemist: a modest, consistent gain

| Dataset · split | v1 RMSE | **v2 RMSE** | v1 R² | **v2 R²** | v1 within ×10 | **v2 within ×10** |
|---|---|---|---|---|---|---|
| AqSolDB · random | 0.835 | **0.793** | 0.867 | **0.880** | 82.0% | **83.7%** |
| AqSolDB · scaffold | 1.012 | **0.942** | 0.787 | **0.816** | 73.7% | **75.8%** |
| ESOL · random | 0.605 | **0.575** | 0.903 | **0.913** | 90.2% | **92.4%** |
| ESOL · scaffold | 0.833 | 0.823 | 0.837 | 0.841 | 81.1% | 78.9% |

The error drops by 5% (random) and 7% (scaffold) on AqSolDB. On the small ESOL scaffold test (227
molecules) v1 and v2 are the same within noise, and v2's cross-validation score there is slightly
worse (0.822 vs 0.806). This is the "modest" part, and section 8 explains why we do not expect
more from this data.

### 6.2 The referee: confidence that means something

AqSolDB test sets, SoluAI v2, grouped by the label the referee gave:

| | Random split | | | | Scaffold split | | | |
|---|---|---|---|---|---|---|---|---|
| **Label** | share | RMSE | R² | **within ×10** | share | RMSE | R² | **within ×10** |
| High confidence | 27.2% | 0.43 | 0.96 | **95.8%** | 1.1% | 0.48 | 0.89 | **95.0%** |
| Medium confidence | 47.3% | 0.74 | 0.88 | **85.3%** | 46.2% | 0.74 | 0.85 | **84.7%** |
| Low confidence | 25.5% | 1.13 | 0.75 | **67.7%** | 52.7% | 1.10 | 0.74 | **67.6%** |
| All | 100% | 0.79 | 0.88 | 83.7% | 100% | 0.94 | 0.82 | 75.8% |

Three things to notice.

* **The labels separate good from bad predictions.** High-confidence predictions have less than
  half the error of low-confidence ones.
* **Each label means the same on both exams** (96% / 85% / 68% against 95% / 85% / 68%). The
  scaffold exam is harder because the referee awards "high" to only 20 of its 1,751 molecules
  and "low" to half of them, not because the labels lie. That is the behaviour we want: when
  the chemistry is unfamiliar the model says so. (With only 20 molecules, the 95% on the scaffold
  side is a rough number.)
* **Large mistakes are rarely labelled confident.** Of the 58 random-split test molecules that
  were wrong by more than 2 log units (a factor of 100), 37 were labelled low, 20 medium and one
  high. On the scaffold split: 60 of 72 low, none high.

Sorted by confidence instead of grouped (random split): the most confident 10% of test molecules
are 96.6% within ×10 (R² 0.98), the most confident half 93.7% (R² 0.94), all of them 83.7%.

### 6.3 The 90% ranges: narrower where it is safe

| AqSolDB test set | Random split | Scaffold split |
|---|---|---|
| v1: width of the range (familiar / unfamiliar) | ± 1.30 / ± 1.93 | ± 1.46 / ± 1.56 |
| v1: test molecules really inside their range | 88.6% | 86.6% |
| v2: width of the range, 5th / 50th / 95th percentile | ± 0.57 / ± 1.03 / ± 2.26 | ± 0.98 / ± 1.42 / ± 2.43 |
| v2: test molecules really inside their range | 88.7% | 90.1% |
| v2: molecules whose range is narrower than one fixed width would be | 66% | 51% |

The typical (median) range on the random split shrinks from ± 1.30 to ± 1.03, which is a factor
of 20 down to a factor of 11, and a quarter of the molecules get ± 0.78 or better. Coverage on
the random split is 88.7%, slightly under the 90% target; on the scaffold split it is on target,
where v1 was 3 points short. Per label, coverage is between 87% and 91% wherever there are enough
molecules to measure it.

The deployed referee is trained on two cross-validations at once (section 3). We checked that
design separately on held-out data: fitted on the training part only, its ranges
contained 89.4% of the random-split test molecules and 90.8% of the scaffold-split ones, and its
"high" label was right within ×10 for 95.8% (24% of the random test) and 95.2% (4% of the scaffold
test) ([research notes](../research/README.md#09--does-the-two-exam-referee-hold-up-yes)).

### 6.4 Against trustworthy lab values

| AqSolDB random-split test molecules | Share | v1 RMSE / R² / within ×10 | **v2 RMSE / R² / within ×10** |
|---|---|---|---|
| Measured 2+ times, labs agree (G3, G5) | 17% | 0.53 / 0.94 / 93.9% | **0.49 / 0.95 / 94.9%** |
| Measured once (G1) | 78% | 0.88 / 0.86 / 80.0% | **0.83 / 0.87 / 82.1%** |
| Measured 2+ times, labs disagree (G2, G4) | 5% | 1.03 / 0.79 / 70.4% | **1.00 / 0.81 / 70.4%** |

On the scaffold split the same reliable group scores RMSE 0.81 / R² 0.86 / 81.9%: a new ring
skeleton stays hard even when the lab value is good.

## 7. Is it 95% accurate?

It depends on what "accurate" means, so here are the honest versions of the answer. Solubility is a
number, not a yes/no label, so there is no single "accuracy". Two measures are common: **R²** (the
share of the variation in solubility that the model explains) and **within ×10** (the share of
predictions within a factor of 10 of the lab value).

| Question | Answer (AqSolDB, random split, molecules never seen) |
|---|---|
| Over *all* test molecules? | **No.** R² = 0.88, within ×10 = 83.7%. |
| For the predictions SoluAI itself labels **high confidence**? | **Yes.** R² = 0.96, within ×10 = 95.8%. That is 27% of the test molecules. |
| Against lab values that were **measured more than once and agree**? | **Yes, just.** R² = 0.95, within ×10 = 94.9%. |
| On the smaller ESOL benchmark? | R² = 0.91, within ×10 = 92.4% overall; all 96 high-confidence molecules (43%) within ×10. |
| For molecules with a **brand-new ring skeleton** (scaffold split)? | **No.** R² = 0.82, within ×10 = 75.8%, and the referee almost never says "high". |

So SoluAI v2 is not "95% accurate" as a blanket statement, and with today's public data nothing
is: when two laboratories measure the same molecule, their values typically differ by 0.5–0.7 log
units, which alone caps R² at roughly 0.9–0.95 for a dataset like this. What v2 can truthfully
say is narrower and more useful: **"when I tell you I am confident, I am right within a factor of
10 about 19 times out of 20, and I tell you so for about a quarter of new molecules."**

Three cautions about that sentence:

1. It was measured on molecules that resemble the training data (random split). The label keeps
   its meaning on new skeletons, but is rarely awarded there.
2. Reliable lab values belong, on average, to simpler and better-studied molecules. Part of the
   0.95 in the third row is "easy molecules", not only "clean labels".
3. Reporting only the best subset would be cherry-picking. That is why every table in this
   document also shows the full test set and the low-confidence group.

## 8. What we tried that did not work

Finding the ceiling took most of the work. Each idea was judged by cross-validation on the
training part (AqSolDB; numbers are CV RMSE, random / scaffold). The scripts and their original
output are in [`research/`](../research/README.md).

| Idea | Result | Verdict |
|---|---|---|
| Starting point: v1 features, one fixed LightGBM setting | 0.834 / 0.947 | |
| **All RDKit descriptors + fingerprint** | 0.804 / 0.906 | **kept** |
| **+ Huber loss** | 0.800 / 0.904 | **kept** (small, but free) |
| Count fingerprints, MACCS keys, atom pairs, torsions, E-state and Crippen atom types | 0.798–0.814 / 0.902–0.915 | no gain |
| A short readable list of 15–40 descriptors instead of all | 0.823–0.803 / 0.930–0.900 | loses the gain |
| Other learners: extra-trees, support-vector regression (two kinds) | 0.849–1.107 / 0.996–1.654 | worse |
| Stacking four models | 0.795 / 0.898 | +0.01 for 4× the cost, no exact SHAP |
| "Analogue correction": add the error made on the most similar training molecules | 0.823 / 0.902 | **worse** than no correction |
| Weighting repeat-measured molecules 2–3× more | 0.803–0.805 / 0.908–0.909 | no gain |
| Down-weighting training values the model strongly disagrees with | test only: 0.800–0.803 / 0.951–0.953 | no gain |
| A bigger model (3,000 trees, 63 leaves) | 0.805 / 0.917 | no gain |

Every reasonable variant lands at CV ≈ 0.80 (random) and ≈ 0.90 (scaffold). When very different
models and features all stop at the same number, the number is a property of the data.

The most instructive failure was the analogue correction. It seemed obvious: if the model
overestimated a molecule's close cousins, lower the prediction. It made predictions worse,
because the cousins' errors are mostly measurement noise and do not carry over. But the *size* of
those errors does carry over: regions of chemistry where the model was badly wrong stay
unpredictable. That observation became trust signal 4 and the referee.

Not tried: graph neural networks (heavy, small published gains on this data, approximate
explanations), a melting-point model as an extra input, and more data. The last one is the most
promising next step.

## 9. Limits

* **The data's noise is still the ceiling.** v2 does not make the average prediction dramatically
  better. It makes it somewhat better, and it tells you which predictions are the good ones.
* **The referee is a statistical judge.** "High confidence" means "molecules that looked like this
  were predicted well". It can be wrong for an individual molecule. About 1 in 20 high-confidence
  test molecules was still off by more than a factor of 10.
* **The 90% is an average.** Conformal prediction guarantees about 90% coverage over many
  molecules like the training data, not for each molecule or each chemical family separately.
* **Molecules in the training data** get narrow ranges because the chemist has seen their lab
  value. The website shows that lab value next to the prediction.
* **Truly alien molecules** (nothing above 20–30% similarity) are rare in the test sets, so the
  referee has little evidence about them. The website still shows the similarity of the closest
  training molecule.
* Everything in the [model card](model_card.md) still applies: no temperature, no pH, no crystal
  structure, organic molecules only.

## 10. Reproduce

```bash
pip install -r requirements-dev.txt && pip install -e .
python -m soluai.train --config config.yaml      # benchmark + final model, about 55 min on 4 cores
pytest                                           # includes tests/test_trust.py
```

The tables in section 6 are written to `results/metrics.md`; the raw numbers are in
`results/metrics.json` and `results/report.json` (key `trust`). The exploratory experiments of
sections 1 and 8 are in [`research/`](../research/README.md).

### References

* Sorkun, Khetan & Er (2019), AqSolDB. *Scientific Data* 6:143.
* Huber (1964), Robust estimation of a location parameter. *Ann. Math. Statist.* 35:73–101.
* Ke et al. (2017), LightGBM. *NeurIPS*.
* Papadopoulos, Vovk & Gammerman (2011), Regression conformal prediction with nearest neighbours.
  *JAIR* 40:815–840. (normalised conformal intervals)
* Lei et al. (2018), Distribution-free predictive inference for regression. *JASA* 113:1094–1111.
* Sheridan (2012), Three useful dimensions for domain applicability in QSAR models using random
  forest. *J. Chem. Inf. Model.* 52:814–823. (similarity and prediction value as error predictors)
* Palmer & Mitchell (2014), Is experimental data quality the limiting factor in predicting the
  aqueous solubility of druglike molecules? *Mol. Pharmaceutics* 11:2962–2972.
