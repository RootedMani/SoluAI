# SoluAI 💧: will it dissolve in water?

**SoluAI predicts how well an organic molecule dissolves in water, and explains *why* in
chemistry terms.** Type a name (`caffeine`), a SMILES string (`CCO`) or pick a molecule. You get:

* the predicted solubility (logS in mol/L, and g/L) with a **90% confidence range**;
* a solubility class on a colour scale (highly soluble → insoluble);
* the 2D structure with **atoms coloured by how they affect solubility** (blue = helps, red = hurts);
* a plain-language explanation, e.g. *"The five –OH groups increase solubility because they form
  hydrogen bonds with water."*;
* a warning when the molecule is **outside the model's experience**.

It is a real, trained machine-learning model (LightGBM gradient boosting on RDKit descriptors and
Morgan fingerprints), trained on 8,754 measured molecules from AqSolDB. It is **not** an LLM.

> **Live site:** <https://soluai.onrender.com>
>
> A chemistry × computer science project. The [presentation guide](docs/presentation_guide.md)
> has a 6-minute talk outline, demo molecules and likely questions.

![Predict page](docs/screenshots/predict-light.png)

| Compare | Explore the data |
|---|---|
| ![Compare](docs/screenshots/compare-light.png) | ![Explore](docs/screenshots/explore-light.png) |
| **How the model works** | **Dark mode** |
| ![Model](docs/screenshots/model-light.png) | ![Dark](docs/screenshots/predict-dark.png) |

## Quick start

Requires **Python 3.11–3.13** (Windows, macOS or Linux).

```bash
git clone https://github.com/RootedMani/soluai.git && cd soluai
```
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```
```bash
uvicorn web.main:app
```

Then open <http://127.0.0.1:8000>. The trained model is included, so no training is needed.

**Retrain everything** (about 30 minutes, fully reproducible):

```bash
pip install -r requirements-dev.txt && pip install -e .
```
```bash
python -m soluai.train --config config.yaml
```

**Run the tests:**

```bash
pytest
```

**Put it online:** see [docs/deployment.md](docs/deployment.md). Deploying on Render takes a few
clicks with the included `render.yaml`.

## Results

Test-set scores. The test molecules were hidden during training and tuning and used only once.
RMSE and MAE are in log units: 1.0 means "off by a factor of 10". R² = share of the variation explained.

| Dataset · split | Delaney equation (2004) | Linear regression | Random forest | **LightGBM** |
|---|---|---|---|---|
| AqSolDB · random | 1.35 / 0.65 | 1.31 / 0.67 | 0.88 / 0.85 | **0.84 / 0.87** |
| AqSolDB · scaffold | 1.44 / 0.57 | 1.41 / 0.59 | 1.07 / 0.76 | **1.01 / 0.79** |
| ESOL · random | 1.07 / 0.70 | 0.89 / 0.79 | 0.62 / 0.90 | **0.61 / 0.90** |
| ESOL · scaffold | 1.24 / 0.64 | 1.21 / 0.66 | 0.81 / 0.85 | **0.83 / 0.84** |

Each cell is test RMSE / R². Random forest and LightGBM use descriptors + fingerprint; linear
regression uses the descriptors. Models were chosen by cross-validation, not by test score. Full
tables, including descriptor-only and fingerprint-only models and CV scores, are in
[results/metrics.md](results/metrics.md).

**What this shows**

* Machine learning cuts the error of the chemist's hand-made formula by 30–43% (RMSE).
* **New ring skeletons are harder** (scaffold split): the error rises from 0.84 to 1.01 on AqSolDB.
  The random split flatters every model.
* **Descriptors carry the big physical trends.** Fingerprints alone are clearly worse; combining
  both is best by cross-validation. On the small ESOL scaffold test, random forest happened to score
  slightly better than the CV-chosen LightGBM (0.81 vs 0.83). With about 230 test molecules that
  difference is within noise.
* The model **rediscovered textbook chemistry**: logP is its most important feature and pushes
  solubility down, while polar surface area and H-bond donors push it up
  ([details](docs/chemistry.md#8-how-this-connects-to-soluai)).
* **Honest caveats:** the 90% range covered 87–89% of test molecules (slightly under target). The
  "unfamiliar molecule" warning is only a weak signal. A good part of the remaining error is noise
  in the measurements themselves. See the [model card](docs/model_card.md) for limitations and
  real failure cases.

## Project structure

```
soluai/
├── config.yaml              all training settings (seed, splits, search space)
├── data/raw/                original datasets + SOURCES.md (URLs, citations, licenses, checksums)
├── data/processed/          cleaned datasets + cleaning report
├── src/soluai/              the Python package
│   ├── data.py              download, clean, canonicalise
│   ├── features.py          RDKit descriptors + Morgan fingerprints
│   ├── splits.py            random split, scaffold split, CV folds
│   ├── baselines.py         Delaney's ESOL equation
│   ├── models.py            linear / random forest / LightGBM factories
│   ├── train.py             one-command training + evaluation
│   ├── evaluate.py          RMSE, MAE, R²
│   ├── uncertainty.py       conformal 90% range + similarity check
│   ├── explain.py           SHAP → atom colours + plain-language sentences
│   ├── groups.py            functional groups (SMARTS)
│   ├── predict.py           load model, predict
│   ├── resolve.py           name/SMILES → molecule, friendly errors, PubChem
│   └── common_molecules.py  offline list of ~60 molecules
├── models/soluai-v1.0.0/    trained model + metadata (versioned)
├── results/                 metrics tables and data for the website
├── web/                     FastAPI server (main.py) + HTML/CSS/JS pages (static/)
├── notebooks/               01_soluai_story.ipynb: the story with charts
├── docs/                    chemistry, model, model card, presentation guide, deployment
├── tests/                   pytest unit tests
└── render.yaml              one-click deployment on Render
```

## Documentation

* [docs/chemistry.md](docs/chemistry.md): the chemistry of solubility, for high school students
* [docs/model.md](docs/model.md): the ML pipeline step by step in simple language
* [docs/model_card.md](docs/model_card.md): intended use, metrics, limitations, failure cases
* [docs/presentation_guide.md](docs/presentation_guide.md): 6-minute talk, demo plan, Q&A
* [docs/deployment.md](docs/deployment.md): putting the website online
* [docs/soluai-documentation-fa.pdf](docs/soluai-documentation-fa.pdf): project documentation in Persian, for teachers (مستندات پروژه)
* [docs/soluai-guide-fa.pdf](docs/soluai-guide-fa.pdf): complete guide in Persian (راهنمای کامل پروژه به فارسی)

## Data and credits

* **AqSolDB:** Sorkun, Khetan & Er (2019), *Scientific Data* 6:143, [doi:10.1038/s41597-019-0151-1](https://doi.org/10.1038/s41597-019-0151-1). CC0.
* **ESOL:** Delaney (2004), *J. Chem. Inf. Comput. Sci.* 44:1000–1005, [doi:10.1021/ci034243x](https://doi.org/10.1021/ci034243x), via MoleculeNet/DeepChem.
* Built with [RDKit](https://www.rdkit.org), [LightGBM](https://lightgbm.readthedocs.io), [scikit-learn](https://scikit-learn.org), [FastAPI](https://fastapi.tiangolo.com) and [Plotly.js](https://plotly.com/javascript/) (bundled, MIT).

Code: MIT license. Datasets: see [data/raw/SOURCES.md](data/raw/SOURCES.md).
