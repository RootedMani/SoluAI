# SoluAI 💧: will it dissolve in water?

**SoluAI predicts how well an organic molecule dissolves in water, and explains *why* in
chemistry terms.** Type a name (`caffeine`, `کافئین`), a SMILES string (`CCO`), draw the molecule or pick one. You get:

* the predicted solubility (logS in mol/L, and g/L) with a **90% range sized for that molecule** and a
  **confidence label** (high / medium / low);
* a solubility class on a colour scale (highly soluble → insoluble);
* the 2D structure with **atoms coloured by how they affect solubility** (blue = helps, red = hurts);
* a plain-language explanation, e.g. *"The five –OH groups increase solubility because they form
  hydrogen bonds with water."*;
* the **lab value** (if known) and the **classic 2004 formula** next to the AI's answer;
* everyday units: grams in a glass of water, and how many litres you need to dissolve 1 g.

Around the model there is a small chemistry toolkit ([details](docs/features.md)):

| | |
|---|---|
| **Draw** a molecule, or type one of 225 offline names (English or Persian) | **Salts:** NaCl, AgCl, CaCO₃… answered with the textbook solubility rules, plus a "will a precipitate form?" mixing lab |
| **Make it more soluble:** the model tries small edits and ranks them | **Trends:** solubility along seven homologous series, prediction vs lab |
| **Quiz:** which one dissolves better? You vs the AI | **Batch:** up to 200 molecules from a list or CSV, results as CSV |
| **Persian / English** interface with right-to-left layout | **Works offline** and prints a one-page PDF report |

It is a real, trained machine-learning model, trained on 8,754 measured molecules from AqSolDB. It is
**not** an LLM. Since version 2.0 it uses the project's own **Chemist–Referee algorithm**
([docs/algorithm.md](docs/algorithm.md)): one model (the *chemist*, robust gradient boosting on 1,242
chemistry features) predicts the solubility, and a second one (the *referee*) looks at the most
similar measured molecules and predicts how far off the chemist is likely to be. When the referee says
"high confidence", the prediction was within a factor of 10 of the lab value for 96% of unseen test
molecules.

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
| **Salts and the mixing lab** | **Trends** |
| ![Salts](docs/screenshots/salts-light.png) | ![Trends](docs/screenshots/trends-light.png) |
| **Quiz** | **Persian (فارسی)** |
| ![Quiz](docs/screenshots/quiz-light.png) | ![Persian](docs/screenshots/predict-fa.png) |

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
python run.py
```

This starts the site and opens <http://127.0.0.1:8000> in your browser (same as `uvicorn web.main:app`).
The trained model is included, so no training is needed. **No internet is needed either**: fonts,
charts and the drawing editor are bundled. Only looking up unusual molecule names uses PubChem.

**Retrain everything** (about 55 minutes on 4 CPU cores, fully reproducible):

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

| Dataset · split | Delaney equation (2004) | Linear regression | Random forest | LightGBM (model v1) | **SoluAI v2** |
|---|---|---|---|---|---|
| AqSolDB · random | 1.35 / 0.65 | 1.31 / 0.67 | 0.88 / 0.85 | 0.84 / 0.87 | **0.79 / 0.88** |
| AqSolDB · scaffold | 1.44 / 0.57 | 1.41 / 0.59 | 1.07 / 0.76 | 1.01 / 0.79 | **0.94 / 0.82** |
| ESOL · random | 1.07 / 0.70 | 0.89 / 0.79 | 0.62 / 0.90 | 0.61 / 0.90 | **0.58 / 0.91** |
| ESOL · scaffold | 1.24 / 0.64 | 1.21 / 0.66 | 0.81 / 0.85 | 0.83 / 0.84 | 0.82 / 0.84 |

Each cell is test RMSE / R². Random forest and LightGBM use descriptors + fingerprint; linear
regression uses the descriptors; SoluAI v2 adds 204 extended descriptors and a robust loss. Models
were chosen by cross-validation, not by test score. Full tables, including descriptor-only and
fingerprint-only models and CV scores, are in [results/metrics.md](results/metrics.md).

**How much can you trust one prediction?** SoluAI v2 labels every prediction. On the AqSolDB test
molecules (never seen in training):

| The referee says | Random split: share · within a factor of 10 | Scaffold split: share · within a factor of 10 |
|---|---|---|
| High confidence | 27% · **96%** | 1% · 95% |
| Medium confidence | 47% · 85% | 46% · 85% |
| Low confidence | 26% · 68% | 53% · 68% |
| All | 100% · 84% | 100% · 76% |

**What this shows**

* Machine learning cuts the error of the chemist's hand-made formula by 35–41% (RMSE).
* **New ring skeletons are harder** (scaffold split): the error rises from 0.79 to 0.94 on AqSolDB.
  The random split flatters every model.
* **v2 is a modest step in accuracy** (5–7% lower error than v1 on AqSolDB, no real change on the
  small ESOL scaffold test) **and a big step in knowing when it is right.** The 90% range used to
  be about ± 1.3–1.5 for every molecule; now it runs from ± 0.6 for easy molecules to ± 2.3 for
  hard ones, and still covers 89–90% of test molecules.
* **Is it 95% accurate? Not overall.** It reaches that level for the predictions it labels "high
  confidence" (96% within a factor of 10, R² 0.96) and against lab values that were measured more
  than once and agree (95%, R² 0.95). Over all molecules it is 84%, and less for new skeletons.
  [The honest answer in full](docs/algorithm.md#7-is-it-95-accurate).
* **The data, not the model, is now the limit.** A dozen further ideas (other models, stacking,
  more features, noise weighting) all stopped at the same score
  ([what did not work](docs/algorithm.md#8-what-we-tried-that-did-not-work)).
* The model **rediscovered textbook chemistry**: logP is its most important feature and pushes
  solubility down, while polar surface area and H-bond donors push it up
  ([details](docs/chemistry.md#8-how-this-connects-to-soluai)).
* **Honest caveats:** the confidence label is a statistical judgement, not a guarantee (about 1 in
  20 high-confidence test predictions was still off by more than a factor of 10). A good part of the
  remaining error is noise in the measurements themselves. See the [model card](docs/model_card.md)
  for limitations and real failure cases.

## Project structure

```
soluai/
├── config.yaml              all training settings (seed, splits, search space)
├── data/raw/                original datasets + SOURCES.md (URLs, citations, licenses, checksums)
├── data/processed/          cleaned datasets + cleaning report
├── src/soluai/              the Python package
│   ├── data.py              download, clean, canonicalise
│   ├── features.py          RDKit descriptors (14 readable + 204 extended) + Morgan fingerprints
│   ├── splits.py            random split, scaffold split, CV folds
│   ├── baselines.py         Delaney's ESOL equation
│   ├── models.py            linear / random forest / LightGBM factories (v2 = LightGBM with Huber loss)
│   ├── train.py             one-command training + evaluation
│   ├── evaluate.py          RMSE, MAE, R²
│   ├── uncertainty.py       the referee (trust model), conformal 90% range, confidence label
│   ├── explain.py           SHAP → atom colours + plain-language sentences
│   ├── groups.py            functional groups (SMARTS)
│   ├── predict.py           load model, predict
│   ├── resolve.py           name/SMILES → molecule, friendly errors, PubChem
│   ├── common_molecules.py  offline list of 225 molecules, with Persian names
│   ├── ionic.py             salts: solubility rules, equations, mixing two solutions
│   └── lab.py               series, "make it more soluble", quiz, batch, kitchen test
├── models/soluai-v2.0.0/    trained model + metadata (versioned; v1.0.0 is kept for comparison)
├── results/                 metrics tables and data for the website
├── web/                     FastAPI server (main.py) + HTML/CSS/JS pages (static/)
│   └── static/vendor/       bundled Plotly, JSME editor and fonts (see THIRD-PARTY.md)
├── run.py                   start the site locally and open the browser
├── notebooks/               01_soluai_story.ipynb: the story with charts (model v1)
├── research/                exploratory experiments behind v2, including the ideas that failed
├── docs/                    chemistry, model, model card, presentation guide, deployment
├── tests/                   pytest unit tests
└── render.yaml              one-click deployment on Render
```

## Documentation

* [docs/chemistry.md](docs/chemistry.md): the chemistry of solubility, for high school students
* [docs/features.md](docs/features.md): the tools around the model (salts, trends, quiz, batch, Persian, offline)
* [docs/algorithm.md](docs/algorithm.md): **the Chemist–Referee algorithm (v2)**: idea, maths, results, what did not work, and the honest answer to "is it 95% accurate?"
* [docs/model.md](docs/model.md): the ML pipeline step by step in simple language
* [docs/model_card.md](docs/model_card.md): intended use, metrics, limitations, failure cases
* [docs/presentation_guide.md](docs/presentation_guide.md): 6-minute talk, demo plan, Q&A
* [docs/deployment.md](docs/deployment.md): putting the website online
* The two Persian PDFs below describe version 1.0 (before the tools in docs/features.md and the v2 algorithm were added).
* [docs/soluai-documentation-fa.pdf](docs/soluai-documentation-fa.pdf): project documentation in Persian, for teachers (مستندات پروژه)
* [docs/soluai-guide-fa.pdf](docs/soluai-guide-fa.pdf): complete guide in Persian (راهنمای کامل پروژه به فارسی)

## Data and credits

* **AqSolDB:** Sorkun, Khetan & Er (2019), *Scientific Data* 6:143, [doi:10.1038/s41597-019-0151-1](https://doi.org/10.1038/s41597-019-0151-1). CC0.
* **ESOL:** Delaney (2004), *J. Chem. Inf. Comput. Sci.* 44:1000–1005, [doi:10.1021/ci034243x](https://doi.org/10.1021/ci034243x), via MoleculeNet/DeepChem.
* **Salt solubilities:** g/100 mL at 20 °C from the Wikipedia [Solubility table](https://en.wikipedia.org/wiki/Solubility_table) (CC BY-SA), which collects handbook values.
* Bundled: [JSME](https://jsme-editor.github.io/) molecule editor (BSD-3), Inter, JetBrains Mono and Vazirmatn fonts (OFL). See [THIRD-PARTY.md](web/static/vendor/THIRD-PARTY.md).
* Built with [RDKit](https://www.rdkit.org), [LightGBM](https://lightgbm.readthedocs.io), [scikit-learn](https://scikit-learn.org), [FastAPI](https://fastapi.tiangolo.com) and [Plotly.js](https://plotly.com/javascript/) (bundled, MIT).

Code: MIT license. Datasets: see [data/raw/SOURCES.md](data/raw/SOURCES.md).
