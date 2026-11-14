# Deploying SoluAI on Render

SoluAI is one small Python web service: FastAPI serves both the JSON API and the
five HTML pages. The trained model is stored in the repository (`models/`), so
Render only installs packages and starts the server. It never trains.

## One-time setup (about 5 minutes)

1. Push this repository to GitHub (already done if you are reading this there).
2. Go to <https://dashboard.render.com> and sign in with GitHub.
3. Click **New → Blueprint**, choose the `soluai` repository and click **Apply**.
   Render reads `render.yaml` and creates a free web service called `soluai`.
4. Wait for the first build (5–10 minutes; RDKit and LightGBM are big downloads).
5. Open the URL Render shows (something like `https://soluai.onrender.com`).

Prefer clicking? **New → Web Service**, pick the repo, and enter:

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn web.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/api/health` |
| Environment variable | `PYTHON_VERSION` = `3.13.18` |
| Instance type | Free |

Every `git push` to `main` redeploys automatically.

## Things to know

* **Cold starts:** free services sleep after about 15 minutes without visitors. The
  next visit takes 30–60 s while the server wakes up. **Open the site a minute
  before your presentation.**
* **Memory:** the free plan has 512 MB. SoluAI uses about 250 MB (measured locally;
  see the README). This is why the website uses LightGBM's built-in SHAP rather
  than the heavier `shap` package.
* **Name lookup** for unusual names (e.g. typing "squalane") calls PubChem over the internet. If PubChem
  is slow or down, the site still works with SMILES, drawing, salts and the built-in list of 225 molecules.
* **No other internet is needed.** Fonts, charts and the drawing editor are bundled in
  `web/static/vendor/`, so the same code runs on a laptop with no connection: `python run.py`.
* **Show it on phones in the classroom** (no Render needed): run `python run.py --host 0.0.0.0` and
  open `http://<your-laptop-IP>:8000` on any device on the same Wi-Fi.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails on `rdkit` or `numpy` | Check `PYTHON_VERSION` is 3.11–3.13 (pandas 3 needs ≥ 3.11). |
| "Application failed to respond" | Check the start command uses `--host 0.0.0.0 --port $PORT`. |
| Page loads but predictions fail | Open `/api/health`. If it errors, check that `models/` and `results/` were pushed (they are not in `.gitignore`). |
| First request very slow | That's the cold start. Wait about a minute. |
