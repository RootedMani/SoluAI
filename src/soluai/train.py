"""The whole experiment in one command.

    python -m soluai.train --config config.yaml

Steps:
1. clean both datasets;
2. for each dataset (AqSolDB, ESOL) and each split type (random, scaffold):
   tune every model with 5-fold cross-validation on the training part only,
   then score it ONCE on the held-out test part. The last model in each table is
   **SoluAI v2**: robust boosting on the extended features (the "chemist") plus
   the trust model (the "referee"), see docs/algorithm.md;
3. check the 90 % confidence intervals and the confidence labels on the test sets;
4. train the final model (SoluAI v2) on all of AqSolDB and save it with its metadata;
5. write results/metrics.json, results/metrics.md and results/report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import date
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import Chem
from sklearn.model_selection import ParameterSampler

from . import MODELS_DIR, RESULTS_DIR, __version__
from .baselines import delaney_matrix, delaney_predict
from .config import config_hash, load_config
from .data import RAW, build_clean_datasets
from .evaluate import metrics
from .features import DESCRIPTOR_NAMES, EXTRA_DESCRIPTOR_NAMES, FAMILIES, PRETTY, Featurizer, family
from .models import make_model
from .splits import cv_splitter, random_split, scaffold_split
from .uncertainty import (ConformalByDomain, TrustModel, conformal_quantile, max_similarity, tanimoto_matrix,
                          trust_report, trust_signals)

log = logging.getLogger("soluai.train")
FEATURE_SETS = ["desc", "fp", "desc+fp"]
V2 = "SoluAI v2"          # row name of the new algorithm in the result tables
V2_FEATURES = "rdkit+fp"


# ----------------------------------------------------------------------------- helpers
def column_slice(kind: str, n_bits: int) -> np.ndarray:
    """Which columns of the full matrix (14 descriptors | extended descriptors | fingerprint) a feature set uses."""
    n_desc, n_extra = len(DESCRIPTOR_NAMES), len(EXTRA_DESCRIPTOR_NAMES)
    desc, fp = np.arange(n_desc), np.arange(n_desc + n_extra, n_desc + n_extra + n_bits)
    return {"desc": desc, "fp": fp, "desc+fp": np.r_[desc, fp], V2_FEATURES: np.arange(n_desc + n_extra + n_bits)}[kind]


def make_trust(cfg: dict) -> TrustModel:
    """An unfitted trust model (the "referee") with the settings from the config file."""
    t = cfg["uncertainty"]["trust"]
    return TrustModel(coverage=cfg["uncertainty"]["coverage"], k=t["neighbours"], floor=t["floor"],
                      tiers=dict(t["tiers"]), seed=cfg["seed"])


def fit_trust(cfg: dict, X: np.ndarray, y: np.ndarray, runs: list[tuple[np.ndarray, list]], n_bits: int):
    """Fit the referee on out-of-fold predictions.

    ``runs`` is a list of ``(out-of-fold predictions, folds)``: one cross-validation in the
    benchmark, two (scaffold folds and random folds) for the final model, so that the referee
    sees both "new skeleton" and "close cousin" situations. For the training molecules,
    neighbours are only taken from *other* folds, exactly like a molecule the model never saw.

    Returns ``(trust, cross-fitted sigma per run, neighbour error per molecule, signal function)``.
    """
    j = {n: i for i, n in enumerate(DESCRIPTOR_NAMES)}
    bits = X[:, column_slice("fp", n_bits)]
    n = len(y)
    nb_err = np.mean([np.abs(oof - y) for oof, _ in runs], axis=0)   # how wrong the model was on each molecule
    trust = make_trust(cfg)

    def signals(sim: np.ndarray, pred: np.ndarray, Xq: np.ndarray) -> np.ndarray:
        return trust_signals(sim, y, nb_err, pred, Xq[:, j["MolLogP"]], Xq[:, j["MolWt"]], Xq[:, j["HeavyAtomCount"]], trust.k)

    base = tanimoto_matrix(bits, bits)
    S, target = [], []
    for oof, folds in runs:
        sims = base.copy()
        for _, va in folds:
            sims[np.ix_(va, va)] = -1.0
        S.append(signals(sims, oof, X))
        target.append(np.abs(oof - y))
    # Both copies of a molecule stay in the same referee fold, so its sigma is always honest.
    ref_folds = [(np.concatenate([tr + r * n for r in range(len(runs))]), np.concatenate([va + r * n for r in range(len(runs))]))
                 for tr, va in runs[0][1]]
    sigma = trust.fit(np.vstack(S), np.concatenate(target), ref_folds)
    return (trust, sigma.reshape(len(runs), n), nb_err,
            lambda Xq, pred: signals(tanimoto_matrix(Xq[:, column_slice("fp", n_bits)], bits), pred, Xq))


def cross_validate(kind: str, params: dict, X: np.ndarray, y: np.ndarray, folds, seed: int) -> tuple[list[float], np.ndarray]:
    """Fold RMSEs and out-of-fold predictions for one model setting."""
    oof = np.zeros(len(y))
    scores = []
    for tr, va in folds:
        m = make_model(kind, params, seed).fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
        scores.append(metrics(y[va], oof[va])["rmse"])
    return scores, oof


def random_search(kind: str, space: dict, n_iter: int, X, y, folds, seed: int) -> dict[str, Any]:
    """Try ``n_iter`` random settings; keep the one with the lowest mean CV RMSE."""
    best: dict[str, Any] = {"cv_rmse": np.inf}
    for params in ParameterSampler(space, n_iter=n_iter, random_state=seed):
        scores, oof = cross_validate(kind, params, X, y, folds, seed)
        if np.mean(scores) < best["cv_rmse"]:
            best = {"params": params, "cv_rmse": float(np.mean(scores)), "cv_sd": float(np.std(scores)), "oof": oof}
    return best


def fold_similarities(fps: list, folds) -> np.ndarray:
    """For each molecule: max similarity to the training part of the fold where it was held out."""
    sim = np.zeros(len(fps))
    for tr, va in folds:
        ref = [fps[i] for i in tr]
        sim[va] = max_similarity([fps[i] for i in va], ref)
    return sim


def fragment_label(mol: Chem.Mol, atom: int, radius: int) -> str:
    """SMILES of the small fragment a fingerprint bit stands for."""
    if radius == 0:
        a = mol.GetAtomWithIdx(atom)
        return f"[{a.GetSymbol()}{'H' * a.GetTotalNumHs() if a.GetTotalNumHs() else ''}]" + (" (aromatic)" if a.GetIsAromatic() else "")
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom)
    atoms = {atom}
    for b in env:
        bond = mol.GetBondWithIdx(b)
        atoms.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), bondsToUse=list(env) or None, canonical=True)


def bit_labels(smiles: list[str], feat: Featurizer, bits: list[int]) -> dict[int, str]:
    """Find an example fragment for each requested fingerprint bit."""
    wanted, labels = set(bits), {}
    for s in smiles:
        if not wanted:
            break
        mol = Chem.MolFromSmiles(s)
        _, info = feat.fingerprint_with_info(mol)
        for b in list(wanted):
            if b in info:
                a, r = info[b][0]
                labels[b] = fragment_label(mol, a, r)
                wanted.discard(b)
    return labels


# ----------------------------------------------------------------------------- experiments
def run_experiments(name: str, df: pd.DataFrame, cfg: dict, X_all: np.ndarray, fps: list) -> tuple[list[dict], dict]:
    """All models x feature sets x split types for one dataset."""
    seed, n_bits = cfg["seed"], cfg["features"]["morgan_bits"]
    smiles, y = df["smiles"].tolist(), df["logS"].to_numpy()
    rows: list[dict] = []
    extras: dict[str, Any] = {}
    for split in ["random", "scaffold"]:
        t0 = time.time()
        if split == "random":
            tr, te = random_split(len(df), cfg["split"]["test_fraction"], seed)
        else:
            tr, te = scaffold_split(smiles, cfg["split"]["test_fraction"], seed)
        tr_smiles = [smiles[i] for i in tr]
        folds = cv_splitter(split, cfg["split"]["cv_folds"], seed, tr_smiles)
        ytr, yte = y[tr], y[te]
        preds: dict[str, np.ndarray] = {}

        def record(model: str, features: str, cv: list[float] | tuple[float, float], test_pred, params=None):
            cv_mean, cv_sd = (float(np.mean(cv)), float(np.std(cv))) if isinstance(cv, list) else cv
            rows.append({
                "dataset": name, "split": split, "model": model, "features": features,
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "cv_rmse": round(cv_mean, 3), "cv_rmse_sd": round(cv_sd, 3),
                **{f"test_{k}": v for k, v in metrics(yte, test_pred).items()},
                "params": params or {},
            })
            log.info("%-8s %-8s %-15s %-7s CV %.3f  test %s", name, split, model, features, cv_mean, metrics(yte, test_pred))

        # (a) Delaney's original equation: nothing to fit.
        D = delaney_matrix(smiles)
        cv = [metrics(ytr[va], delaney_predict([tr_smiles[i] for i in va]))["rmse"] for _, va in folds]
        preds["Delaney (original)"] = delaney_predict([smiles[i] for i in te])
        record("Delaney (original)", "4 ESOL vars", cv, preds["Delaney (original)"])

        # (b') Delaney's four variables, re-fitted.
        Dtr = D[tr]
        cv = [metrics(ytr[va], fit_delaney_refit_from(Dtr[t], ytr[t]).predict(Dtr[va]))["rmse"] for t, va in folds]
        refit = fit_delaney_refit_from(Dtr, ytr)
        preds["Delaney (refit)"] = refit.predict(D[te])
        record("Delaney (refit)", "4 ESOL vars", cv, preds["Delaney (refit)"],
               {"intercept": round(float(refit.intercept_), 3), "coef": [round(float(c), 4) for c in refit.coef_]})

        # (b) Linear regression on all 14 descriptors.
        Xd = X_all[:, column_slice("desc", n_bits)]
        cv, _ = cross_validate("linear", {}, Xd[tr], ytr, folds, seed)
        lin = make_model("linear").fit(Xd[tr], ytr)
        preds["Linear regression"] = lin.predict(Xd[te])
        record("Linear regression", "desc", cv, preds["Linear regression"])

        # Random forest and LightGBM on three feature sets.
        best_by_model: dict[str, tuple[float, str, dict, np.ndarray]] = {}
        for kind, label in [("rf", "Random forest"), ("lgbm", "LightGBM")]:
            for fs in FEATURE_SETS:
                X = X_all[:, column_slice(fs, n_bits)]
                best = random_search(kind, cfg["search"][{"rf": "random_forest", "lgbm": "lightgbm"}[kind]],
                                     cfg["search"]["n_iter"][{"rf": "random_forest", "lgbm": "lightgbm"}[kind]],
                                     X[tr], ytr, folds, seed)
                model = make_model(kind, best["params"], seed).fit(X[tr], ytr)
                p = model.predict(X[te])
                record(label, fs, (best["cv_rmse"], best["cv_sd"]), p, best["params"])
                if label not in best_by_model or best["cv_rmse"] < best_by_model[label][0]:
                    best_by_model[label] = (best["cv_rmse"], fs, best["params"], p, best["oof"])
        for label, (_, fs, params, p, oof) in best_by_model.items():
            preds[f"{label} ({fs})"] = p

        # SoluAI v2, part 1 (the "chemist"): robust boosting on the extended feature set.
        Xtr2, Xte2 = X_all[tr], X_all[te]
        best2 = random_search("lgbm", cfg["search"]["soluai_v2"], cfg["search"]["n_iter"]["soluai_v2"], Xtr2, ytr, folds, seed)
        p_v2 = make_model("lgbm", best2["params"], seed).fit(Xtr2, ytr).predict(Xte2)
        record(V2, V2_FEATURES, (best2["cv_rmse"], best2["cv_sd"]), p_v2, best2["params"])
        preds[f"{V2} ({V2_FEATURES})"] = p_v2

        # SoluAI v2, part 2 (the "referee"): adaptive 90 % ranges and confidence labels.
        trust, _, _, new_signals = fit_trust(cfg, Xtr2, ytr, [(best2["oof"], folds)], n_bits)
        hw_v2 = trust.half_width(new_signals(Xte2, p_v2))
        groups = df["group"].to_numpy()[te]
        reliable = np.isin(groups, cfg["uncertainty"]["trust"]["reliable_groups"])
        trust_rep = trust_report(yte, p_v2, hw_v2, trust, reliable if reliable.any() else None)
        q_fixed = conformal_quantile(np.abs(best2["oof"] - ytr), cfg["uncertainty"]["coverage"])
        trust_rep["fixed_width"] = {"half_width": round(q_fixed, 3),
                                    "coverage": round(float((np.abs(p_v2 - yte) <= q_fixed).mean()), 3)}
        log.info("%s %s trust: %s", name, split, {k: trust_rep[k] for k in ("all", "tiers", "fixed_width")})

        # Uncertainty check on the test set with the best LightGBM (chosen by CV, not by test!).
        _, fs, params, p_lgb, oof = best_by_model["LightGBM"]
        thr = cfg["uncertainty"]["domain_similarity_threshold"]
        sim_oof = fold_similarities([fps[i] for i in tr], folds)
        conf = ConformalByDomain(thr, cfg["uncertainty"]["coverage"]).fit(np.abs(oof - ytr), sim_oof)
        sim_te = max_similarity([fps[i] for i in te], [fps[i] for i in tr])
        hw = conf.half_width(sim_te)
        err = np.abs(p_lgb - yte)
        inside = sim_te >= thr
        bins = [0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01]
        by_sim = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (sim_te >= lo) & (sim_te < hi)
            if m.sum():
                by_sim.append({"range": f"{lo:.1f}-{min(hi, 1):.1f}", "n": int(m.sum()),
                               "rmse": round(float(np.sqrt(np.mean(err[m] ** 2))), 3)})
        extras[split] = {
            "test_idx": te.tolist(),
            "predictions": {k: np.round(v, 3).tolist() for k, v in preds.items()},
            "y_test": np.round(yte, 3).tolist(),
            "similarity_test": np.round(sim_te, 3).tolist(),
            "half_width_v2": np.round(hw_v2, 3).tolist(),
            "trust": trust_rep,
            "uncertainty": {
                "model": f"LightGBM ({fs})",
                "half_width_in": round(conf.q["in"], 3), "half_width_out": round(conf.q["out"], 3),
                "coverage_all": round(float((err <= hw).mean()), 3),
                "coverage_in": round(float((err[inside] <= hw[inside]).mean()), 3) if inside.any() else None,
                "coverage_out": round(float((err[~inside] <= hw[~inside]).mean()), 3) if (~inside).any() else None,
                "share_out_of_domain": round(float((~inside).mean()), 3),
                "rmse_in": round(float(np.sqrt(np.mean(err[inside] ** 2))), 3) if inside.any() else None,
                "rmse_out": round(float(np.sqrt(np.mean(err[~inside] ** 2))), 3) if (~inside).any() else None,
                "error_by_similarity": by_sim,
            },
        }
        log.info("%s %s done in %.0fs; uncertainty %s", name, split, time.time() - t0, extras[split]["uncertainty"])
    return rows, extras


def fit_delaney_refit_from(D: np.ndarray, y: np.ndarray):
    """Linear regression on an already computed Delaney variable matrix."""
    from sklearn.linear_model import LinearRegression

    return LinearRegression().fit(D, y)


# ----------------------------------------------------------------------------- final model
def train_final(df: pd.DataFrame, cfg: dict, X_all: np.ndarray, fps: list, rows: list[dict]) -> dict:
    """Train the deployed model on all of AqSolDB and save it with metadata."""
    seed, n_bits = cfg["seed"], cfg["features"]["morgan_bits"]
    best = next(r for r in rows if r["dataset"] == "aqsoldb" and r["split"] == "scaffold" and r["model"] == V2)
    fs, params = best["features"], best["params"]
    log.info("Final model: %s on %s with %s", V2, fs, params)
    X = X_all[:, column_slice(fs, n_bits)]
    y = df["logS"].to_numpy()
    smiles = df["smiles"].tolist()

    # Out-of-fold errors on the full data -> conformal interval for the final model.
    folds = cv_splitter("scaffold", cfg["split"]["cv_folds"], seed, smiles)
    _, oof = cross_validate("lgbm", params, X, y, folds, seed)
    sim = fold_similarities(fps, folds)
    conf = ConformalByDomain(cfg["uncertainty"]["domain_similarity_threshold"], cfg["uncertainty"]["coverage"])
    conf.fit(np.abs(oof - y), sim)
    # The referee: learns from the same honest out-of-fold errors how big the error usually is.
    # For the final model it studies two exams: the scaffold folds above ("new skeleton") and
    # random folds ("close cousins exist"), because visitors type both kinds of molecules.
    random_folds = cv_splitter("random", cfg["split"]["cv_folds"], seed, smiles)
    _, oof_random = cross_validate("lgbm", params, X, y, random_folds, seed)
    trust, sigma_oof, nb_err, _ = fit_trust(cfg, X, y, [(oof, folds), (oof_random, random_folds)], n_bits)
    hw_oof = trust.q * sigma_oof
    tier_share = {exam: {t: round(float(np.mean([trust.tier(h) == t for h in hw])), 3) for t in ("high", "medium", "low")}
                  for exam, hw in zip(("scaffold", "random"), hw_oof)}

    model = make_model("lgbm", params, seed).fit(X, y)
    feat = Featurizer(fs, cfg["features"]["morgan_radius"], n_bits)

    version_dir = MODELS_DIR / f"soluai-v{cfg['model_version']}"
    version_dir.mkdir(parents=True, exist_ok=True)
    desc = X_all[:, column_slice("desc", n_bits)]
    bundle = {
        "model": model,
        "feature_kind": fs,
        "radius": cfg["features"]["morgan_radius"],
        "n_bits": n_bits,
        "feature_names": feat.names,
        "extra_names": feat.extra_names,
        "trust": trust,
        "conformal": {"threshold": conf.threshold, "coverage": conf.coverage, "q": conf.q},
        "descriptor_ranges": {n: [float(desc[:, i].min()), float(desc[:, i].max())] for i, n in enumerate(DESCRIPTOR_NAMES)},
        "version": cfg["model_version"],
    }
    # Global SHAP values (LightGBM's built-in exact TreeSHAP) for every training molecule.
    contrib = model.predict(X, pred_contrib=True)[:, :-1]
    names = feat.names
    # Average SHAP "slope" of the atom-additive descriptors: how much one unit of
    # logP (or TPSA, MR) shifts the prediction. Used to share their effect among atoms.
    bundle["shap_slopes"] = {
        d: float(np.polyfit(X[:, names.index(d)], contrib[:, names.index(d)], 1)[0])
        for d in ["MolLogP", "TPSA", "MolMR"] if d in names
    }
    joblib.dump(bundle, version_dir / "model.joblib", compress=3)
    # The referee needs to know how wrong the model was on each training molecule when it had not seen it.
    df[["name", "smiles", "logS"]].assign(oof_abs_err=np.round(nb_err, 3)).to_csv(
        version_dir / "training_set.csv", index=False)

    mean_abs = np.abs(contrib).mean(0)
    order = np.argsort(-mean_abs)[:15]
    fp_bits = [int(names[i][3:]) for i in order if names[i].startswith("FP_")]
    labels = bit_labels(smiles, feat, fp_bits)
    importance = [
        {"feature": names[i],
         "label": PRETTY.get(names[i]) or (f"Fragment {labels.get(int(names[i][3:]), names[i])}"
                                            if names[i].startswith("FP_") else f"{names[i]} ({FAMILIES[family(names[i])].lower()})"),
         "mean_abs_shap": round(float(mean_abs[i]), 4)}
        for i in order
    ]
    fp_cols = [i for i, n in enumerate(names) if n.startswith("FP_")]
    group_importance = {
        "descriptors": round(float(mean_abs[[i for i, n in enumerate(names) if n in PRETTY]].sum()), 3),
        **{FAMILIES[f]: round(float(mean_abs[[i for i, n in enumerate(names)
                                              if n not in PRETTY and not n.startswith("FP_") and family(n) == f]].sum()), 3)
           for f in FAMILIES if feat.extra_names},
        "fingerprint fragments": round(float(mean_abs[fp_cols].sum()), 3) if fp_cols else 0.0,
    }
    # SHAP dependence for the chemistry sanity check.
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(y), size=min(1500, len(y)), replace=False)
    dependence, trend = {}, {}
    for d in ["MolLogP", "MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "AromaticProportion"]:
        if d in names:
            j = names.index(d)
            dependence[d] = {"x": np.round(X[sample, j], 3).tolist(), "shap": np.round(contrib[sample, j], 3).tolist()}
            trend[d] = round(float(np.corrcoef(X[:, j], contrib[:, j])[0, 1]), 3)

    meta = {
        "name": "SoluAI", "version": cfg["model_version"], "trained_on": date.today().isoformat(),
        "algorithm": "SoluAI v2: robust LightGBM boosting (chemist) + trust model (referee)",
        "features": fs, "n_features": len(names), "params": params,
        "n_training_molecules": int(len(y)), "config_hash": config_hash(cfg),
        "data_sha256": json.loads((RAW / "SOURCES.json").read_text()) if (RAW / "SOURCES.json").exists() else None,
        "conformal": bundle["conformal"],
        "trust": {"q": round(trust.q, 4), "floor": trust.floor, "neighbours": trust.k, "tiers": trust.tiers,
                  "tier_share_out_of_fold": tier_share,
                  "median_half_width_out_of_fold": {"scaffold": round(float(np.median(hw_oof[0])), 3),
                                                    "random": round(float(np.median(hw_oof[1])), 3)},
                  "random_cv_rmse": metrics(y, oof_random)["rmse"]},
        "scaffold_cv_rmse_of_choice": best["cv_rmse"],
        "packages": {"python": platform.python_version(), "rdkit": rdkit.__version__, "lightgbm": lightgbm.__version__,
                     "scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                     "soluai": __version__},
    }
    (version_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (MODELS_DIR / "LATEST").write_text(f"soluai-v{cfg['model_version']}\n")
    return {"meta": meta, "importance": importance, "group_importance": group_importance,
            "dependence": dependence, "shap_trend_correlation": trend, "oof_rmse": metrics(y, oof)}


def dataset_overlap(aq: pd.DataFrame, esol: pd.DataFrame) -> dict:
    """How many ESOL molecules are also in AqSolDB?

    AqSolDB was partly built from older datasets, including Delaney's. If almost all
    ESOL molecules are in AqSolDB, testing the AqSolDB model on ESOL would be
    "testing on the training data" - so we do NOT report such a cross-dataset score.
    """
    shared = esol["inchikey"].isin(set(aq["inchikey"]))
    return {"n_esol": int(len(esol)), "n_in_aqsoldb": int(shared.sum()), "n_not_in_aqsoldb": int((~shared).sum())}


# ----------------------------------------------------------------------------- reporting
def write_markdown(rows: list[dict], extras: dict, path) -> None:
    """Human-readable metrics tables."""
    lines = ["RMSE and MAE are in log units. *Within ×10* is the share of test molecules predicted within one",
             "log unit, i.e. within a factor of 10 of the measured solubility.", ""]
    pct = lambda v: "–" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    for ds in ["aqsoldb", "esol"]:
        for split in ["random", "scaffold"]:
            sub = [r for r in rows if r["dataset"] == ds and r["split"] == split]
            if not sub:
                continue
            title = f"{ds.upper() if ds == 'esol' else 'AqSolDB'} - {split} split"
            lines += [f"### {title} (train {sub[0]['n_train']}, test {sub[0]['n_test']})", "",
                      "| Model | Features | CV RMSE (mean ± sd) | Test RMSE | Test MAE | Test R² | Within ×10 |",
                      "|---|---|---|---|---|---|---|"]
            for r in sub:
                name = f"**{r['model']}**" if r["model"] == V2 else r["model"]
                lines.append(f"| {name} | {r['features']} | {r['cv_rmse']:.3f} ± {r['cv_rmse_sd']:.3f} | "
                             f"{r['test_rmse']:.3f} | {r['test_mae']:.3f} | {r['test_r2']:.3f} | {pct(r.get('test_within_1'))} |")
            lines.append("")
            t = extras.get(ds, {}).get(split, {}).get("trust")
            if not t:
                continue
            lines += [f"**{V2} on this test set, by confidence label** (90% range: prediction ± q·σ, q = {t['q']}; "
                      f"a fixed-width range would be ± {t['fixed_width']['half_width']} with {pct(t['fixed_width']['coverage'])} coverage)", "",
                      "| Test molecules | Share | RMSE | R² | Within ×10 | Inside their 90% range | Mean half-width |",
                      "|---|---|---|---|---|---|---|"]
            parts = [("All", t["all"])] + [(f"{k.capitalize()} confidence", v) for k, v in t["tiers"].items()]
            parts += [("Measured 2+ times, labs agree", t.get("reliable_measurements")),
                      ("Measured once (or labs disagree)", t.get("single_measurements"))]
            for label, v in parts:
                if v:
                    r2 = "–" if v["r2"] is None else f"{v['r2']:.3f}"
                    lines.append(f"| {label} | {pct(v['share'])} | {v['rmse']:.3f} | {r2} | {pct(v['within_1'])} | "
                                 f"{pct(v['coverage'])} | ± {v['mean_half_width']:.2f} |")
            lines.append("")
    path.write_text("\n".join(lines))


def dataset_points(df: pd.DataFrame, X_all: np.ndarray) -> list[dict]:
    """Per-molecule values used by the Explore page charts."""
    j = {n: i for i, n in enumerate(DESCRIPTOR_NAMES)}
    return [
        {"name": str(r.name)[:60], "smiles": r.smiles, "logS": round(r.logS, 2),
         "logP": round(float(X_all[k, j["MolLogP"]]), 2), "MW": round(float(X_all[k, j["MolWt"]]), 1),
         "HBD": int(X_all[k, j["NumHDonors"]]), "TPSA": round(float(X_all[k, j["TPSA"]]), 1),
         "aromatic": round(float(X_all[k, j["AromaticProportion"]]), 2)}
        for k, r in enumerate(df.itertuples(index=False))
    ]


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description="Train and evaluate SoluAI.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only-final", action="store_true", help="skip the benchmark, only retrain the final model")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    np.random.seed(cfg["seed"])
    t0 = time.time()

    data = build_clean_datasets(cfg)
    full = Featurizer(V2_FEATURES, cfg["features"]["morgan_radius"], cfg["features"]["morgan_bits"])
    X, fps = {}, {}
    for name, df in data.items():
        X[name] = full.transform(df["smiles"])
        fps[name] = [full.fingerprint(Chem.MolFromSmiles(s)) for s in df["smiles"]]

    RESULTS_DIR.mkdir(exist_ok=True)
    if args.only_final:
        rows = json.loads((RESULTS_DIR / "metrics.json").read_text())["results"]
        extras = json.loads((RESULTS_DIR / "report.json").read_text())["splits"]
    else:
        rows, extras = [], {}
        for name in ["esol", "aqsoldb"]:
            r, e = run_experiments(name, data[name], cfg, X[name], fps[name])
            rows += r
            extras[name] = e

    final = train_final(data["aqsoldb"], cfg, X["aqsoldb"], fps["aqsoldb"], rows)
    overlap = dataset_overlap(data["aqsoldb"], data["esol"])
    log.info("ESOL molecules also in AqSolDB: %s", overlap)

    cleaning = json.loads((RAW.parent / "processed" / "cleaning_report.json").read_text())
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(
        {"results": rows, "esol_overlap": overlap, "final_model": final["meta"], "final_oof": final["oof_rmse"],
         "cleaning": cleaning}, indent=2))
    write_markdown(rows, extras, RESULTS_DIR / "metrics.md")
    report = {"splits": extras, "importance": final["importance"], "group_importance": final["group_importance"],
              "dependence": final["dependence"], "shap_trend_correlation": final["shap_trend_correlation"]}
    (RESULTS_DIR / "report.json").write_text(json.dumps(report))
    (RESULTS_DIR / "dataset_points.json").write_text(json.dumps(dataset_points(data["aqsoldb"], X["aqsoldb"])))
    log.info("All done in %.1f minutes.", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
