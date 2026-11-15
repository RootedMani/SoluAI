"""The model zoo: several different models you can choose between and compare.

* ``chemist``   - SoluAI v2.0, the main model (``models/LATEST``): robust boosting ("chemist")
                  plus a "referee" that sizes each prediction's range. The only model with the
                  atom-by-atom explanation and the confidence label.
* ``v1``        - SoluAI v1.0, the original model (``models/soluai-v1.0.0``).
* five models trained on a larger merged dataset (13 public sources, 8,589 molecules):
  ``zoo-lgbm`` (gradient boosting on 217 descriptors + fragment counts), ``zoo-huber`` (the same with a
  robust loss), ``zoo-cat`` (CatBoost), ``zoo-svr`` (a similarity model), ``zoo-mlp`` (neural networks)
* ``consensus`` - a weighted average of those five, weights learned by cross-validation
* ``delaney``   - Delaney's 2004 formula (no machine learning), to show how far ML has come.

Every model gives a prediction for every molecule, so the website can show where they agree
(more trust) and where they disagree (less trust).
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as G

from . import MODELS_DIR

RDLogger.DisableLog("rdApp.*")

ZOO_DIR = MODELS_DIR / "zoo-v2.0.0"
MEMBERS = ["lgbm", "huber", "cat", "svr", "mlp"]
DEFAULT_MODEL = "consensus"
BASE_MODEL = "chemist"  # the model in models/LATEST; it explains atoms and has the referee

#: Shown on the website. (id, English name, Persian name, English description, Persian description)
INFO: list[tuple[str, str, str, str, str]] = [
    ("consensus", "Consensus of 5 models", "اجماع ۵ مدل",
     "Blends the five zoo models below. The most accurate on unseen molecules.",
     "ترکیب پنج مدل زیر (LightGBM، هوبر، CatBoost، SVR و شبکهٔ عصبی). دقیق‌ترین مدل روی مولکول‌های دیده‌نشده."),
    ("chemist", "SoluAI v2.0 · chemist + referee", "SoluAI v2.0 · شیمی‌دان + داور",
     "Robust boosting on 1,242 features, plus a referee that says how sure it is. Explains each atom.",
     "تقویت مقاوم روی ۱۲۴۲ ویژگی، به‌همراه داوری که می‌گوید چقدر مطمئن است. سهم هر اتم را توضیح می‌دهد."),
    ("v1", "SoluAI v1.0", "SoluAI v1.0",
     "The original model: LightGBM on 14 descriptors + a fingerprint.",
     "مدل اصلی: LightGBM روی ۱۴ توصیفگر و یک اثر انگشت."),
    ("zoo-lgbm", "Gradient boosting (LightGBM)", "تقویت گرادیانی (LightGBM)",
     "1,500 small decision trees, each one fixing the mistakes of the ones before it.",
     "۱۵۰۰ درخت تصمیم کوچک که هر کدام اشتباه درخت‌های قبلی را اصلاح می‌کند."),
    ("zoo-huber", "Robust boosting (Huber)", "تقویت مقاوم (هوبر)",
     "Same trees, but large errors count less, so a few wrong lab values can't pull it off course.",
     "همان درخت‌ها، اما خطاهای بزرگ وزن کمتری دارند تا چند داده‌ٔ اشتباه آزمایشگاهی مدل را منحرف نکند."),
    ("zoo-cat", "CatBoost", "CatBoost",
     "Another boosting library with balanced (symmetric) trees: a second opinion from the same family.",
     "کتابخانهٔ تقویتی دیگری با درخت‌های متقارن: نظر دوم از همان خانواده."),
    ("zoo-svr", "Similarity model (SVR)", "مدل شباهت (SVR)",
     "Looks for the known molecules that share the most fragments and predicts from them.",
     "مولکول‌های شناخته‌شده‌ای را پیدا می‌کند که بیشترین قطعهٔ مشترک را دارند و از روی آن‌ها پیش‌بینی می‌کند."),
    ("zoo-mlp", "Neural network", "شبکهٔ عصبی",
     "Three small neural networks (512 → 256 neurons), averaged.",
     "سه شبکهٔ عصبی کوچک (۵۱۲ ← ۲۵۶ نورون) که میانگین آن‌ها گرفته می‌شود."),
    ("delaney", "Classic formula (Delaney 2004)", "فرمول کلاسیک (دلانی ۲۰۰۴)",
     "One line of algebra with 4 numbers (logP, weight, rotatable bonds, aromatic share). No machine learning.",
     "یک رابطهٔ جبری ساده با ۴ عدد (logP، جرم، پیوندهای چرخش‌پذیر، سهم آروماتیک). بدون یادگیری ماشین."),
]
MODEL_IDS = [i[0] for i in INFO]

# ----------------------------------------------------------------------------- features
_morgan = G.GetMorganGenerator(radius=2, fpSize=2048)
_pairs = G.GetAtomPairGenerator(fpSize=1024)
DESCRIPTOR_LIST = [name for name, _ in Descriptors._descList]


def v2_features(mol: Chem.Mol) -> dict[str, np.ndarray]:
    """The four feature blocks the v2 models use, for one molecule (identical at training and prediction time)."""
    full = np.array([fn(mol) for _, fn in Descriptors._descList], dtype=float)
    full = np.nan_to_num(np.clip(full, -1e6, 1e6)).astype(np.float32)
    return {
        "full": full,  # 217 RDKit descriptors
        "mcount": np.clip(_morgan.GetCountFingerprintAsNumPy(mol), 0, 255).astype(np.uint8),  # fragment counts
        "maccs": np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8),  # 167 yes/no structural keys
        "ap": np.clip(_pairs.GetCountFingerprintAsNumPy(mol), 0, 255).astype(np.uint8),  # atom pairs (for similarity)
    }


def v2_feature_matrix(mols: list[Chem.Mol]) -> dict[str, np.ndarray]:
    """Stack ``v2_features`` for many molecules."""
    rows = [v2_features(m) for m in mols]
    return {k: np.array([r[k] for r in rows]) for k in ("full", "mcount", "maccs", "ap")}


def boosting_matrix(F: dict[str, np.ndarray]) -> np.ndarray:
    """Descriptors + fragment counts + MACCS keys side by side (input of the tree models)."""
    return np.hstack([F["full"], F["mcount"].astype(np.float32), F["maccs"].astype(np.float32)])


def unary(C: np.ndarray, levels: int = 256):
    """Write each count as that many 1s ("3" -> 1,1,1,0,0...), as a sparse 0/1 matrix.

    Then (unary(A) @ unary(B).T)[i, j] = sum over fragments of min(A[i], B[j]): the shared count.
    This is exact, fast, and needs far less memory than comparing every pair directly.
    """
    from scipy import sparse

    i, k = np.nonzero(C)
    c = C[i, k].astype(np.int64)
    start = np.repeat(np.cumsum(c) - c, c)
    cols = np.repeat(k.astype(np.int64) * levels, c) + (np.arange(c.sum()) - start)
    return sparse.csr_matrix((np.ones(len(cols), dtype=np.float32), (np.repeat(i, c), cols)),
                             shape=(C.shape[0], C.shape[1] * levels))


def minmax_similarity(A: np.ndarray, B: np.ndarray | None, chunk: int = 512, UB=None, sb=None) -> np.ndarray:
    """Tanimoto similarity for *count* fingerprints: sum(min) / sum(max), for every row of A against every row of B.

    Instead of ``B`` you can pass a cached ``UB = unary(B).T`` and ``sb = B.sum(1)``.
    """
    sa = A.sum(1).astype(np.float32)
    sb = (B.sum(1) if sb is None else sb).astype(np.float32)
    UB = unary(B).T.tocsc() if UB is None else UB
    out = np.empty((len(A), len(sb)), dtype=np.float32)
    for i in range(0, len(A), chunk):
        mn = (unary(A[i:i + chunk]) @ UB).toarray()
        out[i:i + chunk] = mn / (sa[i:i + chunk, None] + sb[None] - mn + 1e-9)
    return out


def svr_kernel(FA: dict[str, np.ndarray], FB: dict | None = None, index: dict | None = None) -> np.ndarray:
    """Similarity used by the SVR: average of fragment-count and atom-pair Tanimoto.

    Compare with the molecules in ``FB``, or with a prepared ``index`` (see ``V2Models._svr``).
    """
    def one(k):
        if index is not None:
            return minmax_similarity(FA[k], None, UB=index[k][0], sb=index[k][1])
        return minmax_similarity(FA[k], FB[k])
    return 0.5 * one("mcount") + 0.5 * one("ap")


def mlp_input(F: dict[str, np.ndarray], qt, keep: np.ndarray) -> np.ndarray:
    """Neural-network input: descriptors rescaled to a bell curve + MACCS keys."""
    return np.hstack([qt.transform(F["full"][:, keep]), F["maccs"].astype(np.float64)])


def mlp_forward(X: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """One neural network: ReLU hidden layers, straight output."""
    h = X.astype(np.float32)
    for k, (W, b) in enumerate(layers):
        h = h @ W + b
        if k < len(layers) - 1:
            h = np.maximum(h, 0)
    return h[:, 0]


def _release_memory() -> None:
    """Give scratch memory used while loading back to the system (matters on a 512 MB server)."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:  # not Linux/glibc: nothing to do
        pass


# ----------------------------------------------------------------------------- the v2 models
class V2Models:
    """The five v2 models and their consensus, loaded from ``models/zoo-v2.0.0``."""

    def __init__(self, folder: Path = ZOO_DIR) -> None:
        self.folder = folder
        self.meta = json.loads((folder / "zoo.json").read_text())
        if self.meta["descriptors"] != DESCRIPTOR_LIST:
            raise RuntimeError("RDKit descriptor list differs from training: install the pinned RDKit version")
        self.weights = self.meta["weights"]

    @cached_property
    def _boost(self) -> dict:
        b = joblib.load(self.folder / "boosting.joblib")
        for m in b.values():
            m.set_params(n_jobs=1)
        return b

    @cached_property
    def _cat(self):
        from catboost import CatBoostRegressor

        m = CatBoostRegressor()
        m.load_model(str(self.folder / "catboost.cbm"))
        return m

    @cached_property
    def _svr(self) -> dict:
        s = dict(np.load(self.folder / "svr.npz"))
        # Prepared once: support molecules as sparse "unary" matrices + their totals (the raw arrays are dropped).
        s["index"] = {k: (unary(s[f"sv_{k}"]).T.tocsc(), s.pop(f"sv_{k}").sum(1)) for k in ("mcount", "ap")}
        return s

    @cached_property
    def _mlp(self) -> dict:
        return joblib.load(self.folder / "mlp.joblib")

    def load_all(self) -> None:
        """Load every file now (the server does this at start so the first visitor doesn't wait)."""
        _ = self._boost, self._cat, self._svr, self._mlp
        _release_memory()

    def predict_members(self, F: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Predictions of all five models plus the consensus, for a feature dict from ``v2_feature_matrix``."""
        B = boosting_matrix(F)
        out = {"lgbm": self._boost["lgbm"].predict(B), "huber": self._boost["huber"].predict(B),
               "cat": self._cat.predict(B)}
        s = self._svr
        K = svr_kernel(F, index=s["index"])
        out["svr"] = K @ s["dual_coef"] + float(s["intercept"])
        mp = self._mlp
        X = mlp_input(F, mp["qt"], mp["keep"])
        out["mlp"] = np.mean([mlp_forward(X, net) for net in mp["nets"]], axis=0)
        out["consensus"] = sum(self.weights[n] * out[n] for n in MEMBERS)
        return {k: np.asarray(v, dtype=float) for k, v in out.items()}


# ----------------------------------------------------------------------------- the zoo
class Zoo:
    """All selectable models. ``base`` is the v1.0 ``Predictor`` (it also holds the training data lookups)."""

    def __init__(self, base, folder: Path = ZOO_DIR, v1_dir: Path = MODELS_DIR / "soluai-v1.0.0") -> None:
        from .predict import Predictor

        self.base = base
        # SoluAI v1.0 stays available even though models/LATEST now points to v2.0.
        self.v1 = base if base.version.startswith("1.") else Predictor(v1_dir)
        self.v2 = V2Models(folder)
        self.metrics = json.loads((folder / "metrics.json").read_text())
        self._views: dict[str, ModelView] = {}

    @staticmethod
    def _tree(pred, mols: list[Chem.Mol]) -> np.ndarray:
        return pred.model.predict(np.array([pred.featurizer.transform_mol(m) for m in mols]))

    def predict_all(self, mols: list[Chem.Mol]) -> dict[str, np.ndarray]:
        """logS from every model, keyed by model id."""
        from .baselines import delaney_one

        if not mols:
            return {i: np.zeros(0) for i in MODEL_IDS}
        p = self.v2.predict_members(v2_feature_matrix(mols))
        out = {"consensus": p["consensus"], "chemist": self._tree(self.base, mols), "v1": self._tree(self.v1, mols),
               "delaney": np.array([delaney_one(m) for m in mols])}
        out.update({f"zoo-{n}": p[n] for n in MEMBERS})
        return {i: np.asarray(out[i], dtype=float) for i in MODEL_IDS}

    def predict(self, model_id: str, mols: list[Chem.Mol]) -> np.ndarray:
        """logS from one model."""
        from .baselines import delaney_one

        if not mols:
            return np.zeros(0)
        if model_id == "chemist":
            return self._tree(self.base, mols)
        if model_id == "v1":
            return self._tree(self.v1, mols)
        if model_id == "delaney":
            return np.array([delaney_one(m) for m in mols])
        p = self.v2.predict_members(v2_feature_matrix(mols))
        return p["consensus" if model_id == "consensus" else model_id.removeprefix("zoo-")]

    def _metric_key(self, model_id: str) -> str:
        return model_id.removeprefix("zoo-")

    def half_widths(self, model_id: str) -> dict[str, float]:
        """90% error range for familiar ("in") and unfamiliar ("out") molecules, from cross-validation."""
        k = self._metric_key(model_id)
        s = self.metrics["splits"]
        return {"in": s["random"]["models"][k]["q90_cv"], "out": s["scaffold"]["models"][k]["q90_cv"]}

    def info(self, lang: str = "en") -> list[dict]:
        """Name, description and held-out test scores of every model (for the website)."""
        out = []
        for mid, en, fa, den, dfa in INFO:
            k = self._metric_key(mid)
            sc = {split: {m: self.metrics["splits"][split]["models"][k][m] for m in ("r2", "rmse", "mae", "w1")}
                  for split in ("random", "scaffold")}
            out.append({"id": mid, "name": fa if lang == "fa" else en, "name_en": en,
                        "description": dfa if lang == "fa" else den, "default": mid == DEFAULT_MODEL,
                        "explains_atoms": mid == BASE_MODEL, "has_referee": mid == BASE_MODEL, "test": sc})
        return out

    def label(self, model_id: str, lang: str = "en") -> str:
        row = next(i for i in INFO if i[0] == model_id)
        return row[2] if lang == "fa" else row[1]

    def view(self, model_id: str) -> "ModelView":
        """A Predictor-like object that answers with ``model_id`` (used by quiz, batch, trends ...)."""
        if model_id not in self._views:
            self._views[model_id] = ModelView(self, model_id)
        return self._views[model_id]


class ModelView:
    """Looks like a ``Predictor`` but predicts with the chosen model.

    Everything that does not depend on the model (training-data lookups, similarity,
    the SHAP explanation) is passed through to the main model (``models/LATEST``).
    For the main model itself the referee's per-molecule range and confidence are kept;
    other models get a fixed 90% range from cross-validation and no confidence label.
    """

    def __init__(self, zoo: Zoo, model_id: str) -> None:
        self.zoo, self.model_id, self.base = zoo, model_id, zoo.base
        self.is_base = model_id == BASE_MODEL
        if self.is_base:
            self.conformal = self.base.conformal
        else:
            hw = zoo.half_widths(model_id)
            self.conformal = {**self.base.conformal, "q": {"in": hw["in"], "out": hw["out"]}}
        self.version = self.base.version if self.is_base else f"{model_id} (zoo 2.0.0)"

    def __getattr__(self, name):  # training_set, training_fps, measured, shap_values, featurizer ...
        return getattr(self.base, name)

    def __hash__(self) -> int:  # so lru_cache can key on it
        return hash(("view", self.model_id))

    def __eq__(self, other) -> bool:
        return isinstance(other, ModelView) and other.model_id == self.model_id

    def predict_mol(self, mol: Chem.Mol) -> float:
        return float(self.zoo.predict(self.model_id, [mol])[0])

    def predict_many(self, smiles: list[str]) -> np.ndarray:
        return self.zoo.predict(self.model_id, [Chem.MolFromSmiles(s) for s in smiles])

    def domain(self, mol: Chem.Mol, k: int = 3, pred: float | None = None) -> dict:
        if self.is_base:
            return self.base.domain(mol, k, pred=pred)
        # The referee judges the main model's errors, not this model's: skip it (pass a dummy prediction).
        d = self.base.domain(mol, k, pred=0.0)
        for key in ("expected_error", "confidence"):
            d.pop(key, None)
        d["half_width"] = self.conformal["q"]["in" if d["max_similarity"] >= self.conformal["threshold"] else "out"]
        return d
