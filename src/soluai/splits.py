"""Ways to split molecules into training and test sets.

* **Random split** - shuffle the molecules and put 20 % aside. Test molecules
  often have a close "cousin" in the training set, so this is the easy exam.
* **Scaffold split** - group molecules by their ring skeleton (Bemis-Murcko
  scaffold) and put *whole groups* aside. The test molecules then have
  skeletons the model has never seen: the hard, more honest exam.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold, KFold, train_test_split


def scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES (rings + linkers). Acyclic molecules give ''."""
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""


def random_split(n: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices for a plain random train/test split."""
    idx = np.arange(n)
    return train_test_split(idx, test_size=test_fraction, random_state=seed)


def scaffold_split(smiles: list[str], test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices for a scaffold split.

    Scaffold groups are shuffled (with a fixed seed) and added to the test set
    until it holds ``test_fraction`` of the molecules. The very large groups
    (e.g. all acyclic molecules, all plain benzenes) are kept in training, as
    is usual, so the test set is made of many different, rarer skeletons.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[scaffold(s)].append(i)
    n_test_target = int(round(test_fraction * len(smiles)))
    big = {k for k, v in groups.items() if len(v) > n_test_target / 2}
    keys = sorted(k for k in groups if k not in big)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    test: list[int] = []
    for k in keys:
        if len(test) >= n_test_target:
            break
        test.extend(groups[k])
    test_set = set(test)
    train = np.array([i for i in range(len(smiles)) if i not in test_set])
    return train, np.array(sorted(test_set))


def cv_splitter(kind: str, n_folds: int, seed: int, smiles: list[str] | None = None):
    """Cross-validation folds matching the split type.

    Returns a list of ``(train_idx, valid_idx)`` pairs.
    """
    if kind == "random":
        return list(KFold(n_folds, shuffle=True, random_state=seed).split(np.arange(len(smiles))))
    groups = np.array([scaffold(s) for s in smiles])
    # Shuffle group labels deterministically so folds are not ordered by scaffold.
    uniq = np.unique(groups)
    perm = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq))))
    gid = np.array([perm[g] for g in groups])
    return list(GroupKFold(n_folds).split(np.arange(len(smiles)), groups=gid))
