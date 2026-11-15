# SoluAI v2 experiment: consensus stacking with a confidence gate

Run order (about 45 min on 2 CPU cores):
`merge.py` → `feats.py` → `kern.py` → `base.py random lgbm,huber,cat,et,svr,mlp` → same with `scaffold` → `final.py`.
`merge.py` expects the source repos (mcsorkun/AqSolDB, Mengjintao/SolCuration) cloned under `/home/claude/newdata`; edit the path `N` for your machine.

## Data
28,714 usable measurements from 13 source files (AqSolDB's 9 raw sets, AQUA, PHYSPROP, OCHEM, ESOL)
→ 8,589 unique organic molecules. Only 135 are molecules that AqSolDB did not already have; the gain is
*repeat measurements*: 1,923 molecules now have 2+ different lab values. Label = median of the distinct values.

## Results (held-out test, 1,718 molecules, never used for tuning)
| | Random split R² | RMSE | within ±1 | Scaffold split R² | RMSE |
|---|---|---|---|---|---|
| v1.0 recipe (same data, same split) | 0.866 | 0.848 | 82.7% | 0.819 | 1.064 |
| **v2 stack, all test molecules** | **0.886** | 0.781 | 85.4% | **0.850** | 0.966 |
| v2, reliably measured molecules only | 0.927 | 0.572 | 90.5% | 0.881 | 0.832 |
| v2, confident predictions only | 0.930 | 0.578 | 92.5% | 0.943 | 0.584 |
| v2, confident + reliably measured | 0.963 (n=168) | 0.384 | 97.6% | 0.944 (n=66) | 0.551 |

Two different lab values for the same molecule agree with each other at only R² = 0.855 (RMSE 0.89, 1,947 molecules).

## Tried and dropped (did not help)
Noise-based sample weights; nearest-analog residual correction (made cross-validation error worse).
A graph neural network was not tested (PyTorch could not be installed in this environment).
