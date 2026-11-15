RMSE and MAE are in log units. *Within ×10* is the share of test molecules predicted within one
log unit, i.e. within a factor of 10 of the measured solubility.

### AqSolDB - random split (train 7003, test 1751)

| Model | Features | CV RMSE (mean ± sd) | Test RMSE | Test MAE | Test R² | Within ×10 |
|---|---|---|---|---|---|---|
| Delaney (original) | 4 ESOL vars | 1.332 ± 0.043 | 1.353 | 0.998 | 0.651 | 61.8% |
| Delaney (refit) | 4 ESOL vars | 1.321 ± 0.040 | 1.336 | 1.001 | 0.660 | 60.8% |
| Linear regression | desc | 1.308 ± 0.033 | 1.311 | 0.983 | 0.672 | 62.1% |
| Random forest | desc | 0.948 ± 0.005 | 0.942 | 0.669 | 0.831 | 77.4% |
| Random forest | fp | 1.268 ± 0.030 | 1.252 | 0.905 | 0.701 | 66.1% |
| Random forest | desc+fp | 0.889 ± 0.007 | 0.877 | 0.621 | 0.854 | 80.0% |
| LightGBM | desc | 0.941 ± 0.005 | 0.935 | 0.672 | 0.833 | 78.1% |
| LightGBM | fp | 1.132 ± 0.006 | 1.146 | 0.822 | 0.750 | 70.3% |
| LightGBM | desc+fp | 0.839 ± 0.010 | 0.835 | 0.590 | 0.867 | 82.0% |
| **SoluAI v2** | rdkit+fp | 0.805 ± 0.011 | 0.793 | 0.551 | 0.880 | 83.7% |

**SoluAI v2 on this test set, by confidence label** (90% range: prediction ± q·σ, q = 2.15; a fixed-width range would be ± 1.243 with 89.3% coverage)

| Test molecules | Share | RMSE | R² | Within ×10 | Inside their 90% range | Mean half-width |
|---|---|---|---|---|---|---|
| All | 100.0% | 0.793 | 0.880 | 83.7% | 88.7% | ± 1.17 |
| High confidence | 27.2% | 0.426 | 0.961 | 95.8% | 89.7% | ± 0.66 |
| Medium confidence | 47.3% | 0.736 | 0.876 | 85.3% | 87.0% | ± 1.06 |
| Low confidence | 25.5% | 1.128 | 0.751 | 67.7% | 91.0% | ± 1.92 |
| Measured 2+ times, labs agree | 17.0% | 0.492 | 0.949 | 94.9% | 95.3% | ± 0.99 |
| Measured once (or labs disagree) | 83.0% | 0.842 | 0.867 | 81.4% | 87.4% | ± 1.21 |

### AqSolDB - scaffold split (train 7003, test 1751)

| Model | Features | CV RMSE (mean ± sd) | Test RMSE | Test MAE | Test R² | Within ×10 |
|---|---|---|---|---|---|---|
| Delaney (original) | 4 ESOL vars | 1.317 ± 0.176 | 1.440 | 1.049 | 0.569 | 58.5% |
| Delaney (refit) | 4 ESOL vars | 1.328 ± 0.151 | 1.412 | 1.034 | 0.586 | 58.8% |
| Linear regression | desc | 1.323 ± 0.164 | 1.409 | 1.023 | 0.588 | 60.5% |
| Random forest | desc | 1.050 ± 0.100 | 1.121 | 0.836 | 0.739 | 68.5% |
| Random forest | fp | 1.887 ± 0.150 | 1.555 | 1.190 | 0.498 | 51.9% |
| Random forest | desc+fp | 1.000 ± 0.100 | 1.070 | 0.796 | 0.762 | 71.0% |
| LightGBM | desc | 1.050 ± 0.093 | 1.075 | 0.807 | 0.760 | 70.1% |
| LightGBM | fp | 1.670 ± 0.173 | 1.413 | 1.077 | 0.585 | 56.8% |
| LightGBM | desc+fp | 0.954 ± 0.097 | 1.012 | 0.752 | 0.787 | 73.7% |
| **SoluAI v2** | rdkit+fp | 0.914 ± 0.095 | 0.942 | 0.699 | 0.816 | 75.8% |

**SoluAI v2 on this test set, by confidence label** (90% range: prediction ± q·σ, q = 2.153; a fixed-width range would be ± 1.427 with 88.5% coverage)

| Test molecules | Share | RMSE | R² | Within ×10 | Inside their 90% range | Mean half-width |
|---|---|---|---|---|---|---|
| All | 100.0% | 0.942 | 0.816 | 75.8% | 90.1% | ± 1.52 |
| High confidence | 1.1% | 0.478 | 0.891 | 95.0% | 80.0% | ± 0.63 |
| Medium confidence | 46.2% | 0.736 | 0.848 | 84.7% | 89.5% | ± 1.19 |
| Low confidence | 52.7% | 1.098 | 0.737 | 67.6% | 90.8% | ± 1.84 |
| Measured 2+ times, labs agree | 15.4% | 0.805 | 0.861 | 81.9% | 91.1% | ± 1.44 |
| Measured once (or labs disagree) | 84.6% | 0.965 | 0.808 | 74.7% | 89.9% | ± 1.54 |

### ESOL - random split (train 892, test 224)

| Model | Features | CV RMSE (mean ± sd) | Test RMSE | Test MAE | Test R² | Within ×10 |
|---|---|---|---|---|---|---|
| Delaney (original) | 4 ESOL vars | 1.108 ± 0.041 | 1.070 | 0.824 | 0.697 | 70.1% |
| Delaney (refit) | 4 ESOL vars | 1.037 ± 0.036 | 0.920 | 0.714 | 0.776 | 73.2% |
| Linear regression | desc | 0.978 ± 0.061 | 0.888 | 0.735 | 0.791 | 74.1% |
| Random forest | desc | 0.683 ± 0.057 | 0.665 | 0.477 | 0.883 | 87.1% |
| Random forest | fp | 1.197 ± 0.045 | 1.194 | 0.885 | 0.623 | 65.2% |
| Random forest | desc+fp | 0.665 ± 0.027 | 0.616 | 0.456 | 0.900 | 88.8% |
| LightGBM | desc | 0.665 ± 0.040 | 0.643 | 0.455 | 0.891 | 88.4% |
| LightGBM | fp | 1.108 ± 0.025 | 1.083 | 0.793 | 0.690 | 74.1% |
| LightGBM | desc+fp | 0.622 ± 0.021 | 0.605 | 0.432 | 0.903 | 90.2% |
| **SoluAI v2** | rdkit+fp | 0.607 ± 0.024 | 0.575 | 0.408 | 0.913 | 92.4% |

**SoluAI v2 on this test set, by confidence label** (90% range: prediction ± q·σ, q = 2.346; a fixed-width range would be ± 0.971 with 92.4% coverage)

| Test molecules | Share | RMSE | R² | Within ×10 | Inside their 90% range | Mean half-width |
|---|---|---|---|---|---|---|
| All | 100.0% | 0.575 | 0.913 | 92.4% | 91.5% | ± 1.00 |
| High confidence | 42.9% | 0.340 | 0.959 | 100.0% | 89.6% | ± 0.58 |
| Medium confidence | 36.6% | 0.602 | 0.919 | 91.5% | 93.9% | ± 1.11 |
| Low confidence | 20.5% | 0.850 | 0.785 | 78.3% | 91.3% | ± 1.69 |

### ESOL - scaffold split (train 889, test 227)

| Model | Features | CV RMSE (mean ± sd) | Test RMSE | Test MAE | Test R² | Within ×10 |
|---|---|---|---|---|---|---|
| Delaney (original) | 4 ESOL vars | 1.069 ± 0.129 | 1.239 | 0.961 | 0.639 | 59.9% |
| Delaney (refit) | 4 ESOL vars | 1.045 ± 0.092 | 1.228 | 0.946 | 0.645 | 61.2% |
| Linear regression | desc | 1.042 ± 0.091 | 1.209 | 0.923 | 0.656 | 63.4% |
| Random forest | desc | 0.853 ± 0.123 | 0.880 | 0.686 | 0.818 | 75.8% |
| Random forest | fp | 2.019 ± 0.326 | 1.510 | 1.164 | 0.463 | 53.3% |
| Random forest | desc+fp | 0.849 ± 0.132 | 0.809 | 0.634 | 0.846 | 78.4% |
| LightGBM | desc | 0.814 ± 0.096 | 0.935 | 0.708 | 0.794 | 78.0% |
| LightGBM | fp | 2.022 ± 0.397 | 1.490 | 1.151 | 0.477 | 55.9% |
| LightGBM | desc+fp | 0.806 ± 0.107 | 0.833 | 0.641 | 0.837 | 81.1% |
| **SoluAI v2** | rdkit+fp | 0.822 ± 0.098 | 0.823 | 0.640 | 0.841 | 78.9% |

**SoluAI v2 on this test set, by confidence label** (90% range: prediction ± q·σ, q = 2.362; a fixed-width range would be ± 1.267 with 88.1% coverage)

| Test molecules | Share | RMSE | R² | Within ×10 | Inside their 90% range | Mean half-width |
|---|---|---|---|---|---|---|
| All | 100.0% | 0.823 | 0.841 | 78.9% | 92.1% | ± 1.63 |
| High confidence | 4.4% | 0.654 | 0.866 | 90.0% | 60.0% | ± 0.70 |
| Medium confidence | 34.8% | 0.764 | 0.880 | 83.5% | 86.1% | ± 1.12 |
| Low confidence | 60.8% | 0.865 | 0.784 | 75.4% | 97.8% | ± 1.98 |
