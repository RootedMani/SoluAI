### AqSolDB - random split (train 7003, test 1751)

| Model | Features | CV RMSE (mean ± sd) | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|---|
| Delaney (original) | 4 ESOL vars | 1.332 ± 0.043 | 1.353 | 0.998 | 0.651 |
| Delaney (refit) | 4 ESOL vars | 1.321 ± 0.040 | 1.336 | 1.001 | 0.660 |
| Linear regression | desc | 1.308 ± 0.033 | 1.311 | 0.983 | 0.672 |
| Random forest | desc | 0.948 ± 0.005 | 0.942 | 0.669 | 0.831 |
| Random forest | fp | 1.268 ± 0.030 | 1.252 | 0.905 | 0.701 |
| Random forest | desc+fp | 0.889 ± 0.007 | 0.877 | 0.621 | 0.854 |
| LightGBM | desc | 0.941 ± 0.005 | 0.935 | 0.672 | 0.833 |
| LightGBM | fp | 1.132 ± 0.006 | 1.146 | 0.822 | 0.750 |
| LightGBM | desc+fp | 0.839 ± 0.010 | 0.835 | 0.590 | 0.867 |

### AqSolDB - scaffold split (train 7003, test 1751)

| Model | Features | CV RMSE (mean ± sd) | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|---|
| Delaney (original) | 4 ESOL vars | 1.317 ± 0.176 | 1.440 | 1.049 | 0.569 |
| Delaney (refit) | 4 ESOL vars | 1.328 ± 0.151 | 1.412 | 1.034 | 0.586 |
| Linear regression | desc | 1.323 ± 0.164 | 1.409 | 1.023 | 0.588 |
| Random forest | desc | 1.050 ± 0.100 | 1.121 | 0.836 | 0.739 |
| Random forest | fp | 1.887 ± 0.150 | 1.555 | 1.190 | 0.498 |
| Random forest | desc+fp | 1.000 ± 0.100 | 1.070 | 0.796 | 0.762 |
| LightGBM | desc | 1.050 ± 0.093 | 1.075 | 0.807 | 0.760 |
| LightGBM | fp | 1.670 ± 0.173 | 1.413 | 1.077 | 0.585 |
| LightGBM | desc+fp | 0.954 ± 0.097 | 1.012 | 0.752 | 0.787 |

### ESOL - random split (train 892, test 224)

| Model | Features | CV RMSE (mean ± sd) | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|---|
| Delaney (original) | 4 ESOL vars | 1.108 ± 0.041 | 1.070 | 0.824 | 0.697 |
| Delaney (refit) | 4 ESOL vars | 1.037 ± 0.036 | 0.920 | 0.714 | 0.776 |
| Linear regression | desc | 0.978 ± 0.061 | 0.888 | 0.735 | 0.791 |
| Random forest | desc | 0.683 ± 0.057 | 0.665 | 0.477 | 0.883 |
| Random forest | fp | 1.197 ± 0.045 | 1.194 | 0.885 | 0.623 |
| Random forest | desc+fp | 0.665 ± 0.027 | 0.616 | 0.456 | 0.900 |
| LightGBM | desc | 0.665 ± 0.040 | 0.643 | 0.455 | 0.891 |
| LightGBM | fp | 1.108 ± 0.025 | 1.083 | 0.793 | 0.690 |
| LightGBM | desc+fp | 0.622 ± 0.021 | 0.605 | 0.432 | 0.903 |

### ESOL - scaffold split (train 889, test 227)

| Model | Features | CV RMSE (mean ± sd) | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|---|
| Delaney (original) | 4 ESOL vars | 1.069 ± 0.129 | 1.239 | 0.961 | 0.639 |
| Delaney (refit) | 4 ESOL vars | 1.045 ± 0.092 | 1.228 | 0.946 | 0.645 |
| Linear regression | desc | 1.042 ± 0.091 | 1.209 | 0.923 | 0.656 |
| Random forest | desc | 0.853 ± 0.123 | 0.880 | 0.686 | 0.818 |
| Random forest | fp | 2.019 ± 0.326 | 1.510 | 1.164 | 0.463 |
| Random forest | desc+fp | 0.849 ± 0.132 | 0.809 | 0.634 | 0.846 |
| LightGBM | desc | 0.814 ± 0.096 | 0.935 | 0.708 | 0.794 |
| LightGBM | fp | 2.022 ± 0.397 | 1.490 | 1.151 | 0.477 |
| LightGBM | desc+fp | 0.806 ± 0.107 | 0.833 | 0.641 | 0.837 |
