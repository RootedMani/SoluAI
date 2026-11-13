# Data sources

Both files are stored here unchanged, exactly as downloaded on 2026-11-12, so the
project works offline. `src/soluai/data.py` only downloads them if they are missing.

| File | Dataset | Compounds | Source URL | License | SHA-256 |
|---|---|---|---|---|---|
| `aqsoldb_curated.csv` | **AqSolDB** | 9,982 | https://raw.githubusercontent.com/mcsorkun/AqSolDB/master/results/data_curated.csv (also on Harvard Dataverse, doi:10.7910/DVN/OVHAW8) | CC0 1.0 (public domain), as published on Harvard Dataverse | `363a42a4…0a805ed` |
| `esol_delaney.csv` | **ESOL** (Delaney) | 1,128 | https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv (MoleculeNet / DeepChem copy) | Data from the supporting information of Delaney (2004); redistributed by MoleculeNet/DeepChem (MIT-licensed project) for research and education. Cite the paper when using it. | `8c06a76f…0f189` |

Full hashes are in `SOURCES.json`.

## Citations

* Sorkun, M. C., Khetan, A., & Er, S. (2019). AqSolDB, a curated reference set of aqueous
  solubility and 2D descriptors for a diverse set of compounds. *Scientific Data*, 6, 143.
  https://doi.org/10.1038/s41597-019-0151-1
* Delaney, J. S. (2004). ESOL: Estimating aqueous solubility directly from molecular
  structure. *Journal of Chemical Information and Computer Sciences*, 44(3), 1000–1005.
  https://doi.org/10.1021/ci034243x
* Wu, Z. et al. (2018). MoleculeNet: a benchmark for molecular machine learning.
  *Chemical Science*, 9, 513–530. https://doi.org/10.1039/C7SC02664A

## Columns used

* AqSolDB: `Name`, `SMILES`, `Solubility` (logS, mol/L, ~25 °C), `Group` (reliability group G1–G5).
* ESOL: `Compound ID`, `smiles`, `measured log solubility in mols per litre`.
