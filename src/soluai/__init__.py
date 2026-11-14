"""SoluAI: predicting the water solubility of molecules with machine learning."""

from pathlib import Path

__version__ = "2.0.0"

#: Root folder of the repository (two levels above this file).
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
