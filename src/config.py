"""Project-wide configuration: paths, dataset schema, and constants.

Everything other modules need to agree on (file locations, column names,
random seed) lives here so there is a single place to change it.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
EDA_DIR = SCREENSHOTS_DIR / "eda"

RAW_DATA_PATH = RAW_DIR / "processed.cleveland.data"
CLEAN_DATA_PATH = PROCESSED_DIR / "heart_disease_clean.csv"

# ---------------------------------------------------------------------------
# Data source (UCI Machine Learning Repository, Cleveland subset)
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
UCI_DATASET_ID = 45  # used by the optional `ucimlrepo` fallback
DOWNLOAD_TIMEOUT_SECONDS = 30
MISSING_VALUE_MARKER = "?"  # how the raw file encodes missing values

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# The raw file has no header row; this is the documented column order.
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
RAW_TARGET = "num"  # 0 = no disease, 1-4 = disease of increasing severity
TARGET = "target"   # binary: 0 = no disease, 1 = disease present

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLASS_LABELS = {0: "No disease", 1: "Disease"}

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1 typical angina, 2 atypical angina, "
          "3 non-anginal pain, 4 asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG result (0 normal, 1 ST-T abnormality, "
               "2 left ventricular hypertrophy)",
    "thalach": "Maximum heart rate achieved (bpm)",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment "
             "(1 upsloping, 2 flat, 3 downsloping)",
    "ca": "Number of major vessels (0-3) coloured by fluoroscopy",
    "thal": "Thalassemia (3 normal, 6 fixed defect, 7 reversible defect)",
}

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
