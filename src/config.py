# src/config.py

from pathlib import Path

# =========================
# RUTAS DEL PROYECTO
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "model"

# Archivo de entrada del dataset
DATASET_FILENAME = "tickets.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME

# Archivo donde se guardará el modelo
MODEL_FILENAME = "naive_bayes_model.pkl"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

# =========================
# COLUMNAS DEL DATASET
# =========================
TEXT_COLUMN = "instruction"
LABEL_COLUMN = "category"

# =========================
# CLASES DEL PROYECTO
# =========================
VALID_CLASSES = [
    "ACCOUNT",
    "CANCEL",
    "CONTACT",
    "DELIVERY",
    "FEEDBACK",
    "INVOICE",
    "ORDER",
    "PAYMENT",
    "REFUND",
    "SHIPPING",
    "SUBSCRIPTION",
]

# =========================
# PARÁMETROS DEL MODELO
# =========================
LAPLACE_ALPHA = 1.0
NUM_FOLDS = 5
RANDOM_SEED = 42

# =========================
# PREPROCESAMIENTO
# =========================
LOWERCASE = True
REMOVE_PUNCTUATION = True
REMOVE_NUMBERS = False
REMOVE_STOPWORDS = True
USE_STEMMING = False
USE_LEMMATIZATION = False
MIN_TOKEN_LENGTH = 2