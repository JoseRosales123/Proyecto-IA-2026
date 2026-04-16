import pickle
from pathlib import Path


def save_model(model, filepath: str):
    """
    Guarda el modelo entrenado en un archivo .pkl
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath: str):
    """
    Carga un modelo previamente guardado desde un archivo .pkl
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo del modelo en: {filepath}")

    with open(path, "rb") as f:
        model = pickle.load(f)

    return model