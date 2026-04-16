# src/data_loader.py

import pandas as pd
from config import DATASET_PATH, TEXT_COLUMN, LABEL_COLUMN, VALID_CLASSES


def load_dataset():
    """
    Carga el dataset desde la ruta configurada y valida que tenga
    las columnas esperadas.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo del dataset en: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = [TEXT_COLUMN, LABEL_COLUMN]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Falta la columna obligatoria '{col}' en el dataset. "
                f"Columnas encontradas: {list(df.columns)}"
            )

    return df


def validate_labels(df):
    """
    Verifica que todas las etiquetas del dataset pertenezcan
    a las clases válidas del proyecto.
    """
    unique_labels = set(df[LABEL_COLUMN].dropna().unique())
    valid_labels = set(VALID_CLASSES)

    invalid_labels = unique_labels - valid_labels

    if invalid_labels:
        raise ValueError(
            "Se encontraron etiquetas no válidas en el dataset: "
            f"{sorted(list(invalid_labels))}"
        )

    return True


def basic_dataset_report(df):
    """
    Genera un pequeño resumen del dataset.
    """
    print("\n=== REPORTE BÁSICO DEL DATASET ===")
    print(f"Cantidad de registros: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
    print("\nDistribución de clases:")
    print(df[LABEL_COLUMN].value_counts())
    print("\nPrimeras filas:")
    print(df.head())