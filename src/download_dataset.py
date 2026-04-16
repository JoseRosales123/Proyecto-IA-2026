# src/download_dataset.py
from datasets import load_dataset
import pandas as pd

print("Descargando dataset de Bitext...")
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

df = pd.DataFrame(dataset["train"])
df.to_csv("../data/raw/tickets.csv", index=False)

print(f"Guardado: {len(df)} filas")
print(f"Columnas: {list(df.columns)}")
print(f"Categorías: {sorted(df['category'].unique().tolist())}")

