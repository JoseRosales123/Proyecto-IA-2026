# src/run_evaluation.py

from data_loader import load_dataset
from preprocessing import preprocess_corpus
from config import TEXT_COLUMN, LABEL_COLUMN, NUM_FOLDS
from kfold import run_k_fold_cross_validation
from metrics import print_metrics_report

def main():
    # Cargar datos
    df = load_dataset()
    texts = df[TEXT_COLUMN].tolist()
    labels = df[LABEL_COLUMN].tolist()

    # Preprocesar
    processed = preprocess_corpus(texts)

    # Correr K-Fold
    results = run_k_fold_cross_validation(processed, labels, k=NUM_FOLDS)

    # Imprimir métricas del último fold como ejemplo
    print_metrics_report(results["fold_results"][-1])

if __name__ == "__main__":
    main()
    