# src/kfold.py

import random
from naive_bayes import MultinomialNaiveBayes
from vocabulary import build_vocabulary
from metrics import compute_all_metrics, print_metrics_report
from config import LAPLACE_ALPHA, NUM_FOLDS, RANDOM_SEED


def k_fold_split(n: int, k: int, seed: int = RANDOM_SEED) -> list:
    """Divide n índices en K folds aleatorios."""
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = n if i == k - 1 else start + fold_size
        folds.append(indices[start:end])
    return folds


def run_k_fold_cross_validation(processed_texts: list, labels: list, k: int = NUM_FOLDS) -> dict:
    """
    Ejecuta K-Fold Cross Validation completo.
    
    Por cada fold:
    1. Separa train/test
    2. Construye vocabulario SOLO con train
    3. Entrena modelo nuevo
    4. Predice sobre test
    5. Calcula métricas
    """
    classes = sorted(list(set(labels)))
    folds = k_fold_split(len(processed_texts), k)

    fold_results = []
    accuracies = []
    macro_f1s = []

    print(f"\n{'='*60}")
    print(f"   K-FOLD CROSS VALIDATION (K={k})")
    print(f"   Total datos: {len(processed_texts)} | Clases: {len(classes)}")
    print(f"{'='*60}")

    for fold_idx in range(k):
        print(f"\n--- Fold {fold_idx + 1}/{k} ---")

        # Separar train y test
        test_indices = set(folds[fold_idx])
        train_texts, train_labels = [], []
        test_texts, test_labels = [], []

        for i in range(len(processed_texts)):
            if i in test_indices:
                test_texts.append(processed_texts[i])
                test_labels.append(labels[i])
            else:
                train_texts.append(processed_texts[i])
                train_labels.append(labels[i])

        print(f"  Train: {len(train_texts)} | Test: {len(test_texts)}")

        # Vocabulario SOLO con train (importante!)
        vocabulary = build_vocabulary(train_texts)
        print(f"  Vocabulario: {len(vocabulary)} palabras")

        # Entrenar
        model = MultinomialNaiveBayes(alpha=LAPLACE_ALPHA)
        model.fit(train_texts, train_labels, vocabulary)

        # Predecir
        predictions = model.predict_many(test_texts)

        # Métricas
        metrics = compute_all_metrics(test_labels, predictions, classes)
        fold_results.append(metrics)
        accuracies.append(metrics["accuracy"])
        macro_f1s.append(metrics["macro_f1"])

        print(f"  Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")

    # Promedios y varianza
    avg_acc = sum(accuracies) / k
    avg_f1 = sum(macro_f1s) / k
    var_acc = sum((a - avg_acc) ** 2 for a in accuracies) / k
    var_f1 = sum((f - avg_f1) ** 2 for f in macro_f1s) / k

    print(f"\n{'='*60}")
    print(f"   RESUMEN K-FOLD")
    print(f"{'='*60}")
    for i in range(k):
        print(f"  Fold {i+1}: Accuracy={accuracies[i]:.4f} | Macro F1={macro_f1s[i]:.4f}")
    print(f"\n  Promedio Accuracy: {avg_acc:.4f} (± {var_acc**0.5:.4f})")
    print(f"  Promedio Macro F1: {avg_f1:.4f} (± {var_f1**0.5:.4f})")
    print(f"  Varianza Accuracy: {var_acc:.6f}")
    print(f"  Varianza Macro F1: {var_f1:.6f}")

    return {
        "fold_results": fold_results,
        "accuracies": accuracies,
        "macro_f1s": macro_f1s,
        "avg_accuracy": avg_acc,
        "avg_macro_f1": avg_f1,
        "var_accuracy": var_acc,
        "var_macro_f1": var_f1,
        "classes": classes
    }
