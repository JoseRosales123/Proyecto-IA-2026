# src/metrics.py

def build_confusion_matrix(y_true: list, y_pred: list, classes: list) -> list:
    """
    Filas = real, Columnas = predicho.
    matrix[i][j] = cuántas veces la clase real i fue predicha como j
    """
    n = len(classes)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    matrix = [[0] * n for _ in range(n)]

    for true_label, pred_label in zip(y_true, y_pred):
        i = class_to_idx[true_label]
        j = class_to_idx[pred_label]
        matrix[i][j] += 1

    return matrix


def calculate_precision(cm: list, classes: list) -> dict:
    """TP / (TP + FP) por clase"""
    precisions = {}
    n = len(classes)
    for j, cls in enumerate(classes):
        tp = cm[j][j]
        fp = sum(cm[i][j] for i in range(n)) - tp
        precisions[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precisions


def calculate_recall(cm: list, classes: list) -> dict:
    """TP / (TP + FN) por clase"""
    recalls = {}
    for i, cls in enumerate(classes):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        recalls[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recalls


def calculate_f1_score(precisions: dict, recalls: dict) -> dict:
    """2 * P * R / (P + R) por clase"""
    f1_scores = {}
    for cls in precisions:
        p = precisions[cls]
        r = recalls[cls]
        f1_scores[cls] = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return f1_scores


def calculate_accuracy(y_true: list, y_pred: list) -> float:
    """Correctas / Total"""
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def calculate_macro_f1(f1_scores: dict) -> float:
    """Promedio de F1 de todas las clases"""
    values = list(f1_scores.values())
    return sum(values) / len(values) if values else 0.0


def compute_all_metrics(y_true: list, y_pred: list, classes: list) -> dict:
    """Calcula todo de una vez y devuelve un diccionario."""
    cm = build_confusion_matrix(y_true, y_pred, classes)
    precision = calculate_precision(cm, classes)
    recall = calculate_recall(cm, classes)
    f1 = calculate_f1_score(precision, recall)
    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": calculate_accuracy(y_true, y_pred),
        "macro_f1": calculate_macro_f1(f1),
        "classes": classes
    }


def print_metrics_report(metrics: dict):
    """Imprime reporte bonito de métricas."""
    classes = metrics["classes"]
    cm = metrics["confusion_matrix"]

    print("\n" + "=" * 65)
    print("              REPORTE DE EVALUACIÓN DEL MODELO")
    print("=" * 65)

    print(f"\n  {'Clase':<25} {'Precisión':>10} {'Recall':>10} {'F1-Score':>10}")
    print("  " + "-" * 55)
    for cls in classes:
        p = metrics["precision"][cls]
        r = metrics["recall"][cls]
        f = metrics["f1_score"][cls]
        print(f"  {cls:<25} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

    print("  " + "-" * 55)
    print(f"  {'Accuracy Global:':<25} {metrics['accuracy']:>10.4f}")
    print(f"  {'Macro F1-Score:':<25} {metrics['macro_f1']:>10.4f}")

    # Matriz de confusión
    print("\n" + "=" * 65)
    print("                    MATRIZ DE CONFUSIÓN")
    print("=" * 65)
    header = "Real \\ Pred"
    print(f"\n  {header:<20}", end="")
    for cls in classes:
        print(f"{cls[:10]:>12}", end="")
    print()
    print("  " + "-" * (20 + 12 * len(classes)))
    for i, cls in enumerate(classes):
        print(f"  {cls[:19]:<20}", end="")
        for j in range(len(classes)):
            print(f"{cm[i][j]:>12}", end="")
        print()

    # Top confusiones
    confusions = []
    for i, cr in enumerate(classes):
        for j, cp in enumerate(classes):
            if i != j and cm[i][j] > 0:
                confusions.append((cr, cp, cm[i][j]))
    confusions.sort(key=lambda x: x[2], reverse=True)

    if confusions:
        print(f"\n  Top confusiones:")
        for real, pred, count in confusions[:10]:
            print(f"    {real} → {pred}: {count} veces")
            