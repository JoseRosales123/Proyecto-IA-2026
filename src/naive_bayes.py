import math
from collections import Counter, defaultdict


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes = []
        self.vocabulary = []
        self.vocabulary_size = 0

        self.class_priors = {}
        self.word_counts_by_class = {}
        self.total_words_by_class = {}
        self.doc_counts_by_class = {}

        self.is_fitted = False

    def fit(self, processed_texts: list[list[str]], labels: list[str], vocabulary: list[str]):
        """
        Entrena el modelo Naïve Bayes Multinomial.
        """
        if len(processed_texts) != len(labels):
            raise ValueError("La cantidad de textos y etiquetas no coincide.")

        self.classes = sorted(list(set(labels)))
        self.vocabulary = vocabulary
        self.vocabulary_size = len(vocabulary)

        total_docs = len(labels)

        # Inicializar estructuras
        self.doc_counts_by_class = {cls: 0 for cls in self.classes}
        self.word_counts_by_class = {cls: defaultdict(int) for cls in self.classes}
        self.total_words_by_class = {cls: 0 for cls in self.classes}

        # Contar documentos y palabras por clase
        for tokens, label in zip(processed_texts, labels):
            self.doc_counts_by_class[label] += 1

            token_counts = Counter(tokens)
            for word, count in token_counts.items():
                self.word_counts_by_class[label][word] += count
                self.total_words_by_class[label] += count

        # Probabilidades a priori P(clase)
        self.class_priors = {
            cls: self.doc_counts_by_class[cls] / total_docs
            for cls in self.classes
        }

        self.is_fitted = True
        return self

    def _word_likelihood(self, word: str, cls: str) -> float:
        """
        Calcula P(word | cls) con Laplace smoothing.
        """
        word_count = self.word_counts_by_class[cls].get(word, 0)
        total_words = self.total_words_by_class[cls]

        return (word_count + self.alpha) / (
            total_words + self.alpha * self.vocabulary_size
        )

    def predict_log_proba(self, tokens: list[str]) -> dict[str, float]:
        """
        Devuelve el score logarítmico por clase.
        """
        if not self.is_fitted:
            raise ValueError("El modelo todavía no ha sido entrenado.")

        token_counts = Counter(tokens)
        log_scores = {}

        for cls in self.classes:
            log_score = math.log(self.class_priors[cls])

            for word, count in token_counts.items():
                likelihood = self._word_likelihood(word, cls)
                log_score += count * math.log(likelihood)

            log_scores[cls] = log_score

        return log_scores

    def predict(self, tokens: list[str]) -> str:
        """
        Predice la clase más probable para un documento tokenizado.
        """
        log_scores = self.predict_log_proba(tokens)
        return max(log_scores, key=log_scores.get)

    def predict_many(self, processed_texts: list[list[str]]) -> list[str]:
        """
        Predice varias instancias.
        """
        return [self.predict(tokens) for tokens in processed_texts]