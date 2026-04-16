from collections import Counter

def build_vocabulary(processed_texts: list[list[str]]) -> list[str]:
    """
    Construye un vocabulario único ordenado alfabéticamente
    a partir de los textos preprocesados.
    """
    vocab = set()

    for tokens in processed_texts:
        vocab.update(tokens)

    return sorted(vocab)


def vectorize_document(tokens: list[str], vocabulary: list[str]) -> list[int]:
    """
    Convierte un documento tokenizado en un vector de frecuencias
    según el vocabulario dado.
    """
    token_counts = Counter(tokens)
    return [token_counts.get(word, 0) for word in vocabulary]


def vectorize_corpus(processed_texts: list[list[str]], vocabulary: list[str]) -> list[list[int]]:
    """
    Convierte todo el corpus en una matriz Bag of Words.
    Cada fila representa un documento y cada columna una palabra del vocabulario.
    """
    return [vectorize_document(tokens, vocabulary) for tokens in processed_texts]


def get_word_frequencies(tokens: list[str]) -> dict[str, int]:
    """
    Devuelve un diccionario con frecuencias de palabras
    para un documento individual.
    """
    return dict(Counter(tokens))