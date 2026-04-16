import re

# Stopwords básicas en inglés para iniciar
STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    "me", "him", "them", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "else",
    "of", "to", "in", "on", "at", "for", "with", "from", "by", "about",
    "as", "into", "over", "after", "before", "between", "through",
    "can", "could", "should", "would", "will", "just",
    "do", "does", "did", "doing", "have", "has", "had",
    "not", "no", "so", "too", "very"
}

def clean_text(text: str) -> str:
    """
    Limpia el texto:
    - convierte a minúsculas
    - elimina caracteres especiales
    - deja solo letras, números y espacios
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> list[str]:
    """
    Divide el texto en tokens por espacios.
    """
    if not text:
        return []
    return text.split()

def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Elimina palabras vacías.
    """
    return [token for token in tokens if token not in STOPWORDS]

def preprocess_text(text: str) -> list[str]:
    """
    Pipeline completo para un texto:
    limpieza -> tokenización -> eliminación de stopwords
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered_tokens = remove_stopwords(tokens)
    return filtered_tokens

def preprocess_corpus(texts: list[str]) -> list[list[str]]:
    """
    Aplica preprocess_text a una lista de textos.
    """
    return [preprocess_text(text) for text in texts]