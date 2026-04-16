from data_loader import load_dataset, validate_labels, basic_dataset_report
from preprocessing import preprocess_corpus, preprocess_text
from vocabulary import build_vocabulary, vectorize_corpus
from naive_bayes import MultinomialNaiveBayes
from model_persistence import save_model, load_model
from config import MODEL_PATH, LAPLACE_ALPHA, TEXT_COLUMN, LABEL_COLUMN

def main():
    print("Iniciando prueba de carga del dataset...")

    try:
        df = load_dataset()
        print("Dataset cargado correctamente.\n")

        validate_labels(df)
        print("Etiquetas validadas correctamente.\n")

        basic_dataset_report(df)

        print("\nIniciando preprocesamiento del texto...")

        texts = df[TEXT_COLUMN].tolist()
        processed_texts = preprocess_corpus(texts)

        print("\n=== TEXTOS PREPROCESADOS ===")
        for i, tokens in enumerate(processed_texts[:5], start=1):
            print(f"{i}. {tokens}")

        print("\nPreprocesamiento completado con éxito.")

        print("\nConstruyendo vocabulario...")
        vocabulary = build_vocabulary(processed_texts)

        print(f"Cantidad de palabras en el vocabulario: {len(vocabulary)}")
        print("Primeras 20 palabras del vocabulario:")
        print(vocabulary[:20])

        print("\nGenerando matriz Bag of Words...")
        bow_matrix = vectorize_corpus(processed_texts, vocabulary)

        print("\n=== PRIMEROS VECTORES BAG OF WORDS ===")
        for i, vector in enumerate(bow_matrix[:3], start=1):
            print(f"Documento {i}: {vector}")

        print("\nPaso 3 completado con éxito.")

        print("\nEntrenando modelo Naïve Bayes...")
        labels = df[LABEL_COLUMN].tolist()

        model = MultinomialNaiveBayes(alpha=LAPLACE_ALPHA)
        model.fit(processed_texts, labels, vocabulary)

        print("Modelo entrenado con éxito.")
        print("\nProbabilidades a priori por clase:")
        for cls, prior in model.class_priors.items():
            print(f"{cls}: {prior:.4f}")

        print("\n=== PRUEBAS DE PREDICCIÓN ===")
        test_examples = [
            "My internet is not working",
            "I need a refund because I was charged twice",
            "Please cancel my account today",
            "Your service is awful and nobody helped me",
            "I want more information about your plans"
        ]

        for text in test_examples:
            processed = preprocess_text(text)
            prediction = model.predict(processed)
            scores = model.predict_log_proba(processed)

            print(f"\nTexto: {text}")
            print(f"Tokens: {processed}")
            print(f"Predicción: {prediction}")
            print("Log-scores:")
            for cls, score in scores.items():
                print(f"  {cls}: {score:.4f}")

        print("\nPaso 4 completado con éxito.")
        print("\nGuardando modelo entrenado...")
        save_model(model, MODEL_PATH)
        print(f"Modelo guardado correctamente en: {MODEL_PATH}")

        print("\nCargando modelo guardado...")
        loaded_model = load_model(MODEL_PATH)
        print("Modelo cargado correctamente.")

        print("\n=== PRUEBA DEL MODELO CARGADO ===")
        sample_text = "My internet connection keeps failing"
        sample_tokens = preprocess_text(sample_text)
        loaded_prediction = loaded_model.predict(sample_tokens)

        print(f"Texto: {sample_text}")
        print(f"Tokens: {sample_tokens}")
        print(f"Predicción con modelo cargado: {loaded_prediction}")

        print("\nPaso 5 completado con éxito.")
        print("\nPrueba completada con éxito.")

    except FileNotFoundError as e:
        print(f"\nERROR DE ARCHIVO: {e}")

    except ValueError as e:
        print(f"\nERROR DE VALIDACIÓN: {e}")

    except Exception as e:
        print(f"\nERROR INESPERADO: {e}")

if __name__ == "__main__":
    main()