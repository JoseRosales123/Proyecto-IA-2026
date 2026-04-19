# Proyecto IA 2026 - Clasificación de Solicitudes a Mesa de Ayuda

## Descripción general
Este proyecto implementa un sistema de clasificación de texto para solicitudes de mesa de ayuda, utilizando el algoritmo **Naïve Bayes Multinomial** desarrollado desde cero en Python.

El objetivo es clasificar automáticamente tickets de soporte en una de las siguientes categorías:

- Soporte Técnico
- Facturación
- Consulta General
- Queja
- Cancelación

Además del motor de inferencia, el proyecto incluye una interfaz web que permite ingresar una solicitud y visualizar la categoría predicha en tiempo real.

---

## Objetivos del proyecto
- Aplicar preprocesamiento de texto.
- Construir un vocabulario con enfoque **Bag of Words**.
- Implementar manualmente **Naïve Bayes Multinomial**.
- Aplicar **Laplace Smoothing**.
- Usar **suma de logaritmos** para evitar underflow numérico.
- Evaluar el modelo con **K-Folds Cross Validation**.
- Guardar y cargar el modelo entrenado.
- Integrar el modelo con una aplicación web.

---

## Tecnologías utilizadas
- Python
- Flask
- Pandas
- NumPy
- NLTK
- HTML
- CSS
- JavaScript

---

## Estructura del proyecto

```bash
Proyecto-IA-2026/
│
├── data/
│   └── raw/
│       └── tickets.csv
│
├── model/
│   └── naive_bayes_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── download_dataset.py
│   ├── kfold.py
│   ├── metrics.py
│   ├── model_persistence.py
│   ├── naive_bayes.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── run_evaluation.py
│   ├── train.py
│   ├── utils.py
│   └── vocabulary.py
│
├── web/
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css
│   │   └── js/
│   │       └── script.js
│   ├── templates/
│   │   └── index.html
│   └── app.py
│
├── README.md
├── requirements.txt
└── .gitignore