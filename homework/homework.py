# flake8: noqa: E501
#
# Este proyecto busca predecir si un cliente entrará en default el mes siguiente,
# utilizando 23 variables explicativas relacionadas con pagos y características demográficas.
#
# LIMIT_BAL: Monto total del crédito otorgado.
# SEX: Género (1=male, 2=female).
# EDUCATION: Nivel educativo (0=N/A; 1=graduate; 2=university; 3=high school; 4=others).
# MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
# AGE: Edad.
# PAY_0 – PAY_6: Historial reciente de pagos (abril-septiembre 2005).
# BILL_AMT1 – BILL_AMT6: Monto facturado para cada mes.
# PAY_AMT1 – PAY_AMT6: Monto efectivamente pagado cada mes.
#
# La columna "default payment next month" es la variable objetivo.
#
# Los conjuntos de entrenamiento y prueba ya vienen pre-separados
# en la carpeta files/input/.
#
# Pasos resumidos:
# 1. Limpieza del dataset.
# 2. Separación X/y.
# 3. Creación del pipeline (OHE → PCA → escalamiento → selección → SVM).
# 4. Optimización de hiperparámetros con validación cruzada.
# 5. Guardado del modelo comprimido.
# 6. Registro de métricas (precision, balanced accuracy, recall, f1).
# 7. Registro de matrices de confusión.

import pandas as pd
import os 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, balanced_accuracy_score,
    recall_score, f1_score, confusion_matrix
)
import gzip
import pickle
import json


def cargar_datos():
    """Carga los datasets comprimidos en formato ZIP."""
    train = pd.read_csv("files/input/train_data.csv.zip")
    test = pd.read_csv("files/input/test_data.csv.zip")
    return train, test


def limpieza(df):
    """Aplica transformación básica y estandariza valores no válidos."""
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"], errors="ignore")
    df = df.dropna()

    # Filtrar valores no válidos
    df = df[(df["EDUCATION"] > 0) & (df["MARRIAGE"] > 0)]

    # Agrupar niveles de educación superiores a 4
    df.loc[df["EDUCATION"] >= 4, "EDUCATION"] = 4

    return df


def separar_datos(df):
    df = df.copy()
    y = df.pop("default")
    X = df
    return X, y


def crear_pipeline(modelo):
    """Construye el pipeline solicitado en el enunciado."""
    categorias = ["SEX", "EDUCATION", "MARRIAGE"]

    transformador = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), categorias)
        ],
        remainder=StandardScaler()
    )

    seleccion = SelectKBest(score_func=f_classif)

    pipeline = Pipeline(
        steps=[
            ("preprocesar", transformador),
            ("pca", PCA(n_components=None)),
            ("seleccion", seleccion),
            ("modelo", modelo),
        ]
    )

    return pipeline


def configurar_grid_search(pipeline, grid, cv=10):
    return GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1
    )


def entrenar_svm(x_train, y_train):
    svm = SVC(random_state=42)

    pipe = crear_pipeline(svm)

    grid = {
        "pca__n_components": [20, 21],
        "seleccion__k": [12],
        "modelo__kernel": ["rbf"],
        "modelo__gamma": [0.099],
    }

    tuned = configurar_grid_search(pipe, grid)
    tuned.fit(x_train, y_train)

    return tuned


def calcular_metricas(modelo, X, y, tipo):
    pred = modelo.predict(X)
    return {
        "type": "metrics",
        "dataset": tipo,
        "precision": precision_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "recall": recall_score(y, pred),
        "f1_score": f1_score(y, pred),
    }


def matriz_confusion(modelo, X, y, tipo):
    pred = modelo.predict(X)
    cm = confusion_matrix(y, pred)

    return {
        "type": "cm_matrix",
        "dataset": tipo,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }


def guardar_modelo(modelo):
    ruta = "files/models/model.pkl.gz"
    os.makedirs(os.path.dirname(ruta), exist_ok=True)

    with gzip.open(ruta, "wb") as f:
        pickle.dump(modelo, f)


def guardar_metricas(model, X_train, y_train, X_test, y_test):
    resultados = [
        calcular_metricas(model, X_train, y_train, "train"),
        calcular_metricas(model, X_test, y_test, "test"),
        matriz_confusion(model, X_train, y_train, "train"),
        matriz_confusion(model, X_test, y_test, "test"),
    ]

    ruta = "files/output/metrics.json"
    os.makedirs(os.path.dirname(ruta), exist_ok=True)

    with open(ruta, "w") as f:
        for fila in resultados:
            f.write(json.dumps(fila) + "\n")


if __name__ == "__main__":
    train, test = cargar_datos()

    train = limpieza(train)
    test = limpieza(test)

    X_train, y_train = separar_datos(train)
    X_test, y_test = separar_datos(test)

    modelo_final = entrenar_svm(X_train, y_train)

    guardar_modelo(modelo_final)
    guardar_metricas(modelo_final, X_train, y_train, X_test, y_test)