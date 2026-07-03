"""
Modelo predictivo supervisado de retraso de actividades.

Entrena un clasificador (Random Forest) sobre el registro histórico y expone
funciones para predecir la probabilidad de retraso, clasificar el riesgo en
bajo / medio / alto y explicar los factores que originan cada alerta.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)

from .data_generator import FEATURES, NOMBRES_ES

# Umbrales de clasificación de riesgo a partir de la probabilidad
UMBRAL_MEDIO = 0.35
UMBRAL_ALTO = 0.60


def clasificar_riesgo(prob):
    if prob >= UMBRAL_ALTO:
        return "Alto"
    if prob >= UMBRAL_MEDIO:
        return "Medio"
    return "Bajo"


def entrenar_modelo(df, test_size=0.25, seed=42):
    """Entrena el clasificador y devuelve (modelo, metricas, datos_test)."""
    X = df[FEATURES]
    y = df["retraso"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=6,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "exactitud": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "matriz_confusion": confusion_matrix(y_test, y_pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return modelo, metricas


def importancia_variables(modelo):
    """Devuelve un DataFrame con la importancia de cada variable."""
    imp = pd.DataFrame({
        "variable": [NOMBRES_ES[f] for f in FEATURES],
        "clave": FEATURES,
        "importancia": modelo.feature_importances_,
    }).sort_values("importancia", ascending=False).reset_index(drop=True)
    return imp


def predecir(modelo, df_actividades):
    """Aplica el modelo a un lote de actividades y añade probabilidad y riesgo."""
    df = df_actividades.copy()
    df["prob_retraso"] = modelo.predict_proba(df[FEATURES])[:, 1]
    df["nivel_riesgo"] = df["prob_retraso"].apply(clasificar_riesgo)
    return df


def factores_alerta(row, top=3):
    """Explica una alerta identificando los factores fuera de rango típico.

    Regla de negocio simple y trazable (no caja negra): compara cada variable
    con su umbral de referencia y devuelve los desvíos más relevantes.
    """
    reglas = []
    if row["rendimiento_cuadrillas"] < 0.70:
        reglas.append(("Rendimiento de cuadrillas bajo",
                       0.70 - row["rendimiento_cuadrillas"]))
    if row["restricciones_pendientes"] >= 2:
        reglas.append((f"{int(row['restricciones_pendientes'])} restricciones sin levantar",
                       row["restricciones_pendientes"] * 0.15))
    if row["disponibilidad_frentes"] < 0.75:
        reglas.append(("Frentes de trabajo no disponibles",
                       0.75 - row["disponibilidad_frentes"]))
    if row["disponibilidad_materiales"] < 0.75:
        reglas.append(("Riesgo en suministro de materiales",
                       0.75 - row["disponibilidad_materiales"]))
    if row["desempeno_subcontratista"] < 0.70:
        reglas.append(("Bajo desempeño del subcontratista",
                       0.70 - row["desempeno_subcontratista"]))
    if row["retrasos_previos"] >= 2:
        reglas.append((f"{int(row['retrasos_previos'])} retrasos previos registrados",
                       row["retrasos_previos"] * 0.12))
    if row["incidencia_geotecnica"] == 1:
        reglas.append(("Incidencia geotécnica reportada", 0.5))
    if row["holgura_total_dias"] <= 1:
        reglas.append(("Holgura cercana a cero (ruta crítica)", 0.4))

    reglas.sort(key=lambda x: x[1], reverse=True)
    if not reglas:
        return ["Sin desvíos significativos en las variables de control"]
    return [r[0] for r in reglas[:top]]


def kpi_deteccion_temprana(df_pred, umbral_alerta=UMBRAL_MEDIO):
    """Calcula el KPI del documento: % de actividades críticas con retraso real
    que fueron alertadas anticipadamente (riesgo medio o alto).

    Requiere que df_pred contenga la columna 'retraso' (verdad de terreno) y
    'prob_retraso'. Se evalúa solo sobre actividades de ruta crítica.
    """
    criticas = df_pred[df_pred["es_ruta_critica"] == 1]
    con_retraso = criticas[criticas["retraso"] == 1]
    if len(con_retraso) == 0:
        return None, 0, 0
    alertadas = con_retraso[con_retraso["prob_retraso"] >= umbral_alerta]
    pct = 100 * len(alertadas) / len(con_retraso)
    return pct, len(alertadas), len(con_retraso)


if __name__ == "__main__":
    from data_generator import generar_dataset
    df = generar_dataset()
    modelo, met = entrenar_modelo(df)
    print("Exactitud:", round(met["exactitud"], 3))
    print("Recall   :", round(met["recall"], 3))
    print("ROC-AUC  :", round(met["roc_auc"], 3))
    print("\nImportancia de variables:")
    print(importancia_variables(modelo).to_string(index=False))
