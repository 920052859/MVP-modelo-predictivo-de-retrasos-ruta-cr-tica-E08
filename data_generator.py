"""
Generador de datos sintéticos para la Estación E08 - Línea 4 Metro de Lima.

Simula un registro histórico de actividades de un proyecto EPC con las
variables descritas en el documento base (rendimiento, holgura, restricciones,
disponibilidad de frentes, desempeño de subcontratistas, incidencias
geotécnicas, etc.). La etiqueta objetivo indica si la actividad presentó o no
un retraso relevante dentro del periodo analizado.

El objetivo NO es reemplazar datos reales, sino ofrecer un entorno de
demostración reproducible para el MVP.
"""

import numpy as np
import pandas as pd

# Paquetes de trabajo de la ruta crítica y su naturaleza de riesgo
PAQUETES = {
    "Ingeniería y diseño":        {"cod": "ING",  "riesgo_base": 0.20, "geotecnia": 0.05},
    "Permisos e interferencias":  {"cod": "PER",  "riesgo_base": 0.35, "geotecnia": 0.05},
    "Procura de suministros":     {"cod": "PRO",  "riesgo_base": 0.30, "geotecnia": 0.02},
    "Pantallas de estación":      {"cod": "PANT", "riesgo_base": 0.45, "geotecnia": 0.45},
    "Excavaciones profundas":     {"cod": "EXC",  "riesgo_base": 0.50, "geotecnia": 0.55},
    "Estructuras principales":    {"cod": "EST",  "riesgo_base": 0.40, "geotecnia": 0.20},
    "Acabados y MEP":             {"cod": "ACB",  "riesgo_base": 0.25, "geotecnia": 0.05},
    "Transferencia final":        {"cod": "TRF",  "riesgo_base": 0.20, "geotecnia": 0.02},
}

FEATURES = [
    "rendimiento_cuadrillas",     # 0-1 (1 = rendimiento óptimo)
    "avance_real_pct",            # 0-100
    "holgura_total_dias",         # 0-30 (0 = ruta crítica)
    "restricciones_pendientes",   # entero
    "disponibilidad_frentes",     # 0-1
    "disponibilidad_materiales",  # 0-1
    "desempeno_subcontratista",   # 0-1
    "retrasos_previos",           # entero (nº de retrasos históricos de la actividad)
    "incidencia_geotecnica",      # 0/1
    "duracion_planificada_dias",  # entero
]

NOMBRES_ES = {
    "rendimiento_cuadrillas": "Rendimiento de cuadrillas",
    "avance_real_pct": "Avance real (%)",
    "holgura_total_dias": "Holgura total (días)",
    "restricciones_pendientes": "Restricciones pendientes",
    "disponibilidad_frentes": "Disponibilidad de frentes",
    "disponibilidad_materiales": "Disponibilidad de materiales",
    "desempeno_subcontratista": "Desempeño de subcontratista",
    "retrasos_previos": "Retrasos previos",
    "incidencia_geotecnica": "Incidencia geotécnica",
    "duracion_planificada_dias": "Duración planificada (días)",
}


def _prob_retraso(row, info):
    """Función lógica que relaciona las variables con la probabilidad de retraso.

    Es una combinación logística de los factores de riesgo. Refleja el criterio
    de ingeniería: bajo rendimiento, restricciones sin levantar, frentes no
    disponibles, mal desempeño del subcontratista, retrasos previos e
    incidencias geotécnicas incrementan la probabilidad de retraso.
    """
    # Formulación centrada en valores típicos del proyecto, para que la tasa
    # global de retraso ronde ~30 % y las variables tengan peso interpretable.
    z = -1.15
    z += (info["riesgo_base"] - 0.33) * 2.0
    z += (0.82 - row["rendimiento_cuadrillas"]) * 4.0
    z += (row["restricciones_pendientes"] - 1.4) * 0.50
    z += (0.85 - row["disponibilidad_frentes"]) * 3.0
    z += (0.83 - row["disponibilidad_materiales"]) * 2.0
    z += (0.80 - row["desempeno_subcontratista"]) * 3.0
    z += (row["retrasos_previos"] - 0.7) * 0.50
    z += row["incidencia_geotecnica"] * 1.2
    # holgura amortigua: menos holgura, mayor impacto neto en el hito
    z += max(0, (5 - row["holgura_total_dias"])) * 0.06
    return 1 / (1 + np.exp(-z))


def generar_dataset(n=600, seed=42):
    """Genera un DataFrame de actividades históricas etiquetadas."""
    rng = np.random.default_rng(seed)
    filas = []
    contador = {cod: 0 for p, i in PAQUETES.items() for cod in [i["cod"]]}

    paquetes_lista = list(PAQUETES.items())
    for _ in range(n):
        paquete, info = paquetes_lista[rng.integers(0, len(paquetes_lista))]
        contador[info["cod"]] += 1
        codigo = f"E08-{info['cod']}-{contador[info['cod']]:03d}"

        rendimiento = float(np.clip(rng.normal(0.82, 0.14), 0.35, 1.0))
        avance = float(np.clip(rng.normal(55, 28), 0, 100))
        # las actividades de ruta crítica tienden a holgura baja
        holgura = float(np.clip(rng.exponential(4.5), 0, 30))
        restricciones = int(np.clip(rng.poisson(1.4), 0, 8))
        disp_frentes = float(np.clip(rng.normal(0.85, 0.15), 0.2, 1.0))
        disp_materiales = float(np.clip(rng.normal(0.83, 0.16), 0.2, 1.0))
        desemp_sub = float(np.clip(rng.normal(0.80, 0.17), 0.3, 1.0))
        retrasos_previos = int(np.clip(rng.poisson(0.7), 0, 6))
        geotecnia = int(rng.random() < info["geotecnia"])
        dur_plan = int(np.clip(rng.normal(35, 18), 5, 120))

        row = {
            "codigo_actividad": codigo,
            "paquete": paquete,
            "rendimiento_cuadrillas": round(rendimiento, 3),
            "avance_real_pct": round(avance, 1),
            "holgura_total_dias": round(holgura, 1),
            "restricciones_pendientes": restricciones,
            "disponibilidad_frentes": round(disp_frentes, 3),
            "disponibilidad_materiales": round(disp_materiales, 3),
            "desempeno_subcontratista": round(desemp_sub, 3),
            "retrasos_previos": retrasos_previos,
            "incidencia_geotecnica": geotecnia,
            "duracion_planificada_dias": dur_plan,
            "es_ruta_critica": int(holgura <= 3),
        }
        p = _prob_retraso(row, info)
        row["retraso"] = int(rng.random() < p)
        filas.append(row)

    return pd.DataFrame(filas)


def generar_actividades_activas(n=40, seed=7):
    """Genera un lote de actividades 'en curso' (sin etiqueta) para el tablero
    de alertas semanales, priorizando paquetes críticos."""
    rng = np.random.default_rng(seed)
    criticos = ["Pantallas de estación", "Excavaciones profundas", "Estructuras principales"]
    filas = []
    contador = {cod: 500 for p, i in PAQUETES.items() for cod in [i["cod"]]}
    for _ in range(n):
        # 70% de la muestra en paquetes críticos
        if rng.random() < 0.7:
            paquete = criticos[rng.integers(0, len(criticos))]
        else:
            paquete = list(PAQUETES.keys())[rng.integers(0, len(PAQUETES))]
        info = PAQUETES[paquete]
        contador[info["cod"]] += 1
        codigo = f"E08-{info['cod']}-{contador[info['cod']]:03d}"
        filas.append({
            "codigo_actividad": codigo,
            "paquete": paquete,
            "rendimiento_cuadrillas": round(float(np.clip(rng.normal(0.80, 0.16), 0.35, 1.0)), 3),
            "avance_real_pct": round(float(np.clip(rng.normal(48, 26), 0, 100)), 1),
            "holgura_total_dias": round(float(np.clip(rng.exponential(3.5), 0, 30)), 1),
            "restricciones_pendientes": int(np.clip(rng.poisson(1.6), 0, 8)),
            "disponibilidad_frentes": round(float(np.clip(rng.normal(0.82, 0.16), 0.2, 1.0)), 3),
            "disponibilidad_materiales": round(float(np.clip(rng.normal(0.81, 0.17), 0.2, 1.0)), 3),
            "desempeno_subcontratista": round(float(np.clip(rng.normal(0.78, 0.18), 0.3, 1.0)), 3),
            "retrasos_previos": int(np.clip(rng.poisson(0.8), 0, 6)),
            "incidencia_geotecnica": int(rng.random() < info["geotecnia"]),
            "duracion_planificada_dias": int(np.clip(rng.normal(35, 18), 5, 120)),
        })
    df = pd.DataFrame(filas)
    df["es_ruta_critica"] = (df["holgura_total_dias"] <= 3).astype(int)
    return df


if __name__ == "__main__":
    df = generar_dataset()
    print(df.head())
    print("\nTasa de retraso global:", round(df["retraso"].mean(), 3))
    print("Retraso por paquete:")
    print(df.groupby("paquete")["retraso"].mean().round(3).sort_values(ascending=False))
    df.to_csv("data/actividades_historicas.csv", index=False)
