"""
MVP · Modelo predictivo de retrasos en la ruta crítica
Estación E08 — Línea 4 del Metro de Lima y Callao (proyecto EPC)

Dashboard de alerta temprana que estima la probabilidad de retraso de cada
actividad, la clasifica en riesgo bajo / medio / alto y explica los factores
que originan la alerta. Datos sintéticos con fines de demostración.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import (
    generar_dataset, generar_actividades_activas, FEATURES, NOMBRES_ES, PAQUETES,
)
from src.model import (
    entrenar_modelo, importancia_variables, predecir, factores_alerta,
    clasificar_riesgo, kpi_deteccion_temprana, UMBRAL_MEDIO, UMBRAL_ALTO,
)

st.set_page_config(
    page_title="E08 · Predictor de Ruta Crítica",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_RIESGO = {"Alto": "#C8102E", "Medio": "#E8A200", "Bajo": "#2E9E5B"}


# ---------------------------------------------------------------------------
# Carga y entrenamiento cacheados
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_datos():
    hist = generar_dataset(n=700, seed=42)
    activas = generar_actividades_activas(n=45, seed=7)
    return hist, activas


@st.cache_resource(show_spinner=False)
def cargar_modelo(_hist):
    modelo, metricas = entrenar_modelo(_hist)
    return modelo, metricas


hist, activas = cargar_datos()
modelo, metricas = cargar_modelo(hist)
activas_pred = predecir(modelo, activas)
hist_pred = predecir(modelo, hist)


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚇 Estación E08 · Línea 4")
    st.caption("Metro de Lima y Callao — Proyecto EPC")
    st.divider()
    seccion = st.radio(
        "Navegación",
        [
            "📊 Tablero de alertas",
            "🧪 Simulador de actividad",
            "🧠 Modelo y desempeño",
            "🗂️ Datos históricos",
            "ℹ️ Acerca del MVP",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.metric("Actividades en seguimiento", len(activas_pred))
    n_alto = int((activas_pred["nivel_riesgo"] == "Alto").sum())
    st.metric("Alertas de riesgo alto", n_alto)
    st.caption("MVP con datos sintéticos · v1.0")


def badge_riesgo(nivel):
    color = COLOR_RIESGO[nivel]
    return (
        f"<span style='background:{color};color:white;padding:3px 10px;"
        f"border-radius:12px;font-size:0.8rem;font-weight:600'>{nivel}</span>"
    )


# ===========================================================================
# 1. TABLERO DE ALERTAS
# ===========================================================================
if seccion == "📊 Tablero de alertas":
    st.title("Tablero de alerta temprana de retrasos")
    st.caption(
        "Actualización semanal · priorización de actividades con riesgo de "
        "retraso en la ruta crítica"
    )

    c1, c2, c3, c4 = st.columns(4)
    total = len(activas_pred)
    alto = int((activas_pred["nivel_riesgo"] == "Alto").sum())
    medio = int((activas_pred["nivel_riesgo"] == "Medio").sum())
    criticas = int(activas_pred["es_ruta_critica"].sum())
    c1.metric("Actividades monitoreadas", total)
    c2.metric("🔴 Riesgo alto", alto)
    c3.metric("🟡 Riesgo medio", medio)
    c4.metric("Ruta crítica", criticas)

    st.divider()

    fcol1, fcol2 = st.columns([1, 1])
    with fcol1:
        filtro_riesgo = st.multiselect(
            "Filtrar por nivel de riesgo",
            ["Alto", "Medio", "Bajo"],
            default=["Alto", "Medio"],
        )
    with fcol2:
        solo_criticas = st.checkbox("Solo actividades de ruta crítica", value=False)

    df_view = activas_pred[activas_pred["nivel_riesgo"].isin(filtro_riesgo)]
    if solo_criticas:
        df_view = df_view[df_view["es_ruta_critica"] == 1]
    df_view = df_view.sort_values("prob_retraso", ascending=False)

    st.markdown(f"#### {len(df_view)} actividades priorizadas")

    if len(df_view) == 0:
        st.info("No hay actividades que cumplan los filtros seleccionados.")
    else:
        for _, row in df_view.iterrows():
            riesgo = row["nivel_riesgo"]
            with st.container(border=True):
                col_a, col_b, col_c = st.columns([2.2, 1, 2.5])
                with col_a:
                    st.markdown(
                        f"**{row['codigo_actividad']}** &nbsp; "
                        f"{badge_riesgo(riesgo)}",
                        unsafe_allow_html=True,
                    )
                    st.caption(row["paquete"])
                with col_b:
                    st.metric("Prob. retraso", f"{row['prob_retraso']*100:.0f}%")
                    if row["es_ruta_critica"]:
                        st.caption("⚠️ Ruta crítica")
                with col_c:
                    st.markdown("**Factores de la alerta:**")
                    for f in factores_alerta(row):
                        st.markdown(f"- {f}")

    st.divider()
    # Distribución por paquete
    st.markdown("#### Riesgo por paquete de trabajo")
    resumen = (
        activas_pred.groupby("paquete")["prob_retraso"].mean().reset_index()
        .sort_values("prob_retraso", ascending=True)
    )
    fig = px.bar(
        resumen, x="prob_retraso", y="paquete", orientation="h",
        labels={"prob_retraso": "Probabilidad media de retraso", "paquete": ""},
        color="prob_retraso", color_continuous_scale=["#2E9E5B", "#E8A200", "#C8102E"],
    )
    fig.update_layout(coloraxis_showscale=False, height=360, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, width='stretch')


# ===========================================================================
# 2. SIMULADOR
# ===========================================================================
elif seccion == "🧪 Simulador de actividad":
    st.title("Simulador de riesgo de una actividad")
    st.caption(
        "Ajuste las variables de control para estimar la probabilidad de "
        "retraso de una actividad específica."
    )

    col1, col2 = st.columns(2)
    with col1:
        paquete = st.selectbox("Paquete de trabajo", list(PAQUETES.keys()))
        rendimiento = st.slider("Rendimiento de cuadrillas", 0.3, 1.0, 0.80, 0.01)
        avance = st.slider("Avance real (%)", 0, 100, 45)
        holgura = st.slider("Holgura total (días)", 0.0, 30.0, 2.0, 0.5)
        restricciones = st.slider("Restricciones pendientes", 0, 8, 2)
    with col2:
        disp_frentes = st.slider("Disponibilidad de frentes", 0.2, 1.0, 0.80, 0.01)
        disp_materiales = st.slider("Disponibilidad de materiales", 0.2, 1.0, 0.80, 0.01)
        desemp_sub = st.slider("Desempeño del subcontratista", 0.3, 1.0, 0.78, 0.01)
        retrasos_previos = st.slider("Retrasos previos", 0, 6, 1)
        dur_plan = st.slider("Duración planificada (días)", 5, 120, 35)
        geotecnia = st.checkbox("Incidencia geotécnica reportada")

    fila = {
        "rendimiento_cuadrillas": rendimiento,
        "avance_real_pct": avance,
        "holgura_total_dias": holgura,
        "restricciones_pendientes": restricciones,
        "disponibilidad_frentes": disp_frentes,
        "disponibilidad_materiales": disp_materiales,
        "desempeno_subcontratista": desemp_sub,
        "retrasos_previos": retrasos_previos,
        "incidencia_geotecnica": int(geotecnia),
        "duracion_planificada_dias": dur_plan,
        "es_ruta_critica": int(holgura <= 3),
    }
    X = pd.DataFrame([fila])[FEATURES]
    prob = float(modelo.predict_proba(X)[:, 1][0])
    nivel = clasificar_riesgo(prob)

    st.divider()
    rc1, rc2 = st.columns([1, 1.4])
    with rc1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": f"Riesgo: {nivel}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLOR_RIESGO[nivel]},
                "steps": [
                    {"range": [0, UMBRAL_MEDIO * 100], "color": "#DFF3E7"},
                    {"range": [UMBRAL_MEDIO * 100, UMBRAL_ALTO * 100], "color": "#FBEFD0"},
                    {"range": [UMBRAL_ALTO * 100, 100], "color": "#F7D6DC"},
                ],
            },
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, width='stretch')
    with rc2:
        st.markdown(f"### Nivel de riesgo: {badge_riesgo(nivel)}", unsafe_allow_html=True)
        st.markdown(f"**Probabilidad estimada de retraso:** {prob*100:.1f}%")
        if fila["es_ruta_critica"]:
            st.warning("Actividad en ruta crítica (holgura ≤ 3 días).")
        st.markdown("**Factores que explican la alerta:**")
        for f in factores_alerta(pd.Series(fila)):
            st.markdown(f"- {f}")


# ===========================================================================
# 3. MODELO Y DESEMPEÑO
# ===========================================================================
elif seccion == "🧠 Modelo y desempeño":
    st.title("Modelo y desempeño")
    st.caption("Clasificador Random Forest entrenado sobre el registro histórico")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Exactitud", f"{metricas['exactitud']*100:.1f}%")
    m2.metric("Recall (sensibilidad)", f"{metricas['recall']*100:.1f}%")
    m3.metric("Precisión", f"{metricas['precision']*100:.1f}%")
    m4.metric("ROC-AUC", f"{metricas['roc_auc']:.3f}")
    st.caption(
        f"Entrenamiento: {metricas['n_train']} actividades · "
        f"Prueba: {metricas['n_test']} actividades"
    )

    st.divider()

    # KPI del documento
    st.markdown("#### KPI · Detección temprana en ruta crítica")
    pct, alertadas, total_c = kpi_deteccion_temprana(hist_pred)
    kc1, kc2 = st.columns([1, 2])
    with kc1:
        st.metric("Actividades críticas alertadas", f"{pct:.0f}%",
                  help="Meta del proyecto: ≥ 80%")
        st.caption(f"{alertadas} de {total_c} actividades críticas con retraso real")
    with kc2:
        meta = 80
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[pct], y=["Detección"], orientation="h",
                             marker_color="#2E9E5B", name="Logrado"))
        fig.add_vline(x=meta, line_dash="dash", line_color="#C8102E",
                      annotation_text="Meta 80%")
        fig.update_layout(height=140, xaxis_range=[0, 100],
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_ticksuffix="%", showlegend=False)
        st.plotly_chart(fig, width='stretch')

    st.divider()

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown("#### Importancia de las variables")
        imp = importancia_variables(modelo)
        fig = px.bar(
            imp.sort_values("importancia"), x="importancia", y="variable",
            orientation="h", color="importancia",
            color_continuous_scale=["#9AB7D6", "#1A3E6E"],
            labels={"importancia": "Importancia", "variable": ""},
        )
        fig.update_layout(coloraxis_showscale=False, height=400,
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')
    with col_b:
        st.markdown("#### Matriz de confusión (test)")
        cm = metricas["matriz_confusion"]
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicho", y="Real", color="Casos"),
            x=["Sin retraso", "Con retraso"], y=["Sin retraso", "Con retraso"],
        )
        fig.update_layout(height=400, coloraxis_showscale=False,
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')

    st.info(
        "El modelo complementa —no reemplaza— el análisis del cronograma y el "
        "juicio profesional. Toda reprogramación o acción contractual continúa "
        "bajo responsabilidad del Project Manager y el equipo competente."
    )


# ===========================================================================
# 4. DATOS HISTÓRICOS
# ===========================================================================
elif seccion == "🗂️ Datos históricos":
    st.title("Registro histórico de actividades")
    st.caption("Base sintética utilizada para el entrenamiento del modelo")

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", len(hist))
    c2.metric("Tasa de retraso", f"{hist['retraso'].mean()*100:.0f}%")
    c3.metric("Actividades críticas", int(hist["es_ruta_critica"].sum()))

    st.divider()
    st.markdown("#### Tasa de retraso por paquete")
    tasa = (hist.groupby("paquete")["retraso"].mean().reset_index()
            .sort_values("retraso", ascending=True))
    fig = px.bar(tasa, x="retraso", y="paquete", orientation="h",
                 color="retraso", color_continuous_scale=["#2E9E5B", "#C8102E"],
                 labels={"retraso": "Tasa de retraso", "paquete": ""})
    fig.update_layout(coloraxis_showscale=False, height=340,
                      margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Muestra de datos")
    cols_mostrar = ["codigo_actividad", "paquete"] + FEATURES + ["es_ruta_critica", "retraso"]
    st.dataframe(
        hist[cols_mostrar].rename(columns={**NOMBRES_ES,
            "codigo_actividad": "Código", "paquete": "Paquete",
            "es_ruta_critica": "Ruta crítica", "retraso": "Retraso"}),
        width='stretch', height=380,
    )

    st.download_button(
        "⬇️ Descargar dataset (CSV)",
        hist.to_csv(index=False).encode("utf-8"),
        "actividades_historicas_E08.csv", "text/csv",
    )


# ===========================================================================
# 5. ACERCA
# ===========================================================================
else:
    st.title("Acerca de este MVP")
    st.markdown(
        """
Este prototipo implementa el modelo predictivo descrito en el ejercicio
**"Optimización de Proyectos con Inteligencia Artificial"** para la Estación
E08 de la Línea 4 del Metro de Lima y Callao.

**Qué hace**
- Estima la probabilidad de retraso de cada actividad mediante un clasificador
  supervisado (Random Forest).
- Clasifica el riesgo en **bajo / medio / alto** y explica los factores que
  originan cada alerta.
- Prioriza las actividades de la **ruta crítica** en un tablero de alerta
  temprana de actualización semanal.
- Mide el **KPI** del documento: % de actividades críticas con retraso
  identificadas anticipadamente (meta ≥ 80%).

**Variables del modelo**
Rendimiento de cuadrillas, avance real, holgura total, restricciones
pendientes, disponibilidad de frentes y materiales, desempeño del
subcontratista, retrasos previos, incidencia geotécnica y duración planificada.

**Alcance y límites**
Los datos son **sintéticos**, generados para demostrar el flujo completo de
manera reproducible. Para uso real, la base debe alimentarse con exportaciones
periódicas de Microsoft Project y los reportes de producción, procura y
restricciones del proyecto. El modelo **no modifica la línea base** ni aprueba
reprogramaciones: es una herramienta de apoyo a la decisión.

---
*Ejercicio de Maestría en Project Management · MVP v1.0*
        """
    )
    with st.expander("Arquitectura del proyecto"):
        st.code(
            "epc-critical-path-predictor/\n"
            "├── app.py                  # Dashboard Streamlit\n"
            "├── src/\n"
            "│   ├── data_generator.py   # Datos sintéticos E08\n"
            "│   └── model.py            # Entrenamiento y predicción\n"
            "├── requirements.txt\n"
            "└── README.md",
            language="text",
        )
