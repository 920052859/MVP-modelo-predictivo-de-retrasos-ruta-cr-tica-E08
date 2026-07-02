# 🚇 Predictor de Ruta Crítica · Estación E08 — Línea 4 Metro de Lima

MVP de un **modelo predictivo de machine learning** que anticipa el riesgo de
retraso de las actividades de la ruta crítica en el proyecto EPC de la Estación
E08 de la Línea 4 del Metro de Lima y Callao.

Implementa la propuesta del ejercicio *"Optimización de Proyectos con
Inteligencia Artificial"*: pasar de un control **reactivo** del cronograma a un
esquema de **alerta temprana** basado en patrones de desempeño.

> ⚠️ Los datos son **sintéticos**, generados para demostrar el flujo completo
> de manera reproducible. No representan información real del proyecto.

---

## ✨ Qué hace

- **Estima la probabilidad de retraso** de cada actividad con un clasificador
  supervisado (Random Forest).
- **Clasifica el riesgo** en 🟢 bajo / 🟡 medio / 🔴 alto y **explica los
  factores** que originan cada alerta (rendimiento bajo, restricciones sin
  levantar, incidencia geotécnica, etc.).
- **Prioriza la ruta crítica** en un tablero de alerta temprana de
  actualización semanal.
- **Simulador** interactivo para evaluar una actividad ajustando sus variables.
- Mide el **KPI del documento**: % de actividades críticas con retraso
  identificadas anticipadamente (meta ≥ 80 %).

## 🧮 Variables del modelo

Rendimiento de cuadrillas · avance real · holgura total · restricciones
pendientes · disponibilidad de frentes · disponibilidad de materiales ·
desempeño del subcontratista · retrasos previos · incidencia geotécnica ·
duración planificada.

## 🗂️ Estructura

```
epc-critical-path-predictor/
├── app.py                  # Dashboard Streamlit (5 secciones)
├── src/
│   ├── data_generator.py   # Datos sintéticos de la Estación E08
│   └── model.py            # Entrenamiento, predicción y explicación
├── .streamlit/config.toml  # Tema visual
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Ejecución local

```bash
git clone https://github.com/<tu-usuario>/epc-critical-path-predictor.git
cd epc-critical-path-predictor
pip install -r requirements.txt
streamlit run app.py
```

Luego abre `http://localhost:8501`.

## ☁️ Despliegue en Streamlit Community Cloud

1. Sube el repositorio a GitHub (ver comandos abajo).
2. Entra a [share.streamlit.io](https://share.streamlit.io) y conecta tu cuenta
   de GitHub.
3. **New app** → selecciona el repositorio, la rama `main` y el archivo
   `app.py`.
4. **Deploy**. Streamlit instala `requirements.txt` automáticamente.

### Subir a GitHub

```bash
git init
git add .
git commit -m "MVP: modelo predictivo de retrasos ruta crítica E08"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/epc-critical-path-predictor.git
git push -u origin main
```

## 🧠 Modelo

- **Algoritmo:** Random Forest (`class_weight="balanced"`, 300 árboles).
- **Métricas típicas:** ROC-AUC ≈ 0.79, recall ≈ 0.63 sobre el conjunto de
  prueba.
- **Explicabilidad:** cada alerta se acompaña de reglas de negocio trazables
  que indican los desvíos que la originan, evitando el sesgo de caja negra.

## ⚖️ Alcance y límites

El modelo **complementa —no reemplaza—** el análisis del cronograma y el juicio
profesional. **No modifica la línea base** ni aprueba reprogramaciones. Para uso
real, la base debe alimentarse con exportaciones periódicas de Microsoft
Project y los reportes de producción, procura y restricciones. Toda decisión
contractual sigue bajo responsabilidad del Project Manager y el equipo
competente.

---

*Ejercicio · Maestría en Project Management · MVP v1.0*
