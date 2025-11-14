# Estructura Profesional de Notebooks para un Proyecto de ML

Esta es una guía de la estructura recomendada para un proyecto de ciencia de datos, separando el flujo de trabajo en notebooks lógicos y reproducibles.

## 1. `01_data_ingestion_and_cleaning.ipynb`

**Propósito:** Cargar los datos crudos, realizar la limpieza más básica y guardar el resultado en un formato intermedio. Esta es la única vez que tocarás los datos "sucios".

**Acciones Clave:**
* Leer los archivos originales (ej. `accepted.csv`, `rejected.csv`) desde `data/raw/`.
* **Importante (para datos grandes):** Usar técnicas de manejo de memoria como `chunksize` o `dtype` en `pd.read_csv()`.
* Unir los datasets si es necesario.
* Manejar valores faltantes iniciales (ej. eliminar columnas con >90% de NaNs).
* Filtrar datos irrelevantes (ej. transacciones con `Quantity < 0`).
* Corregir tipos de datos (ej. convertir fechas a `datetime`).

**Entrada:** `data/raw/accepted.csv`, `data/raw/rejected.csv`
**Salida:** `data/interim/cleaned_loans.parquet` (o `.pkl`)

---

## 2. `02_exploratory_data_analysis.ipynb` (EDA)

**Propósito:** Entender el negocio y los datos. Buscar patrones, correlaciones e insights.

**Acciones Clave:**
* Cargar el dataset limpio intermedio.
* Crear todas las visualizaciones:
    * **Univariadas:** Histogramas, boxplots (para entender la distribución de cada variable).
    * **Bivariadas:** Gráficos de dispersión (scatter plots) para comparar features con el target.
    * **Multivariadas:** Mapas de calor de correlación (heatmap).
* Documentar los hallazgos.

**Entrada:** `data/interim/cleaned_loans.parquet`
**Salida:** Gráficos e insights (no se guardan nuevos archivos de datos).

---

## 3. `03_feature_engineering_and_modeling.ipynb`

**Propósito:** Preparar los datos finales para el modelo, entrenar, optimizar y registrar los experimentos.

**Acciones Clave:**
* Cargar el dataset limpio intermedio.
* **Ingeniería de Características:**
    * Crear características a partir de fechas (mes, año, etc.).
    * Definir el `ColumnTransformer` para el preprocesamiento (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`).
* **División de Datos:**
    * Separar `X` e `y`.
    * Dividir en `X_train`, `X_test` (¡Usar `stratify=y`!).
* **Guardar Datos Procesados:** Guardar `X_train`, `X_test`, `y_train`, `y_test` en `data/processed/`.
* **Modelado:**
    * Entrenar un modelo base (Baseline).
    * Entrenar modelos avanzados (XGBoost, RandomForest, etc.).
* **Optimización de Hiperparámetros:**
    * Usar `Optuna` o `Hyperopt` (con `cross_val_score`) para encontrar los mejores parámetros.
* **Registro (MLflow):**
    * Entrenar el modelo final con los mejores parámetros.
    * Registrar parámetros, métricas (en `X_test`) y el modelo (`log_model` con `signature`).
* **Guardar Modelo:** Guardar el pipeline de preprocesamiento y el modelo final en un solo archivo (ej. `final_model.pkl`) en la carpeta `models/`.

**Entrada:** `data/interim/cleaned_loans.parquet`
**Salida:** `data/processed/X_train.csv`, `data/processed/X_test.csv`, `models/final_model.pkl`, Corrida de MLflow.

---

## 4. `04_model_interpretation_and_fairness.ipynb`

**Propósito:** (El paso más importante para un banco). Auditar el modelo para entender sus decisiones y asegurar que sea justo.

**Acciones Clave:**
* Cargar el `final_model.pkl` de `models/` y los datos de `data/processed/X_test.csv`.
* **Interpretabilidad Global:**
    * `Permutation Importance`: Para ver qué características importan más en el `X_test`.
    * `SHAP Summary Plot`: Para ver el impacto promedio y la dirección de cada característica.
    * `Partial Dependence Plots (PDP)`: Para entender *cómo* el modelo usa una característica (ej. ¿más edad es siempre mejor?).
* **Interpretabilidad Local:**
    * `SHAP Force Plot` o `LIME`: Para explicar por qué un **cliente de ejemplo** fue rechazado.
    * `DiCE`: Para generar un plan de acción ("¿Qué tendría que cambiar este cliente para ser aprobado?").
* **Auditoría de Sesgo (Fairness):**
    * Usar `fairlearn.MetricFrame` para desglosar la precisión (`accuracy`, `recall`) por grupos sensibles (ej. `género`, `raza`).
    * Demostrar que el modelo no discrimina.

**Entrada:** `models/final_model.pkl`, `data/processed/X_test.csv`
**Salida:** Gráficos, análisis de interpretabilidad y reporte de justicia.