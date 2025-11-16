# Credit Risk Model


A model to predict credit default risk using LendingClub data



# Modelo de Predicción de Riesgo Crediticio (Credit Default Risk)

Este proyecto desarrolla un modelo de machine learning para predecir la probabilidad de que un solicitante de préstamo incumpla (`Charged Off`) con sus pagos, utilizando el dataset de LendingClub. El objetivo es construir una herramienta robusta, interpretable y justa para apoyar las decisiones de aprobación de créditos en una entidad financiera.

**Desarrollado por:** `[Ronaldo David Cornejo Valencia]`

**Repositorio de GitHub:** `https://github.com/kerasPro/credit-risk-model`

**Repositorio de App de Monitoreo** `https://github.com/kerasPro/credit-risk-monitoring`

**Repositorio de Experimentos** `https://dagshub.com/kerasPro/credit-risk-model/experiments`

REVISAR LA APLICACIÓN DE MONITOREO DEL MODELO EN SU PROPIO REPOSITORIO

---  

## a. Problema de ML y Diagrama de Flujo del Proyecto

### Problema de Machine Learning

El problema de negocio es **mitigar el riesgo financiero** mediante la identificación proactiva de solicitantes con alta probabilidad de impago.

Esto se traduce en un **problema de clasificación binaria desbalanceada**:
* **Clase 0 (Negativa):** El préstamo fue pagado (`Fully Paid`).
* **Clase 1 (Positiva):** El préstamo falló (`Charged Off`, `Default`, etc.).

La métrica principal de éxito no es la precisión (`accuracy`), sino el **`Recall`** (Sensibilidad), ya que para el banco, el costo de un "Falso Negativo" (aprobar un préstamo malo) es mucho mayor que el de un "Falso Positivo" (rechazar un préstamo bueno).

### Diagrama de Flujo del Proyecto (Pipeline de MLOps)

Este proyecto sigue un flujo de trabajo profesional de MLOps, separando cada etapa lógica:

- El preprocesamiento se incluye en el paso **Ingestión y Limpieza:** , **EDA (Análisis Exploratorio):** y **Ingeniería de Características:**.


Diagrama de Flujo del Proyecto
1.  **Ingestión y Limpieza:** Carga de datos crudos (`data/raw`), eliminación de fuga de datos (data leakage) y ruido.
2.  **EDA (Análisis Exploratorio):** Análisis de variables, identificación de multicolinealidad.
3.  **Ingeniería de Características:** Creación de *features* valiosas.
4.  **Modeling - Optimización y Entrenamiento:**  Preprocesamiento final y búsqueda de hiperparámetros (Optuna) y registro de experimentos (MLflow).
5.  **Análisis y Equidad:** Interpretación del modelo Local y Global.
6.  **(Opcional) Despliegue:** Monitoreo con `docker-compose`, Grafana e InfluxDB.
    `https://github.com/kerasPro/credit-risk-monitoring`

---

## b. Descripción del Dataset

* **Fuente:** [LendingClub Loan Data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
* **Datos Crudos:** Se utilizó el archivo `accepted_2007_to_2018Q4.csv` (2.26 millones de filas, 151 columnas).
* **Datos Limpios (Post-Notebook 01):** Tras eliminar préstamos "Activos" y columnas con fuga de datos o ruido excesivo, nos quedamos con **1,369,566 préstamos** completados (110 columnas).
* **Target (Objetivo):** `loan_status`, mapeado a:
    * **0 (Pagado):** 78.8%
    * **1 (Impago):** 21.2%

### Diccionario de Datos (Características Principales)

*Pega aquí una versión resumida de tu `Lending_Club_Diccionario_Profesional.csv`. Usa una tabla de Markdown.*

| Variable | Descripción (Simplificada) |
| :--- | :--- |
| **`loan_status`** | **(TARGET)** El resultado del préstamo (Pagado, Impago, etc.). |
| **`loan_amnt`** | Cuánto dinero pidió el cliente. |
| **`int_rate`** | La tasa de interés asignada al préstamo. |
| **`dti`** | (Ratio Deuda-Ingreso) Qué porcentaje del ingreso mensual del cliente se destina a pagar deudas. |
| **`fico_range_low`** | El puntaje de crédito FICO (límite inferior) del cliente. |
| **`emp_length`** | Cuántos años lleva en su trabajo actual. |
| **`mths_since_last_delinq`**| Cuántos meses han pasado desde la última vez que el cliente tuvo un pago atrasado. |
| **`nunca_en_mora`** | (Creada) `1` si el cliente nunca ha tenido una morosidad, `0` si sí. |

---

## c. Técnicas de Optimización de Hiperparámetros

Para encontrar el mejor modelo, se utilizó **Optuna**, un framework de optimización bayesiana, en lugar de un `GridSearch` de fuerza bruta.

* **Librería:** `Optuna`
* **Proceso:**
    1.  Se definió una función `objective` que recibe un `trial` (prueba).
    2.  El `trial` sugiere hiperparámetros (ej. `n_estimators`, `max_depth`, `learning_rate`) para un `XGBClassifier`.
    3.  Para obtener una métrica robusta y evitar *data leakage*, la función `objective` utiliza **Validación Cruzada Estratificada** (`StratifiedKFold(n_splits=35`) sobre el conjunto de entrenamiento.
    4.  La métrica a optimizar (maximizar) fue el **`recall`** de la clase 1 (Impago).
    5.  Se ejecutó un estudio (`study.optimize`) con 50 `trials` para encontrar la mejor combinación.

---

## d. Model Card (Ficha del Modelo)

*Esta sección es clave para demostrar una práctica de ML responsable.*

### Detalles del Modelo
* **Tipo de Modelo:** Light Gradient Boosting Machine (LightGBM).
* **Versión del Modelo:** `v1.0.0`
* **Información de Entrenamiento:** El modelo se entrenó con los datos de `data/processed/train_final.parquet` (1,095,652 filas) y se optimizó usando `Optuna` (ver sección anterior).

### Uso Previsto
* **Uso:** Este modelo está diseñado para **apoyar la decisión** de un analista de riesgo en un banco, proporcionando una probabilidad de impago para nuevas solicitudes de préstamo.
* **Fuera de Alcance (Out-of-Scope):** No debe usarse como un sistema de decisión 100% automático. No debe usarse para préstamos fuera de EE.UU. o para tipos de préstamos que no sean personales.

### Métricas de Evaluación
El modelo fue evaluado en un conjunto de prueba (`X_test`) separado (273,914 filas) que nunca vio durante el entrenamiento. Debido al desbalanceo de clases, se priorizó el `Recall`.

* **ROC AUC:** `[Tu valor, ej: 0.6660]`
* **Recall (Clase 1 - Impago):** `[Tu valor, ej: 0.6863]`
* **Precision (Clase 1 - Impago):** `[Tu valor, ej: 0.3430]`
* **F1-Score (Clase 1 - Impago):** `[Tu valor, ej: 0.4574]`
* **Accuracy (General):** `[Tu valor, ej: 0.6543]`

---

## e. Resultados con Métricas (Online y Offline)

### Métricas Offline (Evaluación en `X_test`)
*Como se detalló en el Model Card, el modelo final logró un **Recall de [0.69]%** en el conjunto de prueba, identificando correctamente el porcentaje de los clientes que eventualmente incumplirían su pago.*

![alt text](/reports/figures/image.png)

### Monitoreo Online (Propuesta de Despliegue)

Visualizar el repositorio de Monitoreo -> `https://github.com/kerasPro/credit-risk-monitoring`

Para un despliegue en producción, se implementó un sistema de monitoreo usando un stack de **Docker, InfluxDB y Grafana**.

* **Arquitectura:**
    1.  **MLflow:** Registra el modelo de producción.
    2.  **Evidently AI:** Un notebook que se ejecuta diariamente, comparando los datos de producción (`current_data`) con los datos de referencia (`reference_data`).
    3.  **InfluxDB:** El script envía las métricas de deriva  a una base de datos de series de tiempo.
    4.  **Grafana:** Un dashboard consume los datos de InfluxDB para visualizar el drift del modelo y de las features en tiempo real.

---

## f. Análisis e Interpretación del Algoritmo (SHAP)

Para entender *por qué* el modelo toma sus decisiones, se utilizaron las librerías `SHAP`.

### Interpretación Global (¿Qué le importa al modelo?)

Conclusión: Podemos ver que las variables más importantes para el modelo son "num_loan_amnt" y "num_dti"

![Feature Importance](/reports/figures/feature_importance.png) 

Notablemente, vemos que difiere un poco con feature importance del mismo modelo. Sin embargo, la mayoria de las tendencias se mantiene

![SHAP Summary Plot](/reports/figures/shap_global_feature_importance_beeswarm_plot.png) 


### Interpretación Local (¿Por qué este cliente fue aceptado?)

![SHAP local Plot](/reports/figures/shap_local_waterfall_plot_user_102.png) 

Analizamos un cliente específico que fue aceptado (`target=0`) usando `shap.force_plot`.
* **Predicción Base:** El cliente promedio tiene una probabilidad de impago de `-0.23%`.
* **Factores Positivos:** Su `cat__term 36 months` era muy alto (empujó a la aprobación +40%).
* **Factores Negativos:** Su `num__mths_since_rcnt_il` era mala (redujo el riesgo -1%).
* **Resultado:** Los factores positivos superaron a los negativos, resultando en una predicción final de "Pagador".

---

## g. Conclusiones

1.  **Rendimiento:** El modelo `Ligbm` optimizado demuestra un rendimiento robusto (`Recall: [0.69]`), superando a los modelos base y demostrando ser una herramienta viable para la evaluación de riesgo.
2.  **Interpretabilidad:** El análisis SHAP confirma que el modelo ha aprendido reglas de negocio lógicas (ej. mayor tasa de interés = mayor riesgo), lo que genera confianza en sus decisiones.
3.  **Desafíos:** El principal desafío fue el **desbalanceo de clases**. Se manejó usando `scale_pos_weight` en XGBoost, lo que fue crucial para mejorar el `Recall`.
4.  **Próximos Pasos:** Hacer el repositorio más replicable y desplegarlo en la nube para hacer lo más profesional posible. Asimismo, se va a mejorar el feature engineering para aumentar el recall comparado a esta primera versión.



<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         credit_risk_model and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── credit_risk_model   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes credit_risk_model a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------