# Project Wendsday · Experimento_v5

## Características del dataset

- **Fuente:** `data/competencia.csv` con histórico mensual de clientes bancarios.
- **Identificador temporal:** `foto_mes` (enero 2021 a junio 2021) permite separación Train/Valid/Test.
- **Variable objetivo:** `clase_ternaria` con categorías `CONTINUA`, `BAJA+1`, `BAJA+2`.
- **Cantidad de features:** más de 300 atributos numéricos/categóricos derivados del comportamiento financiero del cliente.
- **Preprocesamiento clave:**
  - Conversión de `clase_ternaria` a binaria mediante `convertir_clase_ternaria_a_target()`.
  - Ingeniería de atributos lag/delta vía `feature_engineering_lag()` y `feature_engineering_delta_lag()`.
  - Posibilidad de aplicar undersampling configurable.

### Distribución del target por mes

| foto_mes | CONTINUA | BAJA+1 | BAJA+2 |
| -------- | -------- | ------ | ------ |
| 202101   | 160080   | 622    | 825    |
| 202102   | 160294   | 831    | 1030   |
| 202103   | 161120   | 1039   | 950    |
| 202104   | 161337   | 955    | 1126   |
| 202105   | 161942   | 1134   | 841    |
| 202106   | 162336   | 843    | 1134   |

## Optimización Bayesiana (BO) con LightGBM

### Hiperparámetros a optimizar

Valores extraídos de `conf.yaml` (`parametros_lgb`):

- **num_leaves:** [10, 300]
- **learning_rate:** [0.01, 0.3]
- **feature_fraction:** [0.4, 1.0]
- **bagging_fraction:** [0.4, 1.0]
- **min_child_samples:** [5, 100]
- **max_depth:** [3, 15]
- **reg_alpha:** [0.0, 10.0]
- **reg_lambda:** [0.0, 10.0]
- **min_data_in_leaf:** [1, 50]
- **num_iterations:** [100, 1000]
- **bagging_freq:** [1, 7]

Parámetros fijos durante la BO:

- **objective:** "binary"
- **metric:** "None" (se usa métrica personalizada)
- **min_gain_to_split:** 0.0
- **verbosity:** -1
- **max_bin:** 31

### Estrategia metodológica

- **Modelo:** `lightgbm.cv` con métrica personalizada `ganancia_evaluator()`.
- **Fold:** 5-fold estratificado (definido en `optimization_BO.py`).
- **Rondas máximas:** 2000 boosting rounds con `early_stopping(200)`.
- **Undersampling:** ratio 0.5 (registrado como `undersampling_ratio_BO` en `main.py`).
- **Semilla:** `SEMILLA[0]` desde configuración.
- **Persistencia:** cada trial se guarda en `resultados/<STUDY_NAME>_iteraciones.json` mediante `guardar_iteracion()`.

### Resultados (completar)

- **Parámetros óptimos:** _pendiente de completar_
- **Ganancia en test (20 tiradas):** _pendiente de completar_
- **Gráfico de 20 tiradas:** _pendiente de completar (ruta de archivo)_

## ZeroShot Parameter Tuning

Valores obtenidos vía `optimization_ZS.py` utilizando FLAML para un arranque sin BO.

- Los parámetros LightGBM se construyen con `_construir_parametros_lightgbm()` y se guardan en `resultados/<STUDY_NAME>_zs_*`.
- Se utiliza la misma función de evaluación `calcular_ganancia()` para estimar la ganancia y sugerir umbral.

### Resultados ZeroShot (completar)

- **Parámetros encontrados:** _pendiente de completar_
- **Ganancia en test (20 tiradas):** _pendiente de completar_
- **Gráfico de 20 tiradas:** _pendiente de completar (ruta de archivo)_

## Búsqueda automática de hiperparámetros (AutoML)

Exploración futura mediante un algoritmo de aprendizaje automático automatizado.

### Resultados AutoML (completar)

- **Parámetros encontrados:** _pendiente de completar_
- **Ganancia en test (20 tiradas):** _pendiente de completar_
- **Gráfico de 20 tiradas:** _pendiente de completar (ruta de archivo)_
