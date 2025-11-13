# 📘 Clase 08.b: Integración de MLflow en el Proyecto DMeyF

## 🔄 Git

```bash
git status
git add Clase08.b.md config.yaml src/config.py src/optimization.py main.py
# Revisa los cambios antes de hacer commit
git diff --staged
git commit -m "feat: Integra MLflow para seguimiento de experimentos"
git push
```

## 🎯 Objetivo

Aprender a configurar y utilizar MLflow para el seguimiento de experimentos en el proyecto DMeyF, registrando métricas, parámetros y modelos en el servidor MLflow remoto (http://52.144.47.145:8080).

Al finalizar esta clase, serás capaz de:

- Configurar MLflow en tu proyecto DMeyF
- Registrar métricas, parámetros y artefactos en MLflow
- Visualizar y comparar experimentos en la interfaz web
- Implementar buenas prácticas de experimentación

## 📑 Índice

1. [Configuración inicial](#-1-configuración-inicial)
   - 1.1. [config.yaml](#-11-configyaml)
   - 1.2. [src/config.py](#-12-srcconfigpy)
2. [Integración en el código](#-2-integración-en-el-código)
   - 2.1. [main.py](#-21-mainpy)
   - 2.2. [optimization.py](#-22-optimizationpy)
   - 2.3. [test_evaluation.py](#-23-testevaluationpy)
3. [Visualización de resultados](#-3-visualización-de-resultados)
   - 3.1. [Navegación en la interfaz](#-31-navegación-en-la-interfaz)
   - 3.2. [Consultas útiles](#-32-consultas-útiles)
4. [Mejores prácticas](#-4-mejores-prácticas)
   - 4.1. [Organización](#-41-organización)
   - 4.2. [Registro de información](#-42-registro-de-información)
   - 4.3. [Artefactos](#-43-artefactos)
   - 4.4. [Monitoreo](#-44-monitoreo)
5. [Solución de problemas](#-5-solución-de-problemas)
   - 5.1. [Error de conexión al servidor MLflow](#-51-error-de-conexión-al-servidor-mlflow)
   - 5.2. [Error de autenticación](#-52-error-de-autenticación)
   - 5.3. [Error de permisos en artefactos](#-53-error-de-permisos-en-artefactos)
   - 5.4. [Error de versión de MLflow](#-54-error-de-versión-de-mlflow)
   - 5.5. [Depuración](#-55-depuración)

## 🛠 1. Configuración inicial

### 1.1. config.yaml

Agrega la siguiente configuración al final de tu archivo `config.yaml` existente, justo después de la variable `STUDY_NAME`:

```yaml
# Agregar al final de config.yaml (después de STUDY_NAME)
mlflow:
  TRACKING_URI: "http://52.144.47.145:8080"
  ARTIFACT_PATH: "./mlruns"
  # Los tags se definen directamente en el código
```

### 1.2. src/config.py

Actualiza el archivo `src/config.py` para cargar la configuración de MLflow. Agrega este código al final del archivo, justo antes de las funciones de carga de configuración:

```python
# =============================================================================
# Configuración de MLflow
# =============================================================================
try:
    # Cargar configuración de MLflow desde config.yaml
    MLFLOW_CFG = _cfgGeneral.get("mlflow", {})

    # Configuración básica
    MLFLOW_TRACKING_URI = MLFLOW_CFG.get("TRACKING_URI", "")

    # Configuración directa sin plantillas
    MLFLOW_EXPERIMENT_NAME = f"DMEyF-{STUDY_NAME}"
    MLFLOW_ARTIFACT_PATH = MLFLOW_CFG.get("ARTIFACT_PATH", "./mlruns")
    MLFLOW_REGISTERED_MODEL_NAME = f"dmeyf-{STUDY_NAME}"

    # Tags fijos
    MLFLOW_TAGS = {
        "proyecto": "DMEyF-Competencia02",
        "equipo": "python",
        "user_name": "tschoppj",
        "comision": "Lunes",
        "version_dataset": "1.0"
    }

    # Crear directorio de artefactos si no existe
    import os
    os.makedirs(MLFLOW_ARTIFACT_PATH, exist_ok=True)

    logger.info(f"Configuración de MLflow cargada - URI: {MLFLOW_TRACKING_URI}")

except Exception as e:
    logger.error(f"Error al cargar configuración de MLflow: {e}")
    raise
```

## 💻 2. Integración en el código

### 2.1. main.py

Modifica el archivo `main.py` para incluir la configuración básica de MLflow:

```python
# =============================================================================
# Configuración de MLflow
# =============================================================================
import mlflow
from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_TAGS

# Configurar MLflow
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Iniciar run de MLflow
    with mlflow.start_run(run_name=f"experimento-{STUDY_NAME}"):
        # Establecer tags
        mlflow.set_tags(MLFLOW_TAGS)

        # ... resto del código de entrenamiento ...

except Exception as e:
    logger.error(f"Error al configurar MLflow: {e}")
    raise
```

### 2.2. optimization.py

En el archivo `src/optimization.py`, agrega el seguimiento de MLflow a la función de optimización:

```python
def optimizar(X, y, test_dataset, kfold, semilla):
    """Función para optimizar hiperparámetros con seguimiento en MLflow."""
    with mlflow.start_run(nested=True):
        # Función objetivo para Optuna
        def objetivo(trial):
            # Parámetros a optimizar

            # Validación cruzada

            # Calcular métricas promedio
            ganancia_promedio = np.mean(metricas)

            # Registrar métricas en MLflow
            mlflow.log_metric("ganancia_cv", ganancia_promedio, step=trial.number)

            return ganancia_promedio

        # Ejecutar optimización
        study = optuna.create_study(direction="maximize")
        study.optimize(objetivo, n_trials=PARAMETROS_LGB.get("n_trial", 100))

        # Registrar los mejores parámetros
        mlflow.log_params({"mejor_" + k: v for k, v in study.best_params.items()})
        mlflow.log_metric("mejor_ganancia", study.best_value)

        return study.best_params
```

### 2.3. test_evaluation.py

Para el archivo `test_evaluation.py`, agrega el seguimiento de métricas y artefactos:

```python
def evaluar_en_test(df: pd.DataFrame, mejores_params: dict, undersampling: float = 1.0, semilla: int = SEMILLA[0]):
    """
    Evalúa el modelo en el conjunto de test y guarda resultados en MLflow.

    Args:
        df: DataFrame con los datos completos
        mejores_params: Diccionario con los mejores parámetros encontrados
        undersampling: Factor de undersampling a aplicar (1.0 = sin undersampling)
        semilla: Semilla para reproducibilidad

    Returns:
        tuple: (ganancia_test, y_pred_proba) ganancia en test y predicciones
    """
    # ... (código existente de preparación de datos) ...

    # Iniciar un run anidado para la evaluación
    with mlflow.start_run(nested=True, run_name="evaluacion-test"):
        # Entrenar modelo final con mejores parámetros

        # Predecir probabilidades

        # Calcular métricas
        ganancia_test = calcular_ganancia(y_test, y_pred_proba)

        # Registrar métricas
        mlflow.log_metric("ganancia_test", ganancia_test)

        # Guardar predicciones
        pred_df = pd.DataFrame({
            'numero_de_cliente': df_test['numero_de_cliente'].values,
            'probabilidad': y_pred_proba,
            'target_real': y_test.values
        })

        # Guardar como artefacto
        mlflow.log_table(pred_df, "predicciones_test.json")

        # Generar y guardar gráfico de ganancia acumulada
        generar_grafico_ganancia(df_test, y_pred_proba, "ganancia_test.png")

        return ganancia_test, y_pred_proba

def generar_grafico_ganancia(df_test, y_pred_proba, filename):
    """Genera y guarda el gráfico de ganancia acumulada."""
    import matplotlib.pyplot as plt

    # Ordenar por probabilidad descendente

    # Calcular ganancia acumulada
    df_plot['ganancia_acumulada'] = df_plot['ganancia'].cumsum()

    # Crear gráfico

    # Guardar como artefacto
    plt.savefig(filename)
    mlflow.log_artifact(filename, "graficos")
    plt.close()
```

## 📊 3. Visualización de resultados

Después de ejecutar tu experimento, podrás ver los resultados en:

```
http://52.144.47.145:8080
```

### 3.1 Navegación en la interfaz

1. **Experimentos**: Lista de todos los experimentos
2. **Ejecuciones**: Filtra y ordena por métricas o parámetros
3. **Comparación**: Selecciona múltiples ejecuciones para comparar
4. **Detalles**: Haz clic en una ejecución para ver todos sus detalles

### 3.2 Consultas útiles

- `metrics.ganancia_test > 1000000` - Filtra por ganancia mínima
- `params.learning_rate < 0.1` - Filtra por hiperparámetros
- `attributes.start_time > '2024-01-01'` - Ejecuciones recientes
- `tags.team = 'python'` - Filtra por tags personalizados

## ✅ 4. Mejores prácticas

### 4.1 Organización

- **Experimentos**: Agrupa ejecuciones relacionadas bajo el mismo experimento
- **Nombres descriptivos**: Usa `run_name` para identificar fácilmente cada ejecución
- **Tags**: Usa tags para clasificar y filtrar ejecuciones

### 4.2 Registro de información

```python
# Parámetros
mlflow.log_params({
    'learning_rate': 0.1,
    'num_leaves': 31,
    'undersampling': 0.5,
    'semilla': 42
})

# Métricas
mlflow.log_metrics({
    'ganancia_train': ganancia_train,
    'ganancia_valid': ganancia_valid,
    'ganancia_test': ganancia_test,
    'umbral_optimo': umbral_optimo
})

# Tags
mlflow.set_tags({
    'proyecto': 'DMEyF-Competencia02',
    'equipo': 'python',
    'user_name': 'tschoppj',
    'comision': 'Lunes',
    'version_dataset': '1.0'
})
```

### 4.3 Artefactos

- **Modelos**: `mlflow.lightgbm.log_model(modelo, "model")`
- **Gráficos**: Guarda figuras y usa `mlflow.log_artifact()`
- **Datos**: Guarda predicciones o datasets de ejemplo
- **Código**: Registra el estado del código con `mlflow.log_artifact(".", "code")`

### 4.4 Monitoreo

- Revisa regularmente las métricas
- Compara diferentes enfoques
- Documenta los hallazgos importantes en las notas de la ejecución

## 🐛 5. Solución de problemas

### 5.1 Error de conexión al servidor MLflow

```
MlflowException: API request to http://52.144.47.145:8080/api/2.0/mlflow/experiments/get-by-name failed
```

**Solución:**

1. Verifica que el servidor MLflow esté en ejecución
2. Comprueba la conectividad: `ping 52.144.47.145`
3. Verifica que el puerto 8080 esté accesible: `telnet 52.144.47.145 8080`
4. Si usas VPN, asegúrate de estar conectado

### 5.2 Error de autenticación

```
MlflowException: Response: {"error_code":"PERMISSION_DENIED","message":"<Error>"}
```

**Solución:**

```python
# Configura las credenciales antes de importar mlflow
import os
os.environ['MLFLOW_TRACKING_USERNAME'] = 'tu_usuario'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'tu_contraseña'

# Ahora importa mlflow
import mlflow
```

### 5.3 Error de permisos en artefactos

```
PermissionError: [Errno 13] Permission denied: '/mlruns/...'
```

**Solución:**

1. Verifica los permisos del directorio de artefactos
2. Asegúrate de que el usuario tenga permisos de escritura
3. O especifica un directorio local con permisos:
   ```python
   mlflow.set_tracking_uri("file:///tmp/mlruns")
   ```

### 5.4 Error de versión de MLflow

```
AttributeError: module 'mlflow' has no attribute 'log_table'
```

**Solución:**
Actualiza MLflow a la versión más reciente:

```bash
pip install --upgrade mlflow
```

### 5.5 Depuración

Para obtener más información sobre los errores:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(mlflow.__name__)
logger.setLevel(logging.DEBUG)
```

## 📦 Dependencias

Asegúrate de tener estas dependencias en tu entorno virtual:

```bash
# Instalar dependencias de MLflow
pip install mlflow>=2.0.0

# O actualiza tu requirements.txt
# mlflow>=2.0.0
# lightgbm>=3.3.0
# optuna>=3.0.0
# scikit-learn>=1.0.0
# pandas>=1.3.0
# numpy>=1.21.0
# pyyaml>=6.0
```

## 📝 Notas adicionales

### Recursos útiles

- [Documentación oficial de MLflow](https://mlflow.org/docs/latest/index.html)
- [Guía de MLflow con LightGBM](https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

### Convenciones del equipo

- **Nombres de experimentos**: `DMEyF-Competencia02-{descripción}`
- **Tags obligatorios**:
  - `proyecto`: Nombre del proyecto (ej: "DMEyF-Competencia02")
  - `equipo`: Nombre del equipo (ej: "python")
  - `user_name`: Tu identificador
  - `comision`: Comisión a la que perteneces (ej: "Lunes")

### Próximos pasos

1. Configura alertas para monitorear el rendimiento de los modelos
2. Explora el uso de MLflow Model Registry para gestionar el ciclo de vida de los modelos
3. Implementa pruebas automatizadas para validar la calidad de los modelos

¡Listo! Ahora tu proyecto DMeyF está configurado para registrar todos los experimentos en el servidor MLflow remoto. Puedes acceder a los resultados en http://52.144.47.145:8080
