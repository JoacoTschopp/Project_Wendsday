# 📘 Clase04.md

## **Entrenamiento Final y Predicción con Mejores Hiperparámetros**

---

## Git - Flujo de trabajo básico

Antes de comenzar con la clase, es importante mantener nuestro repositorio actualizado y sincronizado. Ejecuta los siguientes comandos en la consola:

```bash
git status
git add .
git commit -m "actualizacion"
git pull origin
git status
git add .
git push
```

**Al finalizar la clase, ejecuta nuevamente:**

```bash
git add .
git commit -m "Clase04: Entrenamiento final y predicción con mejores hiperparámetros completada"
git push
```

---

## 🎯 Objetivo

Implementar el **entrenamiento final** del modelo usando los mejores hiperparámetros encontrados por Optuna, entrenar con todos los períodos disponibles y generar predicciones finales para el período objetivo.

---

## 📑 Índice de la clase

0. **Evaluación en conjunto de test** - Validar mejores hiperparámetros en MES_TEST (202104)
1. **Configuración para entrenamiento final** - Usar FINAL_TRAIN y FINAL_PREDIC del config.yaml
2. **Cargar mejores hiperparámetros** - Extraer los mejores parámetros del archivo JSON de Optuna
3. **Script de entrenamiento final** - Crear módulo para entrenamiento con datos completos
4. **Predicción final** - Generar predicciones para el período objetivo
5. **Generación de archivo de salida** - Crear CSV con numero_cliente y predict
6. **Pipeline completo** - Integrar todo en un script ejecutable

---

## 0) Evaluación en conjunto de test

Antes de proceder al entrenamiento final, es **fundamental validar** que nuestros mejores hiperparámetros encontrados por Optuna realmente funcionan bien en un conjunto de datos no visto durante la optimización.

### **¿Por qué evaluar en test?**

- **Validación independiente**: El período `MES_TEST: 202104` no fue usado durante la optimización
- **Detección de overfitting**: Verificar si los hiperparámetros generalizan bien
- **Confianza en el modelo**: Confirmar que el rendimiento es consistente

### **Configuración del test**

El test usa la configuración del `config.yaml`:

```yaml
competencia01:
  MES_TRAIN: [202101, 202102]     # Períodos de entrenamiento
  MES_VALIDACION: 202103          # Período de validación (usado en Optuna)
  MES_TEST: 202104               # Período de test (NO usado en Optuna) ⭐
```

**Para el test:**

- **Entrenamiento**: `MES_TRAIN + MES_VALIDACION` = `[202101, 202102, 202103]`
- **Evaluación**: `MES_TEST` = `202104`

### **Función de evaluación en test**

Ya agregamos las funciones necesarias en `src/optimization.py`:

```python
# src/optimization.py (implementación simplificada)

def evaluar_en_test(df, mejores_params) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.

    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna

    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")

    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]

    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'] == MES_TEST]

    # Entrenar modelo con mejores parámetros
    # ... Implementar entrenamiento y test con la logica de entrenamiento FINAL para mayor detalle
    # recordar realizar todos los df necesarios y utilizar lgb.train()

    # Calcular solo la ganancia
    ganancia_test = calcular_ganancia(y_test, y_pred_binary)

    # Estadísticas básicas
    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100

    resultados = {
        'ganancia_test': float(ganancia_test),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas)
    }

    return resultados

def guardar_resultados_test(resultados_test, archivo_base=None):
    """
    Guarda los resultados de la evaluación en test en un archivo JSON.
    """
    # Guarda en resultados/{STUDY_NAME}_test_results.json
    # ... Implementar utilizando la misma logica que cuando guardamos una iteracion de la Bayesiana
```

---

## 1) Configuración para entrenamiento final

La configuración en `config.yaml` ya incluye los parámetros necesarios para el entrenamiento final:

```yaml
# config.yaml
STUDY_NAME: "lgb_optimization_competencia01"

competencia01:
  # ... configuración anterior ...

  # Configuración para entrenamiento final
  FINAL_TRAIN: ["202101", "202102", "202103", "202104"]  # Todos los períodos para entrenar
  FINAL_PREDIC: "202106"                                 # Período objetivo para predicción
```

**¿Por qué estos períodos?**

- **FINAL_TRAIN**: Incluye todos los períodos disponibles (incluyendo el de validación) para maximizar datos de entrenamiento
- **FINAL_PREDIC**: Período futuro donde queremos hacer predicciones para la competencia

**Actualización en `src/config.py`**:

```python
# src/config.py (agregar estas líneas)
try:
    with open(PATH_CONFIG, "r") as f:
        _cfgGeneral = yaml.safe_load(f)
        _cfg = _cfgGeneral["competencia01"]

        # ... configuración existente ...

        # Configuración para entrenamiento final
        FINAL_TRAIN = _cfg.get("FINAL_TRAIN", ["202101", "202102", "202103", "202104"])
        FINAL_PREDIC = _cfg.get("FINAL_PREDIC", "202106")

except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise
```

---

## 2) Cargar mejores hiperparámetros

Necesitamos extraer los mejores hiperparámetros del archivo JSON generado por Optuna, quien implemento el uso de db puede tomarlo desde ahi:

```python
# src/best_params.py
import json
import logging
from .config import STUDY_NAME

logger = logging.getLogger(__name__)

def cargar_mejores_hiperparametros(archivo_base: str = None) -> dict:
    """
    Carga los mejores hiperparámetros desde el archivo JSON de iteraciones de Optuna.

    Args:
        archivo_base: Nombre base del archivo (si es None, usa STUDY_NAME)

    Returns:
        dict: Mejores hiperparámetros encontrados
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME

    archivo = f"resultados/{archivo_base}_iteraciones.json"

    try:
        with open(archivo, 'r') as f:
            iteraciones = json.load(f)

        if not iteraciones:
            raise ValueError("No se encontraron iteraciones en el archivo")

        # Encontrar la iteración con mayor ganancia
        mejor_iteracion = max(iteraciones, key=lambda x: x['value'])
        mejores_params = mejor_iteracion['params']
        mejor_ganancia = mejor_iteracion['value']

        logger.info(f"Mejores hiperparámetros cargados desde {archivo}")
        logger.info(f"Mejor ganancia encontrada: {mejor_ganancia:,.0f}")
        logger.info(f"Trial número: {mejor_iteracion['trial_number']}")
        logger.info(f"Parámetros: {mejores_params}")

        return mejores_params

    except FileNotFoundError:
        logger.error(f"No se encontró el archivo {archivo}")
        logger.error("Asegúrate de haber ejecutado la optimización con Optuna primero")
        raise
    except Exception as e:
        logger.error(f"Error al cargar mejores hiperparámetros: {e}")
        raise

def obtener_estadisticas_optuna(archivo_base=None):
    """
    Obtiene estadísticas de la optimización de Optuna.

    Args:
        archivo_base: Nombre base del archivo

    Returns:
        dict: Estadísticas de la optimización
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME

    archivo = f"resultados/{archivo_base}_iteraciones.json"

    try:
        with open(archivo, 'r') as f:
            iteraciones = json.load(f)

        ganancias = [iter['value'] for iter in iteraciones]

        estadisticas = {
            'total_trials': len(iteraciones),
            'mejor_ganancia': max(ganancias),
            'peor_ganancia': min(ganancias),
            'ganancia_promedio': sum(ganancias) / len(ganancias),
            'top_5_trials': sorted(iteraciones, key=lambda x: x['value'], reverse=True)[:5]
        }

        logger.info("Estadísticas de optimización:")
        logger.info(f"  Total trials: {estadisticas['total_trials']}")
        logger.info(f"  Mejor ganancia: {estadisticas['mejor_ganancia']:,.0f}")
        logger.info(f"  Ganancia promedio: {estadisticas['ganancia_promedio']:,.0f}")

        return estadisticas

    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        raise
```

---

## 3) Script de entrenamiento final

Creamos un módulo especializado para el entrenamiento final.

Que contiene la secuencia de tres metodos segregados en sus funcionalidades:

preparar_datos_entrenamiento_final() --> entrenar_modelo_final() -->  generar_predicciones_finales()

```python
# src/final_training.py
import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
from .config import FINAL_TRAIN, FINAL_PREDIC, SEMILLA
from .best_params import cargar_mejores_hiperparametros
from .gain_function import ganancia_lgb_binary

logger = logging.getLogger(__name__)

def preparar_datos_entrenamiento_final(df: pd.DataFrame) -> tuple:
    """
    Prepara los datos para el entrenamiento final usando todos los períodos de FINAL_TRAIN.

    Args:
        df: DataFrame con todos los datos

    Returns:
        tuple: (X_train, y_train, X_predict, clientes_predict)
    """
    logger.info(f"Preparando datos para entrenamiento final")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"Período de predicción: {FINAL_PREDIC}")

    # Datos de entrenamiento: todos los períodos en FINAL_TRAIN

    # Datos de predicción: período FINAL_PREDIC

    logger.info(f"Registros de entrenamiento: {len(df_train):,}")
    logger.info(f"Registros de predicción: {len(df_predict):,}")

    #Corroborar que no esten vacios los df

    # Preparar features y target para entrenamiento

    X_train
    y_train

    # Preparar features para predicción
    X_predict
    clientes_predict

    logger.info(f"Features utilizadas: {len(features_cols)}")
    logger.info(f"Distribución del target - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")

    return X_train, y_train, X_predict, clientes_predict

def entrenar_modelo_final(X_train: pd.DataFrame, y_train: pd.Series, mejores_params: dict) -> lgb.Booster:
    """
    Entrena el modelo final con los mejores hiperparámetros.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        mejores_params: Mejores hiperparámetros de Optuna

    Returns:
        lgb.Booster: Modelo entrenado
    """
    logger.info("Iniciando entrenamiento del modelo final")

    # Configurar parámetros del modelo
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'random_state': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        'verbose': -1,
        **mejores_params  # Agregar los mejores hiperparámetros
    }

    logger.info(f"Parámetros del modelo: {params}")

    # Crear dataset de LightGBM

    # Entrenar modelo con lgb.train()

    return modelo

def generar_predicciones_finales(modelo: lgb.Booster, X_predict: pd.DataFrame, clientes_predict: np.ndarray, umbral: float = 0.025) -> pd.DataFrame:
    """
    Genera las predicciones finales para el período objetivo.

    Args:
        modelo: Modelo entrenado
        X_predict: Features para predicción
        clientes_predict: IDs de clientes
        umbral: Umbral para clasificación binaria

    Returns:
        pd.DataFrame: DataFrame con numero_cliente y predict
    """
    logger.info("Generando predicciones finales")

    # Generar probabilidades con el modelo entrenado

    # Convertir a predicciones binarias con el umbral establecido

    # Crear DataFrame de 'resultados' con nombres de atributos que pide kaggle

    # Estadísticas de predicciones
    total_predicciones = len(resultados)
    predicciones_positivas = (resultados['predict'] == 1).sum()
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100

    logger.info(f"Predicciones generadas:")
    logger.info(f"  Total clientes: {total_predicciones:,}")
    logger.info(f"  Predicciones positivas: {predicciones_positivas:,} ({porcentaje_positivas:.2f}%)")
    logger.info(f"  Predicciones negativas: {total_predicciones - predicciones_positivas:,}")
    logger.info(f"  Umbral utilizado: {umbral}")

    return resultados
```

---

## 4) Generación de archivo de salida

Creamos la funcionalidad para guardar las predicciones en el formato requerido:

Nota: este script lo desarrollaremos y agregaremos nuevas funciones la clase que viene.

```python
# src/output_manager.py
import pandas as pd
import os
import logging
from datetime import datetime
from .config import STUDY_NAME

logger = logging.getLogger(__name__)

def guardar_predicciones_finales(resultados_df: pd.DataFrame, nombre_archivo=None) -> str:
    """
    Guarda las predicciones finales en un archivo CSV en la carpeta predict.

    Args:
        resultados_df: DataFrame con numero_cliente y predict
        nombre_archivo: Nombre del archivo (si es None, usa STUDY_NAME)

    Returns:
        str: Ruta del archivo guardado
    """
    # Crear carpeta predict si no existe
    os.makedirs("predict", exist_ok=True)

    # Definir nombre del archivo
    if nombre_archivo is None:
        nombre_archivo = STUDY_NAME

    # Agregar timestamp para evitar sobrescribir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_archivo = f"predict/{nombre_archivo}_{timestamp}.csv"

    # Validar formato del DataFrame

    # Validar tipos de datos

    # Validar valores de predict (deben ser 0 o 1)


    # Guardar archivo
    resultados_df.to_csv(ruta_archivo, index=False)

    logger.info(f"Predicciones guardadas en: {ruta_archivo}")
    logger.info(f"Formato del archivo:")
    logger.info(f"  Columnas: {list(resultados_df.columns)}")
    logger.info(f"  Registros: {len(resultados_df):,}")
    logger.info(f"  Primeras filas:")
    logger.info(f"{resultados_df.head()}")

    return ruta_archivo

```

---

## 5) Pipeline completo - main_final.py

Creamos un script principal que orquesta todo el proceso de entrenamiento final:

```python
# main_final.py
import pandas as pd
import os
import datetime
import logging

from src.loader import cargar_datos, convertir_clase_ternaria_a_target
from src.features import feature_engineering_lag
from src.optimization import optimizar, evaluar_en_test, guardar_resultados_test
from src.best_params import cargar_mejores_hiperparametros
from src.final_training import preparar_datos_entrenamiento_final, generar_predicciones_finales, entrenar_modelo_final
from src.output_manager import guardar_predicciones_finales
from src.best_params import obtener_estadisticas_optuna
from src.config import *

## config basico logging
os.makedirs("logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
monbre_log = f"log_{STUDY_NAME}_{fecha}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{monbre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


## Funcion principal
def main():
    logger.info("Inicio de ejecucion.")

    #00 Cargar datos
    os.makedirs("data", exist_ok=True)
    df = cargar_datos(DATA_PATH)

    #01 Feature Engineering
    atributos = ["mcuentas_saldo", "mtarjeta_visa_consumo", "cproductos"]
    cant_lag = 2
    df_fe = feature_engineering_lag(df, atributos, cant_lag)
    logger.info(f"Feature Engineering completado: {df_fe.shape}")

    #02 Convertir clase_ternaria a target binario
    df_fe = convertir_clase_ternaria_a_target(df_fe)

    #03 Ejecutar optimizacion de hiperparametros
    study = optimizar(df_fe, n_trial=5)

    #04 Análisis adicional
    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")

    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

    #05 Test en mes desconocido
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    # Cargar mejores hiperparámetros
    mejores_params = cargar_mejores_hiperparametros()

    # Evaluar en test
    resultados_test = evaluar_en_test(df_fe, mejores_params)

    # Guardar resultados de test
    guardar_resultados_test(resultados_test)

    # Resumen de evaluación en test
    logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    logger.info(f"✅ Ganancia en test: {resultados_test['ganancia_test']:,.0f}")
    logger.info(f"🎯 Predicciones positivas: {resultados_test['predicciones_positivas']:,} ({resultados_test['porcentaje_positivas']:.2f}%)")


    #06 Entrenar modelo final
    logger.info("=== ENTRENAMIENTO FINAL ===")
    logger.info("Preparar datos para entrenamiento final")
    X_train, y_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final(df_fe)

    # Entrenar modelo final
    logger.info("Entrenar modelo final")
    modelo_final = entrenar_modelo_final(X_train, y_train, mejores_params)

    # Generar predicciones finales
    logger.info("Generar predicciones finales")
    resultados = generar_predicciones_finales(modelo_final, X_predict, clientes_predict)

    # Guardar predicciones
    logger.info("Guardar predicciones")
    archivo_salida = guardar_predicciones_finales(resultados)

    # Resumen final
    logger.info("=== RESUMEN FINAL ===")
    logger.info(f"✅ Entrenamiento final completado exitosamente")
    logger.info(f"📊 Mejores hiperparámetros utilizados: {mejores_params}")
    logger.info(f"🎯 Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"🔮 Período de predicción: {FINAL_PREDIC}")
    logger.info(f"📁 Archivo de salida: {archivo_salida}")
    logger.info(f"📝 Log detallado: logs/{monbre_log}")


    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.")

if __name__ == "__main__":
    main()
```

---

📦 **Dependencias necesarias**:

```bash
pip install pandas lightgbm optuna duckdb matplotlib seaborn pyyaml
```

**Estructura final del proyecto:**

```
Project_Wendsday/
├── main.py                    # Optimización + evaluación en test ⭐ ACTUALIZADO
├── test_evaluation.py         # Evaluación independiente en test ⭐ NUEVO
├── analisis_test.py          # Análisis de resultados de test ⭐ NUEVO
├── main_final.py              # Entrenamiento final (próxima sección)
├── config.yaml               # Configuración completa
├── src/
│   ├── best_params.py        # Cargar mejores hiperparámetros ⭐ NUEVO
│   ├── optimization.py       # Optimización + evaluación en test ⭐ ACTUALIZADO
│   ├── final_training.py     # Entrenamiento final (próxima sección)
│   ├── output_manager.py     # Gestión de archivos de salida (próxima sección)
│   └── ... (módulos anteriores)
├── resultados/               # Resultados de Optuna y test ⭐ ACTUALIZADO
│   ├── {STUDY_NAME}_iteraciones.json
│   └── {STUDY_NAME}_test_results.json ⭐ NUEVO
├── predict/                  # Carpeta de predicciones finales (próxima sección)
└── logs/                     # Logs detallados con timestamps
```
