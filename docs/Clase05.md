# 📘 Clase05.md

## **Optimización Avanzada y Visualización de Resultados**

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
git commit -m "Clase05: Optimización avanzada y visualización de resultados completada"
git push
```

---

## 🎯 Objetivo

Implementar **técnicas avanzadas de optimización** con Cross Validation, **función de ganancia optimizada** usando Polars para máximo rendimiento, y **generación automatizada de gráficos** para análisis visual de resultados.

---

## 📑 Índice de la clase

1. **Nueva función Optimizar() con Cross Validation** - Implementar validación cruzada en LGBM para mayor robustez
2. **Función de ganancia optimizada con Polars** - Calcular ganancia total máxima sin umbral usando Polars
3. **Generación de gráficos avanzados** - Métodos de visualización basados en grafico_test.py
4. **Pipeline completo integrado** - Main.py que orquesta todos los nuevos métodos

---

## 1) Nueva función Optimizar() con Cross Validation en LGBM

La validación cruzada nos permite obtener estimaciones más robustas del rendimiento del modelo, reduciendo la varianza y mejorando la generalización.

### **¿Por qué Cross Validation?**

- **Mayor robustez**: Reduce la dependencia de una sola partición train/validation
- **Mejor estimación**: Promedia resultados sobre múltiples folds
- **Detección de overfitting**: Identifica modelos que no generalizan bien
- **Estabilidad**: Resultados más consistentes entre ejecuciones

HASTA QUE NOS MEUSTRAN ALGO MEJOR!

### **Implementación de Cross Validation**

```python
# src/optimizacion_cv.py
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import json
import os
import logging
from .config import (
    SEMILLA, MES_TRAIN, MES_VALIDACION, STUDY_NAME,
    GANANCIA_ACIERTO, COSTO_ESTIMULO, FEATURES, PARAMETROS_LGB
)
from .gain_function import ganancia_evaluator

logger = logging.getLogger(__name__)

def objetivo_ganancia_cv(trial, df) -> float:
    """
    Función objetivo para Optuna con Cross Validation.
    Utiliza SEMILLA[0] desde configuración para reproducibilidad.

    Args:
        trial: Trial de Optuna
        df: DataFrame con datos

    Returns:
        float: Ganancia promedio del CV
    """
    # Hiperparámetros a optimizar (desde configuración YAML)
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'num_leaves': trial.suggest_int('num_leaves', PARAMETROS_LGB['num_leaves'][0], PARAMETROS_LGB['num_leaves'][1]),
        'learning_rate': trial.suggest_float('learning_rate', PARAMETROS_LGB['learning_rate'][0], PARAMETROS_LGB['learning_rate'][1], log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', PARAMETROS_LGB['feature_fraction'][0], PARAMETROS_LGB['feature_fraction'][1]),
        'bagging_fraction': trial.suggest_float('bagging_fraction', PARAMETROS_LGB['bagging_fraction'][0], PARAMETROS_LGB['bagging_fraction'][1]),
        'min_child_samples': trial.suggest_int('min_child_samples', PARAMETROS_LGB['min_child_samples'][0], PARAMETROS_LGB['min_child_samples'][1]),
        'max_depth': trial.suggest_int('max_depth', PARAMETROS_LGB['max_depth'][0], PARAMETROS_LGB['max_depth'][1]),
        'reg_alpha': trial.suggest_float('reg_alpha', PARAMETROS_LGB['reg_alpha'][0], PARAMETROS_LGB['reg_alpha'][1]),
        'reg_lambda': trial.suggest_float('reg_lambda', PARAMETROS_LGB['reg_lambda'][0], PARAMETROS_LGB['reg_lambda'][1]),
        'bin': trial.suggest_int('bin', PARAMETROS_LGB['bin'][0], PARAMETROS_LGB['bin'][1]),
        'random_state': SEMILLA[0],  # Desde configuración YAML
        'verbosity': -1
    }

    # Preparar datos para CV


    # Features y target


    # Crear dataset de LightGBM


    # Configurar CV con semilla desde configuración
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=100,
        nfold=5,
        stratified=True,
        feval=ganancia_evaluator,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # Extraer ganancia promedio y max

    # Guardar iteración para análisis posterior

    # Agregar nueva iteración

    logger.info(f"Iteración CV {trial.number} guardada - Ganancia: {ganancia_promedio:,.0f} ± {ganancia_std:,.0f}")

    return ganancia_promedio


def optimizar_con_cv(df, n_trials=50) -> optuna.Study:
    """
    Ejecuta optimización bayesiana con Cross Validation.

    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar

    Returns:
        optuna.Study: Estudio de Optuna con resultados de CV
    """
    study_name = f"{STUDY_NAME}"

    logger.info(f"Iniciando optimización con CV - {n_trials} trials")
    logger.info(f"Configuración CV: períodos={MES_TRAIN + [MES_VALIDACION] if isinstance(MES_TRAIN, list) else [MES_TRAIN, MES_VALIDACION]}")

    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)
    )

    # Ejecutar optimización
    study.optimize(lambda trial: objetivo_ganancia_cv(trial, df), n_trials=n_trials)

    # Resultados

    return study
```

---

## 2) Función de ganancia optimizada con Polars

Polars es una biblioteca de manipulación de datos extremadamente rápida que puede acelerar significativamente el cálculo de la ganancia total máxima, especialmente cuando trabajamos con grandes volúmenes de datos.

### **Implementación de ganancia máxima sin umbral**

Función de evaluación personalizada para LightGBM

```python
# src/gain_function.py
import polars as pl
import pandas as pd
import logging
from .config import GANANCIA_ACIERTO, COSTO_ESTIMULO

logger = logging.getLogger(__name__)

def ganancia_evaluator(y_pred, y_true) -> float:
    """
    Función de evaluación personalizada para LightGBM.
    Ordena probabilidades de mayor a menor y calcula ganancia acumulada
    para encontrar el punto de máxima ganancia.

    Args:
        y_pred: Predicciones de probabilidad del modelo
        y_true: Dataset de LightGBM con labels verdaderos

    Returns:
        float: Ganancia total
    """
    y_true = y_true.get_label()

    # Convertir a DataFrame de Polars para procesamiento eficiente
    df_eval = pl.DataFrame({'y_true': y_true,'y_pred_proba': y_pred})

    # Ordenar por probabilidad descendente
    df_ordenado = df_eval.sort('y_pred_proba', descending=True)

    # Calcular ganancia individual para cada cliente
    df_ordenado = df_ordenado.with_columns([pl.when(pl.col('y_true') == 1).then(GANANCIA_ACIERTO).otherwise(-COSTO_ESTIMULO).alias('ganancia_individual')])

    # Calcular ganancia acumulada
    df_ordenado = df_ordenado.with_columns([pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada')])

    # Encontrar la ganancia máxima
    ganancia_maxima = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()

    return 'ganancia', ganancia_maxima, True
```

### **Ventajas de la función de evaluación personalizada**

- **Integración nativa**: Se usa directamente en `lgb.cv()` y `lgb.train()`
- **Optimización en tiempo real**: El modelo se optimiza para maximizar ganancia directamente
- **Eficiencia con Polars**: Procesamiento rápido de grandes volúmenes de datos
- **Sin umbrales fijos**: Encuentra automáticamente el punto óptimo de corte
- **Reproducible**: Resultados consistentes entre ejecuciones

---

## 3) Generación de gráficoS

Los gráficos son fundamentales para entender el comportamiento del modelo y comunicar resultados. Basándonos en `grafico_test.py`, implementamos métodos especializados para visualización.

### **Métodos de visualización**

```python
# src/grafico_test.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime
from .config import STUDY_NAME, GANANCIA_ACIERTO, COSTO_ESTIMULO
from .gain_function_polars import calcular_ganancia_maxima_polars

logger = logging.getLogger(__name__)

def calcular_ganancia_acumulada_optimizada(y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple:
    """
    Calcula la ganancia acumulada ordenando las predicciones de mayor a menor probabilidad.
    Versión optimizada para grandes datasets.

    Args:
        y_true: Valores verdaderos (0 o 1)
        y_pred_proba: Probabilidades predichas

    Returns:
        tuple: (ganancias_acumuladas, indices_ordenados, umbral_optimo)
    """
    logger.info("Calculando ganancia acumulada optimizada...")

    # Ordenar por probabilidad descendente
    indices_ordenados = np.argsort(y_pred_proba)[::-1]
    y_true_ordenado = y_true[indices_ordenados]
    y_pred_proba_ordenado = y_pred_proba[indices_ordenados]

    # Calcular ganancia acumulada vectorizada
    ganancias_individuales = np.where(y_true_ordenado == 1, GANANCIA_ACIERTO, -COSTO_ESTIMULO)
    ganancias_acumuladas = np.cumsum(ganancias_individuales)

    # Encontrar el punto de ganancia máxima
    indice_maximo = np.argmax(ganancias_acumuladas)
    umbral_optimo = y_pred_proba_ordenado[indice_maximo]

    logger.info(f"Ganancia máxima: {ganancias_acumuladas[indice_maximo]:,.0f} en posición {indice_maximo}")
    logger.info(f"Umbral óptimo: {umbral_optimo:.6f}")

    return ganancias_acumuladas, indices_ordenados, umbral_optimo

def crear_grafico_ganancia_avanzado(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   titulo_personalizado: str = None) -> str:
    """
    Crea un gráfico avanzado de ganancia acumulada con múltiples elementos informativos.

    Args:
        y_true: Valores verdaderos
        y_pred_proba: Probabilidades predichas
        titulo_personalizado: Título personalizado para el gráfico

    Returns:
        str: Ruta del archivo del gráfico guardado
    """
    logger.info("Generando gráfico de ganancia avanzado...")

    # Calcular ganancia acumulada
    ganancias_acumuladas, indices_ordenados, umbral_optimo = calcular_ganancia_acumulada_optimizada(y_true, y_pred_proba)

    # Encontrar estadísticas clave
    ganancia_maxima = np.max(ganancias_acumuladas)
    indice_maximo = np.argmax(ganancias_acumuladas)

    # Calcular puntos de referencia
    umbral_025 = 0.025
    clientes_sobre_025 = np.sum(y_pred_proba >= umbral_025)

    # Filtrar datos para visualización (solo mostrar región relevante)
    umbral_ganancia = ganancia_maxima * 0.6  # Mostrar desde 60% de la ganancia máxima
    indices_relevantes = ganancias_acumuladas >= umbral_ganancia
    x_relevante = np.where(indices_relevantes)[0]
    y_relevante = ganancias_acumuladas[indices_relevantes]

    # Configurar estilo del gráfico
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Gráfico principal: Ganancia acumulada
    ax1.plot(x_relevante, y_relevante, color='blue', linewidth=3, label='Ganancia Acumulada', alpha=0.8)

    # Marcar ganancia máxima
    ax1.scatter(indice_maximo, ganancia_maxima, color='red', s=150, zorder=5,
               label=f'Ganancia Máxima: {ganancia_maxima:,.0f}')

    # Líneas de referencia
    ax1.axvline(x=indice_maximo, color='red', linestyle='--', alpha=0.7,
               label=f'Corte Óptimo (cliente {indice_maximo:,})')
    ax1.axvline(x=clientes_sobre_025, color='purple', linestyle='-.', alpha=0.8, linewidth=2,
               label=f'Umbral 0.025 (cliente {clientes_sobre_025:,})')

    # Anotación de ganancia máxima
    ax1.annotate(f'Máximo: {ganancia_maxima:,.0f}\nUmbral: {umbral_optimo:.4f}',
                xy=(indice_maximo, ganancia_maxima),
                xytext=(indice_maximo + len(x_relevante) * 0.15, ganancia_maxima * 1.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='red'))

    # Configurar primer gráfico
    ax1.set_xlabel('Clientes ordenados por probabilidad', fontsize=12)
    ax1.set_ylabel('Ganancia Acumulada', fontsize=12)
    titulo = titulo_personalizado or f'Ganancia Acumulada Optimizada - {STUDY_NAME}'
    ax1.set_title(titulo, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Segundo gráfico: Distribución de probabilidades
    ax2.hist(y_pred_proba, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax2.axvline(x=umbral_optimo, color='red', linestyle='--', linewidth=2,
               label=f'Umbral Óptimo: {umbral_optimo:.4f}')
    ax2.axvline(x=umbral_025, color='purple', linestyle='-.', linewidth=2,
               label=f'Umbral 0.025')

    ax2.set_xlabel('Probabilidad Predicha', fontsize=12)
    ax2.set_ylabel('Densidad', fontsize=12)
    ax2.set_title('Distribución de Probabilidades Predichas', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Ajustar layout
    plt.tight_layout()

    # Guardar gráfico con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("resultados", exist_ok=True)
    ruta_archivo = f"resultados/{STUDY_NAME}_ganancia_avanzado_{timestamp}.png"

    plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # Guardar datos del gráfico en CSV
    ruta_datos = f"resultados/{STUDY_NAME}_datos_ganancia_{timestamp}.csv"
    df_datos = pd.DataFrame({
        'posicion': range(len(ganancias_acumuladas)),
        'ganancia_acumulada': ganancias_acumuladas,
        'probabilidad_ordenada': y_pred_proba[indices_ordenados]
    })
    df_datos.to_csv(ruta_datos, index=False)

    logger.info(f"Gráfico avanzado guardado: {ruta_archivo}")
    logger.info(f"Datos guardados: {ruta_datos}")

    return ruta_archivo

def generar_reporte_visual_completo(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   titulo_estudio: str = None) -> dict:
    """
    Genera un reporte visual completo con todos los gráficos y análisis.

    Args:
        y_true: Valores verdaderos
        y_pred_proba: Probabilidades predichas
        titulo_estudio: Título personalizado para el estudio

    Returns:
        dict: Rutas de todos los archivos generados y estadísticas
    """
    logger.info("=== GENERANDO REPORTE VISUAL COMPLETO ===")

    titulo = titulo_estudio or f"Análisis Completo - {STUDY_NAME}"

    # 1. Gráfico de ganancia avanzado
    ruta_ganancia = crear_grafico_ganancia_avanzado(y_true, y_pred_proba, titulo)

    # 2. Análisis con Polars para estadísticas precisas
    from .gain_function_polars import analisis_ganancia_completo_polars
    analisis_polars = analisis_ganancia_completo_polars(y_true, y_pred_proba)

    # 3. Guardar resumen del reporte
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_resumen = f"resultados/{STUDY_NAME}_reporte_resumen_{timestamp}.json"

    reporte_completo = {
        'metadata': {
            'timestamp': timestamp,
            'titulo_estudio': titulo,
            'total_clientes': len(y_true),
            'distribucion_target': {
                'positivos': int(np.sum(y_true == 1)),
                'negativos': int(np.sum(y_true == 0)),
                'porcentaje_positivos': float(np.mean(y_true) * 100)
            }
        },
        'archivos_generados': {
            'grafico_ganancia': ruta_ganancia,
            'resumen_json': ruta_resumen
        },
        'analisis_polars': analisis_polars,
        'estadisticas_clave': {
            'ganancia_maxima': analisis_polars['ganancia_maxima']['ganancia_maxima'],
            'umbral_optimo': analisis_polars['ganancia_maxima']['umbral_optimo'],
            'clientes_optimos': analisis_polars['ganancia_maxima']['clientes_seleccionados'],
            'mejora_vs_025': analisis_polars['resumen']['mejora_vs_025']
        }
    }

    # Guardar resumen en JSON
    import json
    with open(ruta_resumen, 'w') as f:
        json.dump(reporte_completo, f, indent=2, default=str)

    logger.info("=== REPORTE VISUAL COMPLETADO ===")
    logger.info(f"Archivos generados:")
    logger.info(f"  - Gráfico ganancia: {ruta_ganancia}")
    logger.info(f"  - Resumen JSON: {ruta_resumen}")
    logger.info(f"Ganancia máxima encontrada: {reporte_completo['estadisticas_clave']['ganancia_maxima']:,.0f}")

    return reporte_completo
```

---

## 4) Pipeline completo integrado

Creamos un script principal que orquesta todos los nuevos métodos avanzados: Cross Validation, ganancia con Polars y visualización automatizada.

### **Estructura del pipeline avanzado**

```python
import pandas as pd
import os
import datetime
import logging

from src.loader import cargar_datos, convertir_clase_ternaria_a_target
from src.features import feature_engineering_lag
from src.optimization import optimizar, evaluar_en_test, guardar_resultados_test
from src.optimizacion_cv import optimizar_con_cv
from src.best_params import cargar_mejores_hiperparametros
from src.final_training import preparar_datos_entrenamiento_final, generar_predicciones_finales, entrenar_modelo_final
from src.output_manager import guardar_predicciones_finales
from src.config import *
from src.grafico_test import generar_grafico_test_completo

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
    study = optimizar_con_cv(df_fe, n_trials=10)

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
    resultados_test, y_pred_proba = evaluar_en_test(df_fe, mejores_params)

    # Guardar resultados de test
    guardar_resultados_test(resultados_test)

    # Resumen de evaluación en test
    logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    logger.info(f"✅ Ganancia en test: {resultados_test['ganancia_test']:,.0f}")
    logger.info(f"🎯 Predicciones positivas: {resultados_test['predicciones_positivas']:,} ({resultados_test['porcentaje_positivas']:.2f}%)")

    # Grafico de test
    logger.info("=== GRAFICO DE TEST ===")
    ruta_grafico = generar_grafico_test_completo(df_fe)
    logger.info(f"✅ Gráfico generado: {ruta_grafico}")

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
pip install pandas lightgbm optuna polars matplotlib seaborn pyyaml scikit-learn
```

**Estructura final del proyecto:**

```
Project_Wendsday/
├── main_avanzado.py              # Pipeline avanzado integrado ⭐ NUEVO
├── main_final.py                 # Pipeline de Clase04 (entrenamiento final)
├── config.yaml                   # Configuración completa
├── src/
│   ├── optimization_cv.py        # Optimización con Cross Validation ⭐ NUEVO
│   ├── gain_function_polars.py   # Ganancia optimizada con Polars ⭐ NUEVO
│   ├── visualization.py          # Gráficos avanzados ⭐ NUEVO
│   ├── best_params.py            # Cargar mejores hiperparámetros
│   ├── final_training.py         # Entrenamiento final
│   ├── output_manager.py         # Gestión de archivos de salida
│   └── ... (módulos anteriores)
├── resultados/                   # Resultados y gráficos ⭐ AMPLIADO
│   ├── {STUDY_NAME}_cv_iteraciones.json ⭐ NUEVO
│   ├── {STUDY_NAME}_ganancia_avanzado_*.png ⭐ NUEVO
│   ├── {STUDY_NAME}_reporte_resumen_*.json ⭐ NUEVO
│   └── {STUDY_NAME}_datos_ganancia_*.csv ⭐ NUEVO
├── predict/                      # Predicciones finales optimizadas
└── logs/                         # Logs detallados con timestamps
```

---

## ✅ Resultado esperado

Al final de la clase cada alumno tendrá:

* **Optimización robusta** con Cross Validation en LightGBM
* **Cálculo de ganancia optimizado** usando Polars para máximo rendimiento
* **Visualización automatizada** con gráficos profesionales y análisis visual
* **Pipeline completamente integrado** que combina todas las técnicas avanzadas
* **Umbral óptimo automático** sin necesidad de hardcodear valores
* **Análisis comparativo** entre métodos tradicionales y optimizados
* **Exportación completa** de resultados, gráficos y datos para análisis posterior
