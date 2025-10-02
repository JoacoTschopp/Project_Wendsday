import logging
import lightgbm as lgb
import pandas as pd
import numpy as np
import polars as pl
from .config import *
from .gain_function import calcular_ganancia, ganancia_evaluator
from .loader import convertir_clase_ternaria_a_target
from datetime import datetime
import os
import json

logger = logging.getLogger(__name__)


def evaluar_en_test(df: pd.DataFrame, mejores_params: dict, semilla: int = SEMILLA[0]) -> (float, np.ndarray):
    """
        Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
        Solo calcula la ganancia.
    
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
    
    Returns:
        dict: Ganancia total
        np.ndarray: Probabilidades predichas
    """
    
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
    
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
    
    # Convertir clase_ternaria a target binario
    df_train_completo = convertir_clase_ternaria_a_target(df, baja_2_1=True)
    df_test = convertir_clase_ternaria_a_target(df, baja_2_1=False)
    
    df_train_completo = df_train_completo[df_train_completo['foto_mes'].isin(periodos_entrenamiento)]
    df_test = df_test[df_test['foto_mes'] == MES_TEST]
    
     
    # Usar target (clase_ternaria ya convertida a binaria)
    y_train = df_train_completo['clase_ternaria'].values
    y_test = df_test['clase_ternaria'].values
    
    # Features: usar todas las columnas excepto target
    X_train = df_train_completo.drop(columns=['clase_ternaria'])
    X_test = df_test.drop(columns=['clase_ternaria'])
    
    # Entrenar modelo con función de ganancia personalizada
    train_data = lgb.Dataset(X_train, label=y_train)

    mejores_params['verbose'] = -1 
    mejores_params['seed'] = semilla
    logger.info(f"Semilla de entrenamiento: {semilla}")
    # Entrenar modelo con mejores parámetros
    model = lgb.train(
        mejores_params,
        train_data,
        num_boost_round=300,
        feval=ganancia_evaluator,  # Función de ganancia personalizada
        callbacks=[lgb.log_evaluation(0)]
    )
    
    # Predecir y calcular ganancia
    y_pred_proba = model.predict(X_test)

    # Calcular solo la ganancia
    # Convertir a DataFrame de Polars para procesamiento eficiente
    df_eval = pl.DataFrame({'y_true': y_test, 'y_pred_proba': y_pred_proba})
    
    # Ordenar por probabilidad descendente
    df_ordenado = df_eval.sort('y_pred_proba', descending=True)
    
    # Calcular ganancia individual para cada cliente
    df_ordenado = df_ordenado.with_columns([
        pl.when(pl.col('y_true') == 1).then(GANANCIA_ACIERTO).otherwise(-COSTO_ESTIMULO).alias('ganancia_individual')
    ])
    
    # Calcular ganancia acumulada
    df_ordenado = df_ordenado.with_columns([
        pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada')
    ])
    
    # Encontrar la ganancia máxima
    ganancia_test = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()
    
    
    return ganancia_test, y_pred_proba

def guardar_resultados_test(ganancia_test: float, archivo_base=None):
    """
    Atributos:
        ganancia_test: Resultados de la evaluación en test
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    
    Description:    
        Guarda los resultados de la evaluación en test en un archivo JSON.
    """

    if archivo_base is None:
        archivo_base = STUDY_NAME

    # Guarda en resultados/{STUDY_NAME}_test_results.json
    archivo = f"resultados/{archivo_base}_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Datos de esta iteración
    iteracion_data = {
        'Mes_test': MES_TEST,
        'ganancia_test': float(ganancia_test),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
            'semilla': SEMILLA,
            'meses_train': MES_TRAIN + [MES_VALIDACION],
        },
        'resultados': {
            'ganancia_test': float(ganancia_test),
        }
    }
  
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
  
    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)

    # Guardar resultados
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)

    logger.info(f"Resultados guardados en {archivo}")
        
        