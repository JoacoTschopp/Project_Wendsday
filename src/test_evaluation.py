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

from .undersampling import aplicar_undersampling

logger = logging.getLogger(__name__)


def evaluar_en_test(
    df: pd.DataFrame,
    mejores_params: dict,
    undersampling: float = 1.0,
    semilla: int = SEMILLA[0]
) -> (float, np.ndarray):
    """
        Solo calcula la ganancia.
    
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
        undersampling: Ratio para reducir la clase mayoritaria (1.0 desactiva)
        
    Returns:
        dict: Ganancia total
        np.ndarray: Probabilidades predichas
    """

    logger.info(f"Período de test: {MES_TEST}")
    
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
    
    # Crear copias ANTES de las conversiones para evitar modificar el DataFrame original
    df_train_copy = df.copy()

    # Convertir clase_ternaria a target binario
    df_train_completo = convertir_clase_ternaria_a_target(df_train_copy, baja_2_1=False)
    
    # Filtrar por períodos antes de extraer features para optimizar memoria
    df_train = df_train_completo[df_train_completo['foto_mes'].isin(periodos_entrenamiento)]
    df_test = df_train_completo[df_train_completo['foto_mes'] == MES_TEST]

    if undersampling < 1.0:
        logger.info(f"Aplicando undersampling (ratio={undersampling}) en entrenamiento")
        df_train = aplicar_undersampling(df_train, undersampling, random_state=semilla)
    
    # Extraer target antes de eliminar la columna para evitar problemas de memoria
    y_train = df_train['clase_ternaria'].values.astype(np.int32)
    y_test = df_test['clase_ternaria'].values.astype(np.int32)
    
    # Features: crear copias más pequeñas sin la columna target
    # Usar una lista de columnas en lugar de drop para ser más eficiente en memoria
    feature_columns = [col for col in df_train.columns if col != 'clase_ternaria']
    X_train = df_train[feature_columns].copy()
    X_test = df_test[feature_columns].copy()
    
    # Liberar memoria de los DataFrames grandes
    del df_train_completo, df_test
    
    # Entrenar modelo con función de ganancia personalizada
    train_data = lgb.Dataset(X_train, label=y_train)

    params_entrenamiento = dict(mejores_params)
    params_entrenamiento['verbose'] = -1
    params_entrenamiento['seed'] = semilla
    num_boost_round = params_entrenamiento.pop('num_iterations', params_entrenamiento.pop('num_boost_round', 300))

    logger.info(f"Semilla de entrenamiento: {semilla}")
    # Entrenar modelo con mejores parámetros
    model = lgb.train(
        params_entrenamiento,
        train_data,
        num_boost_round=num_boost_round,
        feval=ganancia_evaluator,  # Función de ganancia personalizada
        callbacks=[lgb.log_evaluation(0)]
    )
    
    # Predecir y calcular ganancia
    y_pred_proba = model.predict(X_test)

    # Calcular solo la ganancia
    ganancia_test, _ = calcular_ganancia(y_true=y_test, y_pred=y_pred_proba)
       
    return ganancia_test, y_pred_proba

def guardar_resultados_test(ganancia_test: float, archivo_base=None) -> str:
    """
    Atributos:
        ganancia_test: Resultados de la evaluación en test
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    
    Description:    
        Guarda los resultados de la evaluación en test en un archivo JSON.
        Retorna la ruta del archivo generado para facilitar su registro externo (MLflow, etc.).
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
    return archivo