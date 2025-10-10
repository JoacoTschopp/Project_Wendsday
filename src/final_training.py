import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
from .config import FINAL_TRAIN, FINAL_PREDIC, SEMILLA
from .optimization_cv import aplicar_undersampling
from .best_params import cargar_mejores_hiperparametros
from .gain_function import ganancia_lgb_binary, ganancia_evaluator
from .loader import convertir_clase_ternaria_a_target

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
    
    # Preparar datos de entrenamiento (TRAIN + VALIDACION + TEST)
    df_train_final = df[df['foto_mes'].isin(FINAL_TRAIN)]
    df_predic_final = df[df['foto_mes'] == FINAL_PREDIC]
 
    # Usar target (clase_ternaria ya convertida a binaria)
    y_train = df_train_final['clase_ternaria'].values
        
    # Features: usar todas las columnas excepto target
    X_train = df_train_final.drop(columns=['clase_ternaria'])
    X_predic = df_predic_final.drop(columns=['clase_ternaria'])
    
    # Clientes para predicción 
    clientes_predic = df_predic_final['numero_de_cliente'].values
    
    logger.info(f"Datos preparados para entrenamiento final")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"X_predic shape: {X_predic.shape}")
    logger.info(f"clientes_predic shape: {clientes_predic.shape}")
    
    return X_train, y_train, X_predic, clientes_predic

def entrenar_modelo_final(
    df: pd.DataFrame,
    mejores_params: dict,
    undersampling: float = 1.0,
    semilla: int = SEMILLA[0]
) -> lgb.Booster:
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
    
    # Preparar datos de entrenamiento (TRAIN + VALIDACION + TEST)
    df_train_final = df[df['foto_mes'].isin(FINAL_TRAIN)]
    df_train_final = convertir_clase_ternaria_a_target(df_train_final, baja_2_1=True)

    if undersampling < 1.0:
        logger.info(f"Aplicando undersampling (ratio={undersampling}) en entrenamiento final")
        df_train_final = aplicar_undersampling(df_train_final, undersampling, random_state=semilla)

    # Usar target (clase_ternaria ya convertida a binaria)
    y_train = df_train_final['clase_ternaria'].values
        
    # Features: usar todas las columnas excepto target
    X_train = df_train_final.drop(columns=['clase_ternaria'])

    # Configurar parámetros del modelo
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'random_state': semilla,
        'verbose': -1,
        **mejores_params  # Agregar los mejores hiperparámetros
    }

    num_boost_round = params.pop('num_iterations', params.pop('num_boost_round', 300))
    
    # Crear dataset de LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Entrenar modelo
    logger.info("Entrenando modelo...")
    modelo = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.log_evaluation(period=100)
        ],
        feval=ganancia_evaluator
    )
    
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
    
    # Generar predicciones    
    predicciones = modelo.predict(X_predict)

    # Binarizar la probabilidad segun el umbral
    predicciones_binary = (predicciones > umbral).astype(int)
    
    # Crear DataFrame de resultados
    resultados = pd.DataFrame({
        'numero_cliente': clientes_predict,
        'predict': predicciones_binary
    })
    
    # Calcular estadísticas
    total_predicciones = len(predicciones_binary)
    predicciones_positivas = np.sum(predicciones_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100

    logger.info(f"Predicciones generadas:")
    logger.info(f"  Total clientes: {total_predicciones:,}")
    logger.info(f"  Predicciones positivas: {predicciones_positivas:,} ({porcentaje_positivas:.2f}%)")
    logger.info(f"  Predicciones negativas: {total_predicciones - predicciones_positivas:,}")
    logger.info(f"  Umbral utilizado: {umbral}")
    
    return resultados
