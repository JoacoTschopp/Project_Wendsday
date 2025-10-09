# src/optimization_cv.py
import optuna
import lightgbm as lgb
import pandas as pd
import polars as pl
import numpy as np
import logging
from datetime import datetime
import json
import os

def _aplicar_undersampling(df: pd.DataFrame, ratio: float, trial_number: int) -> pd.DataFrame:
    if ratio <= 0:
        raise ValueError("El ratio de undersampling debe ser mayor que 0")

    clase_mayoritaria = df[df['clase_ternaria'] == 0]
    clase_minoritaria = df[df['clase_ternaria'] == 1]

    if clase_mayoritaria.empty or clase_minoritaria.empty:
        logger.warning("No se puede aplicar undersampling: una de las clases está vacía")
        return df

    muestra_mayoritaria = int(len(clase_mayoritaria) * ratio)
    muestra_mayoritaria = max(muestra_mayoritaria, len(clase_minoritaria))
    muestra_mayoritaria = min(muestra_mayoritaria, len(clase_mayoritaria))

    mayoritaria_sampleada = clase_mayoritaria.sample(
        n=muestra_mayoritaria,
        random_state=SEMILLA[0] + trial_number,
        replace=False
    )

    df_sampleado = pd.concat([mayoritaria_sampleada, clase_minoritaria], axis=0)
    df_sampleado = df_sampleado.sample(frac=1.0, random_state=SEMILLA[0] + trial_number).reset_index(drop=True)

    logger.debug(
        f"Trial {trial_number}: undersampling aplicado - clase 0: {len(mayoritaria_sampleada)}, clase 1: {len(clase_minoritaria)}"
    )

    return df_sampleado


def objetivo_ganancia_cv(trial, df, undersampling: float = 1.0) -> float:
    """
    Función objetivo con Cross Validation que maximiza ganancia promedio.
    Utiliza lgb.cv() con estratificación interna y métrica personalizada.
  
    Args:
        trial: Trial de Optuna
        df: DataFrame con datos de entrenamiento y validación
        semilla: Semilla para reproducibilidad
  
    Returns:
        float: Ganancia promedio de Cross Validation
    """
    # Hiperparámetros a optimizar
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
  
    # Preparar datos para CV (combinando TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_cv = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_cv = [MES_TRAIN, MES_VALIDACION]

    df_cv = df[df['foto_mes'].isin(periodos_cv)].copy()
    df_cv = convertir_clase_ternaria_a_target(df_cv, baja_2_1=True)
    df_cv['clase_ternaria'] = df_cv['clase_ternaria'].astype(np.int8)

    if undersampling < 1.0:
        df_cv = _aplicar_undersampling(df_cv, undersampling, trial.number)

    # Preparar features y target
    features_cols = [col for col in df_cv.columns if col != 'clase_ternaria']
    X = df_cv[features_cols]
    y = df_cv['clase_ternaria']
  
    logger.debug(f"Trial {trial.number}: CV con {len(df_cv)} registros, {len(features_cols)} features")
  
    # Crear dataset de LightGBM
    dataset = lgb.Dataset(X, label=y)
  
    # Ejecutar Cross Validation con lgb.cv()
    cv_results = lgb.cv(
        params,
        dataset,
        num_boost_round=1000,
        nfold=5,  # 5-fold CV
        stratified=True,  # Estratificación automática
        shuffle=True,
        seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        feval=ganancia_evaluator,  # Métrica personalizada
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
  
    # Extraer resultados de CV
    ganancias_cv = cv_results['valid ganancia-mean']
    ganancia_maxima = np.max(ganancias_cv)
    ganancia_std = np.std(ganancias_cv)
    
    # Obtener el mejor número de iteraciones
    best_iteration = len(ganancias_cv) - 1  # early_stopping ya seleccionó el mejor
    
    logger.debug(f"Trial {trial.number}: Ganancia CV = {ganancia_maxima:,.0f} ± {ganancia_std:,.0f}")
    logger.debug(f"Trial {trial.number}: Mejor iteración = {best_iteration}")
  
    # Guardar iteración con información de CV
    guardar_iteracion_cv(trial, ganancia_maxima)
  
    return ganancia_maxima

def guardar_iteracion_cv(trial, ganancia_maxima, archivo_base=None):
    """
    Guarda cada iteración de CV en archivo JSON con información detallada.
  
    Args:
        trial: Trial de Optuna
        ganancia_promedio: Ganancia promedio de CV
        ganancias_cv: Lista de ganancias por fold
        ganancia_std: Desviación estándar de ganancias
        archivo_base: Nombre base del archivo
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
  
    archivo = f"resultados/{archivo_base}_iteraciones.json"
    os.makedirs("resultados", exist_ok=True)
  
    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia_maxima),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',
    }
  
    # Cargar datos existentes
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
  
    # Guardar archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Iteración CV {trial.number} guardada - Ganancia: {ganancia_maxima:,.0f}")

def optimizar_con_cv(df, n_trials=50, undersampling: float = 1.0) -> optuna.Study:
    """
    Ejecuta optimización bayesiana con Cross Validation.
  
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
  
    Returns:
        optuna.Study: Estudio de Optuna con resultados de CV
    """
    study_name = STUDY_NAME
  
    logger.info(f"Iniciando optimización con CV - {n_trials} trials")
    logger.info(f"Configuración CV: períodos={MES_TRAIN + [MES_VALIDACION] if isinstance(MES_TRAIN, list) else [MES_TRAIN, MES_VALIDACION]}")
  
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0])
    )
  
    # Ejecutar optimización
    study.optimize(lambda trial: objetivo_ganancia_cv(trial, df, undersampling), n_trials=n_trials)
  
    # Resultados
    logger.info(f"Optimización CV completada:")
    logger.info(f"  Mejor ganancia promedio: {study.best_value:,.0f}")
    logger.info(f"  Mejores parámetros: {study.best_params}")
    logger.info(f"  Total trials: {len(study.trials)}")
  
    return study