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

from .config import *
from .gain_function import ganancia_evaluator
from .loader import convertir_clase_ternaria_a_target

logger = logging.getLogger(__name__)


def _obtener_rango_parametro(nombre: str) -> tuple:
    valores = PARAMETROS_LGB.get(nombre)
    if valores is None:
        raise KeyError(f"No se encontró configuración para el parámetro '{nombre}'")

    if isinstance(valores, (int, float)):
        return valores, valores

    if isinstance(valores, (list, tuple)):
        if len(valores) == 0:
            raise ValueError(f"La configuración de '{nombre}' está vacía")
        if len(valores) == 1:
            return valores[0], valores[0]
        return valores[0], valores[1]

    raise TypeError(f"Tipo de configuración no soportado para '{nombre}': {type(valores)}")


def aplicar_undersampling(df: pd.DataFrame, ratio: float, random_state: int) -> pd.DataFrame:
    """Aplica undersampling controlado sobre la clase mayoritaria."""
    if ratio <= 0:
        raise ValueError("El ratio de undersampling debe ser mayor que 0")

    valores_clase = df['clase_ternaria'].to_numpy(copy=False)
    valores_clientes = df['numero_de_cliente'].to_numpy(copy=False)

    mask_mayoritaria = valores_clase == 0
    mask_minoritaria = valores_clase == 1

    if not mask_mayoritaria.any() or not mask_minoritaria.any():
        logger.warning("No se puede aplicar undersampling: una de las clases está vacía")
        return df

    clientes_mayoritaria = np.unique(valores_clientes[mask_mayoritaria])
    clientes_minoritaria = np.unique(valores_clientes[mask_minoritaria])

    muestra_clientes = int(len(clientes_mayoritaria) * ratio)
    muestra_clientes = max(muestra_clientes, len(clientes_minoritaria))
    muestra_clientes = min(muestra_clientes, len(clientes_mayoritaria))

    rng = np.random.default_rng(random_state)
    clientes_sampleados = rng.choice(clientes_mayoritaria, size=muestra_clientes, replace=False)

    mascara_clientes_seleccionados = np.isin(valores_clientes, clientes_sampleados)
    mask_mayoritaria_seleccionada = mask_mayoritaria & mascara_clientes_seleccionados
    mask_final = mask_minoritaria | mask_mayoritaria_seleccionada

    indices_finales = np.flatnonzero(mask_final)
    df_sampleado = df.take(indices_finales).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    clientes_retenidos = len(clientes_sampleados)

    logger.debug(
        "Undersampling aplicado - clientes clase 0: %d, clientes clase 1: %d",
        clientes_retenidos,
        len(clientes_minoritaria)
    )

    return df_sampleado


def _aplicar_undersampling(df: pd.DataFrame, ratio: float, trial_number: int) -> pd.DataFrame:
    base_seed = SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA
    random_state = base_seed + trial_number
    return aplicar_undersampling(df, ratio, random_state=random_state)


def objetivo_ganancia_cv(trial, df, undersampling: float = 1.0) -> float:
    """Evaluación de la ganancia promedio de Cross Validation.
  
    Args:
        trial: Trial de Optuna
        semilla: Semilla para reproducibilidad
  
    Returns:
        float: Ganancia promedio de Cross Validation
    """
    num_leaves_min, num_leaves_max = _obtener_rango_parametro('num_leaves')
    learning_rate_min, learning_rate_max = _obtener_rango_parametro('learning_rate')
    feature_fraction_min, feature_fraction_max = _obtener_rango_parametro('feature_fraction')
    bagging_fraction_min, bagging_fraction_max = _obtener_rango_parametro('bagging_fraction')
    min_child_samples_min, min_child_samples_max = _obtener_rango_parametro('min_child_samples')
    max_depth_min, max_depth_max = _obtener_rango_parametro('max_depth')
    reg_alpha_min, reg_alpha_max = _obtener_rango_parametro('reg_alpha')
    reg_lambda_min, reg_lambda_max = _obtener_rango_parametro('reg_lambda')
    bin_min, bin_max = _obtener_rango_parametro('bin')
    min_data_in_leaf_min, min_data_in_leaf_max = _obtener_rango_parametro('min_data_in_leaf')
    num_iterations_min, num_iterations_max = _obtener_rango_parametro('num_iterations')

    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'num_leaves': trial.suggest_int('num_leaves', num_leaves_min, num_leaves_max),
        'learning_rate': trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', feature_fraction_min, feature_fraction_max),
        'bagging_fraction': trial.suggest_float('bagging_fraction', bagging_fraction_min, bagging_fraction_max),
        'min_child_samples': trial.suggest_int('min_child_samples', min_child_samples_min, min_child_samples_max),
        'max_depth': trial.suggest_int('max_depth', max_depth_min, max_depth_max),
        'reg_alpha': trial.suggest_float('reg_alpha', reg_alpha_min, reg_alpha_max),
        'reg_lambda': trial.suggest_float('reg_lambda', reg_lambda_min, reg_lambda_max),
        'bin': trial.suggest_int('bin', bin_min, bin_max),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', min_data_in_leaf_min, min_data_in_leaf_max),
        'num_iterations': trial.suggest_int('num_iterations', num_iterations_min, num_iterations_max),
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