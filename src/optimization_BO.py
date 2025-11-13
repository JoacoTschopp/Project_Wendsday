import json
import logging
import os
from datetime import datetime
from typing import Optional

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd

from .config import (
    BUCKET_NAME,
    MES_TRAIN,
    MES_VALIDACION,
    PARAMETROS_LGB,
    SEMILLA,
    STUDY_NAME,
)
from .gain_function import ganancia_evaluator
from .loader import convertir_clase_ternaria_a_target
from .undersampling import aplicar_undersampling

logger = logging.getLogger(__name__)


def _preparar_datos_entrenamiento(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara datos combinando TRAIN + VALIDACIÓN para CV."""
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]

    df_train = df[df["foto_mes"].isin(periodos_entrenamiento)].copy()
    df_train = convertir_clase_ternaria_a_target(df_train, baja_2_1=True)
    df_train["clase_ternaria"] = df_train["clase_ternaria"].astype(np.int8)
    return df_train


def objetivo_ganancia(
    trial: optuna.trial.Trial, df: pd.DataFrame, undersampling: float = 1.0
) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos

    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON

    Returns:
    float: ganancia total
    """
    # Hiperparámetros a optimizar (desde conf.yaml)
    params = {
        # Parámetros fijos desde configuración
        "objective": PARAMETROS_LGB["objective"],
        "metric": PARAMETROS_LGB["metric"],
        "min_gain_to_split": PARAMETROS_LGB["min_gain_to_split"],
        "verbosity": PARAMETROS_LGB["verbosity"],
        "max_bin": PARAMETROS_LGB["max_bin"],
        "seed": SEMILLA[0],
        # Parámetros a optimizar
        "num_leaves": trial.suggest_int(
            "num_leaves",
            PARAMETROS_LGB["num_leaves"][0],
            PARAMETROS_LGB["num_leaves"][1],
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            PARAMETROS_LGB["learning_rate"][0],
            PARAMETROS_LGB["learning_rate"][1],
            log=True,
        ),
        "feature_fraction": trial.suggest_float(
            "feature_fraction",
            PARAMETROS_LGB["feature_fraction"][0],
            PARAMETROS_LGB["feature_fraction"][1],
        ),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction",
            PARAMETROS_LGB["bagging_fraction"][0],
            PARAMETROS_LGB["bagging_fraction"][1],
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            PARAMETROS_LGB["min_child_samples"][0],
            PARAMETROS_LGB["min_child_samples"][1],
        ),
        "max_depth": trial.suggest_int(
            "max_depth", PARAMETROS_LGB["max_depth"][0], PARAMETROS_LGB["max_depth"][1]
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", PARAMETROS_LGB["reg_alpha"][0], PARAMETROS_LGB["reg_alpha"][1]
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            PARAMETROS_LGB["reg_lambda"][0],
            PARAMETROS_LGB["reg_lambda"][1],
        ),
        "min_data_in_leaf": trial.suggest_int(
            "min_data_in_leaf",
            PARAMETROS_LGB["min_data_in_leaf"][0],
            PARAMETROS_LGB["min_data_in_leaf"][1],
        ),
        "num_iterations": trial.suggest_int(
            "num_iterations",
            PARAMETROS_LGB["num_iterations"][0],
            PARAMETROS_LGB["num_iterations"][1],
        ),
        "bagging_freq": trial.suggest_int(
            "bagging_freq",
            PARAMETROS_LGB["bagging_freq"][0],
            PARAMETROS_LGB["bagging_freq"][1],
        ),
    }

    df_cv = _preparar_datos_entrenamiento(df)

    if undersampling < 1.0:
        df_cv = aplicar_undersampling(df_cv, ratio=undersampling)

    X = df_cv.drop(columns=["clase_ternaria"])
    y = df_cv["clase_ternaria"].values

    dataset = lgb.Dataset(X, label=y)

    seed = SEMILLA[0] if isinstance(SEMILLA, list) else int(SEMILLA)

    cv_results = lgb.cv(
        params,
        dataset,
        nfold=5,
        num_boost_round=2000,
        stratified=True,
        seed=seed,
        feval=ganancia_evaluator,
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(period=0),
        ],
    )

    metric_candidates = [k for k in cv_results.keys() if k.endswith("-mean")]
    if metric_candidates:
        logger.debug(
            "Trial %s: métricas devueltas por lgb.cv: %s",
            trial.number,
            metric_candidates,
        )

    if not metric_candidates:
        logger.warning(
            "Trial %s: lgb.cv no devolvió métricas con sufijo '-mean'. Keys obtenidas: %s",
            trial.number,
            list(cv_results.keys()),
        )
        ganancia_best = 0.0
        best_iteration = 0
    else:
        metric_name = metric_candidates[0]
        ganancias_mean = np.array(cv_results[metric_name])
        best_iteration = int(np.argmax(ganancias_mean)) + 1
        ganancia_best = float(ganancias_mean[best_iteration - 1])

    guardar_iteracion(trial, ganancia_best, params)

    mlflow.log_metric(
        "ganancia_best", ganancia_best, step=trial.number
    )  # loggear la ganancia en mlflow

    logger.info(
        "Trial %s: Ganancia (promedio 5-fold) = %s | best_iteration=%s",
        trial.number,
        f"{ganancia_best:,.0f}",
        best_iteration,
    )

    return ganancia_best


def guardar_iteracion(trial, ganancia, params_completos=None, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.

    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        params_completos: Diccionario con todos los parámetros (optimizados + fijos)
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME

    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_iteraciones.json"

    # Si se proporcionan params_completos, usarlos; sino, usar solo trial.params
    if params_completos is not None:
        # Combinar parámetros optimizados con fijos
        params_a_guardar = params_completos.copy()
    else:
        params_a_guardar = trial.params

    # Datos de esta iteración
    iteracion_data = {
        "trial_number": trial.number,
        "params": params_a_guardar,
        "value": float(ganancia),
        "datetime": datetime.now().isoformat(),
        "state": "COMPLETE",  # Si llegamos aquí, el trial se completó exitosamente
        "configuracion": {
            "semilla": SEMILLA,
            "mes_train": MES_TRAIN,
            #'mes_validacion': MES_VALIDACION
        },
    }

    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, "r") as f:
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

    # Guardar todas las iteraciones en el archivo
    with open(archivo, "w") as f:
        json.dump(datos_existentes, f, indent=2)

    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f} --- Parámetros: {trial.params}")


def crear_o_cargar_estudio(
    study_name: Optional[str] = None, semilla: Optional[int] = None
) -> optuna.Study:
    """
    Crea un nuevo estudio de Optuna o carga uno existente desde SQLite.

    Args:
        study_name: Nombre del estudio (si es None, usa STUDY_NAME del config)
        semilla: Semilla para reproducibilidad

    Returns:
        optuna.Study: Estudio de Optuna (nuevo o cargado)
    """
    study_name = study_name or STUDY_NAME

    if semilla is None:
        semilla = SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA

    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "optuna_db")
    os.makedirs(path_db, exist_ok=True)

    # Ruta completa de la base de datos
    db_file = os.path.join(path_db, f"{study_name}.db")
    storage = f"sqlite:///{db_file}"

    # Verificar si existe un estudio previo
    if os.path.exists(db_file):
        logger.info(f"⚡ Base de datos encontrada: {db_file}")
        logger.info(f"🔄 Cargando estudio existente: {study_name}")

        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            n_trials_previos = len(study.trials)

            logger.info("✅ Estudio cargado exitosamente")
            logger.info(f"📊 Trials previos: {n_trials_previos}")

            if n_trials_previos > 0:
                logger.info(f"🏆 Mejor ganancia hasta ahora: {study.best_value:,.0f}")

            return study

        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar el estudio: {e}")
            logger.info("🆕 Creando nuevo estudio...")
    else:
        logger.info("🆕 No se encontró base de datos previa")
        logger.info(f"📁 Creando nueva base de datos: {db_file}")

    # Crear nuevo estudio
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=semilla),
        load_if_exists=True,
    )

    logger.info(f"✅ Nuevo estudio creado: {study_name}")
    logger.info(f"💾 Storage: {storage}")

    return study


def optimizar(
    df: pd.DataFrame,
    n_trials: int,
    study_name: Optional[str] = None,
    undersampling: float = 0.01,
) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
        undersampling: Undersampling para entrenamiento

    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado.
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = study_name or STUDY_NAME

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(
        f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}"
    )

    # Crear o cargar estudio desde DuckDB
    base_semilla = SEMILLA[0] if isinstance(SEMILLA, list) else int(SEMILLA)

    study = crear_o_cargar_estudio(study_name=study_name, semilla=base_semilla)

    # Calcular cuántos trials faltan
    trials_previos = len(study.trials)
    trials_a_ejecutar = max(0, n_trials - trials_previos)

    if trials_previos > 0:
        logger.info(f"🔄 Retomando desde trial {trials_previos}")
        logger.info(
            f"📝 Trials a ejecutar: {trials_a_ejecutar} (total objetivo: {n_trials})"
        )
    else:
        logger.info(f"🆕 Nueva optimización: {n_trials} trials")

    # Ejecutar optimización
    if trials_a_ejecutar > 0:
        study.optimize(
            lambda trial: objetivo_ganancia(trial, df, undersampling),
            n_trials=trials_a_ejecutar,
        )
        logger.info(f"🏆 Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Mejores parámetros: {study.best_params}")
    else:
        logger.info(f"✅ Ya se completaron {n_trials} trials")

    return study
