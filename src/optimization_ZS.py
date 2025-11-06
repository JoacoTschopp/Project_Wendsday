import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from flaml.default import preprocess_and_suggest_hyperparams

from .config import *
from .loader import convertir_clase_ternaria_a_target

logger = logging.getLogger(__name__)


def _resolve_seed() -> int:
    if isinstance(SEMILLA, list):
        return SEMILLA[0]
    return int(SEMILLA)


def _split_train_validation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(MES_TRAIN, list):
        df_train = df[df["foto_mes"].isin(MES_TRAIN)].copy()
    else:
        df_train = df[df["foto_mes"] == MES_TRAIN].copy()

    df_val = df[df["foto_mes"] == MES_VALIDACION].copy()

    df_train = convertir_clase_ternaria_a_target(df_train, baja_2_1=True)
    df_val = convertir_clase_ternaria_a_target(df_val, baja_2_1=False)

    df_train["clase_ternaria"] = df_train["clase_ternaria"].astype(np.int8)
    df_val["clase_ternaria"] = df_val["clase_ternaria"].astype(np.int8)

    return df_train, df_val


def _prepare_matrices(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_subset: Optional[Any] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    if feature_subset is not None:
        X_train = df_train[feature_subset].copy()
        X_val = df_val[feature_subset].copy()
    else:
        X_train = df_train.drop(columns=["clase_ternaria"])
        X_val = df_val.drop(columns=["clase_ternaria"])

    y_train = df_train["clase_ternaria"].to_numpy(dtype=np.int8)
    y_val = df_val["clase_ternaria"].to_numpy(dtype=np.int8)

    return X_train, y_train, X_val, y_val


def _calcular_ganancia_desde_probabilidades(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 1]

    orden = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[orden]
    y_pred_sorted = y_pred[orden]

    ganancias_unitarias = np.where(y_true_sorted == 1, GANANCIA_ACIERTO, -COSTO_ESTIMULO)
    ganancia_acumulada = np.cumsum(ganancias_unitarias)

    ganancia_maxima = float(np.max(ganancia_acumulada)) if ganancia_acumulada.size else 0.0
    idx_max = int(np.argmax(ganancia_acumulada)) if ganancia_acumulada.size else 0
    umbral_sugerido = float(y_pred_sorted[idx_max]) if ganancia_acumulada.size else 0.5

    return ganancia_maxima, umbral_sugerido


def preparar_datos_zero_shot(
    df: pd.DataFrame,
    feature_subset: Optional[Any] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    df_train, df_val = _split_train_validation(df)
    return _prepare_matrices(df_train, df_val, feature_subset=feature_subset)


def _sugerir_y_entrenar_con_flaml(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    (
        hyperparams,
        estimator_class,
        X_train_transformed,
        y_train_transformed,
        feature_transformer,
        label_transformer,
    ) = preprocess_and_suggest_hyperparams("classification", X_train, y_train, "lgbm")

    estimator_kwargs = dict(hyperparams)
    estimator_kwargs.setdefault("random_state", _resolve_seed())
    estimator_kwargs.setdefault("n_jobs", -1)

    modelo = estimator_class(**estimator_kwargs)
    modelo.fit(X_train_transformed, y_train_transformed)

    if feature_transformer is not None:
        X_val_transformed = feature_transformer.transform(X_val)
    else:
        X_val_transformed = X_val

    if hasattr(modelo, "predict_proba"):
        proba_val = modelo.predict_proba(X_val_transformed)[:, 1]
    else:
        # Fallback a decision_function si no hay predict_proba
        proba_val = modelo.predict(X_val_transformed)

    if label_transformer is not None:
        y_val_transformed = label_transformer.transform(y_val)
    else:
        y_val_transformed = y_val

    return hyperparams, modelo.get_params(), np.asarray(proba_val), np.asarray(y_val_transformed)


def _construir_parametros_lightgbm(
    hyperparams: Dict[str, Any],
    modelo_params: Dict[str, Any],
) -> Dict[str, Any]:
    rename_map = {
        "n_estimators": "num_iterations",
        "subsample": "bagging_fraction",
        "colsample_bytree": "feature_fraction",
        "min_child_samples": "min_data_in_leaf",
        "n_jobs": "num_threads",
        "random_state": "seed",
    }

    combinados = {**modelo_params, **hyperparams}
    resultado: Dict[str, Any] = {}

    for clave, valor in combinados.items():
        nueva_clave = rename_map.get(clave, clave)
        resultado[nueva_clave] = valor

    resultado["objective"] = "binary"
    resultado["metric"] = "None"
    resultado["verbose"] = -1
    resultado["verbosity"] = -1
    resultado["seed"] = int(resultado.get("seed", _resolve_seed()))

    if "bagging_fraction" not in resultado and "subsample" in combinados:
        resultado["bagging_fraction"] = combinados["subsample"]
    if "feature_fraction" not in resultado and "colsample_bytree" in combinados:
        resultado["feature_fraction"] = combinados["colsample_bytree"]

    # Remover llaves no soportadas por LightGBM nativo
    for clave in ["n_estimators", "subsample", "colsample_bytree", "n_jobs", "random_state"]:
        resultado.pop(clave, None)

    return resultado


def _persistir_resultados(
    archivo_base: str,
    params_flaml: Dict[str, Any],
    params_lightgbm: Dict[str, Any],
    ganancia_validacion: float,
    umbral_sugerido: float,
    proba_val: np.ndarray,
) -> Dict[str, str]:
    os.makedirs("resultados", exist_ok=True)

    iter_path = os.path.join("resultados", f"{archivo_base}_zs_iteraciones.json")
    best_path = os.path.join("resultados", f"{archivo_base}_zs_best_params.json")

    # Obtener número de trial (basado en cantidad de registros existentes)
    if os.path.exists(iter_path):
        try:
            with open(iter_path, "r", encoding="utf-8") as f:
                contenido_existente = json.load(f)
            if not isinstance(contenido_existente, list):
                contenido_existente = []
            trial_number = len(contenido_existente)
        except json.JSONDecodeError:
            contenido_existente = []
            trial_number = 0
    else:
        contenido_existente = []
        trial_number = 0

    # Preparar configuración
    configuracion = {
        "semilla": SEMILLA if isinstance(SEMILLA, list) else [SEMILLA],
        "mes_train": MES_TRAIN if isinstance(MES_TRAIN, list) else [MES_TRAIN],
    }

    # Crear registro en el formato solicitado
    registro = {
        "trial_number": trial_number,
        "params": params_lightgbm,
        "value": float(ganancia_validacion),
        "datetime": datetime.now().isoformat(),
        "state": "COMPLETE",
        "configuracion": configuracion,
    }

    contenido_existente.append(registro)

    with open(iter_path, "w", encoding="utf-8") as f:
        json.dump(contenido_existente, f, indent=2)

    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({"params_lightgbm": params_lightgbm, "params_flaml": params_flaml}, f, indent=2)

    return {"iteraciones": iter_path, "best_params": best_path}


def optimizar_zero_shot(
    df: pd.DataFrame,
    feature_subset: Optional[Any] = None,
    archivo_base: Optional[str] = None,
) -> Dict[str, Any]:
    if archivo_base is None:
        archivo_base = STUDY_NAME

    X_train, y_train, X_val, y_val = preparar_datos_zero_shot(df, feature_subset)

    (
        hyperparams,
        modelo_params,
        proba_val,
        y_val_transformed,
    ) = _sugerir_y_entrenar_con_flaml(X_train, y_train, X_val, y_val)

    ganancia_val, umbral_sugerido = _calcular_ganancia_desde_probabilidades(
        y_val_transformed.astype(np.int32),
        np.clip(proba_val, 0.0, 1.0),
    )

    params_lightgbm = _construir_parametros_lightgbm(hyperparams, modelo_params)

    paths = _persistir_resultados(
        archivo_base,
        hyperparams,
        params_lightgbm,
        ganancia_val,
        umbral_sugerido,
        proba_val,
    )

    logger.info(
        "FLAML Zero-Shot - Ganancia VALID=%s | Umbral=%s",
        f"{ganancia_val:,.0f}",
        f"{umbral_sugerido:.4f}",
    )

    return {
        "ganancia_validacion": ganancia_val,
        "umbral_sugerido": umbral_sugerido,
        "best_params_lightgbm": params_lightgbm,
        "best_params_flaml": hyperparams,
        "paths": paths,
    }


def predecir_zero_shot(
    modelo,
    X: pd.DataFrame,
    umbral: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    proba = modelo.predict(X)
    positive_proba = np.clip(proba, 0.0, 1.0)

    if umbral is None:
        umbral = 0.025

    predicciones = (positive_proba > umbral).astype(int)
    return predicciones, positive_proba
