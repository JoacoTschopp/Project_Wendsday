import numpy as np
import pandas as pd
from .config import GANANCIA_ACIERTO, COSTO_ESTIMULO
import logging
import polars as pl

logger = logging.getLogger(__name__)

def calcular_ganancia(y_pred, y_true):
    """
    Calcula la ganancia máxima acumulada ordenando las predicciones de mayor a menor.

    Args:
        y_true: Valores reales (0 o 1)
        y_pred: Predicciones (probabilidades o scores continuos)

    Returns:
        tuple[float, np.ndarray]: Ganancia máxima acumulada y la serie acumulada completa.
    """
    def _to_polars_series(values, name: str, dtype: pl.DataType | None = None) -> pl.Series:
        # Convertir a lista primero para evitar dtype 'object'
        if isinstance(values, pl.Series):
            series = pl.Series(name, values.to_list())
        elif isinstance(values, pd.Series):
            series = pl.Series(name, values.to_list())
        else:
            # Convertir iterable a lista si no lo es ya
            if not isinstance(values, (list, tuple)):
                values = list(values)
            series = pl.Series(name, values)

        if dtype is not None:
            try:
                series = series.cast(dtype, strict=False)
            except pl.ComputeError:
                # Fallback si no puede castear (ej. Object a Int32)
                series = series.cast(pl.Float64, strict=False)
        return series

    y_true_series = _to_polars_series(y_true, 'y_true', dtype=pl.Float64)
    y_pred_series = _to_polars_series(y_pred, 'y_pred_proba', dtype=pl.Float64)

    if y_true_series.is_empty() or y_pred_series.is_empty():
        logger.debug("Ganancia calculada: 0 (datasets vacíos)")
        return 0.0, np.array([], dtype=float)

    if y_true_series.len() != y_pred_series.len():
        raise ValueError("y_true y y_pred deben tener la misma longitud")

    acumulado_df = (
        pl.DataFrame({
            'y_true': y_true_series,
            'y_pred_proba': y_pred_series
        })
        .sort('y_pred_proba', descending=True)
        .with_columns([
            # Calcular ganancia individual para cada registro (forzar i64 para evitar overflow)
            pl.when(pl.col('y_true').round(0) == 1.0)
              .then(pl.lit(GANANCIA_ACIERTO, dtype=pl.Int64))
              .otherwise(pl.lit(-COSTO_ESTIMULO, dtype=pl.Int64))
              .alias('ganancia_individual')
        ])
        .with_columns([
            # Ganancia acumulada: suma acumulativa de ganancias individuales
            pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada')
        ])
    )
    
    ganancia_acumulada_series = acumulado_df['ganancia_acumulada']
    ganancia_total = ganancia_acumulada_series.max()
    # Evitar overflow: si supera int32, devolver como float
    if ganancia_total > 2_147_483_647:
        ganancia_total = float(ganancia_total)
    ganancias_acumuladas = ganancia_acumulada_series.to_numpy()

    logger.debug(f"Ganancia calculada: {ganancia_total:,.0f} "
                f"(GANANCIA_ACIERTO={GANANCIA_ACIERTO}, COSTO_ESTIMULO={COSTO_ESTIMULO})")

    return ganancia_total, ganancias_acumuladas

def ganancia_lgb_binary(y_pred, y_true):
    """
    Función de ganancia para LightGBM en clasificación binaria.
    Compatible con callbacks de LightGBM.
  
    Args:
        y_pred: Predicciones de probabilidad del modelo
        y_true: Dataset de LightGBM con labels verdaderos
  
    Returns:
        tuple: (eval_name, eval_result, is_higher_better)
    """
    # Obtener labels verdaderos
    y_true_labels = y_true.get_label()
  
    # Convertir probabilidades a predicciones binarias (umbral 0.5)
    y_pred_binary = (y_pred > 0.025).astype(int)
  
    # Calcular ganancia usando configuración
    ganancia_total, _ = calcular_ganancia(y_pred=y_pred_binary, y_true=y_true_labels)

    # Retornar en formato esperado por LightGBM
    return 'ganancia', ganancia_total, True  # True = higher is better

# Función de evaluación personalizada para ganancia con ordenamiento
def ganancia_evaluator(y_pred, y_true) -> tuple:
    """
    Función de evaluación personalizada para LightGBM usando Polars.
    Ordena probabilidades de mayor a menor y calcula ganancia acumulada
    para encontrar el punto de máxima ganancia.
    
    Args:
        y_pred: Predicciones de probabilidad del modelo
        y_true: Dataset de LightGBM con labels verdaderos
    
    Returns:
        tuple: (eval_name, ganancia_maxima, is_higher_better)
    """
    y_true_labels = y_true.get_label()
    
    # Pipeline optimizado con Polars: crear DataFrame, ordenar y calcular en una sola cadena
    ganancia_maxima = (
        pl.DataFrame({
            'y_true': y_true_labels,
            'y_pred_proba': y_pred
        })
        .sort('y_pred_proba', descending=True)
        .with_columns([
            # Calcular ganancia individual y acumulada en una sola operación
            pl.when(pl.col('y_true') == 1)
              .then(pl.lit(GANANCIA_ACIERTO))
              .otherwise(pl.lit(-COSTO_ESTIMULO))
              .cum_sum()
              .alias('ganancia_acumulada')
        ])
        .select(pl.col('ganancia_acumulada').max())
        .item()
    )
        
    return 'ganancia', ganancia_maxima, True

