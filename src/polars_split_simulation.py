import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedShuffleSplit

from .best_params import cargar_mejores_hiperparametros
from .config import (
    COSTO_ESTIMULO,
    GANANCIA_ACIERTO,
    MES_TEST,
    MES_TRAIN,
    MES_VALIDACION,
    SEMILLA,
    STUDY_NAME,
)

logger = logging.getLogger(__name__)


def _ensure_list(value: Sequence[int] | int) -> List[int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [int(value)]


def _convert_target(df: pl.DataFrame, baja_2_1: bool) -> pl.DataFrame:
    if "clase_ternaria" not in df.columns:
        raise ValueError("La columna 'clase_ternaria' es requerida en el DataFrame")

    if baja_2_1:
        return df.with_columns(
            pl.when(pl.col("clase_ternaria") == "CONTINUA")
            .then(pl.lit(0, dtype=pl.Int8))
            .otherwise(pl.lit(1, dtype=pl.Int8))
            .alias("clase_ternaria")
        )

    return df.with_columns(
        pl.when(pl.col("clase_ternaria") == "BAJA+2")
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("clase_ternaria")
    )


def _generate_seeds(base_seed: int, n_iter: int) -> List[int]:
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(low=1, high=2**31 - 1, size=n_iter, endpoint=True)
    return [int(seed) for seed in seeds]


def _compute_gain(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int8, copy=False)
    y_pred = y_pred.astype(np.int8, copy=False)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    ganancia = tp * GANANCIA_ACIERTO - fp * COSTO_ESTIMULO
    return float(ganancia)


def _prepare_cuts(n_rows: int, cut_start: int, cut_end: int, cut_step: int) -> List[int]:
    cuts = list(range(cut_start, cut_end + 1, cut_step))
    return [cut for cut in cuts if cut <= n_rows]


def _undersample_by_unique_client(df: pl.DataFrame, ratio: float, seed: int) -> pl.DataFrame:
    if ratio >= 1.0:
        return df
    if ratio <= 0:
        raise ValueError("El ratio de undersampling debe ser mayor a 0")
    if "numero_de_cliente" not in df.columns:
        raise ValueError("La columna 'numero_de_cliente' es requerida para el undersampling")

    df_mayoritaria = df.filter(pl.col("clase_ternaria") == 0)
    df_minoritaria = df.filter(pl.col("clase_ternaria") == 1)

    if df_mayoritaria.is_empty() or df_minoritaria.is_empty():
        logger.warning("No se puede aplicar undersampling: alguna de las clases está vacía")
        return df

    clientes_mayoritaria = (
        df_mayoritaria.select("numero_de_cliente").unique().to_series().to_list()
    )
    clientes_minoritaria = (
        df_minoritaria.select("numero_de_cliente").unique().to_series().to_list()
    )

    muestra_clientes = int(len(clientes_mayoritaria) * ratio)
    muestra_clientes = max(muestra_clientes, len(clientes_minoritaria))
    muestra_clientes = min(muestra_clientes, len(clientes_mayoritaria))

    rng = np.random.default_rng(seed)
    clientes_seleccionados = rng.choice(clientes_mayoritaria, size=muestra_clientes, replace=False)

    df_mayoritaria_sample = df_mayoritaria.filter(
        pl.col("numero_de_cliente").is_in(clientes_seleccionados)
    )

    df_resultado = pl.concat(
        [df_mayoritaria_sample, df_minoritaria],
        how="vertical_relaxed",
    )

    df_resultado = df_resultado.sample(fraction=1.0, with_replacement=False, seed=seed)

    logger.info(
        "Undersampling aplicado - clientes clase 0 retenidos: %d, clientes clase 1: %d",
        muestra_clientes,
        len(clientes_minoritaria),
    )

    return df_resultado


def run_polars_split_simulation(
    df_features,
    n_iteraciones: int,
    n_tiradas: int = 100,
    base_seed: int = SEMILLA[0],
    cut_start: int = 8000,
    cut_end: int = 12000,
    cut_step: int = 500,
    output_dir: str = "resultados",
    json_filename: Optional[str] = None,
    undersampling: float =1,
) -> str:
    if df_features is None:
        raise ValueError("El DataFrame de features no puede ser None")

    pl_df = pl.from_pandas(df_features)

    periodos_train = _ensure_list(MES_TRAIN)
    if MES_VALIDACION not in periodos_train:
        periodos_train.append(MES_VALIDACION)

    df_train = pl_df.filter(pl.col("foto_mes").is_in(periodos_train))
    df_test = pl_df.filter(pl.col("foto_mes") == MES_TEST)

    if df_train.is_empty():
        raise ValueError("No se encontraron datos para los períodos de entrenamiento")
    if df_test.is_empty():
        raise ValueError("No se encontraron datos para el período de test")

    df_train = _convert_target(df_train, baja_2_1=True)
    df_test = _convert_target(df_test, baja_2_1=False)

    if undersampling < 1.0:
        logger.info(
            "Aplicando undersampling (ratio=%.3f) basado en clientes únicos con semilla %s",
            undersampling,
            base_seed,
        )
        df_train = _undersample_by_unique_client(df_train, undersampling, base_seed)

    y_train = df_train.get_column("clase_ternaria").to_numpy().astype(np.int8)
    X_train = df_train.drop("clase_ternaria").to_pandas()

    y_test = df_test.get_column("clase_ternaria").to_numpy().astype(np.int8)
    X_test = df_test.drop("clase_ternaria").to_pandas()

    cuts = _prepare_cuts(len(y_test), cut_start, cut_end, cut_step)
    if not cuts:
        raise ValueError("Ningún corte es válido para la cantidad de registros disponibles")

    mejores_params = cargar_mejores_hiperparametros()
    seeds = _generate_seeds(base_seed, n_iteraciones)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = json_filename or f"{STUDY_NAME}_polars_split_{timestamp}.json"
    json_path = os.path.join(output_dir, json_filename)

    resultados_iteraciones: List[Dict] = []

    for iter_idx, semilla in enumerate(seeds, start=1):
        logger.info("Iteración %s/%s - entrenando con semilla %s", iter_idx, n_iteraciones, semilla)

        params = dict(mejores_params)
        params["seed"] = semilla
        params["random_state"] = semilla
        params.setdefault("metric", "None")
        params.setdefault("objective", "binary")
        params.setdefault("verbose", -1)

        num_boost_round = params.pop("num_iterations", params.pop("num_boost_round", 300))

        train_dataset = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=num_boost_round,
            callbacks=[lgb.log_evaluation(0)],
        )

        y_pred_proba = model.predict(X_test)
        order_idx = np.argsort(y_pred_proba)[::-1]

        ganancias_por_corte: Dict[str, float] = {}
        predicciones_por_corte: Dict[int, np.ndarray] = {}

        for corte in cuts:
            mascara_pred = np.zeros(len(y_pred_proba), dtype=np.int8)
            top_k = min(corte, len(order_idx))
            mascara_pred[order_idx[:top_k]] = 1
            predicciones_por_corte[corte] = mascara_pred
            ganancias_por_corte[str(corte)] = _compute_gain(y_test, mascara_pred)

        mejor_corte, mejor_ganancia = max(
            ((int(corte), ganancia) for corte, ganancia in ganancias_por_corte.items()),
            key=lambda x: x[1],
        )

        splitter = StratifiedShuffleSplit(
            n_splits=n_tiradas,
            test_size=0.3,
            train_size=0.7,
            random_state=semilla,
        )

        particiones = []
        baseline = np.zeros(len(y_test), dtype=np.float32)
        for partida_idx, (private_idx, public_idx) in enumerate(
            splitter.split(baseline, y_test),
            start=1,
        ):
            resultados_partida: Dict[str, float] = {}
            for corte in cuts:
                predicciones = predicciones_por_corte[corte]
                ganancia_public = _compute_gain(y_test[public_idx], predicciones[public_idx])
                ganancia_private = _compute_gain(y_test[private_idx], predicciones[private_idx])
                resultados_partida[f"{corte}_30"] = ganancia_public
                resultados_partida[f"{corte}_70"] = ganancia_private

            particiones.append(
                {
                    "partida_numero": partida_idx,
                    "resultados": resultados_partida,
                }
            )

        resultados_iteraciones.append(
            {
                "iteracion": iter_idx,
                "semilla_entrenamiento": int(semilla),
                "ganancia_test_completa": {
                    "mejor_corte": int(mejor_corte),
                    "ganancia": float(mejor_ganancia),
                },
                "ganancias_por_corte": ganancias_por_corte,
                "particiones": particiones,
            }
        )

    contenido = {
        "metadata": {
            "study_name": STUDY_NAME,
            "timestamp": timestamp,
            "periodos_train": periodos_train,
            "periodo_test": MES_TEST,
            "n_iteraciones": n_iteraciones,
            "n_tiradas": n_tiradas,
            "cortes": cuts,
            "base_seed": base_seed,
        },
        "iteraciones": resultados_iteraciones,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(contenido, f, indent=2, ensure_ascii=False)

    logger.info("Resultados guardados en %s", json_path)
    return json_path


def plot_public_private_distributions(
    json_path: str,
    output_dir: str = "resultados",
    public_color: str = "#1f77b4",
    private_color: str = "#ff7f0e",
) -> Dict[str, Optional[str]]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encontró el archivo {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        datos = json.load(f)

    iteraciones = datos.get("iteraciones", [])
    if not iteraciones:
        raise ValueError("El archivo JSON no contiene iteraciones para graficar")

    registros: List[Dict[str, object]] = []

    for iteracion in iteraciones:
        iter_idx = iteracion.get("iteracion")
        for particion in iteracion.get("particiones", []):
            partida_numero = particion.get("partida_numero")
            resultados = particion.get("resultados", {})
            for clave_corte, ganancia in resultados.items():
                try:
                    corte_str, porcentaje = clave_corte.split("_")
                    corte = int(corte_str)
                except ValueError:
                    logger.warning("Clave de corte inválida: %s", clave_corte)
                    continue

                subset = "public" if porcentaje == "30" else "private" if porcentaje == "70" else porcentaje
                registros.append(
                    {
                        "iteracion": iter_idx,
                        "particion": partida_numero,
                        "corte": corte,
                        "subset": subset,
                        "ganancia": float(ganancia),
                    }
                )

    if not registros:
        raise ValueError("No se generaron registros para graficar")

    df_registros = pl.DataFrame(registros)
    df_mean = (
        df_registros.group_by(["subset", "corte"])
        .agg(pl.col("ganancia").mean().alias("ganancia_promedio"))
        .sort(["subset", "corte"])
    )

    estudio = datos.get("metadata", {}).get("study_name", STUDY_NAME)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rutas: Dict[str, Optional[str]] = {"public": None, "private": None}

    for subset, color in (("public", public_color), ("private", private_color)):
        df_subset = df_mean.filter(pl.col("subset") == subset)

        if df_subset.is_empty():
            logger.warning("No hay datos para el subset %s", subset)
            continue

        df_plot = df_subset.to_pandas()

        plt.figure(figsize=(10, 6))
        plt.bar(df_plot["corte"], df_plot["ganancia_promedio"], color=color, width=350)
        plt.xlabel("Corte")
        plt.ylabel("Ganancia promedio")
        plt.title(f"Ganancia promedio por corte - {subset.capitalize()} ({estudio})")
        plt.grid(axis="y", alpha=0.3)

        ruta = os.path.join(output_dir, f"{estudio}_ganancias_{subset}_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(ruta, dpi=300)
        plt.close()

        rutas[subset] = ruta

        logger.info("Gráfico de distribución %s guardado en %s", subset, ruta)

    return rutas
