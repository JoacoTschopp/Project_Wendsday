import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    COSTO_ESTIMULO,
    GANANCIA_ACIERTO,
    MES_TRAIN,
    MES_VALIDACION,
    SEMILLA,
)
from .gain_function import ganancia_evaluator
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


@dataclass
class RLHyperparamAgent:
    search_space: Dict[str, List[Any]]
    epsilon: float = 0.1
    gamma: float = 0.9
    alpha: float = 0.2
    seed: Optional[int] = None
    q_table: Dict[str, Dict[Any, float]] = field(default_factory=dict)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        for key, values in self.search_space.items():
            self.q_table[key] = {value: 0.0 for value in values}

    def select_action(self, param_name: str) -> Any:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.search_space[param_name])
        values = self.q_table[param_name]
        return max(values, key=values.get)

    def update(self, param_name: str, action: Any, reward: float) -> None:
        current = self.q_table[param_name][action]
        self.q_table[param_name][action] = current + self.alpha * (reward - current)


class ReinforcementLearningOptimizer:
    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        agent: Optional[RLHyperparamAgent] = None,
        reward_scale: float = 1e-5,
    ) -> None:
        self.search_space = search_space
        self.agent = agent or RLHyperparamAgent(
            search_space=search_space,
            epsilon=0.2,
            seed=_resolve_seed(),
        )
        self.reward_scale = reward_scale

    def suggest_params(self) -> Dict[str, Any]:
        suggestion = {}
        for param in self.search_space:
            suggestion[param] = self.agent.select_action(param)
        return suggestion

    def update_agent(self, params: Dict[str, Any], reward: float) -> None:
        scaled_reward = reward * self.reward_scale
        for param, value in params.items():
            self.agent.update(param, value, scaled_reward)


def preparar_datos_rl(
    df: pd.DataFrame,
    feature_subset: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    df_train, df_val = _split_train_validation(df)

    if feature_subset:
        X_train = df_train[feature_subset].copy()
        X_val = df_val[feature_subset].copy()
    else:
        X_train = df_train.drop(columns=["clase_ternaria"])
        X_val = df_val.drop(columns=["clase_ternaria"])

    y_train = df_train["clase_ternaria"].to_numpy(dtype=np.int8)
    y_val = df_val["clase_ternaria"].to_numpy(dtype=np.int8)

    return X_train, y_train, X_val, y_val


def evaluar_configuracion(
    modelo_fn,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> float:
    modelo = modelo_fn(params)
    modelo.fit(X_train, y_train)
    proba = modelo.predict_proba(X_val)[:, 1]

    orden = np.argsort(proba)[::-1]
    y_sorted = y_val[orden]
    ganancias = np.where(y_sorted == 1, GANANCIA_ACIERTO, -COSTO_ESTIMULO)
    ganancia_acumulada = np.cumsum(ganancias)

    return float(np.max(ganancia_acumulada)) if ganancia_acumulada.size else 0.0


def optimizar_con_rl(
    df: pd.DataFrame,
    modelo_fn,
    search_space: Dict[str, List[Any]],
    episodios: int = 30,
    iteraciones_por_ep: int = 1,
    feature_subset: Optional[List[str]] = None,
) -> Dict[str, Any]:
    X_train, y_train, X_val, y_val = preparar_datos_rl(df, feature_subset)

    optimizador = ReinforcementLearningOptimizer(search_space)
    mejor_config: Dict[str, Any] = {}
    mejor_ganancia = -np.inf

    for episodio in range(episodios):
        params = optimizador.suggest_params()
        ganancia = evaluar_configuracion(
            modelo_fn,
            params,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        optimizador.update_agent(params, ganancia)

        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_config = params

        logger.info(
            "Episodio %s/%s | Ganancia %.2f | Configuracion %s",
            episodio + 1,
            episodios,
            ganancia,
            params,
        )

    logger.info("Ganancia maxima RL: %.2f", mejor_ganancia)
    return mejor_config
