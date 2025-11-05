import logging
from typing import Optional

import pandas as pd
import polars as pl

from .config import SEMILLA

logger = logging.getLogger(__name__)


def aplicar_undersampling(
    df: pd.DataFrame,
    ratio: float,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Aplica undersampling controlado sobre la clase mayoritaria usando Polars.

    Args:
        df: DataFrame que contiene ``clase_ternaria`` y ``numero_de_cliente``.
        ratio: Fracción (0, 1] de clientes de la clase mayoritaria a conservar.
        random_state: Semilla opcional para reproducibilidad. Si se omite, se deriva de ``SEMILLA``.

    Returns:
        pd.DataFrame: DataFrame resultante con todos los clientes minoritarios y
        un subconjunto sampleado de la clase mayoritaria.
    """
    if not 0 < ratio <= 1:
        raise ValueError("El ratio de undersampling debe estar en el rango (0, 1].")

    if df.empty:
        return df

    if random_state is None:
        random_state = SEMILLA[0] if isinstance(SEMILLA, list) else int(SEMILLA)

    df_pl = pl.from_pandas(df)

    majority_clients = (
        df_pl
        .filter(pl.col("clase_ternaria") == 0)
        .select("numero_de_cliente")
        .unique()
    )
    minority_df = df_pl.filter(pl.col("clase_ternaria") == 1)

    if majority_clients.height == 0 or minority_df.height == 0:
        logger.warning("No se puede aplicar undersampling: una de las clases está vacía")
        return df

    sample_size = max(1, int(majority_clients.height * ratio))
    sample_size = min(sample_size, majority_clients.height)

    sampled_clients = majority_clients.sample(
        n=sample_size,
        with_replacement=False,
        shuffle=True,
        seed=random_state,
    )

    majority_sampled = (
        df_pl
        .filter(pl.col("clase_ternaria") == 0)
        .join(sampled_clients, on="numero_de_cliente", how="inner")
    )

    combined = pl.concat([majority_sampled, minority_df], how="vertical")
    combined = combined.sample(
        fraction=1.0,
        with_replacement=False,
        shuffle=True,
        seed=random_state,
    )

    logger.debug(
        "Undersampling aplicado - clientes clase 0 retenidos: %d de %d",
        sample_size,
        majority_clients.height,
    )

    return combined.to_pandas()
