import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


## Funcion para cargar datos
def cargar_datos(path: str) -> pd.DataFrame | None:
    logger.info(f"Cargando dataset desde {path}")
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        raise


def convertir_clase_ternaria_a_target(df: pd.DataFrame, baja_2_1=True) -> pd.DataFrame:
    """
    Convierte clase_ternaria a target binario reemplazando en el mismo atributo:
    - CONTINUA = 0
    y segun los argumentos baja_2_1
    baja_2_1 = true entonces: BAJA+1 y BAJA+2 = 1
    baja_2_1 = false entonces: BAJA+1 = 0 y BAJA+2 = 1

    Args:
        df: DataFrame con columna 'clase_ternaria'
        baja_2_1: Booleano que indica si se considera BAJA+1 como positivo

    Returns:
        pd.DataFrame: DataFrame con clase_ternaria convertida a valores binarios (0, 1)
    """

    # logger.info("Convirtiendo clase_ternaria a target binario")

    # Contar valores originales para logging (antes de modificar)
    n_continua_orig = (df["clase_ternaria"] == "CONTINUA").sum()
    n_baja1_orig = (df["clase_ternaria"] == "BAJA+1").sum()
    n_baja2_orig = (df["clase_ternaria"] == "BAJA+2").sum()
    logger.debug(
        "Distribución original - CONTINUA: %d, BAJA+1: %d, BAJA+2: %d",
        n_continua_orig,
        n_baja1_orig,
        n_baja2_orig,
    )

    # Modificar el DataFrame usando .loc para evitar SettingWithCopyWarning
    if baja_2_1:
        # Convertir clase_ternaria a binario usando numpy.where para mejorar rendimiento
        df.loc[:, "clase_ternaria"] = np.where(
            df["clase_ternaria"] == "CONTINUA", 0, 1
        ).astype(np.int8)
    else:
        # Convertir BAJA+2 a 1, todo lo demás a 0
        df.loc[:, "clase_ternaria"] = np.where(
            df["clase_ternaria"] == "BAJA+2", 1, 0
        ).astype(np.int8)

    # Log de la conversión
    n_ceros = (df["clase_ternaria"] == 0).sum()
    n_unos = (df["clase_ternaria"] == 1).sum()
    logger.debug(
        "Distribución binaria - 0: %d, 1: %d (%.2f%% positivos)",
        n_ceros,
        n_unos,
        (n_unos / (n_ceros + n_unos) * 100) if (n_ceros + n_unos) else 0.0,
    )

    return df
