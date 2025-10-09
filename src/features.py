import pandas as pd
import duckdb
import logging

logger = logging.getLogger("__name__")

def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """

    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df
    
    # Construir la consulta SQL
    sql = "SELECT *"
    
    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")
    
    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df

def feature_engineering_delta_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables delta-lag para los atributos especificados.
    Cada delta corresponde a la diferencia entre el valor actual y su lag i.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos originales.
    columnas : list[str]
        Lista de atributos para los cuales generar delta-lags.
    cant_lag : int, default=1
        Cantidad de lags a considerar para cada delta.

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas delta-lag agregadas.
    """

    logger.info(f"Generando delta-lags con {cant_lag} desfases para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar delta-lags")
        return df

    sql = "SELECT *"

    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += (
                    f", ({attr} - lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)) "
                    f"AS {attr}_delta_lag_{i}"
                )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    sql += " FROM df"

    logger.debug(f"Consulta SQL delta-lag: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    logger.info(f"Delta-lags generados. DataFrame resultante con {df.shape[1]} columnas")

    return df