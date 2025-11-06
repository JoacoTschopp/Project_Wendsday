import json
from src.config import *
import logging

logger = logging.getLogger(__name__)

def cargar_mejores_hiperparametros(archivo_base: str = None) -> dict:
    """
    Carga los mejores hiperparámetros desde el archivo JSON de iteraciones de Optuna.
    
    Args:
        archivo_base: Nombre base del archivo (si es None, usa STUDY_NAME)
    
    Returns:
        dict: Mejores hiperparámetros encontrados
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
    
    archivo = f"resultados/{archivo_base}_iteraciones.json"
    
    try:
        with open(archivo, 'r') as f:
            iteraciones = json.load(f)
        
        if not iteraciones:
            raise ValueError("No se encontraron iteraciones en el archivo")
        
        # Encontrar la iteración con mayor ganancia
        mejor_iteracion = max(iteraciones, key=lambda x: x['value'])
        mejores_params = mejor_iteracion['params']
        mejor_ganancia = mejor_iteracion['value']
        
        logger.info(f"Mejores hiperparámetros cargados desde {archivo}")
        logger.info(f"Mejor ganancia encontrada: {mejor_ganancia:,.0f}")
        logger.info(f"Trial número: {mejor_iteracion['trial_number']}")
        logger.info(f"Parámetros: {mejores_params}")
        
        return mejores_params
        
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo {archivo}")
        logger.error("Asegúrate de haber ejecutado la optimización con Optuna primero")
        raise
    except Exception as e:
        logger.error(f"Error al cargar mejores hiperparámetros: {e}")
        raise


def cargar_mejores_hiperparametros_zs(archivo_base: str = None) -> dict:
    """
    Carga los mejores hiperparámetros desde el archivo JSON de ZeroShot.
    
    Args:
        archivo_base: Nombre base del archivo (si es None, usa STUDY_NAME)
    
    Returns:
        dict: Mejores hiperparámetros encontrados
    """
    if archivo_base is None:
        archivo_base = f"{STUDY_NAME}_zs_iteraciones.json"
    

    archivo = f"resultados/{archivo_base}"
    
    try:
        with open(archivo, 'r') as f:
            iteraciones = json.load(f)
        
        if not iteraciones:
            raise ValueError("No se encontraron iteraciones en el archivo ZeroShot")
        
        # Encontrar la iteración con mayor ganancia
        mejor_iteracion = max(iteraciones, key=lambda x: x['value'])
        mejores_params = mejor_iteracion['params']
        mejor_ganancia = mejor_iteracion['value']
        
        logger.info(f"✅ Mejores hiperparámetros ZeroShot cargados desde {archivo}")
        logger.info(f"✅ Mejor ganancia encontrada: {mejor_ganancia:,.0f}")
        logger.info(f"✅ Trial número: {mejor_iteracion['trial_number']}")
        logger.info(f"✅ Parámetros: {mejores_params}")
        
        return mejores_params
        
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo {archivo}")
        logger.error("Asegúrate de haber ejecutado la optimización ZeroShot primero")
        raise
    except Exception as e:
        logger.error(f"Error al cargar mejores hiperparámetros ZeroShot: {e}")
        raise
