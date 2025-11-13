import json
import logging
import os
from typing import Any, Dict, List, Optional

from src.config import STUDY_NAME

logger = logging.getLogger(__name__)


def cargar_mejores_hiperparametros(
    archivo_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Carga los mejores hiperparámetros desde el archivo JSON de iteraciones.
    Función unificada que maneja tanto archivos de BO como ZeroShot.

    Args:
        archivo_json: Ruta completa al archivo JSON de iteraciones.
                     Si es None, usa resultados/{STUDY_NAME}_iteraciones.json (BO por defecto)

    Returns:
        dict: Mejores hiperparámetros encontrados
    """
    # Si no se proporciona archivo, usar el de BO por defecto
    if archivo_json is None:
        archivo_json = os.path.join("resultados", f"{STUDY_NAME}_iteraciones.json")

    try:
        with open(archivo_json, "r", encoding="utf-8") as f:
            data: Any = json.load(f)

        if not data:
            raise ValueError(
                f"No se encontraron iteraciones en el archivo: {archivo_json}"
            )

        # Transformar según la estructura del archivo
        if isinstance(data, dict):
            # Si es un diccionario, convertir a lista de iteraciones
            iteraciones: List[Any] = list(data.values()) if data else []
        else:
            # Si es una lista, usarla directamente
            iteraciones = list(data)

        # Si los elementos son strings, parsearlos como JSON
        if iteraciones and isinstance(iteraciones[0], str):
            iteraciones = [
                json.loads(item) if isinstance(item, str) else item
                for item in iteraciones
            ]

        if not iteraciones:
            raise ValueError(
                f"No se encontraron iteraciones válidas en el archivo: {archivo_json}"
            )

        # Encontrar la iteración con mayor ganancia
        iteraciones_dict: List[Dict[str, Any]] = []
        for item in iteraciones:
            if isinstance(item, dict):
                iteraciones_dict.append(item)
            else:
                raise ValueError(
                    "Las iteraciones deben ser diccionarios con información de los trials"
                )

        mejor_iteracion = max(iteraciones_dict, key=lambda x: x["value"])
        mejores_params = mejor_iteracion["params"]
        mejor_ganancia = mejor_iteracion["value"]

        logger.info(f"✅ Mejores hiperparámetros cargados desde: {archivo_json}")
        logger.info(f"✅ Mejor ganancia encontrada: {mejor_ganancia:,.0f}")
        logger.info(f"✅ Trial número: {mejor_iteracion['trial_number']}")
        logger.info(f"✅ Parámetros: {mejores_params}")

        return mejores_params

    except FileNotFoundError:
        logger.error(f"❌ No se encontró el archivo: {archivo_json}")
        logger.error("Asegúrate de haber ejecutado la optimización primero")
        raise
    except KeyError as e:
        logger.error(
            f"❌ Error en la estructura del archivo {archivo_json}: falta la clave {e}"
        )
        raise
    except Exception as e:
        logger.error(
            f"❌ Error al cargar mejores hiperparámetros desde {archivo_json}: {e}"
        )
        raise
