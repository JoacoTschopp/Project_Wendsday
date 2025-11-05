import yaml
import os
import logging

logger = logging.getLogger(__name__)

#Ruta del archivo de configuracion
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf.yaml")

try:
    with open(PATH_CONFIG, "r") as f:
        _cfgGeneral = yaml.safe_load(f)
        _cfg = _cfgGeneral["competencia02"]
        PARAMETROS_LGB = _cfgGeneral["parametros_lgb"]
        BUCKET_NAME = _cfgGeneral.get("BUCKET_NAME", "")

        STUDY_NAME = _cfgGeneral.get("STUDY_NAME", "Wendsday")
        DATA_PATH = _cfg.get("DATA_PATH", "../data/competencia.csv")
        SEMILLA = _cfg.get("SEMILLA", [42])
        MES_TRAIN = _cfg.get("MES_TRAIN", "202102")
        MES_VALIDACION = _cfg.get("MES_VALIDACION", "202103")
        MES_TEST = _cfg.get("MES_TEST", "202104")
        GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO", None)
        COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO", None)
        FINAL_TRAIN = _cfg.get("FINAL_TRAIN", [])
        FINAL_PREDIC = _cfg.get("FINAL_PREDIC", "")
        MIN_DATA_IN_LEAF = _cfg.get("MIN_DATA_IN_LEAF", None)
        NUM_ITERATIONS = _cfg.get("NUM_ITERATIONS", None)



except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise

# =============================================================================
# Configuración de MLflow
# =============================================================================
try:
    # Cargar configuración de MLflow desde config.yaml
    MLFLOW_CFG = _cfgGeneral.get("mlflow", {})

    _bucket_root = BUCKET_NAME or os.path.dirname(os.path.dirname(__file__))
    _default_artifact_dir = os.path.abspath(
        MLFLOW_CFG.get("ARTIFACT_PATH", os.path.join(_bucket_root, "mlruns"))
    )

    # Configuración básica
    MLFLOW_TRACKING_URI = MLFLOW_CFG.get("TRACKING_URI")
    if not MLFLOW_TRACKING_URI:
        MLFLOW_TRACKING_URI = f"file://{_default_artifact_dir}"

    # Configuración directa sin plantillas
    MLFLOW_EXPERIMENT_NAME = f"DMEyF-{STUDY_NAME}"
    MLFLOW_ARTIFACT_PATH = _default_artifact_dir
    MLFLOW_REGISTERED_MODEL_NAME = f"dmeyf-{STUDY_NAME}"
  
    # Tags fijos
    MLFLOW_TAGS = {
        "proyecto": "DMEyF-Competencia02",
        "equipo": "python",
        "user_name": "tschoppj",
        "comision": "Lunes",
        "version_dataset": "1.0"
    }
  
    # Crear directorio de artefactos si no existe
    os.makedirs(MLFLOW_ARTIFACT_PATH, exist_ok=True)
  
    logger.info(f"Configuración de MLflow cargada - URI: {MLFLOW_TRACKING_URI}")
  
except Exception as e:
    logger.error(f"Error al cargar configuración de MLflow: {e}")
    raise