import json
import logging
import os
from datetime import datetime

import mlflow
import pandas as pd

from src.best_params import cargar_mejores_hiperparametros
from src.config import (
    BUCKET_NAME,
    DATA_PATH,
    FINAL_PREDIC,
    FINAL_TRAIN,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TAGS,
    MLFLOW_TRACKING_URI,
    STUDY_NAME,
)
from src.features import feature_engineering_delta_lag, feature_engineering_lag
from src.final_training import (
    entrenar_modelo_final,
    generar_predicciones_finales,
    preparar_datos_entrenamiento_final,
)
from src.grafico_test import generar_grafico_test_completo
from src.loader import cargar_datos
from src.optimization_BO import optimizar
from src.optimization_RL import optimizar_rfppo_hpo
from src.optimization_ZS import optimizar_zero_shot
from src.test_evaluation import evaluar_en_test, guardar_resultados_test

## config basico logging
os.makedirs(f"{BUCKET_NAME}/logs", exist_ok=True)

fecha = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
nombre_log = f"log_{STUDY_NAME}_{fecha}.log"

log_path = os.path.join(f"{BUCKET_NAME}/logs/", nombre_log)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8-sig"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def _optimizacion_zs(df_fe: pd.DataFrame):
    """
    Ejecuta la optimización ZeroShot si los archivos JSON no existen.
    """
    logger.info("=== Uso de ZeroShot ===")
    # Verificar si existen los archivos JSON de ZeroShot
    zs_iter_path = os.path.join("resultados", f"{STUDY_NAME}_zs_iteraciones.json")
    zs_best_path = os.path.join("resultados", f"{STUDY_NAME}_zs_best_params.json")

    if os.path.exists(zs_iter_path) and os.path.exists(zs_best_path):
        logger.info("✅ Archivos ZeroShot encontrados. Cargando hiperparámetros...")
        params_lightgbm = cargar_mejores_hiperparametros(archivo_json=zs_iter_path)

        # Cargar información adicional desde el archivo de iteraciones
        with open(zs_iter_path, "r") as f:
            iteraciones = json.load(f)
        mejor_iteracion = max(iteraciones, key=lambda x: x["value"])
        ganancia_val = mejor_iteracion["value"]
        umbral_sugerido = mejor_iteracion.get("umbral_sugerido", 0.5)
        hyperparams = mejor_iteracion.get("params_flaml", {})
        paths = {"iteraciones": zs_iter_path, "best_params": zs_best_path}

        logger.info("=== ANÁLISIS DE RESULTADOS (Cargados desde archivos) ===")
        logger.info(f"✅ Ganancia en validación: {ganancia_val:,.0f}")
        logger.info(f"✅ Umbral sugerido: {umbral_sugerido:.4f}")
        logger.info(
            f"✅ Parámetros LightGBM cargados: {len(params_lightgbm)} parámetros"
        )
        logger.info("✅ Archivos utilizados:")
        logger.info(f"   - Iteraciones: {paths['iteraciones']}")
        logger.info(f"   - Best params: {paths['best_params']}")

    else:
        logger.info("❌ Archivos ZeroShot no encontrados. Ejecutando búsqueda...")
        resultado_zs = optimizar_zero_shot(df_fe)

        # Desempacar resultados del diccionario
        ganancia_val = resultado_zs["ganancia_validacion"]
        umbral_sugerido = resultado_zs["umbral_sugerido"]
        params_lightgbm = resultado_zs["best_params_lightgbm"]
        hyperparams = resultado_zs["best_params_flaml"]
        paths = resultado_zs["paths"]

        logger.info("=== ANÁLISIS DE RESULTADOS ===")
        logger.info(f"✅ Ganancia en validación: {ganancia_val:,.0f}")
        logger.info(f"✅ Umbral sugerido: {umbral_sugerido:.4f}")
        logger.info(f"✅ Parámetros FLAML guardados: {len(hyperparams)} parámetros")
        logger.info(
            f"✅ Parámetros LightGBM guardados: {len(params_lightgbm)} parámetros"
        )
        logger.info("✅ Archivos generados:")
        logger.info(f"   - Iteraciones: {paths['iteraciones']}")
        logger.info(f"   - Best params: {paths['best_params']}")

    # Loggear en MLflow
    mlflow.log_metric("ganancia_validacion", ganancia_val)
    mlflow.log_metric("umbral_sugerido", umbral_sugerido)
    mlflow.log_params({f"best_ZS_{k}": v for k, v in params_lightgbm.items()})
    mlflow.log_artifact(paths["iteraciones"])
    mlflow.log_artifact(paths["best_params"])
    logger.info(f"Ganancia VALID={ganancia_val:,.0f} | Umbral={umbral_sugerido:.4f}")

    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info("Usando hiperparámetros encontrados en ZeroShot")
    ganancia_test, _ = evaluar_en_test(
        df=df_fe, mejores_params=params_lightgbm, undersampling=1
    )
    mlflow.log_metric(
        "ganancia_test_ZS", ganancia_test
    )  # loggear la ganancia en mlflow
    logger.info(f"✅ Ganancia en test ZS: {ganancia_test:,.0f}")

    # Guardar resultados de test
    resultados_path = guardar_resultados_test(ganancia_test)
    mlflow.log_artifact(
        resultados_path
    )  # sube el archivo a mlflow como artifacto .json

    # Resumen de evaluación en test
    logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    logger.info(f"✅ Ganancia en test: {ganancia_test:,.0f}")

    logger.info("=== GRAFICO DE TEST ===")
    ruta_grafico = generar_grafico_test_completo(
        df_fe, params_lightgbm, tiradas=20, undersampling=1
    )
    logger.info(f"✅ Gráfico generado: {ruta_grafico}")
    mlflow.log_artifact(ruta_grafico)  # sube el archivo a mlflow como artifacto .png

    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST COMPLETADA ===")

    return params_lightgbm


## Funcion principal
def main():
    logger.info("Inicio de ejecucion.")

    # 00 Cargar datos
    os.makedirs(f"{BUCKET_NAME}/data", exist_ok=True)
    data_path = os.path.join(BUCKET_NAME, DATA_PATH)

    # 01 Feature Engineering

    #####
    # COMO SE QUE LES GUSTA MUCHO LO ANTERIOR Y LES LLEVA MUCHO TIEMPO 🙄
    #
    # Corroborando si existe antes
    # Cargar el df_fe
    #
    # Guardar el df_fe para no tener que hacerlo de nuevo
    #
    #####

    if os.path.exists(os.path.join(BUCKET_NAME, "data", f"df_fe{STUDY_NAME}.csv")):
        logger.info("✅ df_fe.csv encontrado")
        df_fe = pd.read_csv(os.path.join(BUCKET_NAME, "data", f"df_fe{STUDY_NAME}.csv"))
    else:
        logger.info("❌ df_fe.csv no encontrado")

        df = cargar_datos(data_path)
        # Eliminamos Atributos rotos: mprestamos_personales y cprestamos_personales
        df = df.drop(columns=["mprestamos_personales", "cprestamos_personales"])

        atributos = [col for col in df.columns if col.startswith(("c", "m"))]
        atributos.remove("clase_ternaria")
        cant_lag = 2
        df_fe = feature_engineering_lag(df, atributos, cant_lag)
        df_fe = feature_engineering_delta_lag(df, atributos, cant_lag)

        logger.info(f"Feature Engineering completado: {df_fe.shape}")
        logger.info("Guardando df_fe.csv")

        # Guardamos el df_fe
        df_fe.to_csv(
            os.path.join(BUCKET_NAME, "data", f"df_fe{STUDY_NAME}.csv"), index=False
        )

    # Configurar MLflow y ejecutar pipeline completo
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info(f"MLflow configurado con servidor remoto: {MLFLOW_TRACKING_URI}")

    except Exception as e:
        logger.error(f"Error al configurar MLflow: {e}")
        raise

    with mlflow.start_run(run_name=f"experimento-{STUDY_NAME}"):
        mlflow.set_tags(MLFLOW_TAGS)

        mlflow.log_param("undersampling_ratio_BO", 0.5)
        mlflow.log_param("n_trials_BO", 50)
        mlflow.log_param("Dimenciones_data", df_fe.shape)
        mlflow.log_param("undersampling_ratio_RFPPO", 0.5)
        mlflow.log_param("episodes_RFPPO", 200)
        mlflow.log_param("initial_real_episodes_RFPPO", 100)
        mlflow.log_param("kl_threshold_RFPPO", 0.1)
        # 03 Ejecutar optimizacion de hiperparametros
        study = optimizar(df_fe, n_trials=50, undersampling=0.5)
        mlflow.log_param("n_trials_executed", len(study.trials))

        # Loggear mejores hiperparámetros
        if study.best_params:
            mlflow.log_params({f"best_BO_{k}": v for k, v in study.best_params.items()})

        # 3.1 Análisis adicional
        logger.info("=== ANÁLISIS DE RESULTADOS ===")
        trials_df = study.trials_dataframe()
        if len(trials_df) > 0:
            top_5 = trials_df.nlargest(5, "value")
            logger.info("Top 5 mejores trials:")
            for idx, trial in top_5.iterrows():
                logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")

            # Registrar Trials completos como artefacto
            trials_path = os.path.join("resultados", f"trials_{STUDY_NAME}.csv")
            os.makedirs("resultados", exist_ok=True)
            trials_df.to_csv(trials_path, index=False)
            mlflow.log_artifact(
                trials_path
            )  # sube el archivo a mlflow como artifacto .csv

        logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

        logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
        logger.info("Usando hiperparámetros encontrados en BO")
        archivo_bo = os.path.join("resultados", f"{STUDY_NAME}_iteraciones.json")
        mejores_params = cargar_mejores_hiperparametros(archivo_json=archivo_bo)

        ganancia_test, _ = evaluar_en_test(
            df=df_fe, mejores_params=mejores_params, undersampling=1
        )
        mlflow.log_metric(
            "ganancia_test_BO", ganancia_test
        )  # loggear la ganancia en mlflow
        logger.info(f"✅ Ganancia en test BO: {ganancia_test:,.0f}")

        # Guardar resultados de test
        resultados_path = guardar_resultados_test(ganancia_test)
        mlflow.log_artifact(
            resultados_path
        )  # sube el archivo a mlflow como artifacto .json

        # Resumen de evaluación en test
        logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
        logger.info(f"✅ Ganancia en test: {ganancia_test:,.0f}")

        logger.info("=== GRAFICO DE TEST ===")
        ruta_grafico = generar_grafico_test_completo(
            df=df_fe, mejores_params=mejores_params, tiradas=20, undersampling=1
        )
        logger.info(f"✅ Gráfico generado: {ruta_grafico}")
        mlflow.log_artifact(
            ruta_grafico
        )  # sube el archivo a mlflow como artifacto .png

        logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST COMPLETADA ===")

        # 04 Sin Optimizacion con el uso de ZeroShot
        # 05 Test en mes desconocido - Usando hiperparámetros de ZeroShot
        _optimizacion_zs(df_fe)

        logger.info("====OPTIMIZACION POR RL====")

        mlflow.log_param("undersampling_ratio_RL", 0.5)

        optimizar_rfppo_hpo(
            df_fe,
            undersampling=0.5,
            episodes=200,
            initial_real_episodes=100,
            kl_threshold=0.1,
        )

        logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
        logger.info("Usando hiperparámetros encontrados en RL")
        archivo_RL = os.path.join("resultados", f"{STUDY_NAME}_iteraciones.json")
        mejores_params = cargar_mejores_hiperparametros(archivo_json=archivo_RL)

        ganancia_test, _ = evaluar_en_test(
            df=df_fe, mejores_params=mejores_params, undersampling=1
        )
        mlflow.log_metric(
            "ganancia_test_RL", ganancia_test
        )  # loggear la ganancia en mlflow
        logger.info(f"✅ Ganancia en test RL: {ganancia_test:,.0f}")

        # Guardar resultados de test
        resultados_path = guardar_resultados_test(ganancia_test)
        mlflow.log_artifact(
            resultados_path
        )  # sube el archivo a mlflow como artifacto .json

        # Resumen de evaluación en test
        logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
        logger.info(f"✅ Ganancia en test: {ganancia_test:,.0f}")

        logger.info("=== GRAFICO DE TEST ===")
        ruta_grafico = generar_grafico_test_completo(
            df=df_fe, mejores_params=mejores_params, tiradas=20, undersampling=1
        )
        logger.info(f"✅ Gráfico generado: {ruta_grafico}")
        mlflow.log_artifact(
            ruta_grafico
        )  # sube el archivo a mlflow como artifacto .png

        logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST COMPLETADA ===")

    # 06 Entrenar modelo final
    logger.info("=== ENTRENAMIENTO FINAL ===")

    # Entrenar modelo final
    logger.info("Entrenar modelo final")
    # modelo_final = entrenar_modelo_final(df_fe, mejores_params, undersampling=0.02)

    # Preparar datos y entrenar modelo final
    _, _, X_predict, clientes_predict = preparar_datos_entrenamiento_final(df_fe)
    modelo_final = entrenar_modelo_final(df_fe, mejores_params)

    # Generar predicciones finales
    logger.info("Generar predicciones finales")
    generar_predicciones_finales(modelo_final, X_predict, clientes_predict)

    # Resumen final
    logger.info("=== RESUMEN FINAL ===")
    logger.info("✅ Entrenamiento final completado exitosamente")
    logger.info(f"📊 Mejores hiperparámetros utilizados: {mejores_params}")
    logger.info(f"🎯 Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"🔮 Período de predicción: {FINAL_PREDIC}")
    logger.info(f"📝 Log detallado: logs/{nombre_log}")
    logger.info(">>> Ejecución finalizada. Revisar logs para mas detalles.")


if __name__ == "__main__":
    main()
