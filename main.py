import pandas as pd
import os
import datetime
import logging


from src.config import *
from src.loader import cargar_datos, convertir_clase_ternaria_a_target
from src.features import feature_engineering_lag, feature_engineering_delta_lag
from src.optimization import optimizar

from src.best_params import cargar_mejores_hiperparametros

from src.test_evaluation import evaluar_en_test, guardar_resultados_test
from src.final_training import preparar_datos_entrenamiento_final, generar_predicciones_finales, entrenar_modelo_final
from src.output_manager import guardar_predicciones_finales


## config basico logging
os.makedirs(f"{BUCKET_NAME}/logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
nombre_log = f"log_{STUDY_NAME}_{fecha}.log"

log_path =os.path.join(f"{BUCKET_NAME}/logs/", nombre_log)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8-sig"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


## Funcion principal
def main():
    logger.info("Inicio de ejecucion.")

    #00 Cargar datos
    os.makedirs(f"{BUCKET_NAME}/data", exist_ok=True)
    data_path = os.path.join(BUCKET_NAME, DATA_PATH)
    df = cargar_datos(data_path)       

    #01 Feature Engineering
    
    #####
    #COMO SE QUE LES GUSTA MUCHO LO ANTERIOR Y LES LLEVA MUCHO TIEMPO 🙄
    #
    #Corroborando si existe antes
    #Cargar el df_fe
    #
    #Guardar el df_fe para no tener que hacerlo de nuevo
    #
    #####

    if os.path.exists(os.path.join(BUCKET_NAME, "data", f"df_fe{STUDY_NAME}.csv")):
        logger.info("✅ df_fe.csv encontrado")
        df_fe = pd.read_csv(os.path.join(BUCKET_NAME, "data", f"df_fe{STUDY_NAME}.csv"))
    else:
        logger.info("❌ df_fe.csv no encontrado")
        atributos = [col for col in df.columns if col.startswith(('c', 'm'))]
        atributos.remove("clase_ternaria")
        cant_lag = 2
        df_fe = feature_engineering_lag(df, atributos, cant_lag)
        df_fe = feature_engineering_delta_lag(df, atributos, cant_lag)
        logger.info(f"Feature Engineering completado: {df_fe.shape}")
        logger.info("Guardando df_fe.csv")
        df_fe.to_csv(os.path.join(BUCKET_NAME, "data", f"df_fe{STUDY_NAME}.csv"), index=False)


    #03 Ejecutar optimizacion de hiperparametros
    study = optimizar(df_fe, n_trials=100, undersampling=0.02)
    
    #04 Análisis adicional
    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
    
    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")
    
    #05 Test en mes desconocido
    #logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")


    #mejores_params = cargar_mejores_hiperparametros()
  
    # Evaluar en test
    #ganancia_test = evaluar_en_test(df_fe, mejores_params, undersampling=0.02)
  
    # Guardar resultados de test
    #guardar_resultados_test(ganancia_test[0])
    
    # Resumen de evaluación en test
    #logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    #logger.info(f"✅ Ganancia en test: {ganancia_test[0]:,.0f}")
    
    #logger.info("=== GRAFICO DE TEST ===")
    #ruta_grafico = generar_grafico_test_completo(df_fe, tiradas=30, undersampling=0.02)
    #logger.info(f"✅ Gráfico generado: {ruta_grafico}")
    
    #06 Entrenar modelo final
    #logger.info("=== ENTRENAMIENTO FINAL ===")
 
    # Entrenar modelo final
    #logger.info("Entrenar modelo final")
    #modelo_final = entrenar_modelo_final(df_fe, mejores_params, undersampling=0.02)
  
    # Generar predicciones finales
    #logger.info("Generar predicciones finales")
    #df_predict = df_fe[df_fe['foto_mes'].isin([FINAL_PREDIC])]
    #clientes_predict = df_predict['numero_de_cliente'].values
    #X_predict = df_predict.drop(columns=['clase_ternaria'])
    #resultados = generar_predicciones_finales(modelo_final, X_predict, clientes_predict)

    # Resumen final
    #logger.info("=== RESUMEN FINAL ===")
    #logger.info(f"✅ Entrenamiento final completado exitosamente")
    #logger.info(f"📊 Mejores hiperparámetros utilizados: {mejores_params}")
    #logger.info(f"🎯 Períodos de entrenamiento: {FINAL_TRAIN}")
    #logger.info(f"🔮 Período de predicción: {FINAL_PREDIC}")
    #logger.info(f"📁 Archivo de salida: {archivo_salida}")
    #logger.info(f"📝 Log detallado: logs/{nombre_log}")
    #logger.info(f"📝 Resultados guardados en: {json_path}")
    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.")

if __name__ == "__main__":
    main()