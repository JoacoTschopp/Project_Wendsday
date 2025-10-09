import pandas as pd
import os
import datetime
import logging

from src.loader import cargar_datos, convertir_clase_ternaria_a_target
from src.features import feature_engineering_lag
from src.optimization_cv import optimizar_con_cv
from src.optimization import optimizar
from src.test_evaluation import evaluar_en_test, guardar_resultados_test
from src.best_params import cargar_mejores_hiperparametros
from src.final_training import preparar_datos_entrenamiento_final, generar_predicciones_finales, entrenar_modelo_final
from src.output_manager import guardar_predicciones_finales
from src.config import *
from src.grafico_test import generar_grafico_test_completo

## config basico logging
os.makedirs("logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
nombre_log = f"log_{STUDY_NAME}_{fecha}.log"

log_path =os.path.join("logs/", nombre_log)

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
    os.makedirs("data", exist_ok=True)
    df = cargar_datos(DATA_PATH)       

    #01 Feature Engineering
    atributos = [col for col in df.columns if col.startswith(('c', 'm'))]
    cant_lag = 2
    df_fe = feature_engineering_lag(df, atributos, cant_lag)
    df_fe = feature_engineering_delta_lag(df, atributos, cant_lag)
    logger.info(f"Feature Engineering completado: {df_fe.shape}")

    #02 Convertir clase_ternaria a target binario
    #df_fe = convertir_clase_ternaria_a_target(df_fe)
    
    #03 Ejecutar optimizacion de hiperparametros
    #study = optimizar_con_cv(df_fe, n_trials=50)
    
    #04 Análisis adicional
    #logger.info("=== ANÁLISIS DE RESULTADOS ===")
    #trials_df = study.trials_dataframe()
    #if len(trials_df) > 0:
    #    top_5 = trials_df.nlargest(5, 'value')
    #    logger.info("Top 5 mejores trials:")
    #    for idx, trial in top_5.iterrows():
    #        logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
  
    #logger.info("=== OPTIMIZACIÓN COMPLETADA ===")
    
    #05 Test en mes desconocido
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    mejores_params = cargar_mejores_hiperparametros()
  
    # Evaluar en test
    ganancia_test = evaluar_en_test(df_fe, mejores_params)
  
    # Guardar resultados de test
    guardar_resultados_test(ganancia_test[0])
    
    # Resumen de evaluación en test
    logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    logger.info(f"✅ Ganancia en test: {ganancia_test[0]:,.0f}")
    
    logger.info("=== GRAFICO DE TEST ===")
    ruta_grafico = generar_grafico_test_completo(df_fe, tiradas=20)
    logger.info(f"✅ Gráfico generado: {ruta_grafico}")
    
    #06 Entrenar modelo final
    logger.info("=== ENTRENAMIENTO FINAL ===")
 
    # Entrenar modelo final
    logger.info("Entrenar modelo final")
    modelo_final = entrenar_modelo_final(df_fe, mejores_params)
  
    # Generar predicciones finales
    logger.info("Generar predicciones finales")
    df_predict = df_fe[df_fe['foto_mes'].isin([FINAL_PREDIC])]
    clientes_predict = df_predict['numero_de_cliente'].values
    X_predict = df_predict.drop(columns=['clase_ternaria'])
    resultados = generar_predicciones_finales(modelo_final, X_predict, clientes_predict)
  
    # Guardar predicciones
    logger.info("Guardar predicciones")
    archivo_salida = guardar_predicciones_finales(resultados)
  
   
    # Resumen final
    logger.info("=== RESUMEN FINAL ===")
    logger.info(f"✅ Entrenamiento final completado exitosamente")
    logger.info(f"📊 Mejores hiperparámetros utilizados: {mejores_params}")
    #logger.info(f"🎯 Períodos de entrenamiento: {FINAL_TRAIN}")
    #logger.info(f"🔮 Período de predicción: {FINAL_PREDIC}")
    #logger.info(f"📁 Archivo de salida: {archivo_salida}")
    logger.info(f"📝 Log detallado: logs/{nombre_log}")

    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.")

if __name__ == "__main__":
    main()