import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime

# Importar configuración y funciones existentes
from .config import *
from .gain_function import calcular_ganancia
from .best_params import cargar_mejores_hiperparametros
from .optimization import evaluar_en_test

logger = logging.getLogger(__name__)


def calcular_ganancia_acumulada(y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
    """
    Calcula la ganancia acumulada ordenando las predicciones de mayor a menor probabilidad.
    
    Args:
        y_true: Valores verdaderos
        y_pred_proba: Probabilidades predichas
    
    Returns:
        np.ndarray: Ganancias acumuladas
    """
    # Ordenar por probabilidad descendente
    orden_indices = np.argsort(y_pred_proba)[::-1]
    y_true_ordenado = y_true[orden_indices]
    
    # Calcular ganancia acumulada
    ganancias_acumuladas = np.zeros(len(y_true))
    
    for i in range(1, len(y_true_ordenado) + 1):
        # Tomar los primeros i clientes ordenados
        y_true_subset = y_true_ordenado[:i]
        y_pred_subset = np.ones(i)  # Todos los seleccionados son positivos
        
        # Calcular ganancia usando la función existente
        ganancia_actual = calcular_ganancia(y_true_subset, y_pred_subset)
        ganancias_acumuladas[i-1] = ganancia_actual
    
    return ganancias_acumuladas

def crear_grafico_ganancia_test(y_true: np.ndarray, y_pred_proba: np.ndarray, ganancias_acumuladas: np.ndarray) -> str:
    """
    Crea el gráfico de ganancia acumulada vs predicciones ordenadas con mejoras.
    
    Args:
        y_true: Valores verdaderos
        y_pred_proba: Probabilidades predichas
        ganancias_acumuladas: Ganancias acumuladas calculadas
    
    Returns:
        str: Ruta del archivo guardado
    """
    # Guardar las probabilidades en un archivo CSV
    os.makedirs("resultados", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_probabilidades = f"resultados/{STUDY_NAME}_probabilidades_{timestamp}.csv"
    
    # Crear DataFrame con probabilidades y guardar
    df_probabilidades = pd.DataFrame({
        'probabilidad': y_pred_proba,
        'ganancia_acumulada': ganancias_acumuladas,
        'cliente_ordenado': range(len(y_pred_proba))
    })
    df_probabilidades.to_csv(ruta_probabilidades, index=False)
    logger.info(f"Probabilidades guardadas en: {ruta_probabilidades}")
    
    # Encontrar la ganancia máxima y su índice
    ganancia_maxima = np.max(ganancias_acumuladas)
    indice_maximo = np.argmax(ganancias_acumuladas)
    
    # Calcular el umbral (dos tercios de la ganancia máxima)
    umbral_ganancia = ganancia_maxima * 0.66
    
    # Filtrar datos: solo mantener puntos por encima del umbral
    indices_filtrados = ganancias_acumuladas >= umbral_ganancia
    x_filtrado = np.where(indices_filtrados)[0]
    y_filtrado = ganancias_acumuladas[indices_filtrados]
    
    # Encontrar el punto de corte de 0.025 en probabilidades
    umbral_probabilidad = 0.025
    clientes_sobre_umbral = np.sum(y_pred_proba >= umbral_probabilidad)
    
    # Configurar estilo del gráfico
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(14, 8))
    
    # Gráfico de ganancia acumulada filtrado
    plt.plot(x_filtrado, y_filtrado, 
             color='blue', linewidth=2.5, label='Ganancia Acumulada')
    
    # Marcar la ganancia máxima con un punto rojo y etiqueta
    plt.scatter(indice_maximo, ganancia_maxima, 
               color='red', s=100, zorder=5, label='Ganancia Máxima')
    
    # Agregar etiqueta para la ganancia máxima
    plt.annotate(f'Ganancia Máxima\n{ganancia_maxima:,.0f}', 
                xy=(indice_maximo, ganancia_maxima),
                xytext=(indice_maximo + len(x_filtrado) * 0.1, ganancia_maxima * 1.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Marcar el corte ideal en el eje X (punto de ganancia máxima)
    plt.axvline(x=indice_maximo, color='green', linestyle='--', alpha=0.7, 
                label=f'Corte Ideal (cliente {indice_maximo:,})')
    
    # Marcar el corte de 0.025 en probabilidades
    plt.axvline(x=clientes_sobre_umbral, color='purple', linestyle='-.', alpha=0.8, linewidth=2,
                label=f'Corte 0.025 (cliente {clientes_sobre_umbral:,})')
    
    # Configurar etiquetas y título
    plt.xlabel('Clientes ordenados por probabilidad', fontsize=12)
    plt.ylabel('Ganancia Acumulada', fontsize=12)
    plt.title(f'Ganancia Acumulada por Orden de Predicción (Filtrada) - {STUDY_NAME}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Formatear ejes
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    ruta_archivo = f"resultados/{STUDY_NAME}_grafico_test_{timestamp}.jpg"
    
    plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráfico mejorado guardado en: {ruta_archivo}")
    logger.info(f"Estadísticas del gráfico:")
    logger.info(f"  - Ganancia máxima: {ganancia_maxima:,.0f}")
    logger.info(f"  - Corte ideal en cliente: {indice_maximo:,}")
    logger.info(f"  - Umbral de filtrado: {umbral_ganancia:,.0f} (66% del máximo)")
    logger.info(f"  - Corte 0.025 en cliente: {clientes_sobre_umbral:,}")
    logger.info(f"  - Clientes sobre 0.025: {clientes_sobre_umbral:,} ({clientes_sobre_umbral/len(y_pred_proba)*100:.1f}%)")
    logger.info(f"  - Puntos mostrados: {len(x_filtrado)} de {len(ganancias_acumuladas)} totales")
    
    return ruta_archivo


def generar_grafico_test_completo(df: pd.DataFrame) -> str:
    """
    Función principal que genera el gráfico de test.
    
    Args:
        df: DataFrame con todos los datos
    
    Returns:
        str: Ruta del gráfico generado
    """
    logger.info("=== INICIANDO GENERACIÓN DE GRÁFICO DE TEST ===")
    
    # Cargar mejores hiperparámetros
    mejores_params = cargar_mejores_hiperparametros()
    
    # Obtener predicciones usando evaluar_en_test modificada
    resultados_test, y_pred_proba = evaluar_en_test(df, mejores_params)
    
    # Obtener y_true del conjunto de test
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
    
    df_test = df[df['foto_mes'] == MES_TEST]
    y_true = df_test['clase_ternaria'].values
    
    # Calcular ganancia acumulada
    ganancias_acumuladas = calcular_ganancia_acumulada(y_true, y_pred_proba)
    
    # Crear gráfico
    ruta_grafico = crear_grafico_ganancia_test(y_true, y_pred_proba, ganancias_acumuladas)
    
    logger.info("=== GRÁFICO DE TEST COMPLETADO ===")
    
    return ruta_grafico

if __name__ == "__main__":
    # Configurar logging básico
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Importar y cargar datos
    from .loader import cargar_datos, convertir_clase_ternaria_a_target
    from .features import feature_engineering_lag
    
    logger.info("Cargando datos para generar gráfico de test...")
    
    # Cargar y preparar datos
    df = cargar_datos(DATA_PATH)
    atributos = ["mcuentas_saldo", "mtarjeta_visa_consumo", "cproductos"]
    cant_lag = 2
    df_fe = feature_engineering_lag(df, atributos, cant_lag)
    df_fe = convertir_clase_ternaria_a_target(df_fe)
    
    # Generar gráfico
    ruta_grafico = generar_grafico_test_completo(df_fe)
    
    print(f"\n✅ Gráfico generado: {ruta_grafico}")
