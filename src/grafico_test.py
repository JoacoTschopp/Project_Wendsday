import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import math
from datetime import datetime

# Importar configuración y funciones existentes
from .config import *
from .gain_function import calcular_ganancia
from .best_params import cargar_mejores_hiperparametros
from .test_evaluation import evaluar_en_test
from .loader import convertir_clase_ternaria_a_target
import mlflow

logger = logging.getLogger(__name__)

def _es_primo(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    limite = int(math.isqrt(n))
    i = 5
    while i <= limite:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _siguiente_primo_en_rango(candidato: int, minimo: int, maximo: int) -> int:
    if candidato < minimo:
        candidato = minimo
    if candidato % 2 == 0:
        candidato += 1

    while True:
        if candidato > maximo:
            candidato = minimo if minimo % 2 == 1 else minimo + 1
        if _es_primo(candidato):
            return candidato
        candidato += 2


def _generar_semillas_primas(tiradas: int, semilla_base: int) -> np.ndarray:
    rng = np.random.default_rng(semilla_base)
    digitos = len(str(semilla_base))
    minimo = 10 ** (digitos - 1)
    maximo = 10 ** digitos - 1

    semillas = []
    usados = set()

    while len(semillas) < tiradas:
        candidato = int(rng.integers(minimo, maximo + 1))
        primo = _siguiente_primo_en_rango(candidato, minimo, maximo)
        if primo not in usados:
            semillas.append(primo)
            usados.add(primo)

    return np.array(semillas, dtype=np.int64)

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
    df_probabilidades.sort_values(by='probabilidad', ascending=False, inplace=True)
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
    
    # Calcular cortes por número de clientes (no por umbrales de probabilidad)
    
    # Corte ideal: el punto exacto donde se alcanza la ganancia máxima
    corte_ideal = indice_maximo
    
    # Corte por umbral óptimo: número de clientes con probabilidad >= umbral óptimo
    # El umbral óptimo es la probabilidad del cliente en el punto de ganancia máxima
    orden_indices = np.argsort(y_pred_proba)[::-1]
    y_pred_ordenado = y_pred_proba[orden_indices]
    
    if indice_maximo < len(y_pred_ordenado):
        umbral_optimo_prob = y_pred_ordenado[indice_maximo]
        corte_optimo = np.sum(y_pred_proba >= umbral_optimo_prob)
    else:
        # Fallback si hay algún problema con el índice
        umbral_optimo_prob = np.percentile(y_pred_proba, 95)  # Top 5%
        corte_optimo = np.sum(y_pred_proba >= umbral_optimo_prob)
    
    # Corte fijo: 0.025% de los clientes predichos
    corte_fijo = int(len(y_pred_proba) * 0.025)  # 0.025% de los clientes
    
    # Configurar estilo del gráfico
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(14, 8))
    
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
    plt.axvline(x=corte_ideal, color='green', linestyle='--', alpha=0.7, 
                label=f'Corte Ideal (cliente {corte_ideal:,})')
    
    # Marcar el corte por umbral óptimo
    plt.axvline(x=corte_optimo, color='purple', linestyle='-.', alpha=0.8, linewidth=2,
                label=f'Corte Óptimo (cliente {corte_optimo:,}, prob={umbral_optimo_prob:.4f})')
    
    # Marcar el corte fijo de clientes
    plt.axvline(x=corte_fijo, color='orange', linestyle=':', alpha=0.7, linewidth=2,
                label=f'Corte Fijo 0.025 ({corte_fijo:,} clientes)')
    
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
    logger.info(f"  - Corte ideal en cliente: {corte_ideal:,}")
    logger.info(f"  - Umbral de filtrado: {umbral_ganancia:,.0f} (66% del máximo)")
    logger.info(f"  - Corte por umbral óptimo: {corte_optimo:,} clientes")
    logger.info(f"  - Umbral óptimo de probabilidad: {umbral_optimo_prob:.6f}")
    logger.info(f"  - Corte fijo 0.025% de clientes: {corte_fijo:,} clientes")
    logger.info(f"  - Porcentaje sobre umbral óptimo: {corte_optimo/len(y_pred_proba)*100:.1f}%")
    logger.info(f"  - Porcentaje sobre umbral fijo: {corte_fijo/len(y_pred_proba)*100:.1f}%")
    logger.info(f"  - Diferencia: {corte_fijo - corte_optimo:,} clientes más con umbral fijo")
    logger.info(f"  - Puntos mostrados: {len(x_filtrado)} de {len(ganancias_acumuladas)} totales")
    
    return ruta_archivo


def crear_grafico_comparativo_multiple_semillas(y_true: np.ndarray, resultados_por_semilla: list) -> str:
    """
    Crea un gráfico comparativo con las curvas de ganancia acumulada de múltiples semillas.
    
    Args:
        y_true: Valores verdaderos
        resultados_por_semilla: Lista de diccionarios con resultados por semilla
    
    Returns:
        str: Ruta del archivo guardado
    """
    logger.info("=== CREANDO GRÁFICO COMPARATIVO MÚLTIPLES SEMILLAS ===")
    
    # Configurar estilo del gráfico
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(16, 10))
    
    # Lista para almacenar todas las curvas de ganancia para calcular la media
    todas_las_curvas = []
    max_len = 0  # Para determinar la longitud máxima de las curvas
    
    # Calcular umbral común para filtrado (usar el máximo de todas las ganancias máximas)
    ganancia_maxima_global = max(resultado['ganancia_maxima'] for resultado in resultados_por_semilla)
    umbral_ganancia = ganancia_maxima_global * 0.66
    
    # Graficar cada curva de semilla
    for resultado in resultados_por_semilla:
        semilla = resultado['semilla']
        ganancias_acumuladas = resultado['ganancias_acumuladas']
        color = resultado['color']
        ganancia_maxima = resultado['ganancia_maxima']
        indice_maximo = resultado['indice_maximo']
        
        # Filtrar datos: solo mantener puntos por encima del umbral
        indices_filtrados = ganancias_acumuladas >= umbral_ganancia
        x_filtrado = np.where(indices_filtrados)[0]
        y_filtrado = ganancias_acumuladas[indices_filtrados]
        
        # Almacenar la curva completa para calcular la media después
        todas_las_curvas.append(ganancias_acumuladas)
        max_len = max(max_len, len(ganancias_acumuladas))
        
        # Graficar curva
        plt.plot(x_filtrado, y_filtrado, 
                color=color, linewidth=2, alpha=0.8, 
                label=f'Semilla {semilla} (Max: {ganancia_maxima:,.0f})')
        
        # Marcar punto de ganancia máxima
        plt.scatter(indice_maximo, ganancia_maxima, 
                   color=color, s=80, zorder=5, alpha=0.9)
        
        # Agregar línea vertical para el corte óptimo de esta semilla
        plt.axvline(x=indice_maximo, color=color, linestyle='--', alpha=0.5, linewidth=1)
    
    # Calcular la curva media
    # Primero, asegurar que todas las curvas tengan la misma longitud
    curvas_normalizadas = []
    for curva in todas_las_curvas:
        # Si la curva es más corta que max_len, rellenar con el último valor
        if len(curva) < max_len:
            padding = np.full(max_len - len(curva), curva[-1])
            curva_normalizada = np.concatenate([curva, padding])
        else:
            curva_normalizada = curva
        curvas_normalizadas.append(curva_normalizada)
    
    # Calcular la media de todas las curvas
    curva_media = np.mean(curvas_normalizadas, axis=0)
    
    # Encontrar el punto máximo de la curva media
    indice_maximo_media = np.argmax(curva_media)
    ganancia_maxima_media = curva_media[indice_maximo_media]
    
    # Filtrar la curva media para mostrar solo los puntos relevantes
    indices_filtrados_media = curva_media >= umbral_ganancia
    x_filtrado_media = np.where(indices_filtrados_media)[0]
    y_filtrado_media = curva_media[indices_filtrados_media]
    
    # Graficar la curva media con un estilo destacado
    plt.plot(x_filtrado_media, y_filtrado_media, 
            color='black', linewidth=3.5, alpha=1.0, 
            label=f'Media (Max: {ganancia_maxima_media:,.0f} en {indice_maximo_media:,} clientes)')
    
    # Marcar el punto máximo de la curva media
    plt.scatter(indice_maximo_media, ganancia_maxima_media, 
               color='red', s=120, zorder=10, alpha=1.0, marker='*')
    
    # Agregar línea vertical para el corte óptimo de la curva media
    plt.axvline(x=indice_maximo_media, color='red', linestyle='-', alpha=0.7, linewidth=1.5)
    
    # Agregar anotación con el número de clientes en el punto máximo
    plt.annotate(f'Máximo: {ganancia_maxima_media:,.0f}\n{indice_maximo_media:,} clientes', 
                xy=(indice_maximo_media, ganancia_maxima_media),
                xytext=(indice_maximo_media + max_len*0.05, ganancia_maxima_media),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    # Configurar etiquetas y título
    plt.xlabel('Clientes ordenados por probabilidad', fontsize=12)
    plt.ylabel('Ganancia Acumulada', fontsize=12)
    plt.title(f'Comparación de Ganancia Acumulada por Semilla - {STUDY_NAME}', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Formatear ejes
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_archivo = f"resultados/{STUDY_NAME}_grafico_comparativo_semillas_{timestamp}.jpg"
    
    plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calcular estadísticas comparativas
    ganancias_maximas = [resultado['ganancia_maxima'] for resultado in resultados_por_semilla]
    indices_maximos = [resultado['indice_maximo'] for resultado in resultados_por_semilla]
    
    logger.info(f"Estadísticas comparativas del gráfico:")
    logger.info(f"  - Ganancia máxima global: {ganancia_maxima_global:,.0f}")
    logger.info(f"  - Ganancia máxima promedio: {np.mean(ganancias_maximas):,.0f}")
    logger.info(f"  - Desviación estándar de ganancias: {np.std(ganancias_maximas):,.0f}")
    logger.info(f"  - Corte óptimo promedio: {np.mean(indices_maximos):,.0f} clientes")
    logger.info(f"  - Desviación estándar de cortes: {np.std(indices_maximos):,.0f} clientes")
    logger.info(f"  - Umbral de filtrado: {umbral_ganancia:,.0f} (66% del máximo global)")
    logger.info(f"  - CURVA MEDIA: Ganancia máxima: {ganancia_maxima_media:,.0f} en {indice_maximo_media:,} clientes")
    

    for resultado in resultados_por_semilla:
        logger.info(f"  - Semilla {resultado['semilla']}: Max={resultado['ganancia_maxima']:,.0f}, Corte={resultado['indice_maximo']:,}")
        mlflow.log_metric("ganancia_maxima", resultado['ganancia_maxima'], step=resultado['semilla'])
    
    logger.info(f"Gráfico comparativo guardado en: {ruta_archivo}")
    
    return ruta_archivo


def generar_grafico_test_completo(df: pd.DataFrame, mejores_params: dict = None, tiradas: int = 20, undersampling: float = 1.0, archivo_json: str = None) -> str:
    """
    Función principal que genera el gráfico de test con múltiples entrenamientos usando diferentes semillas.
    
    Args:
        df: DataFrame con todos los datos
        mejores_params: Diccionario con los mejores hiperparámetros (opcional)
        tiradas: Número de semillas/entrenamientos a realizar
        undersampling: Ratio de undersampling para entrenamiento
        archivo_json: Ruta completa al archivo JSON de iteraciones (opcional, si mejores_params es None)
    
    Returns:
        str: Ruta del gráfico generado
    """
    logger.info("=== INICIANDO GENERACIÓN DE GRÁFICO DE TEST CON MÚLTIPLES SEMILLAS ===")
    
    # Cargar mejores hiperparámetros si no se proporcionaron
    if mejores_params is None:
        if archivo_json is not None:
            mejores_params = cargar_mejores_hiperparametros(archivo_json=archivo_json)
        else:
            mejores_params = cargar_mejores_hiperparametros()

    
    # Obtener datos de test (comunes para todas las semillas)
    # Crear copia para evitar modificar el DataFrame original
    df_copy = df.copy()
    df_test = convertir_clase_ternaria_a_target(df_copy, baja_2_1=False)
    df_test = df_test[df_test['foto_mes'] == MES_TEST]
    y_true = df_test['clase_ternaria'].values
    
    # Lista para almacenar resultados de cada semilla
    resultados_por_semilla = []
    colores = ['blue', 'red', 'green', 'orange', 'purple']

    semillas = _generar_semillas_primas(tiradas, SEMILLA[0])

    logger.info(f"Parametros encontrados: {mejores_params}")

    # Realizar 5 entrenamientos con diferentes semillas
    for i, semilla in enumerate(semillas):  # Usar las primeras 5 semillas
        logger.info(f"Entrenando con semilla {semilla} ({i+1}/{tiradas})")
        
        # Obtener predicciones para esta semilla
        ganancia_test, y_pred_proba = evaluar_en_test(
            df=df,
            mejores_params=mejores_params,
            undersampling=undersampling,
            semilla=semilla
        )
        logger.info(f"Ganancia en test de grafico: {ganancia_test:,.0f}")
        # Calcular ganancia acumulada usando el retorno de evaluar_en_test
        # No recalcular, usar directamente ganancia_test
        _, ganancias_acumuladas = calcular_ganancia(y_pred=y_pred_proba, y_true=y_true)
        
        # Encontrar la ganancia máxima y su índice
        ganancia_maxima = np.max(ganancias_acumuladas)
        indice_maximo = np.argmax(ganancias_acumuladas)
        
        # Almacenar resultados
        resultados_por_semilla.append({
            'semilla': semilla,
            'y_pred_proba': y_pred_proba,
            'ganancias_acumuladas': ganancias_acumuladas,
            'ganancia_maxima': ganancia_maxima,
            'indice_maximo': indice_maximo,
            'color': colores[i % len(colores)]
        })
        
        logger.info(f"  - Ganancia máxima con semilla {semilla}: {ganancia_maxima:,.0f}")
    
    # Crear gráfico comparativo
    ruta_grafico = crear_grafico_comparativo_multiple_semillas(y_true, resultados_por_semilla)
    
    logger.info("=== GRÁFICO DE TEST CON MÚLTIPLES SEMILLAS COMPLETADO ===")
    
    return ruta_grafico
