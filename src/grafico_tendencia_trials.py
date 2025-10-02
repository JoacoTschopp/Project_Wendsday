import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cargar_trials_desde_json(ruta_json: str) -> pd.DataFrame:
    """
    Carga un archivo JSON con una lista de trials y devuelve un DataFrame
    con las columnas: trial_number, value (ganancia) y datetime si existe.
    """
    with open(ruta_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("El JSON debe contener una lista de objetos de trials.")

    # Extraer campos relevantes con defaults seguros
    rows: List[Dict] = []
    for item in data:
        trial = {
            'trial_number': item.get('trial_number'),
            'value': item.get('value'),
            'datetime': item.get('datetime'),
            'state': item.get('state')
        }
        # Filtrar nulos o trials sin número/valor
        if trial['trial_number'] is None or trial['value'] is None:
            continue
        rows.append(trial)

    df = pd.DataFrame(rows)

    # Ordenar por numero de trial por si no viene ordenado
    df = df.sort_values('trial_number').reset_index(drop=True)

    # Mantener solo trials completos si hay columna state
    if 'state' in df.columns:
        df = df[df['state'].isin(['COMPLETE', 'Pruned', 'COMPLETE'.lower(), 'complete']) | df['state'].isna()].copy()

    return df


def generar_grafico_tendencia(
    df: pd.DataFrame,
    titulo: str,
    ruta_salida: str,
    ventana_media_movil: int = 5,
    mostrar_puntos: bool = True,
) -> str:
    """
    Genera un gráfico de tendencia (trial vs ganancia) y lo guarda en ruta_salida.
    Solo conecta puntos que representan mejoras incrementales.
    Devuelve la ruta del archivo guardado.
    """
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    x = df['trial_number'].to_numpy()
    y = df['value'].to_numpy(dtype=float)

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Mostrar todos los puntos como puntos individuales
    if mostrar_puntos:
        ax.scatter(x, y, color='#1f77b4', s=20, alpha=0.6, label='Todos los trials')

    # Encontrar puntos de mejora incremental (solo conectar puntos superiores)
    puntos_x_tendencia = [x[0]]
    puntos_y_tendencia = [y[0]]
    
    mejor_valor_actual = y[0]
    mejor_trial_actual = x[0]
    
    for i in range(1, len(y)):
        if y[i] > mejor_valor_actual:
            puntos_x_tendencia.append(x[i])
            puntos_y_tendencia.append(y[i])
            mejor_valor_actual = y[i]
            mejor_trial_actual = x[i]

    # Conectar solo los puntos de mejora incremental
    ax.plot(puntos_x_tendencia, puntos_y_tendencia, color='#2ca02c', linewidth=3, 
            label='Tendencia de mejora incremental', zorder=4)

    # Resaltar los puntos de mejora incremental
    ax.scatter(puntos_x_tendencia, puntos_y_tendencia, color='#2ca02c', s=40, 
               alpha=0.8, zorder=5, edgecolors='darkgreen', linewidth=1)

    # Media móvil opcional
    if ventana_media_movil and ventana_media_movil > 1:
        y_ma = pd.Series(y).rolling(window=ventana_media_movil, min_periods=1).mean().to_numpy()
        ax.plot(x, y_ma, color='#ff7f0e', linewidth=2, linestyle='--', label=f'Media móvil ({ventana_media_movil})')

    # Mejores y peores puntos
    idx_max = int(np.argmax(y))
    idx_min = int(np.argmin(y))
    ax.scatter([x[idx_max]], [y[idx_max]], color='green', s=80, zorder=6, label=f'Máximo: {y[idx_max]:,.0f}')
    ax.scatter([x[idx_min]], [y[idx_min]], color='red', s=60, zorder=5, label=f'Mínimo: {y[idx_min]:,.0f}')

    # Formato ejes y título
    ax.set_xlabel('Trial')
    ax.set_ylabel('Ganancia')
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f'{v:,.0f}'))

    # Leyenda
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return ruta_salida


def nombre_salida_por_defecto(ruta_json: str) -> str:
    base_resultados = os.path.join(os.path.dirname(os.path.dirname(ruta_json)), 'resultados')
    os.makedirs(base_resultados, exist_ok=True)

    base = os.path.splitext(os.path.basename(ruta_json))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(base_resultados, f"tendencia_{base}_{timestamp}.png")


def main():
    parser = argparse.ArgumentParser(description='Generar gráfico de tendencia (trial vs ganancia) desde un JSON de resultados.')
    parser.add_argument('--json', type=str, default=r'c:\\Users\\tschoppj\\proyectos_maestria\\Project_Wendsday\\resultados\\Prueba_completa_BO150_cv_iteraciones.json',
                        help='Ruta al archivo JSON con los trials. Por defecto usa el archivo de resultados actual.')
    parser.add_argument('--salida', type=str, default=None,
                        help='Ruta de salida del PNG/JPG. Si no se especifica, se genera automáticamente en la carpeta resultados/.')
    parser.add_argument('--ventana_ma', type=int, default=5, help='Ventana de media móvil (>=2 para habilitar).')
    args = parser.parse_args()

    ruta_json = args.json
    if not os.path.isfile(ruta_json):
        raise FileNotFoundError(f"No se encuentra el archivo JSON: {ruta_json}")

    df = cargar_trials_desde_json(ruta_json)
    if df.empty:
        raise ValueError('No se encontraron trials válidos con número y valor en el JSON.')

    titulo = 'Tendencia de ganancia por trial'
    ruta_salida = args.salida or nombre_salida_por_defecto(ruta_json)

    salida = generar_grafico_tendencia(
        df=df,
        titulo=titulo,
        ruta_salida=ruta_salida,
        ventana_media_movil=max(1, int(args.ventana_ma)),
        mostrar_puntos=True,
    )

    print(f"Gráfico guardado en: {salida}")


if __name__ == '__main__':
    main()
