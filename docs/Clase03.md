# 📘 Clase03.md

## **Optimización de hiperparámetros con Optuna**

---

## Git - Flujo de trabajo básico

Antes de comenzar con la clase, es importante mantener nuestro repositorio actualizado y sincronizado. Ejecuta los siguientes comandos en la consola:

```bash
git status
git add .
git commit -m "actualizacion"
git pull origin
git status
git add .
git push
```

**Al finalizar la clase, ejecuta nuevamente:**

```bash
git add .
git commit -m "Clase03: Optimización de hiperparámetros con Optuna completada"
git push
```

---

## 🎯 Objetivo

Implementar **optimización bayesiana de hiperparámetros** usando Optuna con función de ganancia personalizada y configuración específica para la competencia.

---

## 📑 Índice de la clase

1. **Configuración específica del proyecto** - Definir parámetros fijos: semillas, períodos y función de ganancia
2. **Manejo de Archivos de Configuracion** - Uso de Archivos Yaml.
3. **Conversión de target binario** - Transformar clase_ternaria a formato binario para optimización
4. **Función de ganancia personalizada** - Implementar la métrica específica de la competencia
5. **Optimización con Optuna** - Configurar búsqueda bayesiana de hiperparámetros
6. **Análisis de resultados** - Evaluar y guardar los mejores parámetros encontrados en un archivo Json

---

## 1) Configuración específica del proyecto

Para garantizar reproducibilidad y usar la configuración correcta de la competencia, presentamos **tres enfoques progresivos** de manejo de configuración:

### **Enfoque A: Configuración directa en main.py**

La forma más simple para proyectos pequeños:

```python
# main.py
import logging
from datetime import datetime

# =====================
# Configuración global
# =====================
SEMILLA = 811157
MES_TRAIN = 202102
MES_VALIDACION = 202103
MES_TEST = 202104
GANANCIA_ACIERTO = 780000
COSTO_ESTIMULO = 20000
# =====================

logger = logging.getLogger(__name__)

def main():
    logger.info(f"Entrenando con SEMILLA={SEMILLA}, TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}")
    # acá llamás a tus funciones de loader, features, train...
    ...
```

**Ventajas**: Simple, todo en un lugar
**Desventajas**: Difícil de reutilizar en otros módulos

### **Enfoque B: Archivo config.py separado**

Mejor organización para proyectos medianos:

```python
# src/config.py
SEMILLA = 811157
MES_TRAIN = 202102
MES_VALIDACION = 202103
MES_TEST = 202104
GANANCIA_ACIERTO = 780000
COSTO_ESTIMULO = 20000
```

```python
# main.py (o cualquier módulo)
from src.config import SEMILLA, MES_TRAIN, MES_VALIDACION, GANANCIA_ACIERTO

def main():
    logger.info(f"Entrenando con SEMILLA={SEMILLA}, TRAIN={MES_TRAIN}")
    # resto del código...
```

**Ventajas**: Reutilizable, centralizado
**Desventajas**: Valores hardcodeados, difícil cambiar sin modificar código

### **Enfoque C: Configuración con YAML (Recomendado para producción)**

Máxima flexibilidad para proyectos grandes:

**STUDY_NAME** es la variable que nos va a identificar la tirada, nos ayudara a identificar la corrida del script y los docs que almacena la app.

```yaml
# config.yaml
STUDY_NAME : "lgb_optimization_competencia01"

competencia01:
    DATA_PATH: "../data/competencia_01.csv"
    SEMILLA: [12,13,14,15,16]
    MES_TRAIN: "202102"
    MES_VALIDACION: "202103"
    MES_TEST: "202104"
    GANANCIA_ACIERTO: 780000
    COSTO_ESTIMULO: 20000
```

```python
# src/config.py
import yaml
import os
import logging

logger = logging.getLogger(__name__)

#Ruta del archivo de configuracion
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf.yaml")

try:
    with open(PATH_CONFIG, "r") as f:
        _cfgGeneral = yaml.safe_load(f)
        _cfg = _cfgGeneral["competencia01"]

        STUDY_NAME = _cfgGeneral.get("STUDY_NAME", "Wendsday")
        DATA_PATH = _cfg.get("DATA_PATH", "../data/competencia.csv")
        SEMILLA = _cfg.get("SEMILLA", [42])
        MES_TRAIN = _cfg.get("MES_TRAIN", "202102")
        MES_VALIDACION = _cfg.get("MES_VALIDACION", "202103")
        MES_TEST = _cfg.get("MES_TEST", "202104")
        GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO", None)
        COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO", None)

except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise

```

**Ventajas**:

- Cambios sin modificar código
- Múltiples configuraciones (dev, test, prod)
- Fácil versionado de configuraciones
- Ideal para equipos de trabajo

**¿Por qué estos valores específicos?**

- **Semilla fija**: Garantiza reproducibilidad en todos los experimentos
- **Períodos específicos**: Train en 202102, validación en 202103 para Optuna, test en 202104
- **Función de ganancia**: Valores reales de la competencia para optimización correcta

**Recomendación**: Usar enfoque A para aprender, B para proyectos personales, C para producción.

---

## 2) Conversión de target binario

La optimización requiere convertir `clase_ternaria` a formato binario:
Aca esta en cada uno la decision que toma, creo lo mas salno es probar incluyendo los BAJA+1 como target igual a CONTINUA y luego como los BAJA+2

Esto lo incluimos en loader.py; dado que es un procesamiento minimo, la decicion de hacer un nuevo script o incluirlo a otro diferente es totalmente valido.

```python
def convertir_clase_ternaria_a_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte clase_ternaria a target binario reemplazando en el mismo atributo:
    - CONTINUA = 0
    - BAJA+1 y BAJA+2 = 1

    Args:
        df: DataFrame con columna 'clase_ternaria'

    Returns:
        pd.DataFrame: DataFrame con clase_ternaria convertida a valores binarios (0, 1)
    """
    # Crear copia del DataFrame para no modificar el original
    df_result = df.copy()

    # Contar valores originales para logging
    n_continua_orig = (df_result['clase_ternaria'] == 'CONTINUA').sum()
    n_baja1_orig = (df_result['clase_ternaria'] == 'BAJA+1').sum()
    n_baja2_orig = (df_result['clase_ternaria'] == 'BAJA+2').sum()

    # Convertir clase_ternaria a binario en el mismo atributo
    df_result['clase_ternaria'] = df_result['clase_ternaria'].map({
        'CONTINUA': 0,
        'BAJA+1': 1,
        'BAJA+2': 1
    })

    # Log de la conversión
    n_ceros = (df_result['clase_ternaria'] == 0).sum()
    n_unos = (df_result['clase_ternaria'] == 1).sum()

    logger.info(f"Conversión completada:")
    logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"  Binario - 0: {n_ceros}, 1: {n_unos}")
    logger.info(f"  Distribución: {n_unos/(n_ceros + n_unos)*100:.2f}% casos positivos")

    return df_result
```

---

## 3) Función de ganancia personalizada

Implementamos la función de ganancia específica de la competencia usando configuración YAML:

Cada una de estos metodos tienen 3 ó 4 lineas de codigo importantes, que ya se vieron en los notebook, estudienla y sientanse libres de modificar, incluso se recomienda agregar otras funciones de ganancias que se ven en al catedra.

```python
# src/gain_function.py
import numpy as np
import pandas as pd
from .config import GANANCIA_ACIERTO, COSTO_ESTIMULO
import logging

logger = logging.getLogger(__name__)

def calcular_ganancia(y_true, y_pred):
    """
    Calcula la ganancia total usando la función de ganancia de la competencia.

    Args:
        y_true: Valores reales (0 o 1)
        y_pred: Predicciones (0 o 1)

    Returns:
        float: Ganancia total
    """
    # Convertir a numpy arrays si es necesario
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Calcular ganancia vectorizada usando configuración
    # Verdaderos positivos: y_true=1 y y_pred=1 -> ganancia
    # Falsos positivos: y_true=0 y y_pred=1 -> costo
    # Verdaderos negativos y falsos negativos: ganancia = 0

    ganancia_total = np.sum(
        (y_true == 1) & (y_pred == 1) * GANANCIA_ACIERTO +  # TP
        (y_true == 0) & (y_pred == 1) * (-COSTO_ESTIMULO)   # FP
    )

    logger.debug(f"Ganancia calculada: {ganancia_total:,.0f} "
                f"(GANANCIA_ACIERTO={GANANCIA_ACIERTO}, COSTO_ESTIMULO={COSTO_ESTIMULO})")

    return ganancia_total

def ganancia_lgb_binary(y_pred, y_true):
    """
    Función de ganancia para LightGBM en clasificación binaria.
    Compatible con callbacks de LightGBM.

    Args:
        y_pred: Predicciones de probabilidad del modelo
        y_true: Dataset de LightGBM con labels verdaderos

    Returns:
        tuple: (eval_name, eval_result, is_higher_better)
    """
    # Obtener labels verdaderos
    y_true_labels = y_true.get_label()

    # Convertir probabilidades a predicciones binarias (umbral 0.5)
    y_pred_binary = (y_pred > 0.025).astype(int)

    # Calcular ganancia usando configuración
    ganancia_total = calcular_ganancia(y_true_labels, y_pred_binary)

    # Retornar en formato esperado por LightGBM
    return 'ganancia', ganancia_total, True  # True = higher is better
```

---

## 4) Optimización con Optuna

Aca tendremos la Optimizacion propiamente dicha, la funcion de ganancia a optimizr por cada iteracion y un metodo extra apra guardar los resultados de cada iteracion.

### Función objetivo para Optuna

```python
# src/optimization.py
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from .gain_function import calcular_ganancia, ganancia_lgb_binary

logger = logging.getLogger(__name__)

def objetivo_ganancia(trial, df) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos

    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON

    Returns:
    float: ganancia total
    """
    # Hiperparámetros a optimizar
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada

	#completar a gusto!!!!!!!

        'random_state': SEMILLA,  # Desde configuración YAML
    }

    # Completar!!!!!!

    ganancia_total = calcular_ganancia(y_val, y_pred_binary)

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_total)

    logger.debug(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")

    return ganancia_total
```

### Guardar Iteracion

```python
def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.

    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME

    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_iteraciones.json"

    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
            'semilla': SEMILLA,
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VALIDACION
        }
    }

    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []

    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)

    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)

    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f}" + "---" + "Parámetros: {params}")
```

### Optimizar

```python
def optimizar(df, n_trials=100) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)

    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado.
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = STUDY_NAME

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")

    # Completar!!!!!!!!

    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")


    return study
```

---

## 6) Main.py integrado

Observemos un par de cosas que se agregaron:
-STUDY_NAME a la creacion del log.

-Uso de variables globales

-Llamamos. al metodo optimizar() con 100 iteraciones y solo le pasamos el df, todo el resto va por parametros globales del Yaml.

-Como salida final las mejores iteraciones de la bayeciana.

```python
# main.py
import logging
from datetime import datetime
import os
import pandas as pd

from src.features import feature_engineering_lag
from src.loader import cargar_datos, convertir_clase_ternaria_a_target
from src.optimization import optimizar
from src.config import *

### Configuración de logging ###
os.makedirs("logs", exist_ok=True)
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_{STUDY_NAME}_{fecha}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("logs/" + nombre_log),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Iniciando programa de optimización con log fechado")

### Manejo de Configuración en YAML ###
logger.info("Configuración cargada desde YAML")
logger.info(f"STUDY_NAME: {STUDY_NAME}")
logger.info(f"DATA_PATH: {DATA_PATH}")
logger.info(f"SEMILLA: {SEMILLA}")
logger.info(f"MES_TRAIN: {MES_TRAIN}")
logger.info(f"MES_VALIDACION: {MES_VALIDACION}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")


### Main ###
def main():
    """Pipeline principal con optimización usando configuración YAML."""
    logger.info("=== INICIANDO OPTIMIZACIÓN CON CONFIGURACIÓN YAML ===")

    # 1. Cargar datos
    df = cargar_datos(DATA_PATH)

    # 2. Feature Engineering
    atributos = ["mcuentas_saldo", "mtarjeta_visa_consumo", "cproductos"]
    cant_lag = 2
    df_fe = feature_engineering_lag(df, atributos, cant_lag)
    logger.info(f"Feature Engineering completado: {df_fe.shape}")

    # 3. Convertir clase_ternaria a binario
    df_fe = convertir_clase_ternaria_a_target(df_fe)

    # 4. Ejecutar optimización (función simple)
    study = optimizar(df_fe, n_trials=100)

    # 5. Análisis adicional
    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")

    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

if __name__ == "__main__":
    main()
```

---

## 7) Análisis de resultados notebook Jupyter

Este codigo sencillo permite realizar un grafico burdo de las iteraciones de la beyeciana, mejoren lo y tambien trabajen en agregarle todas las tiradas con labels segun el nombre del archivo que es el STUDY_NAME.


Tambien mostrar en un grafico o tabla el promedio de las mejores 5 tiradas de cada corrida. (nota: corrida == python main.py)

Pueden mostar mas datos o cargar mas datos si asi lo desean desde el metodo guardar_itreacion()

Luego Usaremos una db, y podran mejorar esta seccion si asi lo desean.

```python
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_json("resultados/lgb_optimization_competencia01_iteraciones.json")
print(df.head())

#Graficar ganancia Optuna eje x trial_number eje y ganancia
# Configurar el estilo
plt.style.use('seaborn-v0_8')
plt.figure(figsize=(12, 6))

# Crear el gráfico
sns.lineplot(data=df, x='trial_number', y='value', marker='o', linewidth=2, markersize=6)

# Personalizar el gráfico
plt.title('Evolución de la Ganancia por Iteración de Optuna', fontsize=16, fontweight='bold')
plt.xlabel('Número de Trial', fontsize=12)
plt.ylabel('Ganancia', fontsize=12)
plt.grid(True, alpha=0.3)

# Formatear el eje y para mostrar valores en millones
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

# Ajustar layout
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Opcional: Mostrar estadísticas básicas
print(f"\nEstadísticas de ganancia:")
print(f"Ganancia máxima: {df['value'].max():,.0f}")
print(f"Ganancia mínima: {df['value'].min():,.0f}")
print(f"Ganancia promedio: {df['value'].mean():,.0f}")
print(f"Trial con mejor ganancia: {df.loc[df['value'].idxmax(), 'trial_number']}")
```

---

📦 **Dependencias necesarias**:

```bash
pip install pandas lightgbm optuna duckdb matplotlib seaborn
```

---

## ✅ Resultado esperado

Al final de la clase cada alumno tendrá:

* Función de conversión de `clase_ternaria` a target binario
* Función de ganancia personalizada de la competencia
* Optimización bayesiana funcionando con Optuna (Casi!)
* Configuración específica: semilla fija y períodos correctos
* Análisis de resultados y mejores parámetros guardados
* Pipeline completo integrado en `main.py`
