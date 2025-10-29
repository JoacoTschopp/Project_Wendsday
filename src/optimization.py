import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from .gain_function import calcular_ganancia, ganancia_lgb_binary, ganancia_evaluator
from .loader import convertir_clase_ternaria_a_target

logger = logging.getLogger(__name__)

def objetivo_ganancia(trial: optuna.trial.Trial, df: pd.DataFrame, undersampling: float = 1) -> float:
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
        'num_leaves': trial.suggest_int('num_leaves', PARAMETROS_LGB['num_leaves'][0], PARAMETROS_LGB['num_leaves'][1]),
        'learning_rate': trial.suggest_float('learning_rate', PARAMETROS_LGB['learning_rate'][0], PARAMETROS_LGB['learning_rate'][1], log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', PARAMETROS_LGB['feature_fraction'][0], PARAMETROS_LGB['feature_fraction'][1]),
        'bagging_fraction': trial.suggest_float('bagging_fraction', PARAMETROS_LGB['bagging_fraction'][0], PARAMETROS_LGB['bagging_fraction'][1]),
        'min_child_samples': trial.suggest_int('min_child_samples', PARAMETROS_LGB['min_child_samples'][0], PARAMETROS_LGB['min_child_samples'][1]),
        'max_depth': trial.suggest_int('max_depth', PARAMETROS_LGB['max_depth'][0], PARAMETROS_LGB['max_depth'][1]),
        'reg_alpha': trial.suggest_float('reg_alpha', PARAMETROS_LGB['reg_alpha'][0], PARAMETROS_LGB['reg_alpha'][1]),
        'reg_lambda': trial.suggest_float('reg_lambda', PARAMETROS_LGB['reg_lambda'][0], PARAMETROS_LGB['reg_lambda'][1]),
        'min_gain_to_split': 0.0,  # Permitir splits con ganancia mínima
        'verbose': -1,  # Reducir verbosidad
        'verbosity': -1,  # Silenciar mensajes adicionales
        'silent': True,  # Modo silencioso
        'bin': 31,
        'random_state': SEMILLA[0],  # Desde configuración YAML
    }
  
    # Preparar datos usando configuración YAML
    # MES_TRAIN ahora puede ser una lista o un solo valor
    if isinstance(MES_TRAIN, list):
        df_train = df[df['foto_mes'].isin(MES_TRAIN)]
    else:
        df_train = df[df['foto_mes'] == MES_TRAIN]
    
    df_val = df[df['foto_mes'] == MES_VALIDACION]
    
    #Convierto a binaria la clase ternaria, 
    # para entrenar el modelo Baja+1 y Baja+2 == 1
    # y calcular la ganancia de validacion Baja+2 solamente en 1
    df_train = convertir_clase_ternaria_a_target(df_train, baja_2_1=True)
    df_val = convertir_clase_ternaria_a_target(df_val, baja_2_1=False)
    df_train['clase_ternaria'] = df_train['clase_ternaria'].astype(np.int8)
    df_val['clase_ternaria'] = df_val['clase_ternaria'].astype(np.int8)

    # Usar target (clase_ternaria ya convertida a binaria)
    y_train = df_train['clase_ternaria'].values
    y_val = df_val['clase_ternaria'].values
    
    # Features: usar todas las columnas excepto target
    X_train = df_train.drop(columns=['clase_ternaria'])
    X_val = df_val.drop(columns=['clase_ternaria'])
    
    # Entrenar modelo con función de ganancia personalizada
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    #####
    #ESTO NO ES OPTIMO, ENTRENAR UN SOLO MODELO!
    #####
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        feval=ganancia_evaluator,  # Función de ganancia personalizada
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Predecir y calcular ganancia
    y_pred_proba = model.predict(X_val)
    
    _, ganancia_total, _ = ganancia_evaluator(y_pred_proba, val_data)

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_total)
  
    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")
  
    return ganancia_total


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
    logger.info(f"Ganancia: {ganancia:,.0f} --- Parámetros: {trial.params}")

def crear_o_cargar_estudio(study_name: str = None, semilla: int = None) -> optuna.Study:
    """
    Crea un nuevo estudio de Optuna o carga uno existente desde SQLite.
    
    Args:
        study_name: Nombre del estudio (si es None, usa STUDY_NAME del config)
        semilla: Semilla para reproducibilidad
    
    Returns:
        optuna.Study: Estudio de Optuna (nuevo o cargado)
    """
    study_name = STUDY_NAME
    
    if semilla is None:
        semilla = SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA
    
    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "optuna_db")
    os.makedirs(path_db, exist_ok=True)
    
    # Ruta completa de la base de datos
    db_file = os.path.join(path_db, f"{study_name}.db")
    storage = f"sqlite:///{db_file}"
    
    # Verificar si existe un estudio previo
    if os.path.exists(db_file):
        logger.info(f"⚡ Base de datos encontrada: {db_file}")
        logger.info(f"🔄 Cargando estudio existente: {study_name}")
        
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            n_trials_previos = len(study.trials)
            
            logger.info(f"✅ Estudio cargado exitosamente")
            logger.info(f"📊 Trials previos: {n_trials_previos}")
            
            if n_trials_previos > 0:
                logger.info(f"🏆 Mejor ganancia hasta ahora: {study.best_value:,.0f}")
            
            return study
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar el estudio: {e}")
            logger.info(f"🆕 Creando nuevo estudio...")
    else:
        logger.info(f"🆕 No se encontró base de datos previa")
        logger.info(f"📁 Creando nueva base de datos: {db_file}")
    
    # Crear nuevo estudio
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=semilla),
        load_if_exists=True
    )
    
    logger.info(f"✅ Nuevo estudio creado: {study_name}")
    logger.info(f"💾 Storage: {storage}")
    
    return study

def optimizar(df: pd.DataFrame, n_trials: int, study_name: str = None, undersampling: float = 0.01) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
        undersampling: Undersampling para entrenamiento
    
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
    
    # Crear o cargar estudio desde DuckDB
    study = crear_o_cargar_estudio(study_name, SEMILLA)

    # Calcular cuántos trials faltan
    trials_previos = len(study.trials)
    trials_a_ejecutar = max(0, n_trials - trials_previos)
    
    if trials_previos > 0:
        logger.info(f"🔄 Retomando desde trial {trials_previos}")
        logger.info(f"📝 Trials a ejecutar: {trials_a_ejecutar} (total objetivo: {n_trials})")
    else:
        logger.info(f"🆕 Nueva optimización: {n_trials} trials")
    
    # Ejecutar optimización
    if trials_a_ejecutar > 0:
        study.optimize(lambda trial: objetivo_ganancia(trial, df, undersampling), n_trials=trials_a_ejecutar)
        logger.info(f"🏆 Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Mejores parámetros: {study.best_params}")
    else:
        logger.info(f"✅ Ya se completaron {n_trials} trials")
  
    return study

def aplicar_undersampling(df: pd.DataFrame, ratio: float, random_state: int = None) -> pd.DataFrame:
    pass