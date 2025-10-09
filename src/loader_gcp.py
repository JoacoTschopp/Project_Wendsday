import pandas as pd
import numpy as np
import logging
from google.cloud import storage
from io import StringIO, BytesIO
import os
from typing import Optional, Union

# Configura el logging
logger = logging.getLogger(__name__)

# Configura el cliente de GCS para usar las credenciales de la instancia
try:
    from google.auth import compute_engine
    from google.auth.exceptions import DefaultCredentialsError
    
    def get_gcs_client():
        """
        Crea y devuelve un cliente de Google Cloud Storage.
        En una instancia de GCP, usa automáticamente las credenciales de la instancia.
        """
        try:
            # Intenta usar las credenciales de la instancia primero
            credentials, project = compute_engine.Credentials(), None
            return storage.Client(credentials=credentials, project=project)
        except DefaultCredentialsError:
            # Si falla, usa el método por defecto (útil para desarrollo local)
            logger.warning("Usando credenciales por defecto. Asegúrate de tener configurado el servicio de cuentas de GCP.")
            return storage.Client()

except ImportError:
    logger.warning("No se pudo importar google.auth. Asegúrate de tener instalado google-auth.")
    logger.info("Ejecuta: pip install google-auth")
    
    def get_gcs_client():
        """
        Versión alternativa en caso de que no esté disponible google.auth
        """
        return storage.Client()

def get_gcs_client():
    """
    Crea y devuelve un cliente de Google Cloud Storage.
    Asegúrate de tener configuradas las credenciales de GCP.
    Las credenciales pueden configurarse mediante la variable de entorno GOOGLE_APPLICATION_CREDENTIALS
    que debe apuntar al archivo JSON de credenciales de servicio.
    """
    return storage.Client()

def cargar_datos(bucket_name: str, blob_name: str, use_binary: bool = False) -> pd.DataFrame:
    """
    Carga un archivo CSV desde un bucket de Google Cloud Storage.
    
    Args:
        bucket_name (str): Nombre del bucket de GCS
        blob_name (str): Ruta completa al archivo dentro del bucket
        use_binary (bool): Si es True, usa descarga binaria (útil para archivos grandes)
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
        
    Raises:
        Exception: Si ocurre un error al cargar los datos
    """
    logger.info(f"Cargando dataset desde gs://{bucket_name}/{blob_name}")
    
    try:
        # Inicializar el cliente de GCS
        client = get_gcs_client()
        
        # Obtener el bucket y el blob
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Verificar si el archivo existe
        if not blob.exists():
            raise FileNotFoundError(f"El archivo gs://{bucket_name}/{blob_name} no existe")
        
        # Opción para manejar archivos grandes con descarga binaria
        if use_binary or blob.size > 10 * 1024 * 1024:  # > 10MB
            logger.info(f"Descargando archivo grande ({blob.size/1024/1024:.2f} MB) en modo binario...")
            content = blob.download_as_bytes()
            data = BytesIO(content)
            df = pd.read_csv(data)
        else:
            # Para archivos pequeños, usar descarga de texto
            content = blob.download_as_text()
            data = StringIO(content)
            df = pd.read_csv(data)
        
        logger.info(f"Dataset cargado exitosamente. Tamaño: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error al cargar el dataset desde GCS: {e}")
        # Intenta dar más información sobre el error
        if '403' in str(e):
            logger.error("Error 403: Permiso denegado. Verifica los permisos de la instancia.")
            logger.error("Asegúrate de que la instancia tenga el rol 'Storage Object Viewer' o superior.")
        raise

def convertir_clase_ternaria_a_target(df: pd.DataFrame, baja_2_1: bool = True) -> pd.DataFrame:
    """
    Convierte clase_ternaria a target binario reemplazando en el mismo atributo:
    - CONTINUA = 0
    y según los argumentos baja_2_1
    baja_2_1 = true entonces: BAJA+1 y BAJA+2 = 1
    baja_2_1 = false entonces: BAJA+1 = 0 y BAJA+2 = 1
    
    Args:
        df: DataFrame con columna 'clase_ternaria'
        baja_2_1: Booleano que indica si se considera BAJA+1 como positivo
    
    Returns:
        pd.DataFrame: DataFrame con clase_ternaria convertida a valores binarios (0, 1)
    """
    logger.info("Convirtiendo clase_ternaria a target binario")
    
    # Contar valores originales para logging (antes de modificar)
    n_continua_orig = (df['clase_ternaria'] == 'CONTINUA').sum()
    n_baja1_orig = (df['clase_ternaria'] == 'BAJA+1').sum()
    n_baja2_orig = (df['clase_ternaria'] == 'BAJA+2').sum()
    
    # Modificar el DataFrame usando .loc para evitar SettingWithCopyWarning
    if baja_2_1:
        # Convertir clase_ternaria a binario usando numpy.where para mejorar rendimiento
        df.loc[:, 'clase_ternaria'] = np.where(df['clase_ternaria'] == 'CONTINUA', 0, 1)
    else:
        # Convertir BAJA+2 a 1, todo lo demás a 0
        df.loc[:, 'clase_ternaria'] = np.where(df['clase_ternaria'] == 'BAJA+2', 1, 0)
        
    # Log de la conversión
    n_ceros = (df['clase_ternaria'] == 0).sum()
    n_unos = (df['clase_ternaria'] == 1).sum()
    
    logger.info(f"Conversión completada:")
    logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"  Binario - 0: {n_ceros}, 1: {n_unos}")
    logger.info(f"  Distribución: {n_unos/(n_ceros + n_unos)*100:.2f}% casos positivos")

    return df

def verificar_permisos_gcs(bucket_name: str) -> None:
    """
    Verifica los permisos de la instancia en el bucket de GCS.
    
    Args:
        bucket_name: Nombre del bucket a verificar
    """
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        
        # Intenta listar los blobs (requiere permisos de lectura)
        blobs = list(client.list_blobs(bucket_name, max_results=1))
        logger.info(f"Permisos de lectura verificados correctamente en el bucket: {bucket_name}")
        
        # Verifica si el bucket existe
        if not bucket.exists():
            logger.warning(f"El bucket {bucket_name} no existe o no tienes permisos para acceder a él.")
            return
            
        logger.info(f"El bucket {bucket_name} existe y es accesible.")
        
    except Exception as e:
        logger.error(f"Error al verificar permisos: {e}")
        if '403' in str(e):
            logger.error("Error 403: Permiso denegado. La instancia no tiene permisos para acceder al bucket.")
            logger.error("Solución: Agrega el rol 'Storage Object Viewer' a la cuenta de servicio de la instancia.")
