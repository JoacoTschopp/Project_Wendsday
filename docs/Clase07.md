# 📘 Clase07.md

## **Infraestructura y Arquitectura de ML: Docker, POO y MLOps**

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
git commit -m "Clase07: Docker, POO y MLOps completada"
git push
```

---

## 🎯 Objetivo

Comprender los **fundamentos de infraestructura moderna** para proyectos de Machine Learning: **contenedorización con Docker**, **diseño orientado a objetos** para pipelines escalables, y **MLOps** con tracking de experimentos en MongoDB.

---

## 📑 Índice de la clase

1. **Docker: Fundamentos y Casos de Uso** - Contenedorización para reproducibilidad
2. **Dockerfile y docker-compose.yml** - Construcción de imágenes y orquestación de servicios
3. **Programación Orientada a Objetos en ML** - Diseño de pipelines modulares y reutilizables
4. **MLOps: Conceptos y Herramientas** - Tracking de experimentos y gestión del ciclo de vida
5. **Implementación Práctica: MLflow + MongoDB** - Registro de experimentos en base de datos local

---

## 1) Docker: Fundamentos y Casos de Uso

### **¿Qué es Docker?**

Docker es una **plataforma de contenedorización** que permite empaquetar aplicaciones y todas sus dependencias en contenedores ligeros, portables y aislados.

### **Conceptos Clave**

- **Contenedor**: Instancia ejecutable de una imagen que corre de forma aislada
- **Imagen**: Plantilla inmutable con el código, runtime, bibliotecas y dependencias
- **Dockerfile**: Archivo de texto con instrucciones para construir una imagen
- **Docker Compose**: Herramienta para definir y ejecutar aplicaciones multi-contenedor
- **Volumen**: Mecanismo para persistir datos fuera del contenedor

### **Beneficios en Proyectos de ML**

```
┌─────────────────────────────────────────────────────────────┐
│                    BENEFICIOS DE DOCKER                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  📦 REPRODUCIBILIDAD                                         │
│  └─ Mismo entorno en desarrollo, testing y producción       │
│                                                               │
│  🔒 AISLAMIENTO                                              │
│  └─ Dependencias encapsuladas, sin conflictos               │
│                                                               │
│  🚀 PORTABILIDAD                                             │
│  └─ Funciona en cualquier sistema (Linux, Mac, Windows)     │
│                                                               │
│  ⚡ EFICIENCIA                                               │
│  └─ Más ligero que VMs, comparte kernel del SO              │
│                                                               │
│  🔄 ESCALABILIDAD                                            │
│  └─ Fácil replicación y orquestación con Kubernetes         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### **Arquitectura de Docker**

```
┌────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA DOCKER                      │
└────────────────────────────────────────────────────────────┘

    USUARIO
       │
       ▼
  ┌─────────┐
  │ Docker  │  ◄─── docker build, docker run, docker-compose
  │  CLI    │
  └────┬────┘
       │
       ▼
  ┌─────────┐
  │ Docker  │  ◄─── Gestiona imágenes, contenedores, redes
  │ Daemon  │
  └────┬────┘
       │
       ├──────────┬──────────┬──────────┐
       ▼          ▼          ▼          ▼
   ┌────────-┐ ┌────-────┐ ┌───-─────┐ ┌────────┐
   │Container│ │Container│ │Container│ │ Image  │
   │   #1    │ │   #2    │ │   #3    │ │Registry│
   └──────-──┘ └───────-─┘ └────────-┘ └────────┘
       │          │            │
       └──────────┴──────--────┘
                 │
                 ▼
         ┌──────────────┐
         │    HOST OS    │
         └──────────────┘
```

### **Casos de Uso en ML**

1. **Desarrollo Local**: Entorno reproducible para experimentación
2. **CI/CD**: Integración y despliegue continuo de modelos
3. **Producción**: Servir modelos con APIs (Flask, FastAPI)

## 2) Dockerfile y docker-compose.yml

### **Dockerfile: Construcción de Imágenes**

Un Dockerfile define paso a paso cómo construir una imagen Docker.

#### **Ejemplo: Dockerfile para Proyecto de ML**

```dockerfile
# Imagen base con Python 3.11
FROM python:3.11-slim

# Información del mantenedor
LABEL maintainer="tu-email@example.com"
LABEL description="Proyecto de Machine Learning con LightGBM"

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivo de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código del proyecto
COPY . .

# Crear directorios necesarios
RUN mkdir -p logs data resultados predict

# Exponer puerto para API (si aplica)
EXPOSE 8000

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV STUDY_NAME="ml_project"

# Comando por defecto
CMD ["python", "main.py"]
```

#### **Instrucciones Principales de Dockerfile**

```
┌──────────────────────────────────────────────────────────────┐
│              INSTRUCCIONES PRINCIPALES                        │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  FROM      → Imagen base (python:3.11, ubuntu:22.04, etc)    │
│  WORKDIR   → Directorio de trabajo dentro del contenedor     │
│  COPY      → Copiar archivos del host al contenedor          │
│  RUN       → Ejecutar comandos durante la construcción       │
│  CMD       → Comando por defecto al iniciar contenedor       │
│  ENTRYPOINT→ Punto de entrada configurable                   │
│  ENV       → Variables de entorno                             │
│  EXPOSE    → Puertos que expone el contenedor                │
│  VOLUME    → Puntos de montaje para persistencia             │
│  ARG       → Variables de construcción (build-time)          │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

#### **Construir y Ejecutar**

```bash
# Construir imagen
docker build -t ml-project:v1.0 .

# Ejecutar contenedor
docker run -v $(pwd)/data:/app/data ml-project:v1.0

# Ejecutar con variables de entorno
docker run -e STUDY_NAME="experimento_01" ml-project:v1.0

# Modo interactivo
docker run -it ml-project:v1.0 bash
```

### **docker-compose.yml: Orquestación de Servicios**

Docker Compose permite definir y ejecutar aplicaciones multi-contenedor.

#### **Ejemplo: docker-compose.yml para ML + MongoDB + MLflow**

```yaml
version: '3.8'

services:
  # MongoDB para tracking de experimentos
  mongodb:
    image: mongo:7.0
    container_name: ml_mongodb
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: admin123
    volumes:
      - mongodb_data:/data/db
      - ./mongo-init:/docker-entrypoint-initdb.d
    networks:
      - ml_network
    restart: unless-stopped

  # MLflow Tracking Server
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.2
    container_name: ml_mlflow
    ports:
      - "5000:5000"
    environment:
      MLFLOW_BACKEND_STORE_URI: mongodb://admin:admin123@mongodb:27017/mlflow?authSource=admin
      MLFLOW_DEFAULT_ARTIFACT_ROOT: /mlflow/artifacts
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    depends_on:
      - mongodb
    networks:
      - ml_network
    command: >
      mlflow server
      --backend-store-uri mongodb://admin:admin123@mongodb:27017/mlflow?authSource=admin
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
    restart: unless-stopped

  # Aplicación de ML
  ml_app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ml_training
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./resultados:/app/resultados
      - ./predict:/app/predict
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MONGO_URI: mongodb://admin:admin123@mongodb:27017/
    depends_on:
      - mongodb
      - mlflow
    networks:
      - ml_network

# Volúmenes persistentes
volumes:
  mongodb_data:
    driver: local
  mlflow_artifacts:
    driver: local

# Red personalizada
networks:
  ml_network:
    driver: bridge
```

#### **Comandos de Docker Compose**

```bash
# Iniciar todos los servicios
docker-compose up -d

# Ver logs de un servicio
docker-compose logs -f mlflow

# Detener servicios
docker-compose down

# Detener y eliminar volúmenes
docker-compose down -v

# Reconstruir imágenes
docker-compose build

# Escalar servicios
docker-compose up -d --scale ml_app=3
```

#### **Arquitectura del Sistema Multi-Contenedor**

```
┌─────────────────────────────────────────────────────────────┐
│              ARQUITECTURA MULTI-CONTENEDOR                  │
└─────────────────────────────────────────────────────────────┘

         ┌──────────────────────────────────────┐
         │        Docker Host (tu PC)           │
         │                                      │
         │   ┌──────────────────────────────┐   │
         │   │      ml_network (bridge)     │   │
         │   │                              │   │
         │   │  ┌────────────────────────┐  │   │
         │   │  │   MongoDB Container    │  │   │
         │   │  │   Puerto: 27017        │  │   │
         │   │  │   Volume: mongodb_data │  │   │
         │   │  └──────────┬─────────────┘  │   │
         │   │             │                │   │
         │   │  ┌──────────▼─────────────┐  │   │
         │   │  │   MLflow Container     │  │   │
         │   │  │   Puerto: 5000         │  │   │
         │   │  │   Volume: artifacts    │  │   │
         │   │  └──────────┬─────────────┘  │   │
         │   │             │                │   │
         │   │  ┌──────────▼─────────────┐  │   │
         │   │  │   ML App Container     │  │   │
         │   │  │   (Training/Predict)   │  │   │
         │   │  │   Volumes: ./data/     │  │   │
         │   │  └────────────────────────┘  │   │
         │   │                              │   │
         │   └──────────────────────────────┘   │
         │                                      │
         └──────────────────────────────────────┘
                        │
                        ▼
              Usuario accede vía:
              - localhost:27017 (MongoDB)
              - localhost:5000 (MLflow UI)
```

---

## 3) Programación Orientada a Objetos en ML

### **¿Por qué POO en Machine Learning?**

La POO permite crear **pipelines modulares, reutilizables y mantenibles** para proyectos de ML.

#### **Beneficios de POO en ML**

```
┌─────────────────────────────────────────────────────────────┐
│                  BENEFICIOS DE POO EN ML                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  🧩 MODULARIDAD                                              │
│  └─ Componentes independientes y reutilizables              │
│                                                               │
│  🔄 REUTILIZACIÓN                                            │
│  └─ Herencia y composición de clases                        │
│                                                               │
│  🧪 TESTING                                                  │
│  └─ Fácil de testear componentes aislados                   │
│                                                               │
│  📝 MANTENIBILIDAD                                           │
│  └─ Código organizado y fácil de extender                   │
│                                                               │
│  🎯 ENCAPSULAMIENTO                                          │
│  └─ Separación de responsabilidades                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### **Diseño de Pipeline de ML con POO**

#### **Arquitectura de Clases**

```
┌─────────────────────────────────────────────────────────────┐
│              ARQUITECTURA DE PIPELINE ML                     │
└─────────────────────────────────────────────────────────────┘

                    ┌──────────────┐
                    │  BaseModel   │ ◄── Clase abstracta base
                    │ (Abstract)   │
                    └───────┬──────┘
                            │
                ┌───────────┴──────────┐
                │                      │
         ┌──────▼──────┐         ┌─────▼──────┐
         │  Classifier │         │ Regressor  │
         │             │         │            │
         └──────┬──────┘         └────────────┘
                │
         ┌──────▼──────────┐
         │  LGBMPipeline   │ ◄── Implementación específica
         └─────────────────┘

                ┌──────────────┐
                │DataProcessor │ ◄── Procesamiento de datos
                └──────┬───────┘
                       │
            ┌──────────┴─────────┐
            │                    │
     ┌──────▼──────┐       ┌─────▼──────┐
     │FeatureEngineer      │ DataLoader │
     │             │       │            │
     └─────────────┘       └────────────┘

                ┌──────────────┐
                │ ModelTracker │ ◄── MLOps tracking
                └──────────────┘
```

### **Ejemplo Básico: Pipeline con POO**

```python
# src/ml_pipeline_basic.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CLASE BASE ABSTRACTA
# ============================================================================

class BaseMLPipeline(ABC):
    """
    Clase base abstracta para pipelines de Machine Learning.
    Define la interfaz común que deben implementar todos los pipelines.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Diccionario con configuración del pipeline
        """
        self.config = config
        self.model = None
        self.is_trained = False
        self._validate_config()
        logger.info(f"{self.__class__.__name__} inicializado")

    @abstractmethod
    def _validate_config(self) -> None:
        """Valida la configuración del pipeline."""
        pass

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Entrena el modelo."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Genera predicciones."""
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evalúa el modelo."""
        pass

    def save_model(self, path: str) -> None:
        """Guarda el modelo entrenado."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        logger.info(f"Modelo guardado en {path}")

    def load_model(self, path: str) -> None:
        """Carga un modelo guardado."""
        logger.info(f"Modelo cargado desde {path}")


# ============================================================================
# PROCESADOR DE DATOS
# ============================================================================

class DataProcessor:
    """
    Clase para procesamiento y transformación de datos.
    Encapsula toda la lógica de feature engineering.
    """

    def __init__(self, feature_config: Dict[str, Any]):
        self.feature_config = feature_config
        self.feature_names = []
        logger.info("DataProcessor inicializado")

    def create_lag_features(self, df: pd.DataFrame,
                           columns: list,
                           lags: list) -> pd.DataFrame:
        """Crea features de lag."""
        df_copy = df.copy()

        for col in columns:
            for lag in lags:
                new_col = f"{col}_lag{lag}"
                df_copy[new_col] = df_copy.groupby('numero_de_cliente')[col].shift(lag)
                self.feature_names.append(new_col)

        logger.info(f"Creados {len(columns) * len(lags)} lag features")
        return df_copy

    def create_rolling_features(self, df: pd.DataFrame,
                               columns: list,
                               windows: list) -> pd.DataFrame:
        """Crea features de ventana móvil."""
        df_copy = df.copy()

        for col in columns:
            for window in windows:
                df_copy[f"{col}_rolling_mean_{window}"] = (
                    df_copy.groupby('numero_de_cliente')[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )

        logger.info(f"Creados rolling features para {len(columns)} columnas")
        return df_copy

    def get_feature_names(self) -> list:
        """Retorna lista de nombres de features creados."""
        return self.feature_names


# ============================================================================
# PIPELINE DE LGBM
# ============================================================================

class LGBMPipeline(BaseMLPipeline):
    """Pipeline específico para LightGBM."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('lgbm_params', {})
        self.training_history = []

    def _validate_config(self) -> None:
        """Valida que la configuración contenga parámetros de LGBM."""
        required_keys = ['lgbm_params', 'features', 'target']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Configuración debe contener '{key}'")

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Entrena el modelo LightGBM."""
        logger.info("Iniciando entrenamiento de LightGBM...")
        logger.info(f"Shape de datos: X={X.shape}, y={y.shape}")

        train_data = lgb.Dataset(X, label=y)

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.config.get('num_boost_round', 100),
            callbacks=[lgb.log_evaluation(period=10)]
        )

        self.is_trained = True
        logger.info("✅ Entrenamiento completado")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Genera predicciones."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Ejecuta train() primero")

        predictions = self.model.predict(X)
        logger.info(f"Generadas {len(predictions)} predicciones")
        return predictions

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evalúa el modelo."""
        predictions = self.predict(X)

        from sklearn.metrics import roc_auc_score, log_loss

        metrics = {
            'roc_auc': roc_auc_score(y, predictions),
            'log_loss': log_loss(y, predictions)
        }

        logger.info(f"Evaluación: {metrics}")
        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Obtiene importancia de features."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()

        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df_importance
```

### **Uso del Pipeline POO**

```python
# main_poo.py
import pandas as pd
import logging
from src.ml_pipeline_basic import LGBMPipeline, DataProcessor

logging.basicConfig(level=logging.INFO)

def main():
    # Configuración
    lgbm_config = {
        'lgbm_params': {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'verbosity': -1
        },
        'features': [],
        'target': 'clase_binaria',
        'num_boost_round': 100
    }

    feature_config = {
        'lag_columns': ['mcuentas_saldo', 'mtarjeta_visa_consumo'],
        'lags': [1, 2, 3]
    }

    # Cargar datos
    df = pd.read_csv('data/competencia_01.csv')

    # Crear componentes
    data_processor = DataProcessor(feature_config)
    pipeline = LGBMPipeline(lgbm_config)

    # Feature engineering
    df = data_processor.create_lag_features(
        df,
        feature_config['lag_columns'],
        feature_config['lags']
    )

    # Preparar datos
    features = data_processor.get_feature_names()
    X_train = df[df['foto_mes'] == 202101][features].fillna(0)
    y_train = df[df['foto_mes'] == 202101]['clase_binaria']

    # Entrenar
    pipeline.train(X_train, y_train)

    # Evaluar
    X_test = df[df['foto_mes'] == 202103][features].fillna(0)
    y_test = df[df['foto_mes'] == 202103]['clase_binaria']
    metrics = pipeline.evaluate(X_test, y_test)

    print(f"Métricas: {metrics}")

if __name__ == "__main__":
    main()
```

---

## 4) MLOps: Conceptos y Herramientas

### **¿Qué es MLOps?**

**MLOps (Machine Learning Operations)** es la práctica de aplicar principios de DevOps al ciclo de vida de modelos de Machine Learning, desde el desarrollo hasta producción.

### **Ciclo de Vida de MLOps**

```
┌─────────────────────────────────────────────────────────────┐
│                 CICLO DE VIDA DE MLOPS                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   DESARROLLO │  ───► │ EXPERIMENTOS│  ───► │ VALIDACIÓN  │
│             │       │   & TUNING  │       │             │
└─────────────┘       └─────────────┘       └─────────────┘
      │                     │                       │
      ▼                     ▼                       ▼
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   CÓDIGO    │       │  TRACKING   │       │   TESTING   │
│   (Git)     │       │  (MLflow)   │       │  (Pytest)   │
└─────────────┘       └─────────────┘       └─────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │   REGISTRO DE   │
                  │    MODELOS      │
                  │  (Model Registry)│
                  └────────┬────────┘
                           │
                           ▼
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│  DEPLOYMENT │  ◄─── │  PACKAGING  │  ◄─── │ APROBACIÓN  │
│  (Docker)   │       │  (Container)│       │             │
└─────────────┘       └─────────────┘       └─────────────┘
      │
      ▼
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│ PRODUCCIÓN  │  ───► │ MONITOREO   │  ───► │ REENTRENAR  │
│             │       │  (Drift)    │       │             │
└─────────────┘       └─────────────┘       └─────────────┘
```

### **Componentes Clave de MLOps**

```
┌─────────────────────────────────────────────────────────────┐
│                COMPONENTES DE MLOPS                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  📝 EXPERIMENT TRACKING                                      │
│  └─ Registrar parámetros, métricas y artifacts              │
│     Herramientas: MLflow, Weights & Biases, Neptune         │
│                                                               │
│  🗄️ MODEL REGISTRY                                           │
│  └─ Versionado de modelos con metadatos                     │
│     Herramientas: MLflow Registry, DVC                       │
│                                                               │
│  🔄 CI/CD PIPELINES                                          │
│  └─ Automatización de pruebas y despliegues                 │
│     Herramientas: GitHub Actions, Jenkins, GitLab CI        │
│                                                               │
│  📊 MONITORING & OBSERVABILITY                               │
│  └─ Detección de data drift y model drift                   │
│     Herramientas: Evidently AI, WhyLogs, Prometheus         │
│                                                               │
│  🐳 CONTAINERIZATION                                         │
│  └─ Empaquetado reproducible con Docker                     │
│     Herramientas: Docker, Kubernetes                         │
│                                                               │
│  📐 FEATURE STORES                                           │
│  └─ Gestión centralizada de features                        │
│     Herramientas: Feast, Tecton, Hopsworks                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### **Librerías MLOps Más Utilizadas**

#### **1. MLflow**

- **Propósito**: Plataforma completa para el ciclo de vida de ML
- **Componentes**:
  - **Tracking**: Registrar experimentos
  - **Projects**: Empaquetar código reproducible
  - **Models**: Gestionar y desplegar modelos
  - **Registry**: Versionado de modelos

#### **2. Weights & Biases (wandb)**

- **Propósito**: Tracking visual avanzado y colaboración
- **Ventajas**: Dashboard interactivo, comparación de experimentos

#### **3. DVC (Data Version Control)**

- **Propósito**: Control de versiones para datos y modelos
- **Ventajas**: Integración con Git, almacenamiento remoto

#### **4. Neptune.ai**

- **Propósito**: Metadata store para experimentos de ML
- **Ventajas**: Búsqueda avanzada, comparación de modelos

#### **5. Evidently AI**

- **Propósito**: Monitoreo de data drift y model performance
- **Ventajas**: Reportes visuales de degradación

### **Comparación de Herramientas**

```
┌──────────────┬────────────┬──────────┬────────────┬────────────┐
│ Herramienta  │ Tracking   │ Registry │ Deployment │ Open Source│
├──────────────┼────────────┼──────────┼────────────┼────────────┤
│ MLflow       │     ✅     │    ✅    │     ✅     │     ✅     │
│ W&B          │     ✅     │    ✅    │     ❌     │     ❌     │
│ Neptune      │     ✅     │    ✅    │     ❌     │     ❌     │
│ DVC          │     ❌     │    ✅    │     ❌     │     ✅     │
│ Kubeflow     │     ✅     │    ✅    │     ✅     │     ✅     │
└──────────────┴────────────┴──────────┴────────────┴────────────┘
```

---

## 5) Implementación Práctica: MLflow + MongoDB

### **Arquitectura de la Solución**

```
┌─────────────────────────────────────────────────────────────┐
│         ARQUITECTURA MLflow + MongoDB + Python               │
└─────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────┐
  │           Python ML Application                   │
  │                                                   │
  │   ┌─────────────────────────────────────────┐   │
  │   │  import mlflow                          │   │
  │   │  mlflow.set_tracking_uri(...)          │   │
  │   │  mlflow.start_run()                     │   │
  │   │  mlflow.log_param("learning_rate", 0.1)│   │
  │   │  mlflow.log_metric("accuracy", 0.95)   │   │
  │   │  mlflow.log_model(model, "model")      │   │
  │   │  mlflow.end_run()                       │   │
  │   └─────────────────────────────────────────┘   │
  └──────────────────┬───────────────────────────────┘
                     │
                     │ HTTP (REST API)
                     ▼
        ┌────────────────────────┐
        │   MLflow Tracking      │
        │       Server           │
        │   (Puerto 5000)        │
        └────────┬───────────────┘
                 │
                 │ MongoDB Connection
                 ▼
    ┌────────────────────────────┐
    │      MongoDB Database       │
    │      (Puerto 27017)         │
    │                             │
    │  Collections:               │
    │  ├─ experiments             │
    │  ├─ runs                    │
    │  ├─ metrics                 │
    │  ├─ params                  │
    │  └─ tags                    │
    └─────────────────────────────┘
```

### **Implementación: Tracking con MLflow y MongoDB**

```python
# src/mlflow_tracker.py
import mlflow
import mlflow.lightgbm
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from pymongo import MongoClient
from datetime import datetime

logger = logging.getLogger(__name__)

class MLflowExperimentTracker:
    """
    Clase para tracking de experimentos con MLflow y MongoDB.
    Integra el tracking de métricas, parámetros y modelos.
    """

    def __init__(self,
                 experiment_name: str,
                 tracking_uri: str = "http://localhost:5000",
                 mongo_uri: str = "mongodb://admin:admin123@localhost:27017/"):
        """
        Args:
            experiment_name: Nombre del experimento
            tracking_uri: URI del servidor de MLflow
            mongo_uri: URI de MongoDB para almacenamiento adicional
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.mongo_uri = mongo_uri

        # Configurar MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Conexión adicional a MongoDB para queries personalizadas
        try:
            self.mongo_client = MongoClient(mongo_uri)
            self.db = self.mongo_client['ml_experiments']
            self.collection = self.db[experiment_name.replace(' ', '_')]
            logger.info(f"Conectado a MongoDB: {mongo_uri}")
        except Exception as e:
            logger.warning(f"No se pudo conectar a MongoDB: {e}")
            self.mongo_client = None

        logger.info(f"MLflowExperimentTracker inicializado: {experiment_name}")
        logger.info(f"Tracking URI: {tracking_uri}")

    def start_run(self, run_name: Optional[str] = None) -> str:
        """
        Inicia una nueva ejecución de experimento.

        Args:
            run_name: Nombre opcional para la ejecución

        Returns:
            ID de la ejecución
        """
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"Run iniciado: {run.info.run_id}")
        return run.info.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Registra parámetros del experimento.

        Args:
            params: Diccionario con parámetros
        """
        mlflow.log_params(params)
        logger.info(f"Parámetros registrados: {len(params)} items")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Registra métricas del experimento.

        Args:
            metrics: Diccionario con métricas
            step: Paso de entrenamiento (opcional)
        """
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"Métricas registradas: {metrics}")

    def log_model(self, model, artifact_path: str = "model") -> None:
        """
        Registra el modelo entrenado.

        Args:
            model: Modelo de LightGBM
            artifact_path: Ruta del artifact
        """
        mlflow.lightgbm.log_model(model, artifact_path)
        logger.info(f"Modelo registrado en: {artifact_path}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Registra un archivo como artifact.

        Args:
            local_path: Ruta local del archivo
            artifact_path: Ruta de destino en MLflow
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Artifact registrado: {local_path}")

    def log_figure(self, figure, artifact_file: str) -> None:
        """
        Registra una figura de matplotlib.

        Args:
            figure: Objeto figure de matplotlib
            artifact_file: Nombre del archivo
        """
        mlflow.log_figure(figure, artifact_file)
        logger.info(f"Figura registrada: {artifact_file}")

    def log_tags(self, tags: Dict[str, str]) -> None:
        """
        Registra tags para el experimento.

        Args:
            tags: Diccionario con tags
        """
        mlflow.set_tags(tags)
        logger.info(f"Tags registrados: {tags}")

    def end_run(self, status: str = "FINISHED") -> None:
        """
        Finaliza la ejecución actual.

        Args:
            status: Estado final (FINISHED, FAILED, KILLED)
        """
        mlflow.end_run(status=status)
        logger.info(f"Run finalizado con estado: {status}")

    def log_to_mongodb(self, run_data: Dict[str, Any]) -> None:
        """
        Guarda datos adicionales en MongoDB para análisis personalizado.

        Args:
            run_data: Datos del experimento
        """
        if self.mongo_client is None:
            logger.warning("MongoDB no disponible")
            return

        try:
            # Agregar timestamp
            run_data['timestamp'] = datetime.now()
            run_data['experiment_name'] = self.experiment_name

            # Insertar en MongoDB
            result = self.collection.insert_one(run_data)
            logger.info(f"Datos guardados en MongoDB: {result.inserted_id}")
        except Exception as e:
            logger.error(f"Error al guardar en MongoDB: {e}")

    def get_best_run(self, metric_name: str = "ganancia",
                     ascending: bool = False) -> Optional[Dict]:
        """
        Obtiene el mejor run según una métrica.

        Args:
            metric_name: Nombre de la métrica
            ascending: Si True, el menor valor es mejor

        Returns:
            Diccionario con información del mejor run
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
        )

        if len(runs) == 0:
            return None

        best_run = runs.iloc[0].to_dict()
        logger.info(f"Mejor run encontrado: {best_run['run_id']}")
        return best_run

    def compare_runs(self, run_ids: list) -> pd.DataFrame:
        """
        Compara múltiples runs.

        Args:
            run_ids: Lista de IDs de runs

        Returns:
            DataFrame con comparación
        """
        runs_data = []
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            runs_data.append({
                'run_id': run_id,
                **run.data.params,
                **run.data.metrics
            })

        df_comparison = pd.DataFrame(runs_data)
        logger.info(f"Comparación de {len(run_ids)} runs completada")
        return df_comparison

    def query_mongodb(self, query: Dict) -> list:
        """
        Ejecuta una query personalizada en MongoDB.

        Args:
            query: Query de MongoDB

        Returns:
            Lista de documentos
        """
        if self.mongo_client is None:
            logger.warning("MongoDB no disponible")
            return []

        results = list(self.collection.find(query))
        logger.info(f"Query ejecutada: {len(results)} documentos encontrados")
        return results

    def close(self) -> None:
        """Cierra conexiones."""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("Conexión a MongoDB cerrada")
```

### **Ejemplo de Uso: Tracking de Experimento**

```python
# main_mlflow.py
import pandas as pd
import lightgbm as lgb
import logging
from src.mlflow_tracker import MLflowExperimentTracker
from src.ml_pipeline_basic import LGBMPipeline, DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Inicializar tracker
    tracker = MLflowExperimentTracker(
        experiment_name="LGBM_Optimization",
        tracking_uri="http://localhost:5000",
        mongo_uri="mongodb://admin:admin123@localhost:27017/"
    )

    # Cargar datos
    df = pd.read_csv('data/competencia_01.csv')

    # Configuración del experimento
    config = {
        'lgbm_params': {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'random_state': 42
        },
        'features': [],
        'target': 'clase_binaria',
        'num_boost_round': 100
    }

    # Iniciar run
    run_id = tracker.start_run(run_name="experimento_01")

    try:
        # Registrar parámetros
        tracker.log_params(config['lgbm_params'])
        tracker.log_tags({
            'framework': 'LightGBM',
            'task': 'binary_classification',
            'dataset': 'competencia_01'
        })

        # Feature engineering
        feature_config = {
            'lag_columns': ['mcuentas_saldo', 'mtarjeta_visa_consumo'],
            'lags': [1, 2, 3]
        }

        data_processor = DataProcessor(feature_config)
        df = data_processor.create_lag_features(
            df,
            feature_config['lag_columns'],
            feature_config['lags']
        )

        # Preparar datos
        features = data_processor.get_feature_names()
        X_train = df[df['foto_mes'] == 202101][features].fillna(0)
        y_train = df[df['foto_mes'] == 202101]['clase_binaria']
        X_test = df[df['foto_mes'] == 202103][features].fillna(0)
        y_test = df[df['foto_mes'] == 202103]['clase_binaria']

        # Entrenar modelo
        pipeline = LGBMPipeline(config)
        pipeline.train(X_train, y_train)

        # Evaluar
        metrics = pipeline.evaluate(X_test, y_test)

        # Registrar métricas
        tracker.log_metrics(metrics)

        # Registrar modelo
        tracker.log_model(pipeline.model)

        # Guardar datos adicionales en MongoDB
        run_data = {
            'run_id': run_id,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'num_features': len(features),
            'params': config['lgbm_params'],
            'metrics': metrics
        }
        tracker.log_to_mongodb(run_data)

        # Finalizar con éxito
        tracker.end_run(status="FINISHED")

        logger.info("✅ Experimento completado exitosamente")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Métricas: {metrics}")

    except Exception as e:
        logger.error(f"❌ Error en experimento: {e}")
        tracker.end_run(status="FAILED")
        raise

    finally:
        tracker.close()

if __name__ == "__main__":
    main()
```

### **Acceso a MLflow UI**

Una vez levantado el stack con `docker-compose up -d`, accede a:

- **MLflow UI**: http://localhost:5000
- **MongoDB**: mongodb://admin:admin123@localhost:27017/

En MLflow UI podrás:

- Ver todos los experimentos
- Comparar runs
- Analizar métricas y parámetros
- Descargar modelos
- Ver artifacts (gráficos, archivos)

---

## 📦 **Instalación de Docker:**

- **Mac**: Descargar Docker Desktop desde https://www.docker.com/products/docker-desktop
- **Linux**: `sudo apt-get install docker.io docker-compose`
- **Windows**: Docker Desktop con WSL2

**Verificar instalación:**

```bash
# Verificar Docker
docker --version
docker-compose --version

# Verificar Python packages
pip list | grep mlflow
pip list | grep pymongo
```

---

## Estructura del proyecto con MLOps

```
Project_Wendsday/
├── Dockerfile                     # ⭐ NUEVO: Imagen de Docker
├── docker-compose.yml             # ⭐ NUEVO: Orquestación de servicios
├── requirements.txt               # Dependencias Python
├── config.yaml                    # Configuración del proyecto
├── main.py                        # Pipeline base
├── main_poo.py                    # ⭐ NUEVO: Pipeline con POO
├── main_mlflow.py                 # ⭐ NUEVO: Pipeline con MLflow tracking
├── src/
│   ├── __init__.py
│   ├── loader.py                  # Carga de datos
│   ├── features.py                # Feature engineering
│   ├── ml_pipeline_basic.py      # ⭐ NUEVO: Clases POO para pipeline
│   ├── mlflow_tracker.py         # ⭐ NUEVO: Tracking con MLflow y MongoDB
│   ├── optimization.py            # Optimización con Optuna
│   ├── best_params.py             # Gestión de mejores parámetros
│   ├── final_training.py          # Entrenamiento final
│   └── output_manager.py          # Gestión de salidas
├── data/                          # Datos del proyecto
├── logs/                          # Logs de ejecución
├── resultados/                    # Resultados y gráficos
├── predict/                       # Predicciones finales
├── mlruns/                        # ⭐ NUEVO: Artifacts de MLflow
└── mongo-init/                    # ⭐ NUEVO: Scripts de inicialización MongoDB
```

---

## ✅ Resultado esperado

### **Conocimientos Teóricos**

* ✅ **Comprensión de Docker**: Contenedorización, imágenes, contenedores y volúmenes
* ✅ **Docker Compose**: Orquestación de múltiples servicios (ML app, MongoDB, MLflow)
* ✅ **POO aplicada a ML**: Diseño de pipelines modulares y reutilizables
* ✅ **MLOps fundamentals**: Ciclo de vida, herramientas y mejores prácticas
* ✅ **Experiment Tracking**: Registro sistemático de experimentos con MLflow

### **Habilidades Prácticas**

* 🐳 **Dockerfile funcional** para tu proyecto de ML
* 🔧 **docker-compose.yml** orquestando MongoDB + MLflow + ML App
* 🎯 **Pipeline POO** con clases base abstractas y herencia
* 📊 **MLflow tracking** integrado con MongoDB para análisis personalizado
* 🔍 **Queries en MongoDB** para recuperar y analizar experimentos

### **Arquitectura Implementada**

```
┌─────────────────────────────────────────────────────────────┐
│              ARQUITECTURA FINAL CLASE 07                     │
└─────────────────────────────────────────────────────────────┘

        TU CÓDIGO PYTHON (POO)
               │
               ├─── BaseMLPipeline (Abstract)
               ├─── LGBMPipeline (Concrete)
               ├─── DataProcessor
               └─── MLflowExperimentTracker
                          │
                          ▼
                   ┌─────────────┐
                   │   MLflow    │ ◄─── UI en localhost:5000
                   │   Server    │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   MongoDB   │ ◄─── Persistencia
                   │   Database  │
                   └─────────────┘
                          │
                  TODO EN DOCKER
```

### **Beneficios Obtenidos**

1. **Reproducibilidad Total**: Tu entorno corre igual en cualquier máquina
2. **Trazabilidad Completa**: Todos los experimentos registrados y comparables
3. **Código Mantenible**: Arquitectura POO facilita testing y extensiones
4. **Escalabilidad**: Fácil transición a producción con Docker
5. **Colaboración**: Equipo puede reproducir experimentos exactos


## 🎓 Conceptos Clave para Recordar

### **Docker**

- Los contenedores son **ligeros y portables**
- Las imágenes son **inmutables** (se reconstruyen, no se modifican)
- Los volúmenes permiten **persistencia de datos**
- Docker Compose facilita **orquestación multi-contenedor**

### **POO en ML**

- **Abstracción**: Clases base definen interfaces comunes
- **Encapsulamiento**: Cada clase tiene responsabilidades claras
- **Herencia**: Reutilización de código entre modelos similares
- **Polimorfismo**: Mismo código funciona con diferentes implementaciones

### **MLOps**

- **Tracking**: Registrar TODO (parámetros, métricas, código, datos)
- **Versioning**: Modelos y datos deben estar versionados
- **Automation**: CI/CD para pruebas y despliegues automáticos
- **Monitoring**: Detectar degradación del modelo en producción

### **Mejores Prácticas**

```
┌─────────────────────────────────────────────────────────────┐
│                    MEJORES PRÁCTICAS                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ✓ Usa .dockerignore para excluir archivos innecesarios     │
│  ✓ Multi-stage builds para imágenes más pequeñas            │
│  ✓ No hardcodees credenciales en Dockerfiles                │
│  ✓ Usa variables de entorno para configuración              │
│  ✓ Documenta tus clases con docstrings detallados           │
│  ✓ Implementa logging en todos los métodos importantes      │
│  ✓ Registra TODOS los experimentos, incluso los fallidos    │
│  ✓ Usa tags para organizar experimentos relacionados        │
│  ✓ Guarda artifacts (gráficos, configs) en MLflow           │
│  ✓ Implementa health checks en tus contenedores             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 Recursos Adicionales

### **Documentación Oficial**

- [Docker Documentation](https://docs.docker.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MongoDB Documentation](https://www.mongodb.com/docs/)
- [Python OOP Tutorial](https://docs.python.org/3/tutorial/classes.html)

### **Tutoriales Recomendados**

- [Docker for Data Science](https://docker-curriculum.com/)
- [MLflow Tutorial](https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)
- [Design Patterns in Python](https://refactoring.guru/design-patterns/python)

### **Comunidades**

- [MLOps Community](https://mlops.community/)
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Docker Community Forums](https://forums.docker.com/)

---

**¡Felicidades!** 🎉

¡Nos vemos en la próxima clase! 🚀
