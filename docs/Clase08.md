# 📘 Clase 08: Configuración de Entorno de Producción

## 🔄 Git

```bash
git status
git add Clase08.md
git commit -m "Agrega Clase 08: Configuración de entorno de producción"
git push
```

## 🎯 Objetivo

Aprender a configurar un entorno de producción en un VPS con Ubuntu Server, PostgreSQL como backend para MLflow, y configuraciones básicas de seguridad y rendimiento.

## 📑 Índice

1. [Configuración de VPS con Ubuntu Server y PostgreSQL](#-registro--instalación-base-de-ubuntu-server--postgresql-mlflow-backend)
2. [Preparación del sistema](#-1-preparación-inicial-del-sistema)
3. [Instalación de PostgreSQL](#-2-instalación-de-postgresql)
4. [Configuración de base de datos para MLflow](#-3-creación-de-base-de-datos-y-usuario-para-mlflow)
5. [Verificación de conexión](#-4-verificación-de-conexión)
6. [Ajustes de rendimiento](#-5-opcional-ajustes-de-rendimiento-mínimo-2-gb-ram)
7. [Instalación de MLflow en entorno virtual](#-3-instalar-mlflow-en-entorno-virtual)
8. [Configuración de carpetas de artefactos](#-4-crear-carpeta-de-artefactos)
9. [Prueba de ejecución manual](#-5-probar-ejecución-manual)
10. [Configuración del servicio systemd](#-6-crear-servicio-permanente)
11. [Activación y verificación del servicio](#-7-activar-y-verificar-el-servicio)
12. [Verificación final](#-8-verificación-final)
13. [Resultado final](#-resultado-final)

## 🧾 REGISTRO — Instalación base de Ubuntu Server + PostgreSQL (MLflow backend)

### 📌 Datos de entorno

| Recurso                  | Valor                                               |
| ------------------------ | --------------------------------------------------- |
| **SO**             | Ubuntu Server 24.04 LTS                             |
| **Usuario admin**  | mladmin (no-root, con sudo)                         |
| **Acceso SSH**     | Clave pública autorizada, root deshabilitado       |
| **Firewall (ufw)** | Activo — Puertos abiertos: 22 (SSH), 8080 (MLflow) |
| **Proveedor VPS**  | VPS con 1 vCPU, 2 GB RAM, 10 GB SSD                 |
| **IP pública**    | (la de tu instancia, ej. 52.x.x.x)                  |

## ⚙️ 1. Preparación inicial del sistema

Actualización del sistema:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl ufw python3-venv python3-pip
```

### (Opcional) Agregar 1 GB de swap

```bash
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Configuración del firewall

```bash
sudo ufw allow OpenSSH
sudo ufw allow 8080/tcp
sudo ufw enable
sudo ufw status
```

## 🔑 2. Configuración de acceso SSH seguro

### 2.1 Crear un usuario administrativo

```bash
sudo adduser mladmin
sudo usermod -aG sudo mladmin
```

🔸 Este usuario será el que administre el servidor y levante MLflow.

> 🔸 Asegurate de establecer una contraseña temporal si la pedirá.

### 2.2 Configurar acceso SSH por clave pública

Desde  **tu máquina local (Verificar SO)** , generá tu clave si aún no la tenés:

```bash
ssh-keygen -t ed25519 -C tu_email@ejemplo.com
```

Luego copiá tu clave al servidor:

```bash
cat ~/.ssh/id_ed25519.pub
```

> y pegar su contenido dentro del servidor en:
>
> ```bash
> sudo nano /home/mladmin/.ssh/authorized_keys
> ```

Asegurá los permisos:

```bash
sudo chmod 700 /home/mladmin/.ssh
sudo chmod 600 /home/mladmin/.ssh/authorized_keys
sudo chown -R mladmin:mladmin /home/mladmin/.ssh
```

### 2.3 Probar la conexión

Desde tu máquina local:

```bash
ssh mladmin@<IP_DEL_SERVIDOR>
```

Si entra sin pedir contraseña, tu configuración SSH por clave funciona correctamente. ✅

## 🗄️ 3. Instalación de PostgreSQL

```bash
sudo apt install -y postgresql postgresql-contrib
sudo systemctl enable --now postgresql
sudo systemctl status postgresql --no-pager
```

### 🧩 3.1. Creación de base de datos y usuario para MLflow

Entrar al motor:

```bash
sudo -u postgres psql
```

Y ejecutar dentro de psql:

```sql
CREATE DATABASE mlflow;
CREATE ROLE mlflow LOGIN PASSWORD '123456';
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
\q
```

## 🧪 4. Verificación de conexión

```bash
psql -h 127.0.0.1 -U mlflow -d mlflow
```

Ingresar la contraseña → Debe mostrar el prompt `mlflow=>`
Salir con `\q`.

## 🪜 5. Instalar MLflow en entorno virtual

```bash
sudo mkdir -p /opt/mlflow-venv
sudo python3 -m venv /opt/mlflow-venv
sudo chown -R $USER:$USER /opt/mlflow-venv

source /opt/mlflow-venv/bin/activate
pip install --upgrade pip
pip install mlflow psycopg2-binary
mlflow --version
```

## 🪜 6. Crear carpeta de artefactos

```bash
sudo mkdir -p /srv/mlflow/artifacts
sudo chown -R $USER:$USER /srv/mlflow
```

## 🪜 7. Probar ejecución manual

```bash
export MLFLOW_BACKEND_URI="postgresql://mlflow:1q2w3e4r5t@127.0.0.1:5432/mlflow"
export ARTIFACT_ROOT="/srv/mlflow/artifacts"

mlflow server \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-store-uri ${MLFLOW_BACKEND_URI} \
  --default-artifact-root file:${ARTIFACT_ROOT} \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"
```

### ✅ Resultado esperado:

```
INFO:     Uvicorn running on http://0.0.0.0:8080
```

Luego probar en navegador:

```
http://<tu-ip-pública>:8080
```

## 🪜 8. Crear servicio permanente

Crear archivo de servicio:

```bash
sudo nano /etc/systemd/system/mlflow.service
```

Contenido del archivo de servicio:

```ini
[Unit]
Description=MLflow Tracking Server
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
Type=simple
User=mladmin
Group=mladmin

# Parámetros principales
Environment="MLFLOW_BACKEND_URI=postgresql://mlflow:1q2w3e4r5t@127.0.0.1:5432/mlflow"
Environment="ARTIFACT_ROOT=/srv/mlflow/artifacts"

WorkingDirectory=/srv/mlflow

ExecStart=/opt/mlflow-venv/bin/mlflow server \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-store-uri ${MLFLOW_BACKEND_URI} \
  --default-artifact-root file:${ARTIFACT_ROOT} \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"

Restart=on-failure
RestartSec=5

# Seguridad básica del proceso
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

## 🪜 9. Activar y verificar el servicio

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mlflow
sudo systemctl status mlflow --no-pager
```

Ver logs en vivo:

```bash
sudo journalctl -u mlflow -f
```

## 🧪 10. Verificación final

### Interfaz Web:

```
http://<tu-ip-pública>:8080
```

### Estado del servicio:

```bash
sudo systemctl status mlflow
```

### Ver logs recientes:

```bash
sudo journalctl -u mlflow -n 30 --no-pager
```

## ✅ Resultado final

| Componente       | Estado                                 |
| ---------------- | -------------------------------------- |
| PostgreSQL       | OK — Base mlflow, user mlflow         |
| MLflow           | OK — corriendo en /opt/mlflow-venv    |
| Artefactos       | /srv/mlflow/artifacts                  |
| Servicio systemd | mlflow.service activo                  |
| Accesibilidad    | Puerto 8080 abierto a cualquier IP     |
| Web UI           | http://<tu-ip-pública>:8080 funcional |

## 📦 Dependencias

```bash
# Instaladas durante el proceso
- postgresql
- postgresql-contrib
- git
- curl
- ufw
- python3-venv
- python3-pip
- mlflow
- psycopg2-binary
```

## 📝 Notas adicionales

1. **Seguridad**:

   - Cambiar la contraseña del usuario `mlflow` en producción
   - Considerar usar SSL/TLS para las conexiones
   - Restringir el acceso a la IP de tu red local si es posible
2. **Mantenimiento**:

   - Configurar copias de seguridad regulares de la base de datos
   - Monitorear el uso de disco de los artefactos
   - Actualizar regularmente los paquetes de Python
3. **Escalabilidad**:

   - Para mayor carga, considerar usar S3/MinIO para almacenar artefactos
   - Ajustar parámetros de PostgreSQL según los recursos disponibles
   - Considerar usar un proxy inverso como Nginx
4. **Monitoreo**:

   - Configurar alertas para el servicio MLflow
   - Monitorear el uso de CPU, memoria y disco
   - Revisar logs regularmente para detectar problemas
