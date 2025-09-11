# 📘 Clase01.md

## **Puesta en marcha: Python fuera del Notebook**

---

## 📑 Índice de la clase

1. Diferencia entre Jupyter Notebook y Script ejecutable.
2. Estructura mínima de un proyecto.
3. Creación de entorno virtual `.venv`.
4. El `main()` y `if __name__ == "__main__":`.
5. Cargar un CSV con  **Polars** .
6. Primer log en archivo.
7. Ejercicio integrador: correr `python main.py`.
8. Crear repositorio y hacer `git push`.

---

## 📝 Desarrollo de la clase

---

### **1. Notebook vs Script ejecutable**

* **Notebook (Jupyter):**
  * Ideal para exploración, pruebas rápidas, visualización.
  * Se ejecuta celda por celda, no siempre en orden → resultados poco reproducibles.
  * Suele quedar mucho "código basura" de prueba.
* **Script `.py`:**
  * Se ejecuta completo de arriba a abajo.
  * Fácil de **automatizar** (`python archivo.py`).
  * Mejor para producción (en servidores, pipelines, cron jobs).

⚖️ **Comparación:**

* Notebook = pizarra para pruebas.
* Script = receta de cocina: siempre se ejecuta igual.

---

### **2. Estructura mínima de proyecto**

Creamos carpeta de trabajo `Project_Wendsday`/:

<pre class="overflow-visible!" data-start="1478" data-end="1571"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>Project_Wendsday/
├── </span><span>main</span><span>.py</span><span>
├── data/
│    └── competencia_01</span><span>.csv</span><span>
├── logs/
└── requirements</span><span>.txt</span><span>
</span></span></code></div></div></pre>

* `main.py`: punto de entrada.
* `data/`: datasets de entrada.
* `logs/`: registros de ejecución.
* `requirements.txt`: lista de dependencias (se genera después).

---

### **3. Crear un entorno virtual (.venv)**

¿Por qué? → Para aislar librerías y no romper el sistema.

**Windows (PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas
pip freeze > requirements.txt
```

**Mac/Linux (bash/zsh):**

```bash
python3 -m venv .venv
source venv/bin/activate
pip install pandas
pip freeze > requirements.txt

```

Para desactivar:

```bash
deactivate
```

### **4. El `main()` y `if __name__ == "__main__":`**

Explicar:

* `main()` es nuestra función principal.
* `if __name__ == "__main__":` asegura que solo se ejecute si corremos `python main.py`, no si lo importamos en otro archivo.

Ejemplo mínimo (`main.py`):

```python
def main():
    print("Hola mundo desde main!")

if __name__ == "__main__":
    main()

```

Ejecutar:

```bash
python main.py
```

---

### **5. Cargar un dataset CSV**

En `data/competencia_01.csv` usar un dataset simple.

Código en `main.py`:

```python
import pandas as pd

def main():
    print("Inicio de ejecución")

    # Cargar dataset
    df = pd.read_csv("data/competencia_01.csv")
    print(df.head())
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

if __name__ == "__main__":
    main()

```

---

### **6. Primer log en archivo**

Idea: guardar mensajes para revisar después.

Ejemplo dentro de `main()`:

```python
with open("logs/logs.txt", "a") as f:
    f.write(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas\n")
```

* `"a"` → append (agregar al final).
* Queda registro aunque cierres la terminal.

---

### **7. Ejercicio integrador**

Armar `main.py` con estos pasos:

1. Imprimir inicio.
2. Cargar dataset con Pandas.
3. Guardar log con filas y columnas.
4. Mostrar primeras 5 filas.

```python
import pandas as pd
from datetime import datetime
import os


def main():
    print(">>> Inicio de ejecución")

    # Asegurar que exista la carpeta de logs
    os.makedirs("logs", exist_ok=True)

    # Cargar dataset desde carpeta data
    try:
        df = pd.read_csv("data/competencia_01.csv")
    except FileNotFoundError:
        print("No se encontró el archivo data/competencia_01.csv")
        return

    # Mostrar primeras filas en consola
    print(df.head())

    # Información básica
    filas, columnas = df.shape
    mensaje = f"[{datetime.now()}] Dataset cargado con {filas} filas y {columnas} columnas\n"

    # Guardar log en archivo
    with open("logs/logs.txt", "a", encoding="utf-8") as f:
        f.write(mensaje)

    print(">>> Ejecución finalizada. Revisa logs/logs.txt")


if __name__ == "__main__":
    main()


```

Ejecutar:

```bash
python main.py
```

Revisar en `logs/logs.txt` que se escribió el mensaje.

---

### **8. Crear repositorio y hacer push**

**Inicializar Git (dentro de la carpeta del proyecto):**

```bash
git init
git add.
git commit -m "Clase 01 - Proyecto inicial"
```

**En GitHub:** crear repo vacío `Project_Wendsday` NO colocar licencia de ningun tipo.

**Conectar y subir:**

```bash
git remote add origin git@github.com:[UsuarioGitlab]/Project_Wendsday.git
git branch -M main
git push -u origin main
```

⚠️ Recordar: agregar `.gitignore` para no subir `.venv/` ni `logs/` ni `data/`. Usar SSH es una buena practica que requiere de menor intervencion en el trabajo diario, se recomienda.

Ejemplo de `.gitignore`:

```bash
.venv/
__pycache__/
logs/
data/
```



Nota: Si existe interes y alguien estuvo investigando en lo que hay en el repo sobre git van a ver que lo ideal es usar ramas distintas de main, pueden consultar si tienen dudas, en estas clases es posible que no tengamos tiempo de ahondar en ese tema, lo tendre en mente si sobra algo de tiempo.

---

## ✅ Resultado esperado

Al final de la clase cada alumno tendrá:

* Carpeta `Project_Wendsday/` con:
  * `main.py` funcionando.
  * `data/competencia_01.csv` cargado.
  * `logs/logs.txt` con registros.
  * `requirements.txt` generado.
* Repositorio en GitHub con el proyecto.
