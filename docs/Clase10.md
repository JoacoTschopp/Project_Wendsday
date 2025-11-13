# 📘 Clase10.md

## 🔄 Pre-commit (ejecución opcional en el proyecto)

Para mantener la calidad del código sin obligar su ejecución antes de cada commit, configuramos **pre-commit** con una selección estándar de hooks (Black, isort, Ruff, Mypy y validaciones básicas). Este paso puede ejecutarse manualmente cuando quieras revisar el estado del repositorio.

1. Instalar la herramienta (una sola vez en tu entorno):

   ```bash
   pip install pre-commit
   pre-commit --version
   ```
2. Ejecutar todos los hooks sobre el proyecto completo cuando quieras validar el código:

   ```bash
   pre-commit run --all-files
   ```
3. (Opcional) Si preferís que los hooks corran automáticamente antes de cada `git commit`, podés habilitar la integración local:

   ```bash
   pre-commit install
   ```

> 📌 El archivo `.pre-commit-config.yaml` ya está en la raíz del proyecto con una configuración estándar. Si necesitás ajustar reglas (por ejemplo, cambiar el `line-length`, habilitar/deshabilitar un hook o modificar las reglas de Mypy), editá ese archivo y volvé a ejecutar `pre-commit run --all-files` para validar.

## Git - Flujo de trabajo sugerido

```bash
git status
# (opcional) Ejecutar pre-commit run --all-files
pre-commit run --all-files

git add .
git commit -m "Clase09: Automatización de calidad con pre-commit"
git pull origin main
git push origin main
```

- `pre-commit run --all-files` permite detectar conflictos, formatos incorrectos o linting pendiente antes de dejar cambios listos para versionar.
- Si en algún momento querés omitir un hook puntual, usá `SKIP=hook-id pre-commit run --all-files`.

---

## 🎯 Objetivo

Automatizar controles de calidad de código en el proyecto DMeyF mediante pre-commit y herramientas complementarias (Black, isort y Ruff) para garantizar consistencia, estilo y detección temprana de errores.

---

## 📑 Índice de la clase

1. [Contexto y motivación](#1-contexto-y-motivación)
2. [Hooks configurados](#2-hooks-configurados)
3. [Estrategia de uso en el flujo Git](#3-estrategia-de-uso-en-el-flujo-git)
4. [Integración con herramientas del proyecto](#4-integración-con-herramientas-del-proyecto)
5. [Checklist de calidad previa a experimentos](#5-checklist-de-calidad-previa-a-experimentos)
6. [Taller práctico: aplicar hooks sobre cambios reales](#6-taller-práctico-aplicar-hooks-sobre-cambios-reales)
7. [Preguntas frecuentes](#7-preguntas-frecuentes)
8. [Certificaciones recomendadas](#8-certificaciones-recomendadas)

---

## 1) Contexto y motivación

- El crecimiento del proyecto implica múltiples contribuciones, ramas y experimentos.
- Errores comunes (identación desigual, imports desordenados, trailing whitespace) hacen más difícil el seguimiento de cambios.
- Automatizar verificaciones minimiza la deuda técnica acumulada y acelera las revisiones de código.

### Beneficios

- **Consistencia**: mismo estilo sin depender de revisiones manuales.
- **Confianza**: cambios listos para producción después de pasar los hooks.
- **Tiempo**: se reducen correcciones repetitivas durante code reviews.

---

## 2) Hooks configurados

### 2.1 Hooks base (`pre-commit-hooks`)

| Hook                        | Descripción                                    | Uso práctico                            |
| --------------------------- | ----------------------------------------------- | ---------------------------------------- |
| `check-added-large-files` | Evita subir archivos muy pesados por error      | Protege el repo de datasets gigantes     |
| `check-merge-conflict`    | Detecta restos de conflictos (`<<<<<<< HEAD`) | Imprescindible después de merges        |
| `check-yaml`              | Valida sintaxis YAML                            | Especialmente útil para `config.yaml` |
| `end-of-file-fixer`       | Garantiza salto de línea al final              | Evita diffs molestos                     |
| `trailing-whitespace`     | Remueve espacios al final de línea             | Limpia notebooks convertidos a texto     |

### 2.2 Hooks de formato, linting y tipado

- **Black (`psf/black`)**: formatea código Python con `line-length` 88 (estándar del proyecto).
- **isort (`pycqa/isort`)**: ordena imports automáticamente con el perfil de Black.
- **Ruff (`astral-sh/ruff-pre-commit`)**: inspecciona estilo y posibles errores; se ejecuta con `--fix` para correcciones rápidas.
- **Mypy (`pre-commit/mirrors-mypy`)**: verifica el tipado estático del proyecto. Actualmente se ejecuta con `--ignore-missing-imports` para evitar falsos positivos en dependencias externas.

#### Ejemplo de configuración destacada en `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black
        args: ["--line-length", "88"]
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.5
    hooks:
      - id: ruff
        args: ["--fix"]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        args: ["--ignore-missing-imports"]

```

> ⚠️ Ruff no reemplaza una revisión humana: para cambios complejos seguí apoyándote en los logs y pruebas del proyecto.

---

## 3) Estrategia de uso en el flujo Git

1. Trabajá en tu rama feature:

   ```bash
   git checkout -b feature/pre-commit-calidad
   ```
2. Realizá cambios y ejecutá pruebas del proyecto (`python main.py`, scripts de validación, etc.).
3. Ejecutá `pre-commit run --all-files` para detectar ajustes necesarios.
4. Revisá los archivos modificados automáticamente por los hooks (`git status` te mostrará cambios nuevos).
5. Agregá y commiteá con mensajes descriptivos.
6. Sincronizá con `git pull origin main` para minimizar conflictos antes del push.

> 💡 Si trabajás con muchos archivos, podés limitarte al área modificada: `pre-commit run` (sin `--all-files`) revisa solo lo staged.

---

## 4) Integración con herramientas del proyecto

- **`src/features.py` y módulos de ETL**: Black e isort mantienen el estilo homogéneo para que las transformaciones sean más legibles.
- **`main.py`**: Ruff posibilita detectar imports sin usar o ramas de código inalcanzables tras refactors.
- **`config.yaml` y `config.py`**: `check-yaml` evita errores sutiles en parámetros de experimentos.
- **Scripts de experimentos (`test_evaluation.py`, notebooks convertidos)**: los hooks limpian espacios y conflictos en formato texto, reduciendo ruido en pull requests.

---

## 5) Checklist de calidad previa a experimentos

1. ¿El dataset utilizado está documentado en `data/` o referenciado en `README`/`Clase` correspondiente?
2. ¿Los cambios de configuración (`config.yaml`) se reflejan en las notas de la clase?
3. ¿Los scripts principales (`main.py`, `src/*.py`) pasan:
   - `pre-commit run --files path/al/script.py`
   - `python -m compileall path/al/script.py` (detección temprana de errores sintácticos)
4. ¿Se actualizaron logs o resultados en `resultados/`? Verificá evitar subir archivos temporales.
5. ¿Se documentaron los nuevos pasos en la sección correspondiente de la clase?

## 6) Taller práctico: aplicar hooks sobre cambios reales

1. Modificá un archivo con imports desordenados y espacios extra.
2. Ejecutá:

   ```bash
   pre-commit run --all-files
   ```
3. Observá cómo Black formatea el código, isort reordena imports y Ruff corrige linting.
4. Revisá diffs con `git diff` para comprender cada ajuste automático.
5. Documentá en tu commit lo realizado y relacioná la limpieza con el experimento o feature que estás desarrollando.

## 7) Preguntas frecuentes

- **¿Puedo desactivar algún hook temporalmente?** Sí, con `SKIP=hook-id pre-commit run --all-files` o ajustando `.pre-commit-config.yaml`.
- **¿Qué pasa si Black cambia el estilo que necesitaba para una celda específica?** Podés excluir bloques con comentarios `# fmt: off` / `# fmt: on`, pero usalo con moderación.
- **¿Es necesario ejecutar pre-commit en los notebooks?** Los hooks afectan archivos `.py` y `.md`. Para notebooks `.ipynb`, convertí a `.py` con `jupyter nbconvert --to script` si necesitás limpieza automática.
- **¿Ruff rompe builds si encuentra errores?** Solo si lo configurás como obligatorio en CI. En tu entorno local, revisá los reportes y corregí manualmente cuando sea necesario.
- **¿Cómo actualizo los hooks?** Modificá las versiones en `.pre-commit-config.yaml` y corré `pre-commit autoupdate`.

---

## 8) Certificaciones recomendadas

| Plataforma                          | Enfoque                                                                                                                           | Enlace                                                                                  |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Google Cloud (GCP)**        | Rutas de certificación orientadas a Data Engineering, Machine Learning Engineer y arquitecturas escalables en la nube de Google. | [https://cloud.google.com/certification](https://cloud.google.com/certification)           |
| **Amazon Web Services (AWS)** | Certificaciones de nivel Associate y Professional para arquitectos, data engineers y especialistas en soluciones analíticas.     | [https://aws.amazon.com/certification/](https://aws.amazon.com/certification/)             |
| **Microsoft Azure**           | Trayectos para Data Scientist, Data Engineer y AI Engineer, integrando servicios de Azure Machine Learning y Synapse.             | [https://learn.microsoft.com/certifications/](https://learn.microsoft.com/certifications/) |
| **DeepLearning.AI**           | Especializaciones enfocadas en IA aplicada, MLOps y LLMs desarrolladas junto a expertos del ecosistema.                           | [https://www.deeplearning.ai](https://www.deeplearning.ai)                                 |

> ℹ️ Cada enlace lleva a la plataforma oficial, donde podés revisar planes de estudio actualizados, prerequisitos y recursos complementarios.

---

## 📎 Recursos adicionales

- Documentación oficial: [https://pre-commit.com](https://pre-commit.com)
- Black: [https://black.readthedocs.io](https://black.readthedocs.io)
- isort: [https://pycqa.github.io/isort](https://pycqa.github.io/isort)
- Ruff: [https://docs.astral.sh/ruff](https://docs.astral.sh/ruff)
- Curso recomendado: *Automatización de calidad en proyectos ML* (capítulo 3 sobre flujos de trabajo colaborativos).

---

## ✅ Resultado final

- El repositorio cuenta con `.pre-commit-config.yaml` listo para usar.
- Sabés ejecutar los hooks manualmente sin volverlos obligatorios.
- Incorporaste una guía práctica para integrar los resultados de los hooks con el flujo de trabajo del proyecto.

> 🎉 ¡Con esto aseguramos código limpio y consistente antes de continuar con nuevos experimentos o despliegues!
