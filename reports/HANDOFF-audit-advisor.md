# Handoff de Contexto: Auditoría, Simplificación y Rendimiento en Cython (Clypto)

**Destinatario:** `cy-audit-advisor-code` (`orca_session_id:opencode2_23077bf4_da54_4c4c_a995_ba16ea326f83`)  
**Emisor:** `antigravity` (`orca_session_id:antigravity_3c5a507b_d3ef_41f9_8d4c_84d9c3382509`)  
**Fecha:** 2026-11-11  
**Rama de trabajo:** `refactoring/audit-simplifications` (worktree: `/home/deal/orca/workspaces/clypto/audit-simplifications`)  
**Rama base release:** `origin/release/v2026b2` (commit `b1c6f6ff`)  
**Rama objetivo final:** `master` / `release/v2026b2`  

---

## 1. Plan y Alcance de la Auditoría

El objetivo de esta auditoría es aplicar la filosofía **Ponytail** (anti-sobreingeniería, eliminación de abstracciones innecesarias, poda de código muerto y maximización del rendimiento a nivel C en Cython).

### Hallazgos y Estado de Ejecución
- **Fase 1 (Cerrada - commit `f4732b50`):**
  - *Problema:* Redundancia entre suites de pruebas (`tests/test_compiler_and_cli.py` vs `tests/test_compiler_and_tools.py`) y warning de ruff en `tests/test_problem_decorator.py`.
  - *Solución:* Se eliminó el archivo de test redundante y se corrigió el warning (`O` -> `objectives`).
  - *Criterio de aceptación:* `ruff check .` con 0 errores y suite de compilador pasando al 100%.

- **Fase 2 (Cerrada - commit `bc919a95`):**
  - *Problema:* Overhead de float-boxing en `Population.fitness`, `Population.solutions` y `argsort` en `clypto/optimizer/native/population.pyx`, que creaban listas intermedias de Python y objetos `float` por cada agente y dimensión.
  - *Solución:* 
    - `@property def fitness`: Preasigna `np.empty(n, dtype=np.float64)` y escribe directamente mediante puntero C `double* data = <double*>cnp.PyArray_DATA(res)` para agentes nativos `Agent`. Incluye fallback seguro para duck-typing (`RuntimeAgent` de los decoradores).
    - `@property def solutions`: Copia por bloques de memoria en C vía `memcpy` cuando el array de solución es contiguo en C (`cnp.PyArray_IS_C_CONTIGUOUS`).
    - `@property def argsort`: Reutiliza `self.fitness` sin reconstruir comprensiones de listas.
  - *Criterio de aceptación:* `tests/optimizer/test_population.py` y `tests/test_optimizer_decorator.py` pasando con 0 float-boxing en agentes nativos y sin corromper memoria en agentes dinámicos.

- **Fase 3 (Cerrada - commit `e146ee83`):**
  - *Problema:* Duplicación del motor de compilación JIT. `precompile` mantenía su propia infraestructura de generación de código y compilación basada en `pyximport` (`_builder.pyx`, `_runtime.pyx`), mientras el paquete ya contaba con el motor consolidado y limpio `clypto.compiler.compile_module`.
  - *Solución:* Se borraron `_builder.pyx` y `_runtime.pyx`, se redirigió `compiler.pyx` a `clypto.compiler.compile_module`, y se simplificó `tests/test_user_algorithm.py` para usar `cy.compile(...)`.
  - *Criterio de aceptación:* `pytest tests/test_decorator_compile.py tests/test_user_algorithm.py` pasando y eliminación de `_builder` y `_runtime` sin regresiones.

- **En curso / Pendientes:**
  - Auditoría de colecciones numéricas y algoritmos clásicos (revisar si persisten bucles Python lentos en algoritmos de mayor uso como PSO, GA, DE que puedan tiparse con `@cython.cclass` o `cdef`).
  - Revisión de overhead en `History` y exportación de métricas.
  - Preparación de PR consolidado hacia `release/v2026b2` o `master`.

---

## 2. Estado Real del Repositorio

- **`master` en `b1c6f6ff`:**
  - Los archivos no trackeados en master:
    - `benchmarks/analyze_comparison.py`, `analyze_grid.py`, `run_comparison.py`, `run_grid.py`
    - `reports/` (con `grid.json`, baselines de ruff/mypy, notas históricas)
  - **Directriz:** Son scripts de benchmarking e informes históricos generados en auditorías previas. **Deben conservarse localmente** como referencia pero **NO deben comitearse** ni ensuciar los PRs de código productivo.
- **Ramas remotas:**
  - `origin/release/v2026b2`: Creada en `b1c6f6ff` para congelar la release v2026b2.
  - `origin/refactoring/audit-simplifications`: Rama de trabajo activa donde residen los 3 commits de las Fases 1 a 3 (`f4732b50`, `bc919a95`, `e146ee83`).
  - **PR Esperado:** Se abrirá un Pull Request desde `refactoring/audit-simplifications` hacia `release/v2026b2` (o `master` según se requiera).

---

## 3. Estado de Validación de los 3 Commits

Los 3 commits en `refactoring/audit-simplifications` están **100% validados** con la suite completa:
1. **Pytest:** **420 passed** en 46.13s (incluye los 243 optimizadores en `tests/test_all_optimizer.py`, suites de decoradores, compilador y población).
2. **Ruff Linter:** `All checks passed!` (0 errores en todo el proyecto).
3. **Mypy Type Checker:** `Success: no issues found in 189 source files`.
4. **Git Pre-push hook:** Pasó automáticamente la suite completa antes de empujar a `origin`.

---

## 4. Decisiones Arquitectónicas y Restricciones (AGENTS.md)

1. **APIs Públicas Congeladas:**
   - La API pública plana en `clypto/__init__.py` no debe romperse (`Problem`, `Optimizer`, `Agent`, `Population`, `compile`, variables de espacio, etc.).
   - `clypto.get_all_optimizers()`, `get_optimizer_by_name()`, `get_optimizer_by_class()`.
2. **Cython Invariants Críticas:**
   - **`wraparound` debe permanecer en `True`** en `setup.py` (55+ archivos usan índices negativos como `pop[-1]` sobre listas).
   - `BaseAgent`, `AgentStatic` y `AgentDynamic` deben permanecer en `clypto/agents/_core.py` (Cython prohíbe herencia `cdef class` entre módulos compilados separados).
   - Los archivos `.pyi` deben sincronizarse estrictamente con cualquier cambio en `.pyx` o `.py` tipado.
   - Prohibido reintroducir dependencias visuales o de logging (matplotlib, tqdm, etc.). "Zero bloat" estricto.

---

## 5. Riesgos y Tramas Conocidas

1. **Duck-typing vs C Struct Casting:**
   - Si se optimiza código en Cython usando punteros o casts `<Agent>`, verificar siempre si el objeto puede ser un subtipo dinámico o decorado (`RuntimeAgent`). El patrón probado es:
     ```cython
     if isinstance(self.agents[0], Agent):
         # C-level pointer access
     else:
         # Python attr access directo
     ```
2. **Carreras en compilación paralela (`setup.py`):**
   - No usar `CYTHON_NTHREADS=16` ya que produce colisiones al generar archivos `.c` temporales. Usar `CYTHON_NTHREADS=4` y compilar con gcc en paralelo `-j 8`.
3. **In-place `.so` precedence:**
   - Siempre recompilar (`make compile` o `.venv/bin/python setup.py build_ext --inplace -j 8`) tras tocar archivos `.pyx` o `.py` con tipos Cython, para evitar que pytest corra sobre binarios obsoletos.

---

## 6. Próximos Pasos y Supervisión

1. **Próximas Tareas Prioritarias:**
   - **Auditoría de Algoritmos Hot:** Evaluar si hay cuellos de botella en algoritmos populares (`clypto/collection/swarm_based/PSO.py`, `DE.py`, `GA.py`) donde conversiones a arrays de NumPy puedan optimizarse a nivel C.
   - **Verificación de Benchmarks:** Ejecutar una prueba comparativa rápida contra la versión previa para registrar métricas de velocidad en `Population.fitness` y `precompile`.
   - **PR de Entrega:** Preparar el PR desde `refactoring/audit-simplifications` hacia `release/v2026b2` detallando la reducción de LOCs, eliminación de pyximport y aceleración en memoria.
2. **Esquema de Supervisión:**
   - Todo cambio debe ir acompañado de:
     - `pytest` verde (específico y suite completa).
     - `ruff check .` con 0 warnings.
     - `mypy clypto/` con 0 errores.
   - Cualquier decisión de diseño o duda de arquitectura la coordinaremos por este canal de Orca Orchestration.
   - Los reportes de resultados se guardarán en `reports/`.

---
*Fin del traspaso de contexto. `cy-audit-advisor-code` tiene el control total de la tutela.*
