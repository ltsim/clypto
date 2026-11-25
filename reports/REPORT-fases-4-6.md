# Reporte: Auditoría de Simplificación y Rendimiento (Clypto)

**Sesión:** `cy-audit-advisor-code`
**Fecha:** 2026-11-14
**Rama:** `refactoring/audit-simplifications` (11 commits sobre `b1c6f6ff`, sin push)
**Estado:** Fases 1–6 cerradas y verificadas; 2 correcciones nuevas pendientes de decisión.

---

## 1. Resumen

Sobre las Fases 1–3 (compilador, population, tests) se auditó el resto de
`clypto/native/collection/` (243 clases) buscando trabajo invariante al bucle,
argumentos redundantes y duplicación. Sólo se aplicaron cambios con equivalencia
numérica demostrable; se verificó fitness final **bit a bit idéntico** en procesos
frescos.

Después se revisó el trabajo de optimización de hot-path (PSO, population, utils,
bounds, DE) que ya estaba en la rama, y en esa revisión aparecieron **dos bugs
reales**, uno de ellos de reproducibilidad que afectaba a cuatro algoritmos.

---

## 2. Commits

### Fases 4–5 — simplificación y hoisting

| Commit | Tipo | Alcance |
|---|---|---|
| `049d0d35` | `perf(collection)` | Hoisting de agregados invariantes en `DevGCO`, `OriginalEVO` |
| `3e69ba7a` | `chore(collection)` | Elimina 9 argumentos redundantes `cls=cy.Population` (firma pública intacta) |
| `5ea34989` | `perf(collection)` | Hoisting de ~20 agregados fuera de bucles por individuo en `OriginalHHO`, `IncrementalGWO`, `OriginalAGTO`, `MGTO`, `OriginalESO`, `OriginalINFO`, `OriginalSOO` |

### Fase 6 — hot path (revisada y verificada en esta sesión)

| Commit | Contenido |
|---|---|
| `d0adefb0` | `update_velocity` de PSO: bucle con punteros C y un solo `np.clip` |
| `2bce0578` | `reset_solution`/`correct_solution`: punteros y clip único para bounds float |
| `0aa483a7` | `kway_tournament`: `np.argsort(kind="stable")` en vez de `sorted` + listas |
| `94d1aa6a` | `NumberBounds.generate`: `low + (up-low)*random(n)` inline |
| `f746a501` | DE/JADE/SHADE: pools por lista en vez de `set` restado |

### Correcciones nuevas

| Commit | Contenido |
|---|---|
| `4f2c0832` | **fix(bounds):** los fast paths inline producían `inf`/`nan` cuando `up - low` desbordaba, donde `Generator.uniform` lanza `OverflowError`. Se exige intervalo finito en el flag y también en `reset_solution`. |
| `7654d659` | **fix(collection):** `JADE`, `OriginalSHADE`, `L_SHADE` y `OriginalLSHADEcnEpSin` usaban `scipy.stats.{cauchy,norm}.rvs` sin `random_state`, consumiendo el RNG global y **rompiendo `solve(seed=...)`**. Los 8 call sites ahora pasan `random_state=self.generator`. |

---

## 3. Verificación

### 3.1 Puertas de calidad (HEAD)

```
make build-ext                              OK
uv run --no-sync pytest tests/ -q           426 passed
ruff check clypto/ tests/                   clean
mypy clypto/ --config-file pyproject.toml   clean (189 archivos)
```

### 3.2 Equivalencia de las primitivas nuevas

Script `/tmp/opencode/equiv_check.py` compara cada fast path contra la expresión
NumPy/Python exacta que sustituyó, con dos generadores sembrados igual, y además
comprueba que el **stream del RNG queda en la misma posición**:

- `NumberBounds.generate` (incluido `low == up`): bit a bit.
- `kway_tournament` (min/max × reverse × k, con empates): mismos índices.
- `reset_solution` (con y sin valores fuera de rango): bit a bit.
- `update_velocity` (w escalar float/int/vector, con y sin clip): bit a bit.

Resultado: **ALL BIT-EXACT**.

### 3.3 Extremo a extremo contra `5ea34989` (28 algoritmos, seed 42, proceso fresco)

25 de 28 reproducen fitness y solución exactos. Los 3 restantes son
`JADE`, `OriginalSHADE`, `L_SHADE`, que cambian **por el fix intencional** de
`rvs` (antes no eran reproducibles entre procesos; ver §5). Una vez fijado
`np.random.seed(0)` en ambos lados, los 7 algoritmos afectados por DE/SHADE
coinciden exactamente, lo que demuestra que `f746a501` es equivalente:

```
PASS JADE            7.093926607887578e-08 | 7.093926607887578e-08
PASS SADE            3.0949103670452784e-09 | 3.0949103670452784e-09
PASS SAP_DE          7.90112316254054      | 7.90112316254054
PASS OriginalSHADE   1.6267968889980405e-09 | 1.6267968889980405e-09
PASS L_SHADE         2.789431429263855e-10 | 2.789431429263855e-10
PASS OriginalDE      17.59063104139374     | 17.59063104139374
PASS LSHADEcnEpSin   1.0580163531120979e-07 | 1.0580163531120979e-07
```

**Cautela sobre tiempos:** la máquina estuvo cargada (variaciones de 650–1720 ms
entre corridas del mismo algoritmo). No se reportan multiplicadores de velocidad;
sólo el trabajo estructural eliminado.

---

## 4. Optimizaciones evaluadas y descartadas

| Candidato | Motivo |
|---|---|
| `population.sort()` por iteración en SSA/NRO | La población sí muta en modo secuencial. |
| Media de `CHIO` hoisteable | La población cambia por índice dentro del bucle. |
| Quitar `sort` en `after_evolve` | ~0.3–0.5 % y riesgo de cambiar identidad de empates. |
| Fast-path de evaluación por lotes | 2–5 % extremo a extremo; toca el contrato de evaluación. |
| Fusión de operadores PSO/DE/GA | Riesgo alto, ganancia modesta, difícil de probar bit a bit. |
| Overhead de `History` | Cero cuando el tracking está desactivado (verificado). |

---

## 5. Hallazgo: `solve(seed=...)` no reproducía en 4 algoritmos

`JADE`, `OriginalSHADE`, `L_SHADE` y `OriginalLSHADEcnEpSin` llamaban a
`scipy.stats.cauchy.rvs` / `norm.rvs` sin `random_state`. scipy usa entonces el
RNG global de NumPy, que `solve(seed=...)` no siembra: dos procesos con la misma
semilla daban fitness distintos. Reproducido con `PYTHONHASHSEED` fijo (no era
hash randomization) y localizado en la primera evaluación tras la población
inicial. Arreglado en `7654d659`; test de regresión en
`tests/test_all_optimizer.py::test_seeded_solve_ignores_the_global_numpy_rng`.

Esto también explica los "mismatches" que aparecían al comparar esas tres
algoritmos: eran ruido, no regresiones.

---

## 6. Riesgo previo: fuga de estado entre solves en un mismo proceso

La observación antigua (DevGCO daba `580.19` si antes corría OriginalEVO en el
mismo intérprete) **ya no se reproduce en HEAD**. Probado con el caso exacto
(pop=100, dims=30, epoch=200, seed 42), ambos órdenes, mismo proceso y procesos
separados: `DevGCO=544.2988788667016`, `OriginalEVO=4.5258559696622775` en los
cuatro casos. Riesgo rebajado; sigue pendiente la auditoría de memoria planificada
en `reports/AUDIT-MEMORY-LEAKS-PLAN.md` (valgrind no está instalado en el host).

---

## 7. Pendientes

1. Decidir si se conserva `7654d659` (cambia la trayectoria de 4 algoritmos; los
   hace deterministas, pero quien haya guardado resultados de esos algoritmos verá
   otros números).
2. Preparar PR de `refactoring/audit-simplifications` → `release/v2026b2` / `master`
   (requiere aprobación; sin push).
3. Opcional: dedupe de `EliteMultiGA`/`EliteSingleGA`, huecos en `docs/REFERENCES`.
4. Ejecutar la auditoría de memoria (`AUDIT-MEMORY-LEAKS-PLAN.md`) cuando haya
   valgrind o se decida ir sólo con ASan/RSS.
