# PLAN.md — Revisión científica v0.4 del estudio de kernels bajo distribution shift

## 0. Propósito

Este plan convierte la revisión metodológica externa y la auditoría posterior del código en una ronda de trabajo ejecutable.

El objetivo **no** es conservar a toda costa el titular actual. El objetivo es obtener la conclusión más sólida que permitan los datos, con un protocolo que pueda resistir una revisión exigente de *EPJ Quantum Technology* o de una revista comparable.

Resultados finales posibles, todos científicamente válidos:

1. La familia clásica extendida supera a la cuántica bajo presupuesto y selección comparables.
2. No existe una diferencia robusta entre las familias una vez eliminados los sesgos del protocolo.
3. La familia cuántica conserva alguna ventaja bajo controles estrictos.
4. No existe una ventaja familiar general, pero determinadas geometrías predicen mejor comportamiento OOD independientemente de que sean clásicas o cuánticas.

**Regla de trabajo:** no redactar una conclusión más fuerte que la soportada por los análisis confirmatorios definidos antes de observar sus nuevos resultados.

---

# 1. Principios no negociables

1. **Congelar la versión actual.**
   - No modificar ni borrar la release `v0.3.0`.
   - Registrar el `HEAD`, tag, fecha, entorno y hashes de los CSV actuales.
   - Todos los resultados nuevos deben escribirse bajo `results/v4/` o una ruta claramente versionada.
   - Los scripts antiguos pueden conservarse como `legacy`, pero no deben alimentar silenciosamente tablas nuevas.

2. **Separar análisis confirmatorios y exploratorios.**
   - Antes de ejecutar los dos análisis decisivos, crear `docs/ANALYSIS_SPEC_V4.md`.
   - Declarar en ese documento:
     - endpoint primario;
     - protocolo primario;
     - familia comparada;
     - unidad de análisis;
     - tratamiento de dependencias;
     - presupuesto de candidatos;
     - semillas del remuestreo;
     - estadísticos que se reportarán;
     - criterios para elegir el titular final.
   - Cualquier análisis añadido después debe marcarse como exploratorio.

3. **No usar el OOD test para seleccionar en el protocolo principal.**
   - P1 puede usar un conjunto ID de validación.
   - El OOD test debe permanecer intocable.
   - P2 puede mantenerse, pero debe denominarse explícitamente protocolo supervisado con etiquetas OOD de realizaciones previas.

4. **No confundir repetición computacional con independencia estadística.**
   - Las 15 ejecuciones son realizaciones del pipeline, no 15 datasets independientes.
   - La inferencia debe respetar dataset, escenario, master seed, q-split y tamaño.

5. **No presentar valores p imposibles de justificar.**
   - Retirar el titular `p = 2 × 10^-4`.
   - No sustituirlo por otro valor p hasta auditar las unidades intercambiables.
   - Con pocos datasets, priorizar tamaños de efecto, intervalos y consistencia entre escenarios.

---

# 2. Endpoint primario recomendado

Definir como endpoint confirmatorio principal:

> Diferencia de balanced accuracy OOD entre la mejor configuración cuántica y la mejor configuración clásica extendida, seleccionadas mediante validación ID, bajo un presupuesto igual de candidatos y evaluadas en un OOD test no utilizado para selección.

Convención:

```text
delta = BAcc_OOD(quantum) - BAcc_OOD(classical_extended)
```

- `delta > 0`: favorable a la familia cuántica.
- `delta < 0`: favorable a la familia clásica.
- Reportar media, mediana, intervalo y distribución por dataset.
- SVC será el clasificador primario.
- GPC será un análisis secundario de robustez y de incertidumbre probabilística.
- P1 con ID-validation será el protocolo primario.
- P2 y P3 serán secundarios:
  - P2: `cross-realization OOD-supervised selection`.
  - P3: `OOD oracle upper bound`.

No definir como endpoint primario el número de settings ganados ni un valor p aislado.

---

# 3. Orden obligatorio de ejecución

## GATE 0 — Auditoría y congelación

### Tareas

- [ ] Crear rama de trabajo, por ejemplo `paper-v4-methodological-audit`.
- [ ] Guardar:
  - commit de partida;
  - tag de partida;
  - `environment.yml`;
  - versión de Python;
  - versiones de NumPy, pandas, SciPy, scikit-learn y Qiskit;
  - hashes SHA-256 de los CSV usados.
- [ ] Crear `results/v4/MANIFEST.json`.
- [ ] Crear una tabla de procedencia:
  - dataset;
  - escenario;
  - shift;
  - master seed;
  - q-split seed;
  - model seed;
  - tamaño;
  - fichero de resultados;
  - número de candidatos por familia.
- [ ] Identificar configuraciones duplicadas antes de contar presupuestos.
  - Atención especial a aliases como `linear__svc__d4` y `linear__svc_C1__d4`.
  - Deduplicar por definición funcional, no solo por nombre textual.

### Salida

```text
results/v4/audit/
  manifest.json
  file_hashes.csv
  candidate_inventory.csv
  duplicate_candidates.csv
  experiment_hierarchy.csv
```

### Criterio de aceptación

No continuar si el número real de candidatos únicos o la procedencia de algún resultado no puede reconstruirse.

---

# 4. GATE 1 — Control decisivo de presupuesto de búsqueda

## Problema

La comparación actual selecciona el mejor resultado de aproximadamente 115 configuraciones clásicas frente a 60 cuánticas. Esto puede favorecer a la familia con mayor cobertura del hiperespacio.

## Objetivo

Determinar si la paridad o ventaja clásica permanece cuando ambas familias reciben exactamente el mismo presupuesto.

## Implementación

Crear:

```text
scripts/analysis/budget_matched_selection.py
tests/test_budget_matched_selection.py
```

### 4.1 Inventario y definición de candidatos

La unidad de candidato debe incluir, como mínimo:

```text
family
kernel_base
kernel_variant
dimension
length_or_angle_scale
C
classifier
preprocessing_variant
```

Excluir duplicados exactos.

### 4.2 Comparación 60-vs-60

Para cada setting y clasificador:

1. Mantener las 60 configuraciones cuánticas únicas.
2. Muestrear 60 configuraciones clásicas únicas de la familia extendida.
3. Ejecutar la selección P1 sobre el subconjunto.
4. Evaluar el candidato seleccionado en OOD.
5. Repetir el submuestreo clásico al menos 5.000 veces usando una semilla fijada en `ANALYSIS_SPEC_V4.md`.
6. Repetir el análisis con:
   - muestreo uniforme;
   - muestreo estratificado por kernel base;
   - muestreo estratificado por kernel base × dimensión.

El resultado principal no debe depender de una sola forma arbitraria de submuestreo.

### 4.3 Curvas de presupuesto

Calcular curvas para:

```text
B ∈ {5, 10, 20, 30, 40, 50, 60}
```

Para cada familia:

- submuestrear `B` candidatos;
- seleccionar por ID-validation;
- evaluar OOD;
- repetir;
- mostrar rendimiento esperado frente a presupuesto.

Para la familia clásica puede añadirse una extensión descriptiva:

```text
B ∈ {80, 100, 115}
```

pero las comparaciones entre familias deben limitarse al rango común `B ≤ 60`.

### 4.4 Comparaciones estructuralmente emparejadas

Añadir al menos dos análisis adicionales:

1. **Base-kernel matched**
   - mismo número de kernels base por familia;
   - mismo número de dimensiones;
   - mismo número de escalas por base.

2. **Complexity-matched**
   - construir subconjuntos aproximadamente emparejados por effective rank o concentración;
   - comparar desempeño evitando que una familia cubra regiones geométricas que la otra no intenta cubrir.

### Salidas

```text
results/v4/budget/
  candidate_inventory_unique.csv
  budget_resamples_by_setting.csv
  budget_summary_by_dataset.csv
  budget_summary_global.csv
  budget_curves.csv
  matched_grid_results.csv
  fig_budget_curves.pdf
  fig_delta_distribution_60v60.pdf
```

### Estadísticos a reportar

- distribución empírica de `delta`;
- mediana y percentiles 2.5/97.5;
- proporción de remuestreos con `delta > 0`;
- proporción con `delta < 0`;
- resultados por dataset y escenario;
- sensibilidad al método de muestreo.

### Decisión editorial del GATE 1

#### Caso A — ventaja clásica robusta

Solo puede hablarse de ventaja clásica si:

- la dirección negativa aparece en la gran mayoría de remuestreos;
- es consistente en más de un dataset;
- no depende de un único kernel o escenario;
- y permanece bajo grids emparejados.

Titular posible:

> Under matched search budgets, the extended classical family equals or exceeds the fidelity-kernel family in OOD performance.

#### Caso B — intervalo centrado cerca de cero o cruza ampliamente cero

Titular:

> No robust family-level OOD advantage remains after matching search budgets and selection protocols.

Este resultado sigue siendo publicable y probablemente es el más defendible.

#### Caso C — ventaja cuántica reaparece

Titular:

> A residual quantum-kernel advantage survives matched-budget, no-OOD-test selection in specific regimes.

No generalizar fuera de los escenarios donde se observe.

### Criterio de aceptación

No continuar usando “reversal” o “classical wins” en título, abstract o conclusiones hasta cerrar este gate.

---

# 5. GATE 2 — Auditoría de independencia e inferencia válida

## Problema

El script actual cambia signos por setting, aunque los settings comparten datasets, pools, semillas y tamaños. Esto no preserva la dependencia.

## 5.1 Auditar la jerarquía real

Crear:

```text
scripts/analysis/audit_experimental_dependence.py
```

Medir y guardar:

- solapamiento de train, ID y OOD entre q-split seeds;
- solapamiento entre master seeds;
- relación de anidamiento entre tamaños;
- reutilización de muestras entre escenarios;
- hashes de índices;
- si cada master seed crea un master pool verdaderamente distinto;
- correlación intraclúster de los deltas.

Salidas:

```text
results/v4/dependence/
  sample_overlap_qsplit.csv
  sample_overlap_master_seed.csv
  nested_size_audit.csv
  intracluster_correlations.csv
  hierarchy_report.md
```

Publicar también la auditoría que ya muestra aproximadamente 0–1.2% de solapamiento OOD y 0–0.4% de train entre q-split seeds, indicando claramente cómo se calculó.

## 5.2 Eliminar el test actual como evidencia confirmatoria

- Conservar `hierarchical_stats.py` como `legacy` o marcar su output como obsoleto.
- No llamar “hierarchical permutation test” a un test que cambia signos por setting.
- No utilizar `p = 2 × 10^-4` en ningún artefacto v4.

## 5.3 Estrategia inferencial recomendada

No elegir el método definitivo hasta completar la auditoría de dependencia.

### Opción preferida si master seeds son razonablemente independientes

Usar un modelo jerárquico o bootstrap multinivel:

```text
dataset
  └── scenario/shift
       └── master_seed
            └── qsplit_seed
                 └── pipeline realization
```

El tamaño no debe tratarse automáticamente como réplica independiente si las muestras están anidadas.

Posibles enfoques:

- modelo mixto con dataset y escenario como efectos fijos, y master seed como efecto aleatorio;
- bootstrap multinivel remuestreando primero master seeds dentro de cada dataset/escenario;
- inferencia por randomization solo en unidades cuya intercambiabilidad esté demostrada.

### Opción conservadora si la independencia no puede defenderse

Tratar los datasets como cuatro estudios de caso fijos:

- efecto por dataset;
- intervalo por master seed o bootstrap de clúster cuando sea legítimo;
- media global ponderada por dataset únicamente como resumen descriptivo;
- leave-one-dataset-out;
- sin un valor p global de población.

Con cuatro datasets, no vender un test exacto por dataset como evidencia fuerte. La discreción del test debe reconocerse explícitamente.

## 5.4 Nuevo script

Crear:

```text
scripts/analysis/hierarchical_effect_estimation.py
tests/test_hierarchical_effect_estimation.py
```

Debe producir:

- efectos por dataset;
- efectos por escenario;
- media dataset-equal;
- intervalos multinivel;
- sensibilidad al nivel de clustering;
- leave-one-dataset-out;
- heterogeneidad;
- ninguna pseudo-replicación.

### Salidas

```text
results/v4/inference/
  dataset_effects.csv
  scenario_effects.csv
  hierarchical_effects.csv
  lodo_effects.csv
  sensitivity_to_clustering.csv
  fig_forest_effects.pdf
```

### Criterio de aceptación

Cada fila inferencial debe documentar:

```text
unit_of_resampling
number_of_independent_clusters
number_of_lower_level_observations
method
assumptions
```

No reportar un valor p sin esa información.

---

# 6. Separar ID-validation e ID-test

## Problema

P1 selecciona y reporta ID sobre `id_test`. Ese conjunto es realmente validación y no permite confirmar la supuesta ventaja cuántica ID.

## Diseño

Usar cuatro particiones:

```text
train
id_validation
id_test
ood_test
```

### Requisitos

- `id_validation` se usa para:
  - selección de configuración;
  - selección de C;
  - cualquier decisión de hiperparámetros.

- `id_test` se usa solo para:
  - confirmar separabilidad ID;
  - probabilistic metrics ID;
  - comparar degradación ID→OOD.

- `ood_test` no se usa para selección en P1.

### Estrategia computacional

1. Comprobar si existen predicciones o Gram blocks por muestra.
2. Si existen matrices completas para el antiguo ID set:
   - dividir determinísticamente sus índices;
   - reutilizar/slicing sin recomputar estados.
3. Si solo existen métricas agregadas:
   - regenerar las evaluaciones;
   - evitar recomputar kernels de train si pueden reutilizarse;
   - computar únicamente los nuevos bloques ID-validation/train e ID-test/train.

### Proporción

Usar una división estratificada y fijada antes de mirar resultados, por ejemplo:

```text
50% ID-validation / 50% ID-test
```

o mantener tamaños suficientes para estabilidad. Documentar la elección y no cambiarla después.

### Cambios de código

Actualizar:

```text
src/utils/**/split*
scripts/**/run_*grid.py
scripts/analysis/honest_selection_analysis.py
```

Renombrar columnas para que no haya ambigüedad:

```text
bacc_id_val
bacc_id_test
bacc_ood_test
```

### Resultado

La afirmación “quantum ID separability” solo podrá mantenerse si aparece en `id_test`, no en `id_validation`.

---

# 7. Renombrar y auditar P1, P2 y P3

## Nombres definitivos

### P1 — ID-validation deployment selection

```text
Select on ID-validation, evaluate on ID-test and OOD-test.
No OOD labels used.
```

### P2 — Cross-realization OOD-supervised selection

```text
Select using labeled OOD data from other q-split realizations,
evaluate on the held-out realization.
```

No llamarlo:

- honest no-OOD-label;
- deployment without OOD labels;
- unsupervised OOD selection.

### P3 — Same-test OOD oracle

```text
Select and evaluate on the same OOD target.
Upper bound only.
```

## Auditoría de solapamiento de P2

Crear una tabla y una figura con:

- porcentaje de muestras repetidas entre folds;
- train/ID/OOD por separado;
- media, máximo y distribución;
- interpretación.

Un solapamiento del 0–1.2% reduce la preocupación, pero no convierte P2 en un protocolo sin etiquetas OOD.

---

# 8. Regularización simétrica en SVC

## Problema

Un `C=1` fijo no representa la misma regularización funcional para kernels con espectros diferentes. El control existente es parcial y asimétrico.

## Diseño

Ejecutar el sweep completo para **todos los candidatos clásicos y cuánticos utilizados en el análisis confirmatorio**.

Grid recomendado:

```text
C ∈ {0.01, 0.1, 1, 10, 100}
```

Puede reducirse únicamente si existe una justificación previa basada en saturación, no en los resultados finales.

### Selección

- seleccionar `kernel configuration + C` usando solo `id_validation`;
- evaluar en `id_test` y `ood_test`;
- igualar el presupuesto total de combinaciones entre familias en el análisis principal;
- repetir el control 60-vs-60 considerando `C` como parte del candidato.

### Cómputo

No recomputar Gram matrices. Solo reentrenar SVC sobre matrices existentes.

### Salidas

```text
results/v4/c_sweep/
  all_candidates_c_sweep.csv
  p1_selected_by_setting.csv
  budget_matched_with_c.csv
  ordering_stability.csv
  fig_c_sensitivity.pdf
```

## GPC

No fingir una simetría inexistente.

Opciones aceptables:

1. mantener el GPC como análisis secundario con hiperparámetros fijados y declararlo;
2. realizar sensibilidad a amplitud/prior si el modelo lo permite;
3. no usar GPC para el claim familiar principal.

La vía recomendada es **SVC como endpoint primario y GPC como análisis secundario**.

---

# 9. Ablación de preprocesamiento

## Objetivo

Determinar cuánto depende la geometría clásica del mapeo angular `[0, π]`.

## Condiciones mínimas

Para los kernels clásicos:

1. SVD coordinates antes del mapeo angular.
2. Representación mapeada a `[0, π]`.

Para quantum:

3. Quantum feature maps con su entrada angular canónica.

Dejar claro que el vector reducido `z` es compartido, pero el mapeo angular forma parte de la definición del feature map cuántico. Evitar decir literalmente que las entradas numéricas finales son idénticas si no lo son.

## Métricas

- balanced accuracy ID/OOD;
- effective rank;
- KTA_ID;
- KTA_OOD;
- delta KTA;
- concentración;
- orden de kernels;
- resultado familiar P1.

## Requisito editorial

La ablación debe aparecer en el cuerpo principal, no solo en apéndice, porque ya se ha observado que el rango del kernel lineal cambia sustancialmente.

---

# 10. Replanteamiento del análisis geométrico

## Tesis permitida

> Effective rank and OOD kernel-target alignment are robustly associated with OOD performance across the evaluated kernels and shifts.

## Tesis no permitida sin evidencia adicional

> A single causal geometric mechanism explains the entire pattern.

## 10.1 Terminología

Separar siempre:

```text
KTA_OOD = alineamiento absoluto en OOD
KTA_survival = KTA_OOD - KTA_ID
```

No usar “alignment survival” para referirse a `KTA_OOD`.

## 10.2 Análisis adicionales

Crear:

```text
scripts/analysis/mechanism_robustness_v4.py
```

Realizar:

1. correlaciones dentro de setting;
2. correlaciones dentro de familia;
3. modelos parciales controlando:
   - familia;
   - kernel base;
   - dimensión;
   - escala;
   - dataset;
   - shift;
   - clasificador;
4. leave-one-dataset-out:
   - ajustar relación geométrica en tres datasets;
   - evaluar capacidad predictiva en el cuarto;
5. matching aproximado por effective rank:
   - comparar kernels clásicos y cuánticos con rangos similares;
   - comprobar si la etiqueta de familia aporta información residual;
6. análisis de mediación exploratorio:
   - escala → effective rank → KTA_OOD → accuracy OOD.

## 10.3 Cross-fitting y permutación

Mantener los controles existentes, pero describirlos con precisión:

- la permutación rechaza el alineamiento esperado bajo etiquetas aleatorias;
- el cross-fit evita compartir exactamente las mismas muestras entre KTA y accuracy;
- ninguno de los dos demuestra causalidad completa.

## 10.4 Endpoint geométrico confirmatorio

Elegir uno:

```text
rho(effective_rank_train, KTA_OOD)
```

o

```text
predictive performance of train-only geometry for OOD accuracy
```

La segunda opción sería más fuerte si puede construirse sin etiquetas OOD.

---

# 11. Nombre de los shifts

## EMBER

Usar:

- `within-class sparsity-tail shift`;
- `within-class train-geometry-tail shift`;
- o el término compacto acordado.

No llamarlos covariate shift salvo demostrar `P(Y|X)` invariante.

## Network

Reemplazar de forma uniforme:

```text
natural regime drift
```

por:

```text
held-out attack-campaign and capture-partition shift
```

o:

```text
attack-campaign regime shift
```

Explicar:

- ataque objetivo ausente del train;
- cambio de partición de captura en UNSW;
- prevalencias balanceadas;
- no es un split temporal por timestamp;
- es el mecanismo más operacional de los incluidos, pero sigue siendo construido.

Buscar y corregir en:

```text
manuscript/sn-article.tex
README.md
CITATION.cff
figuras
tablas
captions
scripts/reporting/
nombres visibles de columnas
```

Los nombres internos de carpetas pueden mantenerse por compatibilidad si se documenta el alias.

---

# 12. Incertidumbre y calibración

## Terminología

Sustituir:

```text
calibrated uncertainty
calibrated predictive probabilities
```

por:

```text
probabilistic uncertainty estimates
Laplace-approximation predictive probabilities
```

salvo que se aplique y valide un procedimiento de calibración separado.

## Protocolo

- seleccionar configuración por P1 usando `id_validation`;
- evaluar en `id_test` y `ood_test`;
- nunca seleccionar por Best-by-OOD para las figuras principales de incertidumbre.

## Métricas

- NLL/log-loss;
- Brier score;
- ECE con intervalos;
- reliability diagrams;
- predictive entropy;
- risk–coverage o selective prediction;
- opcionalmente adaptive ECE para comprobar sensibilidad al binning.

No interpretar únicamente “más entropía OOD” como buena incertidumbre.

---

# 13. Repeticiones y semillas

Crear una subsección explícita:

## Sources of variability

Documentar qué modifica:

- `master_seed`;
- `qsplit_seed`;
- `model_seed`;
- SVD;
- median heuristic;
- clasificador;
- quantum kernel.

Usar en todo el texto:

```text
15 pipeline realizations
```

o:

```text
5 q-split realizations × 3 model-seed realizations
```

No usar “15 independent repetitions” ni permitir que el lector infiera independencia.

---

# 14. Rediseño del manuscrito

## 14.1 Historia principal recomendada

### Movimiento 1 — Baselines débiles fabrican una ventaja aparente

- quantum vs linear+RBF;
- mostrar que la conclusión optimista se reproduce;
- P3 solo como auditoría de cómo surge el resultado.

### Movimiento 2 — Controles honestos eliminan una conclusión universal

- selección ID-validation;
- presupuesto 60-vs-60;
- regularización simétrica;
- resultados por dataset;
- conclusión determinada por GATE 1.

### Movimiento 3 — La geometría cruza las fronteras de familia

- effective rank;
- KTA_OOD y KTA_survival separados;
- asociación, no causalidad;
- fidelidad situada dentro del continuo clásico;
- implicaciones para diseño de kernels.

## 14.2 Título

Mantener provisionalmente el título conservador actual:

> Quantum and Classical Kernels under Distribution Shift: A Controlled Study of Kernel Geometry and Out-of-Distribution Robustness

No cambiarlo hasta cerrar GATE 1 y GATE 2.

## 14.3 Abstract

El abstract final debe contener:

- problema;
- protocolo;
- principal control de selección;
- resultado budget-matched;
- asociación geométrica;
- limitación de simulación exacta;
- ninguna afirmación de ventaja tecnológica.

Evitar:

- más de un valor p;
- “single mechanism explains”;
- “fully fair”;
- “calibrated”;
- “natural drift”;
- “reversal” si el resultado no sobrevive 60-vs-60.

## 14.4 Longitud

Objetivo:

- manuscrito principal: aproximadamente 25–35 páginas incluyendo referencias;
- material suplementario para:
  - resultados per-setting;
  - grids completos;
  - tablas de semillas;
  - métricas probabilísticas secundarias;
  - detalles exhaustivos del pipeline;
  - P3 oracle;
  - análisis exploratorios.

## 14.5 Endpoint declarado

Incluir una frase explícita en métodos:

> The primary confirmatory comparison is the budget-matched P1 SVC OOD balanced-accuracy difference between the quantum and extended-classical families.

El resto debe identificarse como secundario o exploratorio.

---

# 15. Figuras y tablas v4

Crear exclusivamente desde `results/v4/`:

```text
scripts/reporting/make_v4_tables.py
scripts/reporting/make_v4_figures.py
```

## Figuras principales

1. Diagrama de protocolo con:
   - train;
   - ID-validation;
   - ID-test;
   - OOD-test;
   - P1/P2/P3 correctamente nombrados.

2. Curva de rendimiento frente a presupuesto.

3. Forest plot de efectos por dataset.

4. Distribución 60-vs-60.

5. Continuo geometría:
   - effective rank;
   - KTA_OOD;
   - familias diferenciadas;
   - sin interpretar causalidad.

6. Ablación angular/no angular.

## Tablas principales

1. Diseño experimental.
2. Endpoint primario por dataset.
3. Sensibilidad al presupuesto.
4. Sensibilidad al clustering/inferencia.
5. Resultados ID-test y OOD-test.
6. Coste computacional.

## Regla

Cada tabla debe contener una columna indicando:

```text
selection data
evaluation data
candidate budget
unit of uncertainty
```

---

# 16. Reproducibilidad y CI

## 16.1 Comando maestro

Crear:

```text
python scripts/reproduce_v4.py --stage audit
python scripts/reproduce_v4.py --stage budget
python scripts/reproduce_v4.py --stage inference
python scripts/reproduce_v4.py --stage manuscript
```

El comando debe:

- validar inputs;
- comprobar hashes;
- no sobrescribir outputs v0.3;
- registrar logs;
- producir manifest;
- fallar ante candidatos duplicados inesperados.

## 16.2 Tests mínimos

- selección P1 nunca consulta OOD;
- P2 usa folds distintos y se etiqueta como OOD-supervised;
- P3 es el único same-test oracle;
- budget sampling devuelve exactamente B candidatos únicos;
- resultados reproducibles con semilla fija;
- agrupación jerárquica no permuta observaciones por debajo del clúster declarado;
- nombres KTA no se intercambian;
- ninguna tabla v4 lee rutas legacy;
- ausencia de `calibrated uncertainty` y `natural drift` en texto público.

## 16.3 README

Actualizar solo al cerrar los resultados.

Debe incluir:

- commit/tag exactos;
- endpoint primario;
- resultado final sin exageración;
- límites de simulación;
- comandos de reproducción;
- disclaimer de datasets.

---

# 17. Análisis de coste y relevancia cuántica

Para mejorar el encaje en *EPJ Quantum Technology*, añadir:

1. coste de construcción de Gram matrices clásicas y cuánticas;
2. memoria;
3. crecimiento con `n` y `d/qubits`;
4. coste relativo por candidato;
5. sensibilidad a shots simulados o perturbación del kernel.

## Control mínimo de shots/noise

Sin necesidad de hardware real:

- partir de kernels statevector;
- simular estimación con presupuestos de shots;
- proyectar/corregir PSD cuando proceda;
- medir estabilidad de:
  - accuracy;
  - effective rank;
  - KTA;
  - selección de configuración.

Shots sugeridos:

```text
{128, 512, 2048, 8192}
```

Este análisis puede realizarse en un subconjunto representativo si el coste total es excesivo, pero debe estar predefinido.

La conclusión debe ser:

- qué hallazgos geométricos son ideales/statevector;
- cuáles sobreviven estimación finita;
- y qué no puede extrapolarse a hardware.

---

# 18. Criterios finales para elegir el framing

## Framing 1 — No robust family advantage

Usar si:

- 60-vs-60 centra el delta alrededor de cero;
- la dirección cambia entre datasets;
- o los intervalos honestos no permiten una ventaja global.

Mensaje:

> Apparent quantum advantages are highly sensitive to baseline breadth, search budget, regularization, and selection protocol; after controlling them, no robust family-level advantage remains.

## Framing 2 — Classical parity/advantage

Usar si:

- la dirección clásica permanece bajo presupuesto igual;
- aparece en varios datasets;
- es robusta a C, preprocessing y clustering;
- no depende de un único kernel.

Mensaje:

> Heavy-tailed classical kernels reproduce or exceed the OOD geometry of fidelity kernels under matched evaluation conditions.

## Framing 3 — Regime-specific quantum advantage

Usar si:

- la ventaja cuántica permanece en determinados datasets;
- no aparece como universal;
- sobrevive presupuesto, regularización y selección.

Mensaje:

> Fidelity kernels provide a regime-specific benefit rather than a general OOD advantage.

## En los tres casos

La contribución central puede seguir siendo:

> una auditoría controlada de cómo la geometría y el protocolo determinan las conclusiones sobre quantum kernels bajo shift.

---

# 19. Orden de commits recomendado

1. `audit: freeze v0.3 inputs and document v4 analysis specification`
2. `analysis: add candidate deduplication and budget-matched selection`
3. `analysis: audit split overlap and experimental hierarchy`
4. `stats: replace pseudo-hierarchical permutation with cluster-aware effect estimation`
5. `protocol: introduce ID validation and untouched ID test`
6. `experiments: complete symmetric SVC C sweep`
7. `analysis: add preprocessing and mechanism robustness controls`
8. `paper: rename protocols shifts and uncertainty claims`
9. `reporting: generate v4 tables and figures from versioned outputs`
10. `docs: shorten manuscript and pin reproducibility metadata`
11. `release: publish audited v0.4 artifact`

No mezclar resultados, código y reescritura masiva en un único commit.

---

# 20. Definition of Done

La revisión v4 estará terminada únicamente cuando:

- [ ] El resultado principal usa presupuesto igual.
- [ ] P1 selecciona en ID-validation y evalúa en ID-test/OOD-test.
- [ ] P2 está correctamente nombrado.
- [ ] La auditoría de solapamientos está publicada.
- [ ] El test pseudo-jerárquico y `p=2e-4` han desaparecido.
- [ ] La incertidumbre se cuantifica con unidades legítimas.
- [ ] C se ajusta simétricamente para SVC.
- [ ] La ablación angular aparece en el cuerpo principal.
- [ ] KTA_OOD y KTA_survival están separados.
- [ ] El mecanismo se presenta como asociación salvo evidencia adicional.
- [ ] “Natural drift” se ha sustituido por un nombre exacto.
- [ ] “Calibrated uncertainty” se ha eliminado salvo calibración real.
- [ ] Las 15 ejecuciones se describen como realizaciones del pipeline.
- [ ] El manuscript principal se ha reducido.
- [ ] Todas las tablas y figuras se generan desde `results/v4/`.
- [ ] El commit/tag exacto aparece en el paper.
- [ ] Existe un comando reproducible end-to-end.
- [ ] El abstract coincide literalmente con el resultado del GATE 1 y GATE 2.
- [ ] Una lectura crítica final no encuentra ninguna afirmación que dependa de un análisis oracle presentado como deployable.

---

# 21. Primera tarea para Claude Code en plan mode

Antes de implementar, Claude debe devolver:

1. mapa exacto de los archivos que modificará;
2. jerarquía experimental deducida del código;
3. inventario real y deduplicado de candidatos;
4. disponibilidad de índices por muestra y Gram matrices reutilizables;
5. coste estimado en número de fits clásicos y kernels cuánticos nuevos;
6. diseño concreto del control 60-vs-60;
7. propuesta inferencial condicionada a la auditoría de independencia;
8. riesgos de compatibilidad con los resultados v0.3;
9. secuencia de commits;
10. criterios automáticos de validación.

No comenzar por reescribir el manuscrito. Empezar por **GATE 0, GATE 1 y GATE 2**, porque esos resultados determinan la historia final del paper.
