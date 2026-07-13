# Plan de revisión completa (respuesta al informe de revisión externa)

Autorizado por Roberto (2026-07-13): ejecutar Fases A+B+C completas para blindar el
paper antes de enviarlo a EPJ QT. Este documento es la especificación ejecutable;
el estado vivo se refleja al final.

## Contexto
Una revisión externa (GPT, simulando revisor de EPJ QT) señaló problemas verificados:
selección Best-by-OOD sobre el test OOD (compare_extended_families.py:86-88), sweep de
bandwidth asimétrico (solo RBF recibe factores gamma; Laplaciano/Matérn a mediana fija),
circularidad parcial del mecanismo (kta_ood vs accuracy OOD), preprocesado angular
compartido, naming del natural drift (pools ref/cur de paper 2 = drift de composición de
campaña), C=1 fijo, pseudoreplicación en Wilcoxon, wording "calibrated".

Ventaja estructural: cada run guarda `extended_kernels_qsplits__summary.csv` con TODAS
las configs × splits × métricas → los protocolos de selección honesta son re-análisis puro.

Datos congelados: EMBER main (270 runs), EMBER sweep (270), netflow main (810).
Netflow sweep (405) en ejecución — 18 workers paralelos, monitor armado.

## FASE A — Re-análisis y reencuadre (sin cómputo cuántico nuevo)

### A1. Protocolos de selección honesta [script: scripts/analysis/honest_selection_analysis.py]
- **P1 (deployment, comparación principal nueva)**: por run, familia y modelo, elegir la
  config con mejor balanced accuracy en id_test; reportar su bacc en ood_test. Agregar
  por setting (media±std sobre 15 runs). Sin etiquetas OOD en la selección.
- **P2 (cross-seed robustness)**: por setting, 5-fold sobre q-split seeds: seleccionar
  por media OOD en 4 seeds, evaluar en el seed held-out. Responde "¿puedo elegir un
  kernel robusto con ejemplos previos del shift?" sin tocar el test de evaluación.
- **P3 (oracle)**: el Best-by-OOD actual, degradado a "oracle upper bound with target
  labels", sin inferencia formal.
- Familias: quantum; classical_ext; classical_orig (kernel linear|rbf_gscale*).
- Nota búsqueda: en el sweep los presupuestos quedan ~igualados (55 vs 60 configs);
  reportar recuentos de candidatos por familia y, si hay tiempo, curva rendimiento vs
  nº de configs.

### A2. Estadística jerárquica [script: scripts/analysis/hierarchical_stats.py]
- Deltas por setting bajo P1/P2; Wilcoxon POR dataset/escenario (EMBER, unsw_dos,
  unsw_recon, toniot) separadamente, no pooled.
- Test de permutación por bloques (permutar etiqueta familia dentro de setting;
  estadístico = media de deltas por escenario, luego media entre escenarios).
- Leave-one-dataset-out para los claims globales.
- CIs bootstrap (por setting, remuestreando runs) para las medias por escenario.
- Retirar "effect>1" como criterio inferencial; conservarlo solo como descriptor.

### A3. Controles del mecanismo [script: scripts/analysis/mechanism_controls.py]
- Null por permutación: kta_ood con etiquetas OOD permutadas (100×) → distribución nula
  de rho(kta_ood, bacc); reportar la correlación real contra el null.
- Cross-fitting: partir el OOD de cada run en dos mitades; KTA en mitad A, accuracy en
  mitad B. Requiere Gram OOD… NO están guardadas → aproximación: usar kta_ood de un
  q-split seed y accuracy de OTRO seed del mismo setting (cross-seed, mismos datos que A1).
- Predictores train-only: rho(eff_rank_train, bacc_ood) y rho(kta_train, bacc_ood) ya
  computables de kernel_geometry CSVs; presentarlos como la parte PREDICTIVA del
  mecanismo, con kta_ood como parte DIAGNÓSTICA.
- Correlaciones dentro de cada familia por separado (no solo mezclando).
- Reescribir §mecanismo: "post-hoc geometric association" + parte predictiva train-only;
  distinguir explícitamente kta_ood absoluto vs supervivencia ID→OOD.

### A4. Renombrados y prosa
- m1/m2: "label-conditional tail shifts" (constructed OOD splits), no "covariate shift".
- natural drift → "attack-campaign regime shift" (documentar ref=sin la campaña,
  cur=con ella); suavizar "uses no synthetic construction at all".
- "calibrated uncertainty/probabilities" → "probabilistic predictions with calibration
  assessment". Añadir reliability diagrams si es barato; NLL/Brier con CIs.
- Título: quitar "Governs". Candidato: "Quantum and Classical Kernels under Distribution
  Shift: A Controlled Study of Kernel Geometry and Out-of-Distribution Robustness"
  (decidir con Roberto en el pase de prosa).
- Abstract/conclusiones: liderar con P1 (selección honesta); Best-by-OOD como oracle;
  claims por-benchmark, no universales; p-values por escenario.
- "symmetric bandwidth tuning" → descripción literal (gamma para RBF, escalas de ángulo
  para mapas cuánticos, longitud de escala para Laplaciano/Matérn tras B1).

### A5. Reproducibilidad de pools de red
- Copiar de C:/Users/masteria.DOMINE/RF/paper_2 el pipeline raw→ref/cur (UNSW-NB15,
  ToN-IoT) a este repo (src/utils/netflow/ o scripts/data/), con documentación del
  criterio exacto (qué campaña se excluye del ref, timestamps/sesiones si existen),
  hashes de los CSVs y actualización del README.

## FASE B — Cómputo clásico nuevo (sin statevector; lanzar cuando el sweep libere CPU)

### B1. Sweep de longitud de escala clásico [nuevo runner ligero]
- Script que, por cada run existente (los 4 roots), reconstruye el embedding
  (determinista: make_embedding_pipeline(dim, seed)), computa Laplaciano y Matérn(1.5,2.5)
  con l = f × l_median, f ∈ {0.1,0.3,3,10}, entrena SVC y GPC y añade filas a
  `summary_classical_lsweep.csv` en el mismo run dir. El análisis A1 los fusiona.
- Coste: clásico puro, n≤4000; con 18 workers, horas.

### B2. Ablación de preprocesado
- Mismo runner con representación alternativa: MaxAbs+SVD+Standard SIN MinMax[0,π].
  Solo kernels clásicos (los cuánticos requieren ángulos por construcción).
- Comparar: ranking de kernels clásicos y descriptores geométricos (eff rank del lineal!)
  en ambas representaciones → responde si el rango casi-1 del lineal es artefacto del
  angle mapping.
- Subconjunto suficiente: EMBER main grid completo + un escenario de red (unsw_dos),
  15 runs. Ampliar si los resultados difieren.

### B3. Sensibilidad a C (clásica completa)
- SVC con C ∈ {0.01,0.1,1,10,100} para todos los kernels clásicos (recomputando Grams,
  barato). Curvas rendimiento-vs-C; mostrar si el orden entre kernels depende de C=1.

## FASE C — Sensibilidad a C cuántica (subconjunto)
- C ∈ {0.01,0.1,1,10,100} para los 4 mapas cuánticos en un subconjunto representativo:
  EMBER m1+m2 tamaño S (q1000, 5 dims) y unsw_dos natural S, 1 master seed × 5 qs × 3 s.
  Statevector n=1000 — asequible. Si el orden es estable, generalizar por argumento;
  si no, ampliar.

## Cierre (tras A+B+C)
1. Reescritura de manuscrito (título, abstract, métodos-selección, resultados con P1/P2,
   mecanismo reformulado, stats por escenario, limitaciones actualizadas).
2. Tablas/figuras nuevas desde los análisis (extender make_v2_tables o script nuevo).
3. Recompilar, verificación numérica prosa↔CSVs.
4. README + CITATION 0.3.0, commit, tag, push, release v0.3.0 → Zenodo.
5. Informe final a Roberto con el veredicto de cada protocolo honesto.

## Estado vivo
- [x] Plan aprobado (13-jul)
- [x] A1 script (`honest_selection_analysis.py`) + resultados en
      `results/honest_selection/` para ember_main, ember_sweep, netflow_main
- [ ] A1 netflow sweep (espera fin del sweep)
- [x] A3 parcial (`mechanism_controls.py` → `results/mechanism_controls/`);
      pendiente: null por permutación de etiquetas + cross-fitting (necesita
      recomputar Grams en subset q1000 — unir al batch de Fase B)
- [x] A2 (`hierarchical_stats.py` → `results/honest_selection/hier_stats.csv`).
      TITULAR REVISADO: netflow GPC vs classical_ext sobrevive P1 (+0.017,
      perm p=1e-4, LODO +0.015..+0.022) y P2 (+0.020, p=1e-4). SVC: P1 nulo
      (p=0.29), P2 pequeño (+0.006, p=7e-4) → claim dependiente de clasificador.
      EMBER vs ext: agregado nulo en P1/P2 (m2 positivo y m1 negativo se
      cancelan) → contar la historia POR MECANISMO de shift, no pooled.
- [ ] A4 prosa (tras B), A5 pipeline de pools
- [ ] B1, B2, B3 (esperan CPU libre), C
- [ ] Cierre

## Hallazgos clave hasta ahora (13-jul noche)
**A1 — la ventaja cuántica bajo selección honesta se vuelve REGIME-DEPENDENT pero real:**
- EMBER m2 (centroide): sobrevive con claridad — P1/P2 positivos, 8-9/9 wins,
  p=0.004-0.03 vs classical_ext, en main y sweep.
- EMBER m1 (sparsity): SE INVIERTE vs classical_ext bajo P1 (svc sweep: 0/9,
  p=0.004 a favor del clásico). El oracle P3 enmascaraba esto.
- Netflow vs classical_ext, GPC: P2 positivo en LOS 6 grupos (p<=0.04); P1 positivo
  y significativo en natural drift de los 3 escenarios. SVC: dependiente de régimen
  (se invierte en toniot natural P1: 0/9). GPC es el clasificador donde la historia
  cuántica aguanta.
- Presupuestos de candidatos: main 35 clásicas vs 20 cuánticas (favorece clásico);
  sweep 55 vs 60 (~igualado). Reportar en el paper.
**A3 — el mecanismo se REFUERZA al separar honesto/diagnóstico:**
- spec_train_eff_rank (train-only, honesto): mediana rho 0.59-0.92 y frac>0 >=0.87
  en 12/16 grupos×modelo (débil en unsw m2_centroid ~0.15, como ya admitía el paper).
  A menudo MÁS fuerte que kta_ood. → La parte predictiva del mecanismo es honesta.
- kta_drop_id_to_ood: rho mediana -0.2..-0.78 (más caída → peor OOD) — la
  supervivencia como diagnóstico funciona en la dirección correcta.
- kta_train/kta_id NO predicen (rho ~0, a veces negativa) → el resultado negativo
  del paper (sin regla a priori por alineamiento) confirmado con protocolo limpio.
- Dentro de quantum-only en unsw m2_centroid, eff_rank frac>0 = 0 → reportar como
  límite del mecanismo.
