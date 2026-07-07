# Paper v2 — Expansion Roadmap

Branch: `paper-v2-expansion`. Objetivo: responder a las tres críticas de la carta de
rechazo de Quantum Machine Intelligence (2026) y reenviar el manuscrito ampliado.
La v1 se publica en arXiv tal cual; la v2 se subirá como revisión del mismo arXiv.

## Contexto

- Rechazo 1 (Machine Learning): contribución demasiado aplicada para revista
  generalista de ML. No se reenvía ahí.
- Rechazo 2 (Quantum Machine Intelligence): tres objeciones concretas que
  estructuran esta ampliación (ver workstreams).
- El Paper 2 ("Validate Before Commit", NIDS) cita este trabajo como motivación
  ligera; se citará la versión arXiv (placeholder [30] de su bibliografía).

## Workstream A — Análisis mecanístico (prioridad 1, sin experimentos nuevos)

Responde a: *"no theoretical account of why fidelity-based quantum kernels should
outperform classical kernels under distribution shift"*.

No se busca un teorema, sino cantidades geométricas computables que **predigan**
los regímenes clásico-favorables ya observados (concentrados en m2 / seed 42):

- [x] Geometric difference g(K_C, K_Q) de Huang et al. (2021), por setting y dimensión.
- [x] Espectro de las matrices de kernel (decaimiento de autovalores) en ID vs OOD.
- [x] Concentración del kernel (media/varianza de entradas off-diagonal, efecto
      bandwidth) en ID vs OOD, por familia.
- [x] Kernel–target alignment en ID vs OOD.
- [x] Test de la hipótesis sobre el grid (qsplit seed 42): el rango efectivo del
      kernel cuántico seleccionado (AUC 0.89, MW p=0.018, Pearson r=0.61 con
      ΔOOD) y el drop relativo de alignment ID→OOD (AUC 0.14, p=0.035) están
      asociados a los settings clásico-favorables. Los 3 settings m2/ms42
      corresponden a PauliXZ con rango efectivo colapsado (9–15), geometría
      casi clásica. Código: `src/analysis/` + `scripts/analysis/`.
- [x] Robustez: geometría con los 5 qsplit seeds (90 combos). El rango efectivo
      del kernel cuántico seleccionado se REFUERZA: AUC 0.929 (p=0.008) sobre
      medias entre seeds, rango por seed [0.84, 0.95]. El drop de KTA se
      debilita (AUC ~0.73 en dirección consistente, no significativo). La
      geometric difference a λ=0.01 correlaciona NEGATIVAMENTE con ΔOOD
      (r=−0.52, p=0.026): grande-g es necesario pero no suficiente (consistente
      con Huang et al.).
- [x] Variante a priori (selección por KTA de train): NO predice (AUC 0.38-0.46).
      Resultado negativo honesto — el mecanismo es diagnóstico/explicativo, no
      un criterio de despliegue. Encaja con la lección del Paper 2 (la señal
      pre-despliegue barata no existe; hace falta validación).

Insumo: matrices de kernel ya generadas por el pipeline existente (results/).

## Workstream B — Baselines probabilísticos y más kernels (prioridad 2)

Responde a: *"probabilistic approaches (GPs, DKL, Bayesian SVMs) entirely absent;
gains over two narrow classical baselines"*.

- [x] Gaussian Process classifier (Laplace, R&W Alg. 3.1/3.2) con kernel
      precomputado, mismo protocolo. Métricas: log-loss, Brier, ECE, entropía
      predictiva ID vs OOD. Runner: `src/experiments/ember/extended/`.
- [x] Kernels clásicos adicionales: poly 2/3, Laplaciano, Matérn 3/2 y 5/2
      (median heuristic train-only).
- [x] Resultado (18 settings, 1 run/celda, qs42/ms42): la ventaja OOD cuántica
      se sostiene vs linear+RBF (SVC 15/18 +0.035; GPC 16/18 +0.031) pero vs la
      familia ampliada queda en tablas con SVC (7/18, −0.002) y en mayoría débil
      con GPC (12/18, +0.005). El challenger es el Laplaciano (seleccionado
      ~13/18). La ventaja ID sobrevive a todo (16-18/18). Geometría: gradiente
      monótono rango efectivo ↔ supervivencia del KTA dentro de la familia
      clásica (linear 1.2/−0.017 → Laplaciano 17.7/+0.019) — el mecanismo del
      workstream A explica el resultado del B.
- [x] Runs repetidos completados (18 settings × 15 runs = 270). Números finales
      (Best-by-OOD, medias de 15 runs): cuántico vs linear+RBF gana 17/18 (SVC,
      Δ+0.037, effect>1 en 12) y 15/18 (GPC, Δ+0.028, effect>1 en 11); vs
      familia ampliada mantiene mayoría de wins (SVC 15/18, GPC 11/18) pero el
      Δ medio colapsa a +0.002/+0.004 y el effect>1 desaparece (0-1/18) — la
      ventaja OOD es estadísticamente indistinguible del ruido contra el
      Laplaciano (seleccionado 13-14/18, casi siempre d12). Best-by-ID:
      cuántico 18/18 contra AMBAS familias, effect>1 en 11-18 — la ventaja de
      separabilidad ID es el resultado robusto.
- [ ] Opcional: justificar en el paper por qué GPC-Laplace cubre la objeción de
      incertidumbre (no hace falta DKL/SVM bayesiano).

## Workstream C — Datasets adicionales (prioridad 3, mayor riesgo)

Responde a: *"empirical scope confined to a single dataset family (EMBER)"*.

Reutilizar los pipelines del Paper 2 (features numéricas estandarizadas + reducción
a baja dimensión, formato ya compatible con el kernel-swap):

- [x] UNSW-NB15 (DoS, Reconnaissance) y ToN-IoT (Scanning) desde los pools
      ref/cur del Paper 2. Pipeline: `src/utils/netflow/` + `scripts/netflow/`.
- [x] Mecanismos de shift: `m2_centroid` (análogo fiel del m2) y `natural_cur`
      (drift natural ref→cur, mecanismo nuevo que EMBER no tenía).
- [x] Grid 54 settings (3 escenarios × 2 shifts × 3 ms × 3 tamaños, 1 run/celda):
      cuántico vs linear+RBF OOD 51-52/54; **vs familia ampliada 40/54 (SVC,
      Δ+0.009) y 51/54 (GPC, Δ+0.022)** — en flujos de red la ventaja sobrevive
      a la familia ampliada, al contrario que en EMBER. Con SVC el drift natural
      favorece más al cuántico (23/27) que el sintético (17/27).
- [x] Test cross-dataset del mecanismo (correlaciones dentro de setting entre
      geometría y OOD bacc, ~70 settings, 40-55 celdas c/u): **kta_ood es el
      predictor universal** (mediana ρ 0.46-0.74, positivo en 94-100% de
      settings en los 4 datasets y ambos clasificadores); eff_rank es fuerte en
      EMBER/ToN-IoT (ρ~0.8) y débil en UNSW (ρ 0.14-0.38). Refinamiento: rango
      alto = capacidad estructural (a priori), supervivencia del alignment =
      predictor próximo (diagnóstico). `scripts/analysis/mechanism_generalization.py`.
- [ ] Pendiente opcional: CICIDS2017 (requiere prep desde raw), runs repetidos
      del grid netflow.

Riesgo asumido: la ventaja cuántica puede no replicar fuera de EMBER (el Paper 2
sugiere escepticismo en la tarea de monitorización). Si no replica, el titular pasa
a "mapa de regímenes": la conclusión regime-dependent sigue siendo publicable y
consistente con el framing conservador de la v1.

## Checklist arXiv (v1, previo a todo lo anterior)

- [x] Recuperar el proyecto compilable completo: ahora en `manuscript/`
      (`sn-article.tex`, `sn-bibliography.bib`, `sn-jnl.cls`, `sn-mathphys-ay.bst`,
      3 figuras `fig_*.png` y el PDF compilado).
- [ ] Compilar y añadir el `.bbl` (arXiv no ejecuta BibTeX).
- [ ] Subir a arXiv: categoría primaria `quant-ph`, cross-list `cs.LG` (y opcional
      `cs.CR`). Licencia recomendada: arXiv non-exclusive license.
- [ ] Al recibir el identificador, rellenar [30] en el Paper 2 y actualizar
      README/CITATION.cff de este repo con la referencia al preprint.

## Venues candidatos para la v2

1. Machine Learning: Science and Technology (IOP) — mejor encaje.
2. EPJ Quantum Technology.
3. Quantum Information Processing / IEEE Trans. on Quantum Engineering.
4. Alternativa con reframing de seguridad: Computers & Security, JISA.
