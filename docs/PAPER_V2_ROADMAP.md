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

- [ ] Geometric difference g(K_C, K_Q) de Huang et al. (2021), por setting y dimensión.
- [ ] Espectro de las matrices de kernel (decaimiento de autovalores) en ID vs OOD.
- [ ] Concentración del kernel (media/varianza de entradas off-diagonal, efecto
      bandwidth) en ID vs OOD, por familia.
- [ ] Kernel–target alignment en ID vs OOD.
- [ ] Test de la hipótesis: ¿alguna de estas cantidades, computada a priori,
      separa los 4 settings donde gana el clásico de los 14 donde gana el cuántico?

Insumo: matrices de kernel ya generadas por el pipeline existente (results/).

## Workstream B — Baselines probabilísticos y más kernels (prioridad 2)

Responde a: *"probabilistic approaches (GPs, DKL, Bayesian SVMs) entirely absent;
gains over two narrow classical baselines"*.

- [ ] Gaussian Process classifier con kernel precomputado, mismo protocolo
      kernel-swap (mismo preprocesado, mismos splits). Aporta además
      incertidumbre bajo shift (calibración ID vs OOD como métrica secundaria).
- [ ] Kernels clásicos adicionales: polinomial, Laplaciano, Matérn (ν = 1/2, 3/2, 5/2).
- [ ] Opcional (si hay tiempo): SVM bayesiano o DKL pequeño; si no, justificar
      por qué GPC cubre la objeción de cuantificación de incertidumbre.

## Workstream C — Datasets adicionales (prioridad 3, mayor riesgo)

Responde a: *"empirical scope confined to a single dataset family (EMBER)"*.

Reutilizar los pipelines del Paper 2 (features numéricas estandarizadas + reducción
a baja dimensión, formato ya compatible con el kernel-swap):

- [ ] CICIDS2017 (flujos de red).
- [ ] UNSW-NB15 (flujos de red).
- [ ] Definir mecanismos de shift análogos a m1/m2 sobre estos datasets
      (extremos de sparsity / distancia a centroide de train) + si es posible un
      split temporal natural.

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
