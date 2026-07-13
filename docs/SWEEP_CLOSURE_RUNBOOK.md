# Runbook: cierre del paper tras el sweep de netflow (15 runs/setting)

> **EN PAUSA PARCIAL (2026-07-13 tarde):** una revisión externa simulada (GPT) señaló
> problemas metodológicos verificados en el código (selección Best-by-OOD sobre el test
> OOD, sweep de bandwidth no simétrico para Laplaciano/Matérn, circularidad parcial del
> KTA-OOD). Al completarse el sweep ejecutar SOLO los pasos 0-2 (verificación, agregación,
> tablas) y committear resultados. **NO ejecutar los pasos 3-7** (prosa, release v0.3.0,
> Zenodo, "listo para enviar") hasta que Roberto decida el alcance de la revisión.
> Los datos per-run del sweep (summary.csv por run con todas las configs) son la base
> del re-análisis con selección honesta (cross-seed / select-on-ID), sin cómputo nuevo.

Objetivo: al completarse los 405 runs de `results/netflow_bandwidth_sweep/extended_kernels/`,
ejecutar el cierre completo y dejar el repo listo para enviar a EPJ QT.

Estado al armarlo (2026-07-13 ~15:30): 18 workers paralelos corriendo
(logs `results/netflow_bandwidth_sweep/_logs/worker__*.log`); 54/405 runs completos.
Un run está completo si existe `<dir>/extended_kernels_qsplits__summary.csv`.

## 0. Verificación
- Contar runs completos; objetivo 405 (27 settings x 15 = 5 qsplit-seeds x 3 model-seeds).
- Si los workers murieron con <405: revisar `worker__*.err.log` y el tail del
  `extended__<label>.log` del run fallido; relanzar el shard con el mismo comando
  (es resume-safe) y re-armar el watcher.

## 1. Agregación
```
<env-python> scripts/analysis/compare_extended_families.py \
  --root results/netflow_bandwidth_sweep/extended_kernels \
  --out-dir results/netflow_bandwidth_sweep
```
(env-python = miniconda3/envs/kernel-shift-framework/python.exe; revisar antes los
args reales del script — puede necesitar flags extra; mirar cómo se generó el CSV
de EMBER sweep si hay dudas.)
Sanity check del CSV resultante: 8 filas (2 modelos x 2 vistas x 2 referencias),
n_settings=27, y n_runs=15 en `family_comparison_by_setting.csv`.

## 2. Tablas y figuras
- `python scripts/reporting/make_v2_tables.py` → regenera `results/tables_v2/`
  (la tabla `table_family_sweep_netflow.tex` cambia).
- Figuras: make_v2_figures.py NO lee el sweep de netflow (solo el de EMBER, línea ~49)
  → en principio no cambian. Verificar con grep antes de decidir regenerar.

## 3. Prosa del manuscrito (`manuscript/sn-article.tex`)
- §res_bandwidth (~línea 491): actualizar la frase de netflow:
  hoy dice "(one run per setting)" con 24/27 GP p=5.2e-5 y 19/27 SVC p=9.8e-4.
  Sustituir por los nuevos wins/27, mean Delta, Holm p y effect>1 de ambos
  clasificadores, y "(15 runs per setting, as in the main grid)".
- `grep -n "one run" manuscript/sn-article.tex` para no dejar restos
  (revisar también tab:protocol_summary y §methods por menciones del sweep).
- Verificar TODO número de la prosa afectada contra el CSV nuevo.
- Si el veredicto se debilita (menos wins, p mayor): redactar honestamente,
  el framing del paper (arco de tres veredictos) lo tolera; avisar a Roberto
  del cambio de conclusión en el resumen final.

## 4. Recompilar
`latexmk -pdf -interaction=nonstopmode sn-article.tex` en `manuscript/`.
Checks: 0 undefined citations, 0 bibtex warnings, sin "Float too large" nuevos.

## 5. GitHub
- README.md: revisar si menciona recuentos de runs del sweep o "one run";
  actualizar la sección de resultados si procede.
- CITATION.cff: bump `version:` 0.2.0 → 0.3.0 (el DOI es el de concepto, no cambia).
- `git add` de: run dirs nuevos del sweep (~378 dirs, ~190 MB — consistente con el
  freeze de EMBER en 338f97e), CSVs agregados, tablas_v2, manuscrito, README, CITATION.
- Commit (mensaje: freeze del sweep netflow a 15 runs + actualización del manuscrito),
  tag `v0.3.0`, `git push origin main --tags`.

## 6. Zenodo
- `gh release create v0.3.0 --title ... --notes ...` → el webhook GitHub–Zenodo
  archiva la nueva versión bajo el concept DOI 10.5281/zenodo.19147649.
- Verificar a los ~5-10 min vía API pública de Zenodo que aparece la versión nueva;
  si no aparece, avisar a Roberto (puede requerir token/manual).

## 7. Informe final a Roberto
Resumen con: veredicto del sweep (¿cambió la conclusión?), números nuevos vs viejos,
estado del PDF, commit/tag/release, DOI de la versión Zenodo, y confirmación de
que no queda nada pendiente para el envío a EPJ QT.
