# v1.1 novelty-consolidation specification

Status: frozen before the v1.1 manuscript rewrite.

Date: 2 August 2026.

## Purpose

Version 1.1 strengthens the theoretical and editorial presentation of the
existing v1.0 evidence. It does not add a retrospective performance sweep,
change a selected model, reopen a target-label audit, or alter any frozen v0.9
or v1.0 endpoint.

The central claim is a sharp finite-target identified set for the predictive
advantage of one fixed model against the best member of a prespecified finite
reference family. Accuracy is the closed-form special case of an exact
finite-label, bounded-loss construction. The quantum-kernel experiments are
the substantive use case and falsification setting, not the source of a claim
that disagreement itself is new.

## Locked v1.1 changes

1. State a general sharp partial-label envelope for additive bounded losses
   over a finite label space. The lower endpoint is separable; the upper
   endpoint is an exact finite max--min program. State information optimality:
   any assumption-free interval based on the same observed predictions and
   labels must contain the identified endpoints.
2. Present the existing zero--one accuracy theorems as a closed-form
   corollary, without changing their estimand or numerical results.
3. Add direct comparisons with co-validation, Active Testing, weak-supervision
   partial identification, limited-label model selection, Huang's geometric
   screening, classical QML surrogates, and the concurrent 2026 kernel-swap
   preprint.
4. Promote the already generated 30/60/115 reference-breadth result. The
   levels and 5,000 block-ordering summaries are inherited from the frozen
   v0.9 analysis; no new tier or outcome-dependent ordering is introduced.
5. Rewrite the npj Quantum Information cover letter around the identified set,
   three-gate framework, prospective Gate-2 transfer, entanglement audit, and
   finite-shot result.
6. Use `internally locked prospective replication`; do not call the v1.0
   experiment publicly preregistered or independently timestamped.

## Contribution-preservation ledger

No pre-v1.1 contribution is deleted from the scientific record. Each item must
retain a quantitative headline or explicit interpretation in the main text,
with full results in the main paper or Supplementary Information and its
method retained in the main Methods section.

| Existing contribution | Required v1.1 location |
| --- | --- |
| Oracle-to-controlled sign reversal and P1/P1-prime estimands | Main Results and Discussion; full tables in main/SI |
| Equal-budget and full-reference comparisons | Main Results; full sensitivities in SI |
| Within-v4 factorial and shortcut ablation | Main Results headline; full tables in SI |
| External protocol-sensitivity corroboration | Main Results and Discussion; full task results in SI |
| Product versus entangling-ZZ audit | Main Results and Gate-3 interpretation; full strata in SI |
| Circuit and sampling resources | Main table/Methods; detailed resources in SI |
| Effective-rank matching, KTA, and geometry associations | Main quantitative summary; full diagnostics in SI |
| Repeated finite-shot perturbation | Main result and figure; full cells in SI |
| Sharp zero- and partial-label accuracy certificates | Central main theory/result |
| Balanced-accuracy MILP and train/target PSD construction | Main Methods; result details in SI |
| Internally locked prospective Gate-2 replication | Central main result; complete cells and failure record in SI |

## Validation gates

- Exhaustive small-batch enumeration must match the general bounded-loss
  endpoints, including non-zero--one loss tables and arbitrary partial labels.
- The zero--one specialization must reproduce the existing accuracy helpers.
- The main manuscript must retain all preservation-ledger topics, no more than
  ten display items, an abstract of at most 150 words, and the v0.9/v1.0
  numerical integrity gates.
- The main and Supplementary PDFs must compile without unresolved references,
  citations, or overfull boxes and pass complete visual inspection.
- Public release, Git tagging, and Zenodo deposition remain author-controlled
  submission actions and are not part of this consolidation.
