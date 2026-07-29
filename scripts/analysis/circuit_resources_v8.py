"""Generate the frozen v0.8 circuit-family and finite-shot resource audit."""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import qiskit
from qiskit import transpile
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap


DIMENSIONS = (4, 6, 8, 10, 12)
SHOTS = (128, 512, 2048, 8192)
BASIS = ("rz", "sx", "x", "cx")
OPTIMIZATION_LEVEL = 0
N_MEASUREMENT_REPLICATES = 30
N_FIXED_CASES = 8


def feature_maps(dimension: int):
    return {
        "zz_r1_full": (
            ZZFeatureMap(
                feature_dimension=dimension,
                reps=1,
                entanglement="full",
            ),
            "entangling_zz",
            1,
            "Z,ZZ",
        ),
        "zz_r2_full": (
            ZZFeatureMap(
                feature_dimension=dimension,
                reps=2,
                entanglement="full",
            ),
            "entangling_zz",
            2,
            "Z,ZZ",
        ),
        "pauli_xz_r1_full": (
            PauliFeatureMap(
                feature_dimension=dimension,
                reps=1,
                paulis=["X", "Z"],
                entanglement="full",
            ),
            "separable_product",
            1,
            "X,Z",
        ),
        "zmap_r2": (
            ZFeatureMap(feature_dimension=dimension, reps=2),
            "separable_product",
            2,
            "Z",
        ),
    }


def operation_counts(circuit) -> tuple[int, int]:
    counts = circuit.count_ops()
    two_qubit = int(counts.get("cx", 0))
    one_qubit = int(
        sum(
            count
            for operation, count in counts.items()
            if operation not in {"cx", "barrier", "measure"}
        )
    )
    return one_qubit, two_qubit


def bind_distinct_inputs(feature_map):
    parameters = sorted(feature_map.parameters, key=lambda parameter: parameter.name)
    first = np.linspace(0.17, 1.31, len(parameters))
    second = np.linspace(0.29, 1.57, len(parameters))
    left = feature_map.assign_parameters(dict(zip(parameters, first)))
    right = feature_map.assign_parameters(dict(zip(parameters, second)))
    return left.compose(right.inverse())


def circuit_table() -> pd.DataFrame:
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        for dimension in DIMENSIONS:
            for name, (feature_map, stratum, reps, paulis) in feature_maps(
                dimension
            ).items():
                compiled_map = transpile(
                    feature_map,
                    basis_gates=list(BASIS),
                    optimization_level=OPTIMIZATION_LEVEL,
                )
                fidelity = bind_distinct_inputs(feature_map)
                compiled_fidelity = transpile(
                    fidelity,
                    basis_gates=list(BASIS),
                    optimization_level=OPTIMIZATION_LEVEL,
                )
                map_one, map_two = operation_counts(compiled_map)
                fidelity_one, fidelity_two = operation_counts(compiled_fidelity)
                rows.append({
                    "feature_map": name,
                    "map_stratum": stratum,
                    "dimension_qubits": dimension,
                    "repetitions": reps,
                    "pauli_strings": paulis,
                    "product_factorized": stratum == "separable_product",
                    "feature_map_depth": int(compiled_map.depth()),
                    "feature_map_1q_gates": map_one,
                    "feature_map_cx_gates": map_two,
                    "fidelity_template_depth": int(compiled_fidelity.depth()),
                    "fidelity_template_1q_gates": fidelity_one,
                    "fidelity_template_cx_gates": fidelity_two,
                    "qiskit_version": qiskit.__version__,
                    "basis_gates": ",".join(BASIS),
                    "optimization_level": OPTIMIZATION_LEVEL,
                    "device_routing": False,
                })
    return pd.DataFrame(rows)


def shot_resource_table() -> pd.DataFrame:
    n_train = 1000
    n_id_val = 250
    n_id_test = 250
    n_ood = 500
    block_counts = {
        "train_upper_off_diagonal": n_train * (n_train - 1) // 2,
        "id_validation_to_train": n_id_val * n_train,
        "id_test_to_train": n_id_test * n_train,
        "ood_test_to_train": n_ood * n_train,
        "ood_square_upper_off_diagonal": n_ood * (n_ood - 1) // 2,
    }
    estimates = sum(block_counts.values())
    rows = []
    for shots in SHOTS:
        rows.append({
            "shots_per_fidelity": shots,
            **block_counts,
            "distinct_fidelity_estimates_per_case_replicate": estimates,
            "circuit_invocations_per_case_replicate": estimates,
            "total_shots_per_case_replicate": estimates * shots,
            "n_fixed_cases": N_FIXED_CASES,
            "n_measurement_replicates": N_MEASUREMENT_REPLICATES,
            "projected_circuit_invocations_full_sensitivity": (
                estimates * N_FIXED_CASES * N_MEASUREMENT_REPLICATES
            ),
            "projected_shots_full_sensitivity": (
                estimates
                * shots
                * N_FIXED_CASES
                * N_MEASUREMENT_REPLICATES
            ),
            "projection_conditions_reuse_same_measurements": True,
            "hardware_executed": False,
        })
    output = pd.DataFrame(rows)
    total = {
        "shots_per_fidelity": "all_levels",
        **{key: np.nan for key in block_counts},
        "distinct_fidelity_estimates_per_case_replicate": (
            estimates * len(SHOTS)
        ),
        "circuit_invocations_per_case_replicate": estimates * len(SHOTS),
        "total_shots_per_case_replicate": estimates * sum(SHOTS),
        "n_fixed_cases": N_FIXED_CASES,
        "n_measurement_replicates": N_MEASUREMENT_REPLICATES,
        "projected_circuit_invocations_full_sensitivity": (
            estimates
            * len(SHOTS)
            * N_FIXED_CASES
            * N_MEASUREMENT_REPLICATES
        ),
        "projected_shots_full_sensitivity": (
            estimates
            * sum(SHOTS)
            * N_FIXED_CASES
            * N_MEASUREMENT_REPLICATES
        ),
        "projection_conditions_reuse_same_measurements": True,
        "hardware_executed": False,
    }
    return pd.concat([output, pd.DataFrame([total])], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/v8/reviewer_revision"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    circuits = circuit_table()
    shots = shot_resource_table()
    circuits.to_csv(args.out_dir / "circuit_resources.csv", index=False)
    shots.to_csv(args.out_dir / "finite_shot_resources.csv", index=False)
    print(circuits.to_string(index=False))
    print(shots.to_string(index=False))
    print(f"[OK] wrote circuit and finite-shot resource audit to {args.out_dir}")


if __name__ == "__main__":
    main()
