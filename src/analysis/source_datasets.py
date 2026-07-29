"""Canonical source-dataset mapping for the security scenario groups."""
from __future__ import annotations


def source_dataset_for_group(group: str) -> str:
    """Map every security scenario-group to one of three source datasets."""
    if group.startswith("ember_"):
        return "ember"
    if group.startswith("unsw_"):
        return "unsw"
    if group.startswith("toniot_"):
        return "toniot"
    raise ValueError(f"Unrecognized security scenario-group: {group}")
