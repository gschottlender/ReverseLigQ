from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def generate_rev_ligq_databases(data_dir: str | Path) -> Tuple[Dict[str, Any], Dict[str, list[str]], Dict[str, list[str]]]:
    """
    Build and persist ligand↔Pfam mappings and per-organization ligand lists.

    Inputs (expected under data_dir):
      - rev_ligq/fam_prot_dict.pkl
      - merged_databases/binding_data_merged.parquet
      - merged_databases/uncurated_binding_data.parquet
      - compound_data/pdb_chembl/ligands.parquet

    Outputs (written under data_dir/rev_ligq):
      - ligs_fams_curated.pkl    : dict[chem_comp_id] -> list[unique pfam_id]
      - ligs_fams_possible.pkl   : dict[chem_comp_id] -> list[unique pfam_id]
      - ligand_lists.pkl         : dict[org] -> list[unique chem_comp_id] (sorted)

    Parameters
    ----------
    data_dir : str | Path
        Base directory containing the expected subfolders/files.

    Returns
    -------
    ligand_lists : dict
        dict[org] -> list of ligand IDs to consider for that organism.
    ligs_fams_curated : dict
        dict[chem_comp_id] -> list of unique Pfam IDs (curated evidence).
    ligs_fams_possible : dict
        dict[chem_comp_id] -> list of unique Pfam IDs (uncurated/possible evidence).
    """
    data_dir = Path(data_dir)

    lq_rev_data_dir = data_dir / "rev_ligq"
    tables_dir = data_dir / "merged_databases"
    compound_data_dir = data_dir / "compound_data" / "pdb_chembl"

    # --- Load inputs ---
    with (lq_rev_data_dir / "fam_prot_dict.pkl").open("rb") as f:
        fam_prot_db = pickle.load(f)

    ligs_curated_table = pd.read_parquet(tables_dir / "binding_data_merged.parquet")
    ligs_possible_table = pd.read_parquet(tables_dir / "uncurated_binding_data.parquet")
    ligs_val_db = pd.read_parquet(compound_data_dir / "ligands.parquet")

    # --- Keep only ligands that exist in the validated ligand index ---
    valid_ids = set(ligs_val_db["chem_comp_id"].astype(str))
    ligs_curated_table = ligs_curated_table[ligs_curated_table["chem_comp_id"].astype(str).isin(valid_ids)]
    ligs_possible_table = ligs_possible_table[ligs_possible_table["chem_comp_id"].astype(str).isin(valid_ids)]

    # Optional: drop rows with missing keys (prevents weird None/NaN keys in dicts)
    ligs_curated_table = ligs_curated_table.dropna(subset=["chem_comp_id", "pfam_id"])
    ligs_possible_table = ligs_possible_table.dropna(subset=["chem_comp_id", "pfam_id"])

    # --- Build ligand -> unique Pfam list dicts (curated / possible) ---
    ligs_fams_curated: Dict[str, list[str]] = (
        ligs_curated_table.groupby("chem_comp_id")["pfam_id"]
        .unique()
        .map(list)
        .to_dict()
    )

    ligs_fams_possible: Dict[str, list[str]] = (
        ligs_possible_table.groupby("chem_comp_id")["pfam_id"]
        .unique()
        .map(list)
        .to_dict()
    )

    # Persist these dictionaries
    with (lq_rev_data_dir / "ligs_fams_curated.pkl").open("wb") as f:
        pickle.dump(ligs_fams_curated, f)

    with (lq_rev_data_dir / "ligs_fams_possible.pkl").open("wb") as f:
        pickle.dump(ligs_fams_possible, f)

    # --- Efficient per-organism ligand lists ---
    # Precompute pfam -> set(ligands) once, so we don't re-filter large tables per organism.
    pfam2lig_curated: Dict[str, set[str]] = (
        ligs_curated_table.groupby("pfam_id")["chem_comp_id"]
        .agg(lambda s: set(s.astype(str)))
        .to_dict()
    )

    pfam2lig_possible: Dict[str, set[str]] = (
        ligs_possible_table.groupby("pfam_id")["chem_comp_id"]
        .agg(lambda s: set(s.astype(str)))
        .to_dict()
    )

    ligand_lists: Dict[str, list[str]] = {}

    for org, fam_dict in fam_prot_db.items():
        # fam_dict keys are Pfam IDs for that organism
        ligs_set: set[str] = set()
        for pfam_id in fam_dict.keys():
            ligs_set |= pfam2lig_curated.get(pfam_id, set())
            ligs_set |= pfam2lig_possible.get(pfam_id, set())

        # Keep deterministic output ordering (np.unique would also sort)
        ligand_lists[org] = sorted(ligs_set)

    with (lq_rev_data_dir / "ligand_lists.pkl").open("wb") as f:
        pickle.dump(ligand_lists, f)

    return ligand_lists, ligs_fams_curated, ligs_fams_possible

