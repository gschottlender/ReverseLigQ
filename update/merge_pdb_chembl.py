
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from typing import Optional, Tuple, Dict
from pathlib import Path

from search_tools.compound_helpers import (
    unify_pdb_chembl,
    build_ligand_index,
    build_morgan_representation,
    build_chemberta_representation,
    LigandStore,
)

from update.database_curation.curation_helpers import curate_chembl_possible_by_pdb_similarity

def merge_databases(data_dir,tanimoto_curation_threshold=0.35):
    pdb_data_dir = f'{data_dir}/pdb/'
    chembl_data_dir = f'{data_dir}/chembl/'

    # merge compound and smiles data

    ligs_smiles_pdb = pd.read_parquet(f'{pdb_data_dir}/pdb_ligand_smiles.parquet')
    ligs_smiles_chembl = pd.read_parquet(f'{chembl_data_dir}/chembl_ligand_smiles.parquet')
    ligs_smiles_merged, chembl_to_pdb_id = unify_pdb_chembl(ligs_smiles_pdb, ligs_smiles_chembl)

    # Build ligand index under some root directory
    root = Path(f"{data_dir}/compound_data/pdb_chembl")
    build_ligand_index(ligs_smiles_merged, root)

    # Build Morgan fingerprint representation (1024 bits, radius 2)
    build_morgan_representation(root, n_bits=1024, radius=2, batch_size=20000)

    # Build ChemBERTa representation (default 768 bits)
    build_chemberta_representation(root, n_bits=768, batch_size=128)

    # Later, anywhere: load and query fingerprints by comp_id
    store = LigandStore(root)
    morgan_repr = store.load_representation("morgan_1024_r2")

    binding_data_pdb = pd.read_parquet(f'{pdb_data_dir}/pdb_binding_data.parquet')
    binding_data_chembl_curated = pd.read_parquet(f'{chembl_data_dir}/chembl_binding_data_curated.parquet')
    binding_data_chembl_possible = pd.read_parquet(f'{chembl_data_dir}/chembl_binding_data_possible.parquet')

    # Build a set of mapping keys (fast membership checks)
    keys_set = set(chembl_to_pdb_id.keys())

    # ---------------------------
    # CURATED
    # ---------------------------

    # Boolean mask: only rows whose chem_comp_id is in the mapping
    mask_cur = binding_data_chembl_curated["chem_comp_id"].isin(keys_set)

    # Compute mapped values for those rows only
    mapped_cur = binding_data_chembl_curated.loc[mask_cur, "chem_comp_id"].map(chembl_to_pdb_id)

    # Assign back in place
    binding_data_chembl_curated.loc[mask_cur, "chem_comp_id"] = mapped_cur.values


    mask_pos = binding_data_chembl_possible["chem_comp_id"].isin(keys_set)
    mapped_pos = binding_data_chembl_possible.loc[mask_pos, "chem_comp_id"].map(chembl_to_pdb_id)
    binding_data_chembl_possible.loc[mask_pos, "chem_comp_id"] = mapped_pos.values

    # Curate chembl ligands with multi-domain targets using Tanimoto similarity against pdb ligands with known binding domain
    curated_from_possible, remaining_possible = curate_chembl_possible_by_pdb_similarity(
    binding_pdb=binding_data_pdb,
    binding_chembl_possible=binding_data_chembl_possible,
    store=store,
    rep_name="morgan_1024_r2",
    tanimoto_threshold=tanimoto_curation_threshold,
    )

    binding_data_chembl = pd.concat([
    binding_data_chembl_curated,
    curated_from_possible
    ], axis=0)

    binding_data_chembl_possible = remaining_possible

    # Merge PDB and ChEMBL data
    binding_data_merged = pd.concat(
    [binding_data_pdb, binding_data_chembl], 
    axis=0,
    ignore_index=True
    )

    return ligs_smiles_merged, binding_data_merged, binding_data_chembl_possible