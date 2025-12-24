import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from search_tools.compound_helpers import unpack_bits

# Import these from your existing module if this lives in a separate file:
# from compound_helpers import LigandStore, unpack_bits


def _tanimoto_from_bits_matrix(
    query_bits: np.ndarray,
    matrix_bits: np.ndarray,
    matrix_sum: np.ndarray,
) -> np.ndarray:
    """
    Compute Tanimoto similarity between a single query bit vector
    and all rows in a bit matrix.

    Parameters
    ----------
    query_bits : np.ndarray
        1D array of shape (dim,) with 0/1 bits for the query ligand.
    matrix_bits : np.ndarray
        2D array of shape (N, dim) with 0/1 bits for candidate ligands.
    matrix_sum : np.ndarray
        1D array of shape (N,) containing precomputed bit counts for
        each row in matrix_bits.

    Returns
    -------
    np.ndarray
        1D array of shape (N,) with Tanimoto similarity scores.
    """
    # Ensure correct dtypes
    q = query_bits.astype(np.uint8, copy=False)
    m = matrix_bits.astype(np.uint8, copy=False)

    # Bit counts
    q_sum = int(q.sum())
    m_sum = matrix_sum  # already precomputed for each row

    # Intersection: since bits are 0/1, elementwise product is equivalent to logical AND
    intersection = (m * q).sum(axis=1)

    # Union = sum(A) + sum(B) - intersection
    union = m_sum + q_sum - intersection

    # Avoid division by zero
    valid = union > 0
    tanimoto = np.zeros_like(union, dtype=np.float64)
    tanimoto[valid] = intersection[valid] / union[valid]
    return tanimoto


def curate_chembl_possible_by_pdb_similarity(
    binding_pdb: pd.DataFrame,
    binding_chembl_possible: pd.DataFrame,
    store: "LigandStore",
    rep_name: str = "morgan_1024_r2",
    tanimoto_threshold: float = 0.4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Curate possible ChEMBL binding domains using PDB ligands and Tanimoto similarity.

    This function:
      - Uses a Morgan fingerprint representation stored on disk (via LigandStore).
      - For each row in binding_chembl_possible, checks if there is *any* PDB
        ligand that:
          * binds to the same Pfam (pfam_id) in binding_pdb, and
          * has Tanimoto similarity >= tanimoto_threshold to the ChEMBL ligand.
      - If such PDB ligands exist, the corresponding row is considered "curated":
        the Pfam is accepted as a valid binding domain for that ChEMBL ligand.
      - Rows that do not find any supporting PDB ligand remain in the "possible"
        set.

    Parameters
    ----------
    binding_pdb : pd.DataFrame
        PDB binding data with at least:
          - 'chem_comp_id' : ligand ID (must exist in LigandStore's comp_id space)
          - 'pfam_id'      : Pfam domain identifier
        Other columns (e.g. 'pdb_id', 'uniprot_id', 'source') are preserved but
        not used directly in the similarity search.
    binding_chembl_possible : pd.DataFrame
        ChEMBL binding data with uncertain domains, with at least:
          - 'chem_comp_id' : ChEMBL ligand ID
          - 'pfam_id'      : candidate Pfam domain
          - 'uniprot_id'   : target protein (for context)
        Other columns (e.g. 'pchembl', 'mechanism', 'source') are preserved.
    store : LigandStore
        LigandStore instance pointing to the root directory where:
          - ligands.parquet
          - reps/<rep_name>.dat
          - reps/<rep_name>.meta.json
        are stored.
    rep_name : str
        Name of the representation to use (default: 'morgan_1024_r2').
    tanimoto_threshold : float
        Minimum Tanimoto similarity to consider a PDB ligand as "supporting"
        the domain assignment (default: 0.4).

    Returns
    -------
    curated_from_possible : pd.DataFrame
        Subset of binding_chembl_possible where the Pfam domain could be
        validated by at least one similar PDB ligand (Tanimoto >= threshold).
        This can be appended to your existing binding_data_chembl_curated.
    remaining_possible : pd.DataFrame
        Subset of binding_chembl_possible where no supporting PDB ligand was
        found; i.e. domains that remain "possible" / uncurated.
    """
    # ------------------------------------------------------------------
    # 1) Load the representation (e.g. Morgan fingerprints)
    # ------------------------------------------------------------------
    morgan = store.load_representation(rep_name)
    dim = morgan.dim

    # ------------------------------------------------------------------
    # 2) Build Pfam -> PDB ligand ID mapping
    #    (which ligands in PDB have been observed binding to each Pfam?)
    # ------------------------------------------------------------------
    pdb_df = binding_pdb.dropna(subset=["pfam_id", "chem_comp_id"]).copy()

    # For each Pfam, we keep the unique chem_comp_id values
    pfam_to_pdb_ids: Dict[str, List[str]] = (
        pdb_df.groupby("pfam_id")["chem_comp_id"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )

    # ------------------------------------------------------------------
    # 3) Precompute fingerprints for all unique ChEMBL ligands in "possible"
    # ------------------------------------------------------------------
    possible_df = binding_chembl_possible.copy()
    unique_chembl_ids = possible_df["chem_comp_id"].dropna().unique().tolist()

    # Retrieve fingerprints for all these ligands at once (0/1 bits)
    # Assumes all chem_comp_id exist in the LigandStore index.
    chembl_fps = morgan.get_by_ids(unique_chembl_ids, as_float=False)
    # Build lookup dicts: comp_id -> bits, comp_id -> bitcount
    chembl_fp_map: Dict[str, np.ndarray] = dict(zip(unique_chembl_ids, chembl_fps))
    chembl_fp_sum: Dict[str, int] = {
        cid: int(fp.sum()) for cid, fp in chembl_fp_map.items()
    }

    # ------------------------------------------------------------------
    # 4) Pfam-level cache: for each Pfam, precompute PDB ligand fingerprints
    # ------------------------------------------------------------------
    pfam_cache: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}

    def _get_pdb_fp_matrix_for_pfam(pfam_id: str):
        """
        For a given Pfam, return:
          - fps_pdb  : (N, dim) 0/1 bit matrix for PDB ligands bound to this Pfam
          - sum_pdb  : (N,) bit counts for each PDB ligand
          - pdb_ids  : list of chem_comp_id strings used to build the matrix

        If there is no PDB ligand for this Pfam, or if none of the PDB
        chem_comp_id are present in the LigandStore index, returns None.
        """
        if pfam_id in pfam_cache:
            return pfam_cache[pfam_id]

        pdb_ids = pfam_to_pdb_ids.get(pfam_id)
        if not pdb_ids:
            pfam_cache[pfam_id] = None
            return None

        # Map PDB chem_comp_id -> ligand index in the memmap (ignore missing IDs)
        idxs = [
            store.id_to_idx[cid]
            for cid in pdb_ids
            if cid in store.id_to_idx
        ]

        if len(idxs) == 0:
            pfam_cache[pfam_id] = None
            return None

        # Read packed bits directly from the memmap and unpack them
        packed = morgan.memmap[idxs]  # shape: (N, packed_dim)
        fps_pdb = unpack_bits(packed, dim)  # shape: (N, dim)
        sum_pdb = fps_pdb.sum(axis=1)

        pfam_cache[pfam_id] = (fps_pdb, sum_pdb, pdb_ids)
        return pfam_cache[pfam_id]

    # ------------------------------------------------------------------
    # 5) Main loop: for each row in binding_chembl_possible, decide if it can be curated
    # ------------------------------------------------------------------
    curated_idx: List[int] = []  # indices of rows to move to "curated"

    # We group by Pfam so that we reuse the same PDB FP matrix for all rows
    # sharing that Pfam.
    for pfam_id, group in possible_df.groupby("pfam_id", sort=False):
        # Get the PDB fingerprint matrix for this Pfam
        pfam_data = _get_pdb_fp_matrix_for_pfam(pfam_id)
        if pfam_data is None:
            # No PDB information for this Pfam -> cannot curate any row in this group
            continue

        fps_pdb, sum_pdb, _ = pfam_data  # we don't actually need pdb_ids here

        # For each ChEMBL ligand in this Pfam group, compute Tanimoto to all PDB ligands
        for idx, row in group.iterrows():
            chem_id = row["chem_comp_id"]
            if pd.isna(chem_id) or chem_id not in chembl_fp_map:
                # No fingerprint available for this ligand -> leave as "possible"
                continue

            fp_query = chembl_fp_map[chem_id]
            # Safety check on dimensionality
            if fp_query.shape[0] != dim:
                raise ValueError(
                    f"Fingerprint dimension mismatch for ligand '{chem_id}' "
                    f"(expected {dim}, got {fp_query.shape[0]})."
                )

            # Compute Tanimoto similarities to all PDB ligands for this Pfam
            tanimoto = _tanimoto_from_bits_matrix(fp_query, fps_pdb, sum_pdb)

            # If any Tanimoto >= threshold, we consider this Pfam validated for this ChEMBL ligand
            if np.any(tanimoto >= tanimoto_threshold):
                curated_idx.append(idx)

# ------------------------------------------------------------------
    # 6) Build the output DataFrames
    # ------------------------------------------------------------------
    # Unique indices of curated rows (per ligand–Pfam)
    curated_idx_unique = sorted(set(curated_idx))

    # Rows where at least one Pfam was validated by PDB similarity
    curated_from_possible = possible_df.loc[curated_idx_unique].copy()

    # Now enforce the ligand–protein logic:
    # if a (chem_comp_id, uniprot_id) pair has ANY curated Pfam,
    # we do NOT keep any of its other Pfams in remaining_possible.
    #
    # In other words:
    #   - curated_from_possible: all validated (ligand, protein, Pfam) rows
    #   - remaining_possible: only (ligand, protein, Pfam) rows for
    #     ligand–protein pairs that have ZERO validated Pfams.

    # Build a list of (ligand, protein) pairs for all rows
    all_pairs = list(zip(
        possible_df["chem_comp_id"],
        possible_df["uniprot_id"],
    ))

    # Set of (chem_comp_id, uniprot_id) pairs that had at least one curated Pfam
    curated_pairs = {
        all_pairs[i] for i in curated_idx_unique
    }

    # Mask: keep only rows whose (ligand, protein) pair is NOT in curated_pairs
    mask_remaining = [
        pair not in curated_pairs for pair in all_pairs
    ]

    remaining_possible = possible_df.loc[mask_remaining].copy()

    # Optionally, mark the curation method
    curated_from_possible["curation_method"] = "pdb_similarity_tanimoto"

    return curated_from_possible, remaining_possible
