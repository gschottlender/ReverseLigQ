"""
Helpers for compound unification and numerical representations.

This module provides:

- Unification of PDB and ChEMBL ligand tables using InChIKey
  while preserving all rows (no drops for missing InChIKey).
- Construction of a ligand index (ligands.parquet) with a dense
  integer index (lig_idx) for efficient array storage.
- Efficient computation and storage of Morgan fingerprints
  (packed bits in a memmap on disk).
- A small API to retrieve representations (e.g. Morgan) by comp_id
  without loading the full matrix into RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple

import json
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import inchi, AllChem, DataStructs
from rdkit import RDLogger
import torch
from transformers import AutoModel, AutoTokenizer

# Silence RDKit warnings (invalid SMILES, sanitization issues, etc.)
RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# 1. Basic utilities: InChIKey, Morgan fingerprints, bit packing
# ---------------------------------------------------------------------------

def smiles_to_inchikey(smiles: str) -> Optional[str]:
    """
    Convert a SMILES string to an InChIKey. Returns None if it fails.

    This is helpful to structurally unify compounds coming from different
    sources (PDB, ChEMBL, etc.) even when SMILES differ in notation.
    """
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def morgan_fp_bits(
    smiles: str,
    n_bits: int = 1024,
    radius: int = 2,
) -> Optional[np.ndarray]:
    """
    Compute a Morgan fingerprint as a 0/1 numpy array of shape (n_bits,).

    Returns None if the SMILES cannot be parsed or the fingerprint
    cannot be generated.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    n_bits : int
        Fingerprint length in bits (default: 1024).
    radius : int
        Morgan fingerprint radius (default: 2).
    """
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


def pack_bits(arr: np.ndarray) -> np.ndarray:
    """
    Pack a 0/1 array of bits along the last axis using numpy.packbits.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (N, n_bits) with values 0/1.

    Returns
    -------
    np.ndarray
        Packed array of shape (N, n_bits / 8), dtype=uint8.
    """
    return np.packbits(arr, axis=-1)


def unpack_bits(packed: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Unpack bits from the last axis back to an array of length n_bits.

    Parameters
    ----------
    packed : np.ndarray
        Packed array of shape (N, n_bytes).
    n_bits : int
        Desired number of bits in the output.

    Returns
    -------
    np.ndarray
        Unpacked array of shape (N, n_bits), dtype=uint8 with values 0/1.
    """
    unpacked = np.unpackbits(packed, axis=-1)
    # In case there are extra bits, truncate to the desired length
    if unpacked.shape[-1] > n_bits:
        unpacked = unpacked[..., :n_bits]
    return unpacked


# ---------------------------------------------------------------------------
# 2. Unify PDB and ChEMBL ligand tables
# ---------------------------------------------------------------------------


def unify_pdb_chembl(
    ligs_smiles_pdb: pd.DataFrame,
    ligs_smiles_chembl: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Unify PDB and ChEMBL ligands using InChIKey and build an ID mapping.

    Both input tables are expected to use the same ID column name:
      - 'chem_comp_id' : ligand identifier (PDB 3-letter codes, CHEMBL IDs, etc.)
      - 'smiles'       : canonical SMILES string

    The function performs a structure-based unification:

      - For each non-null InChIKey, all ligands (from PDB and/or ChEMBL)
        sharing that InChIKey are grouped together.
      - A single canonical ID is chosen for the group:
          * Prefer a PDB ID (source == 'pdb') if present.
          * Otherwise, choose one of the ChEMBL IDs (lexicographically smallest).
      - All original IDs in the group are mapped to this canonical ID.

      - For rows with null InChIKey, no structure-based unification is
        possible. Each chem_comp_id becomes canonical for itself.

    Returns
    -------
    final_ligs : pd.DataFrame
        DataFrame with one row per canonical ligand, columns:
          - 'chem_comp_id' : canonical ID
          - 'smiles'       : canonical SMILES (taken from one representative)
        This is the table that should be used to build ligands.parquet.

    id_mapping : dict
        Dictionary mapping ANY original ID (from ligs_smiles_pdb or
        ligs_smiles_chembl) to its canonical ID in final_ligs:
          { original_chem_comp_id -> canonical_chem_comp_id }

        - For canonical IDs themselves, the mapping will simply be
          id_mapping[canonical_id] == canonical_id.
        - For structurally duplicated IDs (same InChIKey), all will map
          to the same canonical ID.
    """

    pdb_df = ligs_smiles_pdb.copy()
    chembl_df = ligs_smiles_chembl.copy()

    # Basic column checks
    for name, df in [("ligs_smiles_pdb", pdb_df), ("ligs_smiles_chembl", chembl_df)]:
        if "chem_comp_id" not in df.columns:
            raise ValueError(f"{name} must have a 'chem_comp_id' column.")
        if "smiles" not in df.columns:
            raise ValueError(f"{name} must have a 'smiles' column.")

    # Compute InChIKey for both tables
    pdb_df["inchikey"] = pdb_df["smiles"].map(smiles_to_inchikey)
    chembl_df["inchikey"] = chembl_df["smiles"].map(smiles_to_inchikey)

    # Tag source
    pdb_df["source"] = "pdb"
    chembl_df["source"] = "chembl"

    # Keep only the relevant columns
    pdb_df = pdb_df[["chem_comp_id", "smiles", "inchikey", "source"]]
    chembl_df = chembl_df[["chem_comp_id", "smiles", "inchikey", "source"]]

    # Concatenate both tables
    combined = pd.concat([pdb_df, chembl_df], ignore_index=True)

    # Split into non-null InChIKey (can be unified) and null (cannot)
    non_null = combined[combined["inchikey"].notna()].copy()
    null_rows = combined[combined["inchikey"].isna()].copy()

    # Mapping from original ID -> canonical ID
    id_mapping: Dict[str, str] = {}
    canonical_rows = []

    # ------------------------------------------------------------------
    # 1. Handle non-null InChIKey groups (structure-based unification)
    # ------------------------------------------------------------------
    if not non_null.empty:
        for inchikey, grp in non_null.groupby("inchikey", sort=False):
            # Prefer PDB ligands as canonical, if available
            pdb_grp = grp[grp["source"] == "pdb"]
            if not pdb_grp.empty:
                # Choose one PDB ligand as canonical (e.g. lexicographically smallest ID)
                canon_row = pdb_grp.sort_values("chem_comp_id").iloc[0]
            else:
                # No PDB ligand: choose one ChEMBL ligand as canonical
                canon_row = grp.sort_values("chem_comp_id").iloc[0]

            canon_id = canon_row["chem_comp_id"]
            canonical_rows.append(canon_row)

            # Map all IDs in this group to the canonical ID
            for cid in grp["chem_comp_id"].unique():
                id_mapping[cid] = canon_id

    # ------------------------------------------------------------------
    # 2. Handle null InChIKey rows (no structural unification possible)
    # ------------------------------------------------------------------
    # For these, each chem_comp_id is its own canonical ID, unless
    # it was already assigned in the previous step.
    if not null_rows.empty:
        # Drop exact duplicates to avoid adding the same canonical row twice
        null_rows = null_rows.drop_duplicates(subset=["chem_comp_id", "smiles", "inchikey", "source"])

        for _, row in null_rows.iterrows():
            cid = row["chem_comp_id"]
            if cid not in id_mapping:
                # This ID was not part of any non-null InChIKey group
                id_mapping[cid] = cid
                canonical_rows.append(row)

    # ------------------------------------------------------------------
    # 3. Build final_ligs from canonical rows
    # ------------------------------------------------------------------
    canonical_df = pd.DataFrame(canonical_rows)

    # Ensure uniqueness by canonical ID (in case of any accidental duplicates)
    canonical_df = canonical_df.sort_values("chem_comp_id").drop_duplicates(
        subset=["chem_comp_id"],
        keep="first",
    )

    final_ligs = canonical_df[["chem_comp_id", "smiles"]].reset_index(drop=True)

    return final_ligs, id_mapping


# ---------------------------------------------------------------------------
# 3. Build ligand index and Morgan representation
# ---------------------------------------------------------------------------

def build_ligand_index(
    final_ligs: pd.DataFrame,
    root: str | Path,
) -> Path:
    """
    Build the ligand index table (ligands.parquet) with a dense integer index.

    The resulting table contains:
      - chem_comp_id   : final unified ligand ID (PDB or ChEMBL)
      - smiles    : canonical SMILES used downstream
      - inchikey  : structure-based key (may be null)
      - lig_idx   : dense integer index [0..N-1] used to index arrays on disk

    Parameters
    ----------
    final_ligs : pd.DataFrame
        Unified ligand table with at least ['chem_comp_id', 'smiles'].
    root : str or Path
        Directory where ligands.parquet will be written.

    Returns
    -------
    Path
        Path to the written ligands.parquet file.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    df = final_ligs.copy()
    if "inchikey" not in df.columns:
        df["inchikey"] = df["smiles"].map(smiles_to_inchikey)

    df = df.reset_index(drop=True)
    df["lig_idx"] = np.arange(len(df), dtype=np.int64)

    out_path = root / "ligands.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def build_morgan_representation(
    root: str | Path,
    n_bits: int = 1024,
    radius: int = 2,
    batch_size: int = 10000,
    name: str = "morgan_1024_r2",
) -> None:
    """
    Compute Morgan fingerprints for all ligands in ligands.parquet and
    store them as a packed bit matrix in a memmap on disk.

    Files written under `root`:

      - ligands.parquet (must already exist)
      - reps/<name>.dat         : memmap with packed bits, shape (N, n_bits/8)
      - reps/<name>.meta.json   : metadata describing the representation

    Parameters
    ----------
    root : str or Path
        Root directory containing ligands.parquet. Results are written
        under root / 'reps'.
    n_bits : int
        Number of bits in the Morgan fingerprint (default: 1024).
    radius : int
        Morgan fingerprint radius (default: 2).
    batch_size : int
        Number of ligands processed per batch to control RAM usage.
    name : str
        Name of the representation (used in file names).
    """
    root = Path(root)
    reps_dir = root / "reps"
    reps_dir.mkdir(exist_ok=True, parents=True)

    ligs_path = root / "ligands.parquet"
    ligs = pd.read_parquet(ligs_path)
    n = len(ligs)

    if n == 0:
        raise ValueError("ligands.parquet is empty, nothing to process.")

    n_bytes = n_bits // 8
    data_path = reps_dir / f"{name}.dat"

    # Create an empty memmap on disk
    mm = np.memmap(
        data_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n, n_bytes),
    )

    # Process ligands batch by batch
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = ligs.iloc[start:end]
        smiles_list = batch["smiles"].tolist()

        fps_bits_list: List[np.ndarray] = []
        for smi in smiles_list:
            arr = morgan_fp_bits(smi, n_bits=n_bits, radius=radius)
            if arr is None:
                # Do not drop the ligand: if fingerprint fails, use a zero vector
                arr = np.zeros((n_bits,), dtype=np.uint8)
            fps_bits_list.append(arr)

        fps_bits = np.stack(fps_bits_list, axis=0)     # (batch_size, n_bits)
        fps_packed = pack_bits(fps_bits)               # (batch_size, n_bytes)

        mm[start:end, :] = fps_packed

    mm.flush()

    # Metadata describing this representation
    meta = {
        "name": name,
        "file": f"{name}.dat",
        "dtype": "uint8",
        "dim": n_bits,
        "radius": radius,
        "packed_bits": True,
        "packed_dim": n_bytes,
        "n_ligands": int(n),
    }
    meta_path = reps_dir / f"{name}.meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

def build_chemberta_representation(
    root: str | Path,
    n_bits: int = 768,
    radius: int = 2,
    batch_size: int = 128,
    name: str = "chemberta_zinc_base_768",
) -> None:
    """
    Compute ChemBERTa embeddings for all ligands in ligands.parquet and
    store them as a dense float matrix in a memmap on disk.

    Files written under `root`:

      - ligands.parquet (must already exist)
      - reps/<name>.dat         : memmap dense float matrix, shape (N, dim)
      - reps/<name>.meta.json   : metadata describing the representation

    Parameters
    ----------
    root : str | Path
        Root directory containing ligands.parquet. Results are written
        under root / 'reps'.
    n_bits : int
        Embedding dimension expected (default: 768 for ChemBERTa-zinc-base-v1).
        Kept as `n_bits` to be API-identical to build_morgan_representation.
    radius : int
        Kept for API-compatibility (not used for embeddings).
    batch_size : int
        Number of ligands processed per batch. NOTE: for transformer embeddings,
        10000 is usually too large; use something like 64-1024 depending on GPU/CPU.
    name : str
        Name of the representation (used in file names).
    """
    root = Path(root)
    reps_dir = root / "reps"
    reps_dir.mkdir(exist_ok=True, parents=True)

    ligs_path = root / "ligands.parquet"
    ligs = pd.read_parquet(ligs_path)
    n = len(ligs)

    if n == 0:
        raise ValueError("ligands.parquet is empty, nothing to process.")

    # -----------------------------
    # Load ChemBERTa
    # -----------------------------
    model_id = "seyonec/ChemBERTa-zinc-base-v1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Generating ChemBERTa embeddings using {device}')

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    # Validate dimensionality matches the API parameter `n_bits`
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise ValueError("Could not infer hidden_size from model.config.")
    if int(n_bits) != hidden_size:
        raise ValueError(
            f"n_bits={n_bits} does not match ChemBERTa hidden_size={hidden_size}. "
            f"Use n_bits={hidden_size} (or switch to a model whose hidden size matches)."
        )

    dim = hidden_size
    data_path = reps_dir / f"{name}.dat"

    # Store as float16 on disk (smaller). Change to float32 if you prefer.
    mm = np.memmap(
        data_path,
        mode="w+",
        dtype=np.float16,
        shape=(n, dim),
    )

    smiles_all = ligs["smiles"].astype(str).tolist()

    # -----------------------------
    # Embedding helper (mean pooling)
    # -----------------------------
    def _embed_batch(smiles_list: list[str]) -> np.ndarray:
        """
        Returns (B, dim) float32 numpy.
        Uses attention-mask mean pooling on last_hidden_state.
        """
        enc = tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            out = model(**enc)  # BaseModelOutput
            last = out.last_hidden_state  # (B, T, H)
            attn = enc.get("attention_mask", None)

            if attn is None:
                pooled = last.mean(dim=1)
            else:
                attn_f = attn.unsqueeze(-1).to(last.dtype)          # (B, T, 1)
                summed = (last * attn_f).sum(dim=1)                 # (B, H)
                denom = attn_f.sum(dim=1).clamp(min=1.0)            # (B, 1)
                pooled = summed / denom                             # (B, H)

        return pooled.detach().cpu().to(torch.float32).numpy()

    # -----------------------------
    # Process ligands batch by batch
    # -----------------------------
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_smiles = smiles_all[start:end]

        try:
            emb = _embed_batch(batch_smiles)  # (B, dim)
            if emb.shape != (end - start, dim):
                raise RuntimeError(f"Unexpected embedding shape {emb.shape} vs {(end-start, dim)}")
        except Exception:
            emb_rows = []
            for smi in batch_smiles:
                try:
                    e = _embed_batch([smi])[0]
                except Exception:
                    e = np.zeros((dim,), dtype=np.float32)
                emb_rows.append(e)
            emb = np.stack(emb_rows, axis=0)

        mm[start:end, :] = emb.astype(np.float16, copy=False)

    mm.flush()

    # -----------------------------
    # Metadata describing this representation
    # -----------------------------
    meta: Dict = {
        "name": name,
        "file": f"{name}.dat",
        "dtype": "float16",
        "dim": int(dim),
        "radius": int(radius),           # kept only for API compatibility
        "packed_bits": False,
        "packed_dim": None,
        "n_ligands": int(n),
        "model_id": model_id,
        "pooling": "mean_attention_mask",
    }
    meta_path = reps_dir / f"{name}.meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

# ---------------------------------------------------------------------------
# 4. Access representations by chem_comp_id: LigandStore & Representation
# ---------------------------------------------------------------------------

class Representation:
    """
    Numerical representation of ligands stored as a memmap on disk.

    This class abstracts over details such as bit packing and allows
    retrieving vectors by chem_comp_id without loading the full N x D matrix
    into RAM.
    """

    def __init__(
        self,
        name: str,
        memmap: np.memmap,
        meta: Dict,
        id_to_idx: Dict[str, int],
    ):
        self.name = name
        self.memmap = memmap
        self.meta = meta
        self.id_to_idx = id_to_idx

    @property
    def dim(self) -> int:
        """Return the dimensionality of the representation (number of features)."""
        return int(self.meta["dim"])

    def _indices_from_ids(self, comp_ids: List[str]) -> np.ndarray:
        """
        Convert a list of chem_comp_id strings into an array of integer indices (lig_idx).

        Raises KeyError if any chem_comp_id is not found.
        """
        idxs = []
        for cid in comp_ids:
            try:
                idxs.append(self.id_to_idx[cid])
            except KeyError:
                raise KeyError(f"chem_comp_id '{cid}' not found in ligand index.")
        return np.array(idxs, dtype=np.int64)

    def get_by_ids(
        self,
        comp_ids: List[str],
        as_float: bool = False,
    ) -> np.ndarray:
        """
        Retrieve the representation vectors for a list of comp_ids.

        For bit-packed representations (e.g. Morgan), this:
          - reads the packed rows from the memmap
          - unpacks them to 0/1 arrays of shape (n_ids, dim)

        Parameters
        ----------
        comp_ids : list of str
            Ligand IDs (final comp_id) to fetch.
        as_float : bool
            If True, convert the result to float32 (useful for ML models).
            If False, keep the native dtype (e.g. uint8 0/1 for bits).

        Returns
        -------
        np.ndarray
            Array of shape (len(comp_ids), dim).
        """
        if len(comp_ids) == 0:
            return np.zeros((0, self.dim), dtype=np.float32 if as_float else np.uint8)

        idxs = self._indices_from_ids(comp_ids)
        raw = self.memmap[idxs]  # (n_ids, dim_packed) or (n_ids, dim)

        if self.meta.get("packed_bits", False):
            arr = unpack_bits(raw, self.dim)  # (n_ids, dim)
        else:
            arr = np.asarray(raw)

        if as_float:
            return arr.astype(np.float32)
        return arr


class LigandStore:
    """
    Simple manager for ligands and their numerical representations.

    Expected directory structure under `root`:

      root/
        ligands.parquet
        reps/
          <name>.dat
          <name>.meta.json

    The ligands.parquet file must contain:
      - 'chem_comp_id'
      - 'lig_idx'
    """

    def __init__(self, root: str | Path):
        root = Path(root)
        self.root = root

        ligs_path = root / "ligands.parquet"
        if not ligs_path.exists():
            raise FileNotFoundError(f"ligands.parquet not found at {ligs_path}")

        self.ligands = pd.read_parquet(ligs_path)

        if "chem_comp_id" not in self.ligands.columns or "lig_idx" not in self.ligands.columns:
            raise ValueError("ligands.parquet must contain 'chem_comp_id' and 'lig_idx' columns.")

        # Map comp_id -> lig_idx for fast lookup
        self.id_to_idx: Dict[str, int] = dict(
            zip(self.ligands["chem_comp_id"], self.ligands["lig_idx"])
        )

    def load_representation(self, name: str) -> Representation:
        """
        Load a representation stored under root / 'reps'.

        The corresponding meta file (<name>.meta.json) defines:
          - dtype
          - dim
          - packed_bits
          - packed_dim (if packed_bits=True)
          - n_ligands

        Parameters
        ----------
        name : str
            Representation name (e.g. 'morgan_1024_r2').

        Returns
        -------
        Representation
            A Representation object bound to this LigandStore.
        """
        reps_dir = self.root / "reps"
        meta_path = reps_dir / f"{name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata for representation '{name}' not found at {meta_path}"
            )

        with meta_path.open() as f:
            meta = json.load(f)

        data_path = reps_dir / meta["file"]
        dtype = np.dtype(meta["dtype"])

        if meta.get("packed_bits", False):
            shape = (meta["n_ligands"], meta["packed_dim"])
        else:
            shape = (meta["n_ligands"], meta["dim"])

        mm = np.memmap(
            data_path,
            mode="r",
            dtype=dtype,
            shape=shape,
        )

        return Representation(
            name=name,
            memmap=mm,
            meta=meta,
            id_to_idx=self.id_to_idx,
        )
