from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from huggingface_hub import snapshot_download
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from transformers import AutoTokenizer, AutoModel

# -------------------------------------------------------------------------
# Defaults requested
# -------------------------------------------------------------------------
DEFAULT_STORE_ROOT = Path("databases/compound_data/pdb_chembl")
DEFAULT_LIGAND_LISTS_PATH = Path("databases/rev_ligq/ligand_lists.pkl")

CHEMBERTA_REP_NAME = "chemberta_zinc_base_768"
MORGAN_REP_NAME = "morgan_1024_r2"

DEFAULT_TANIMOTO_THRESHOLD = 0.4
DEFAULT_CHEMBERTA_THRESHOLD = 0.8
DEFAULT_K_MAX = 1000

def get_dataset(local_dir: str = "databases") -> None:
    """
    Download the ReverseLigQ dataset from Hugging Face into a local directory.

    Parameters
    ----------
    local_dir : str
        Local directory where the dataset will be stored.
    """
    snapshot_download(
        repo_id="gschottlender/reverse_ligq",
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )


from search_tools.compound_helpers import LigandStore, unpack_bits  # noqa: E402


# -------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------
def _load_ligand_lists(path: str | Path) -> Dict[str, List[str]]:
    path = Path(path)
    with path.open("rb") as f:
        d = pickle.load(f)
    # Normalize keys to strings
    return {str(k): list(v) for k, v in d.items()}


def _build_idx_to_id(store: LigandStore) -> np.ndarray:
    """
    idx_to_id[lig_idx] = chem_comp_id (as str)
    Requires lig_idx to be dense [0..N-1] and consistent with representation rows.
    """
    ligs = store.ligands
    if "lig_idx" not in ligs.columns or "chem_comp_id" not in ligs.columns:
        raise ValueError("ligands.parquet must contain 'lig_idx' and 'chem_comp_id'.")

    ligs_sorted = ligs.sort_values("lig_idx", kind="mergesort")
    idxs = ligs_sorted["lig_idx"].to_numpy(dtype=np.int64)
    if not np.array_equal(idxs, np.arange(len(ligs_sorted), dtype=np.int64)):
        raise ValueError("lig_idx is not a dense 0..N-1 range; cannot map idx->id safely.")

    return ligs_sorted["chem_comp_id"].astype(str).to_numpy()


def _build_smiles_lookup(store: LigandStore) -> Dict[str, str]:
    ligs = store.ligands
    if "smiles" not in ligs.columns:
        raise ValueError("ligands.parquet must contain a 'smiles' column to return SMILES in results.")
    return dict(zip(ligs["chem_comp_id"].astype(str), ligs["smiles"].astype(str)))


def _org_indices_from_ids(
    rep,
    org_ids: List[str],
    strict: bool = False,
) -> np.ndarray:
    """
    Convert list of chem_comp_id -> np.ndarray of lig_idx using rep.id_to_idx.
    """
    id_to_idx = rep.id_to_idx
    if strict:
        missing = [cid for cid in org_ids if cid not in id_to_idx]
        if missing:
            raise KeyError(
                f"{len(missing)} ids from ligand_lists not found in ligands.parquet "
                f"(examples: {missing[:5]})"
            )

    # Fast path: include only those present
    return np.fromiter(
        (int(id_to_idx[cid]) for cid in org_ids if cid in id_to_idx),
        dtype=np.int64,
    )


@dataclass(frozen=True)
class SearchConfig:
    """
    Common search configuration for both searchers.
    """
    min_score: float
    k_max: int
    chunk_size: int = 50_000
    strict_ligands: bool = False


def _update_topk_with_threshold(
    scores: np.ndarray,
    idxs: np.ndarray,
    min_score: float,
    best_scores: np.ndarray,
    best_indices: np.ndarray,
    k_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only candidates with score >= min_score and maintain the global top-k_max
    among those candidates.

    Parameters
    ----------
    scores : (M,) float32
    idxs   : (M,) int64    global indices aligned with scores
    min_score : float
    best_scores : (K,) float32
    best_indices: (K,) int64
    k_max : int

    Returns
    -------
    (best_scores, best_indices) updated, each shape (<=k_max,)
    """
    if scores.size == 0:
        return best_scores, best_indices

    mask = scores >= min_score
    if not np.any(mask):
        return best_scores, best_indices

    s = scores[mask]
    i = idxs[mask]

    # If too many in this chunk, keep only top k_max locally
    if s.size > k_max:
        local = np.argpartition(-s, k_max - 1)[:k_max]
        s = s[local]
        i = i[local]

    all_scores = np.concatenate([best_scores, s])
    all_indices = np.concatenate([best_indices, i])

    # Keep top k_max overall
    k_eff = min(k_max, all_scores.size)
    keep = np.argpartition(-all_scores, k_eff - 1)[:k_eff]
    best_scores = all_scores[keep]
    best_indices = all_indices[keep]

    return best_scores, best_indices


def _finalize_results(
    best_scores: np.ndarray,
    best_indices: np.ndarray,
    idx_to_id: np.ndarray,
    smiles_lookup: Dict[str, str],
    organism: str,
) -> List[Dict[str, Any]]:
    """
    Sort by score desc, tie-break by idx_global asc for determinism,
    then build the final list-of-dicts schema.
    """
    if best_scores.size == 0:
        return []

    # Sort: (-score, idx)
    order = np.lexsort((best_indices.astype(np.int64), -best_scores.astype(np.float32)))
    scores = best_scores[order]
    indices = best_indices[order]

    results: List[Dict[str, Any]] = []
    rank = 1
    for idx_global, score in zip(indices, scores):
        if idx_global < 0:
            continue
        comp_id = str(idx_to_id[int(idx_global)])
        results.append(
            {
                "rank": rank,
                "idx_global": int(idx_global),
                "comp_id": comp_id,
                "score": float(score),
                "smiles": smiles_lookup.get(comp_id),
                "organism": organism,
            }
        )
        rank += 1
    return results


# -------------------------------------------------------------------------
# ChemBERTa (cosine) searcher
# -------------------------------------------------------------------------
class ChembertaSearcher:
    """
    Cosine similarity search using a ChemBERTa embedding representation,
    restricted to ligands belonging to a given organism.

    - Representation is loaded via LigandStore (memmap on disk).
    - Organism candidates are selected via ligand_lists.pkl (list of chem_comp_id).
    - Scoring streams over memmap in chunks to limit RAM.
    - Selection keeps only hits with score >= min_score, capped to top k_max hits.
    """

    def __init__(
        self,
        store: LigandStore,
        ligand_lists: Dict[str, List[str]],
        organism: str,
        rep_name: str = CHEMBERTA_REP_NAME,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        device: Optional[str] = None,
        max_length: int = 256,
        config: Optional[SearchConfig] = None,
    ) -> None:
        self.store = store
        self.rep = store.load_representation(rep_name)
        self.organism = str(organism)

        self.idx_to_id = _build_idx_to_id(store)
        self.smiles_lookup = _build_smiles_lookup(store)

        if self.organism not in ligand_lists:
            raise KeyError(f"Organism {self.organism!r} not found in ligand_lists.")
        org_ids = ligand_lists[self.organism]
        self.org_indices = _org_indices_from_ids(self.rep, org_ids, strict=False)

        # Model for query embedding
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Rep metadata
        self.dim = int(self.rep.meta["dim"])
        # If you ever store normalized embeddings, set meta["normalized"]=True for speed
        self.embs_are_normalized = bool(self.rep.meta.get("normalized", False))

        if config is None:
            config = SearchConfig(
                min_score=DEFAULT_CHEMBERTA_THRESHOLD,
                k_max=DEFAULT_K_MAX,
                chunk_size=50_000,
                strict_ligands=False,
            )
        self.config = config

    @classmethod
    def from_defaults(
        cls,
        organism: str,
        store_root: str | Path = DEFAULT_STORE_ROOT,
        ligand_lists_path: str | Path = DEFAULT_LIGAND_LISTS_PATH,
        rep_name: str = CHEMBERTA_REP_NAME,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        device: Optional[str] = None,
        max_length: int = 256,
        min_score: float = DEFAULT_CHEMBERTA_THRESHOLD,
        k_max: int = DEFAULT_K_MAX,
        chunk_size: int = 50_000,
    ) -> "ChembertaSearcher":
        store = LigandStore(store_root)
        ligand_lists = _load_ligand_lists(ligand_lists_path)
        cfg = SearchConfig(min_score=min_score, k_max=k_max, chunk_size=chunk_size)
        return cls(
            store=store,
            ligand_lists=ligand_lists,
            organism=str(organism),
            rep_name=rep_name,
            model_name=model_name,
            device=device,
            max_length=max_length,
            config=cfg,
        )

    @torch.no_grad()
    def embed_smiles(self, smiles: str, pooling: str = "mean", normalize: bool = True) -> np.ndarray:
        """
        Compute query embedding (float32, shape (dim,)) using attention-mask mean pooling (default).
        """
        encoded = self.tokenizer(
            [smiles],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        outputs = self.model(**encoded)
        last_hidden = outputs.last_hidden_state  # [1, T, H]

        if pooling == "cls":
            vec = last_hidden[:, 0, :]  # [1, H]
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).type_as(last_hidden)  # [1, T, 1]
            summed = (last_hidden * mask).sum(dim=1)                              # [1, H]
            counts = mask.sum(dim=1).clamp(min=1.0)                               # [1, 1]
            vec = summed / counts                                                 # [1, H]

        vec = vec.squeeze(0)  # [H]
        if normalize:
            vec = F.normalize(vec, p=2, dim=0)

        arr = vec.detach().cpu().numpy().astype(np.float32)
        if arr.shape[0] != self.dim:
            raise RuntimeError(f"Query embedding dim mismatch: got {arr.shape[0]}, expected {self.dim}")
        return arr

    def search(
        self,
        query_smiles: str,
        min_score: Optional[float] = None,
        k_max: Optional[int] = None,
        chunk_size: Optional[int] = None,
        pooling: str = "mean",
    ) -> List[Dict[str, Any]]:
        """
        Threshold-based cosine search with an additional neighbor cap.

        - Keep only hits with score >= min_score
        - If too many hits, keep only top k_max by score
        - Results are returned sorted by score desc (tie-break idx asc)

        Defaults:
          min_score = 0.8
          k_max     = 1000
        """
        cfg = self.config
        min_score = cfg.min_score if min_score is None else float(min_score)
        k_max = cfg.k_max if k_max is None else int(k_max)
        chunk_size = cfg.chunk_size if chunk_size is None else int(chunk_size)

        q = self.embed_smiles(query_smiles, pooling=pooling, normalize=True)  # unit norm
        org_idx = self.org_indices
        n_org = len(org_idx)

        best_scores = np.empty((0,), dtype=np.float32)
        best_indices = np.empty((0,), dtype=np.int64)

        for start in range(0, n_org, chunk_size):
            end = min(start + chunk_size, n_org)
            idx_chunk = org_idx[start:end]

            # (M, dim) float32
            chunk_embs = np.asarray(self.rep.memmap[idx_chunk], dtype=np.float32)

            # cosine = dot/(||x||*||q||); ||q||=1
            sims = chunk_embs @ q
            if not self.embs_are_normalized:
                norms = np.linalg.norm(chunk_embs, axis=1)
                norms = np.where(norms > 0, norms, 1.0)
                sims = sims / norms

            best_scores, best_indices = _update_topk_with_threshold(
                scores=sims.astype(np.float32, copy=False),
                idxs=idx_chunk.astype(np.int64, copy=False),
                min_score=min_score,
                best_scores=best_scores,
                best_indices=best_indices,
                k_max=k_max,
            )

        return _finalize_results(
            best_scores=best_scores,
            best_indices=best_indices,
            idx_to_id=self.idx_to_id,
            smiles_lookup=self.smiles_lookup,
            organism=self.organism,
        )


# -------------------------------------------------------------------------
# Morgan (Tanimoto) searcher
# -------------------------------------------------------------------------
class MorganTanimotoSearcher:
    """
    Tanimoto similarity search using a Morgan fingerprint representation,
    restricted to ligands belonging to a given organism.

    - Representation is loaded via LigandStore (memmap on disk).
    - Organism candidates are selected via ligand_lists.pkl (list of chem_comp_id).
    - Scoring streams over memmap in chunks to limit RAM.
    - Selection keeps only hits with score >= min_score, capped to top k_max hits.

    IMPORTANT:
    - This implementation assumes your Morgan representation is packed_bits=True
      (as in your build_morgan_representation). We compute intersection safely by
      unpacking chunk bits via your existing unpack_bits helper (same one used by
      Representation.get_by_ids), ensuring bit order consistency with your packing.
    """

    def __init__(
        self,
        store: LigandStore,
        ligand_lists: Dict[str, List[str]],
        organism: str,
        rep_name: str = MORGAN_REP_NAME,
        radius: int = 2,
        config: Optional[SearchConfig] = None,
    ) -> None:
        self.store = store
        self.rep = store.load_representation(rep_name)
        self.organism = str(organism)
        self.radius = int(radius)

        self.idx_to_id = _build_idx_to_id(store)
        self.smiles_lookup = _build_smiles_lookup(store)

        if self.organism not in ligand_lists:
            raise KeyError(f"Organism {self.organism!r} not found in ligand_lists.")
        org_ids = ligand_lists[self.organism]
        self.org_indices = _org_indices_from_ids(self.rep, org_ids, strict=False)

        # Rep metadata
        self.fp_dim = int(self.rep.meta["dim"])
        self.packed_bits = bool(self.rep.meta.get("packed_bits", False))
        self.packed_dim = int(self.rep.meta.get("packed_dim") or (self.fp_dim // 8))

        if config is None:
            config = SearchConfig(
                min_score=DEFAULT_TANIMOTO_THRESHOLD,
                k_max=DEFAULT_K_MAX,
                chunk_size=50_000,
                strict_ligands=False,
            )
        self.config = config

    @classmethod
    def from_defaults(
        cls,
        organism: str,
        store_root: str | Path = DEFAULT_STORE_ROOT,
        ligand_lists_path: str | Path = DEFAULT_LIGAND_LISTS_PATH,
        rep_name: str = MORGAN_REP_NAME,
        radius: int = 2,
        min_score: float = DEFAULT_TANIMOTO_THRESHOLD,
        k_max: int = DEFAULT_K_MAX,
        chunk_size: int = 50_000,
    ) -> "MorganTanimotoSearcher":
        store = LigandStore(store_root)
        ligand_lists = _load_ligand_lists(ligand_lists_path)
        cfg = SearchConfig(min_score=min_score, k_max=k_max, chunk_size=chunk_size)
        return cls(
            store=store,
            ligand_lists=ligand_lists,
            organism=str(organism),
            rep_name=rep_name,
            radius=radius,
            config=cfg,
        )

    def fp_from_smiles_bits(self, smiles: str) -> np.ndarray:
        """
        Compute query Morgan fingerprint as a 0/1 uint8 vector of shape (fp_dim,).
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")

        bitvect = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.radius,
            nBits=self.fp_dim,
        )
        arr = np.zeros((self.fp_dim,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bitvect, arr)
        return arr

    def search(
        self,
        query_smiles: str,
        min_score: Optional[float] = None,
        k_max: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Threshold-based Tanimoto search with an additional neighbor cap.

        - Keep only hits with score >= min_score
        - If too many hits, keep only top k_max by score
        - Results are returned sorted by score desc (tie-break idx asc)

        Defaults:
          min_score = 0.4
          k_max     = 1000
        """
        cfg = self.config
        min_score = cfg.min_score if min_score is None else float(min_score)
        k_max = cfg.k_max if k_max is None else int(k_max)
        chunk_size = cfg.chunk_size if chunk_size is None else int(chunk_size)

        q_bits = self.fp_from_smiles_bits(query_smiles).astype(np.uint8)
        bits_q = float(q_bits.sum())

        org_idx = self.org_indices
        n_org = len(org_idx)

        best_scores = np.empty((0,), dtype=np.float32)
        best_indices = np.empty((0,), dtype=np.int64)

        for start in range(0, n_org, chunk_size):
            end = min(start + chunk_size, n_org)
            idx_chunk = org_idx[start:end]

            # Read packed rows from memmap, then unpack with your canonical helper
            raw = np.asarray(self.rep.memmap[idx_chunk], dtype=np.uint8)  # (M, packed_dim) if packed_bits
            if self.packed_bits:
                fps_chunk = unpack_bits(raw, self.fp_dim).astype(np.uint8, copy=False)  # (M, fp_dim) 0/1
            else:
                # In case you stored 0/1 directly (not expected, but supported)
                fps_chunk = raw.astype(np.uint8, copy=False)  # (M, fp_dim)

            # intersection = sum(fp & q)
            inter = np.bitwise_and(fps_chunk, q_bits).sum(axis=1).astype(np.float32)
            bits_chunk = fps_chunk.sum(axis=1).astype(np.float32)
            union = bits_q + bits_chunk - inter
            denom = np.where(union > 0, union, 1.0)
            tanimoto = inter / denom

            best_scores, best_indices = _update_topk_with_threshold(
                scores=tanimoto,
                idxs=idx_chunk.astype(np.int64, copy=False),
                min_score=min_score,
                best_scores=best_scores,
                best_indices=best_indices,
                k_max=k_max,
            )

        return _finalize_results(
            best_scores=best_scores,
            best_indices=best_indices,
            idx_to_id=self.idx_to_id,
            smiles_lookup=self.smiles_lookup,
            organism=self.organism,
        )

def attach_domains_to_ligands(
    ligand_results: List[Dict[str, Any]],
    base_dir: str | Path,
) -> List[Dict[str, Any]]:
    """
    Annotate ligand search results with curated/possible domains.

    This function automatically loads:
      - ligs_fams_curated.pkl
      - ligs_fams_possible.pkl
    from `base_dir`, and enriches each ligand result with:

      - 'domains': list of {'domain_id': <Pfam>, 'tag': 'curated'/'possible'}
      - 'tags'   : list of tags present at the ligand level

    Parameters
    ----------
    ligand_results : list of dict
        Output of ChembertaSearcher.search or MorganTanimotoSearcher.search.
    base_dir : str or Path
        Directory containing ligs_fams_curated.pkl and ligs_fams_possible.pkl.

    Returns
    -------
    List[dict]
        New list of ligand results with domain annotations.
    """
    base_dir = Path(base_dir)

    with open(base_dir / "ligs_fams_curated.pkl", "rb") as f:
        ligs_fams_curated: Dict[str, List[str]] = pickle.load(f)

    with open(base_dir / "ligs_fams_possible.pkl", "rb") as f:
        ligs_fams_possible: Dict[str, List[str]] = pickle.load(f)

    annotated: List[Dict[str, Any]] = []

    for result in ligand_results:
        lig_id = result["comp_id"]
        curated_domains = ligs_fams_curated.get(lig_id, [])
        possible_domains = ligs_fams_possible.get(lig_id, [])

        # domain_id -> tag ('curated' or 'possible')
        domain_tags: Dict[str, str] = {}

        # Curated has priority over possible
        for dom_id in curated_domains:
            domain_tags[dom_id] = "curated"

        for dom_id in possible_domains:
            if dom_id not in domain_tags:
                domain_tags[dom_id] = "possible"

        domains_list = [
            {"domain_id": dom_id, "tag": tag}
            for dom_id, tag in domain_tags.items()
        ]

        ligand_tags: List[str] = []
        if curated_domains:
            ligand_tags.append("curated")
        if possible_domains:
            ligand_tags.append("possible")

        new_entry = dict(result)
        new_entry["domains"] = domains_list
        new_entry["tags"] = ligand_tags

        annotated.append(new_entry)

    return annotated


def build_candidate_proteins_table(
    annotated_ligands: List[Dict[str, Any]],
    base_dir: str | Path,
    organism: str | int,
    max_domain_ranks: Optional[int] = 50,   # <-- allow None
    include_only_curated: bool = False,
    show_only_proteins_with_description: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build the final candidate proteins table from ligand-domain annotations.

    If max_domain_ranks is None, domain rank filtering is disabled and all
    domains found are kept.
    """
    base_dir = Path(base_dir)
    organism_str = str(organism)

    # 1) Load family->protein mapping and protein descriptions
    with open(base_dir / "fam_prot_dict.pkl", "rb") as f:
        fam_prot_dict = pickle.load(f)

    if organism_str not in fam_prot_dict:
        raise KeyError(f"Organism {organism_str!r} not found in fam_prot_dict.pkl")

    fam_prot_org: Dict[str, List[str]] = fam_prot_dict[organism_str]

    prot_descriptions: Dict[str, Any] = {}  # <-- avoid UnboundLocalError
    prot_desc_path = base_dir / "prot_descriptions.pkl"
    if prot_desc_path.exists():
        with open(prot_desc_path, "rb") as f:
            prot_descriptions = pickle.load(f)
        # Expected: {'description': {protein_id: description, ...}}
        prot_descriptions = prot_descriptions.get("description", {}) or {}

    # 2) Select best reference ligand for each domain
    domain_stats: Dict[str, Dict[str, Any]] = {}

    for ligand_entry in annotated_ligands:
        ligand_id = ligand_entry["comp_id"]
        ligand_score = ligand_entry["score"]
        ligand_smiles = ligand_entry.get("smiles")

        for dom in ligand_entry.get("domains", []):
            domain_id = dom["domain_id"]
            tag = dom["tag"]  # 'curated' or 'possible'

            if include_only_curated and tag != "curated":
                continue

            # Domain must exist for this organism
            if domain_id not in fam_prot_org:
                continue

            info = domain_stats.get(domain_id)
            if info is None or ligand_score > info["reference_ligand_score"]:
                domain_stats[domain_id] = {
                    "reference_ligand_id": ligand_id,
                    "reference_ligand_score": ligand_score,
                    "reference_ligand_smiles": ligand_smiles,
                    "domain_tag": tag,
                }

    if not domain_stats:
        return []

    # 3) Rank domains by reference ligand score
    domain_list: List[Dict[str, Any]] = []
    for domain_id, info in domain_stats.items():
        entry = {"domain_id": domain_id}
        entry.update(info)
        domain_list.append(entry)

    domain_list.sort(key=lambda d: -d["reference_ligand_score"])

    current_rank = 0
    last_score: Optional[float] = None
    for entry in domain_list:
        score = entry["reference_ligand_score"]
        if last_score is None or score < last_score:
            current_rank += 1
            last_score = score
        entry["rank"] = current_rank

    # NEW: apply rank filter only if max_domain_ranks is not None and > 0
    if max_domain_ranks is not None and int(max_domain_ranks) > 0:
        mdr = int(max_domain_ranks)
        domain_list = [d for d in domain_list if d["rank"] <= mdr]

    # 4) Expand to protein-level rows
    rows: List[Dict[str, Any]] = []

    for entry in domain_list:
        domain_id = entry["domain_id"]
        rank = entry["rank"]
        domain_tag = entry["domain_tag"]
        ref_lig_id = entry["reference_ligand_id"]
        ref_lig_score = entry["reference_ligand_score"]
        ref_lig_smiles = entry["reference_ligand_smiles"]

        protein_ids = fam_prot_org.get(domain_id, [])
        if not protein_ids:
            continue

        if show_only_proteins_with_description:
            protein_ids = [p for p in protein_ids if p in prot_descriptions]
            if not protein_ids:
                continue

        for protein_id in protein_ids:
            description = prot_descriptions.get(protein_id)

            row = {
                "rank": rank,
                "protein_id": protein_id,
                "protein_description": description,
                "domain_id": domain_id,
                "domain_tag": domain_tag,
                "reference_ligand_id": ref_lig_id,
                "reference_ligand_score": ref_lig_score,
                "reference_ligand_smiles": ref_lig_smiles,
            }
            rows.append(row)

    rows.sort(
        key=lambda r: (
            r["rank"],
            -r["reference_ligand_score"],
            r["protein_id"],
        )
    )

    return rows

def build_ligand_summary_dataframe(
    annotated_ligands: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a ligand-level summary table from the output of attach_domains_to_ligands.

    Each row corresponds to a single ligand and includes:
      - rank
      - comp_id
      - score
      - smiles
      - curated_domains       (comma-separated string)
      - possible_domains      (comma-separated string)
      - domain_summary        (human-readable summary)

    Parameters
    ----------
    annotated_ligands : list of dict
        Output of attach_domains_to_ligands. Each entry must contain:
          - 'rank'
          - 'comp_id'
          - 'score'
          - 'smiles'
          - 'domains': list of { 'domain_id': str, 'tag': 'curated'/'possible' }

    Returns
    -------
    pandas.DataFrame
        Ligand-level summary table.
    """
    rows = []

    for lig in annotated_ligands:
        comp_id = lig.get("comp_id")
        rank = lig.get("rank")
        score = lig.get("score")
        smiles = lig.get("smiles")

        domains = lig.get("domains", [])

        curated = sorted(
            d["domain_id"] for d in domains if d.get("tag") == "curated"
        )
        possible = sorted(
            d["domain_id"] for d in domains if d.get("tag") == "possible"
        )

        curated_str = ", ".join(curated) if curated else ""
        possible_str = ", ".join(possible) if possible else ""

        summary_parts = []
        if curated:
            summary_parts.append(f"curated: {curated_str}")
        if possible:
            summary_parts.append(f"possible: {possible_str}")
        domain_summary = " | ".join(summary_parts) if summary_parts else ""

        rows.append(
            {
                "rank": rank,
                "comp_id": comp_id,
                "score": score,
                "smiles": smiles,
                "curated_domains": curated_str,
                "possible_domains": possible_str,
                "domain_summary": domain_summary,
            }
        )

    df = pd.DataFrame(rows)
    # Optional: sort by rank then score descending
    df = df.sort_values(by=["rank", "score"], ascending=[True, False]).reset_index(drop=True)
    return df