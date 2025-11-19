from __future__ import annotations

import pickle
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


def get_dataset(local_dir: str = "data") -> None:
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


class ChembertaSearcher:
    """
    Similarity searcher based on ChemBERTa embeddings for a single organism.

    This class:
      - loads global ChemBERTa embeddings (memory-mapped),
      - restricts them to the ligands associated with a given organism,
      - computes a ChemBERTa embedding for a query SMILES,
      - performs cosine-similarity k-NN search only within that organism.
    """

    def __init__(
        self,
        embs: np.memmap,
        id_to_idx: Dict[str, int],
        idx_to_id: List[str],
        smiles_dict: Dict[str, str],
        organism: str,
        org_indices: np.ndarray,
        tokenizer,
        model,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        """
        Parameters
        ----------
        embs : np.memmap
            Global embedding matrix of shape (N_total, H), memory-mapped.
        id_to_idx : dict
            Global mapping from compound ID (e.g. CHEMBL ID) to global index.
        idx_to_id : list
            Global list mapping index -> compound ID.
        smiles_dict : dict
            Subset of global SMILES dictionary, restricted to this organism.
        organism : str
            Organism identifier (e.g. "1", "13", ...).
        org_indices : np.ndarray
            Global indices (into `embs`) that belong to this organism.
        tokenizer : transformers.PreTrainedTokenizer
            ChemBERTa tokenizer.
        model : transformers.PreTrainedModel
            ChemBERTa model in eval mode.
        device : str
            Device where the model lives ("cpu" or "cuda").
        max_length : int
            Maximum token length for ChemBERTa input.
        """
        self.embs = embs  # (N_total, H) float32/float16 memmap
        self.id_to_idx = id_to_idx
        self.idx_to_id = idx_to_id
        self.smiles_dict = smiles_dict
        self.organism = organism
        self.org_indices = org_indices
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_length = max_length

        # Global info (not only for this organism)
        self.N_total, self.H = embs.shape
        self.N_org = len(org_indices)

    # --------- convenience constructor ---------
    @classmethod
    def from_paths(
        cls,
        base_dir: str | Path,
        organism: str,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        device: Optional[str] = None,
    ) -> "ChembertaSearcher":
        """
        Build a ChembertaSearcher for a single organism from disk.

        This method:
          - loads global embeddings (memmap),
          - loads id_to_idx / idx_to_id mappings,
          - loads global SMILES,
          - loads ligand_lists and extracts the ligands for this organism,
          - builds a restricted SMILES dict for the organism,
          - loads ChemBERTa model and tokenizer.

        Parameters
        ----------
        base_dir : str or Path
            Directory containing:
              - comps_embs.npy
              - id_to_idx.pkl
              - idx_to_id.pkl
              - comps_smiles.pkl
              - ligand_lists.pkl
        organism : str
            Organism identifier (e.g. "1", "13", ...).
        model_name : str
            Hugging Face model name or path for ChemBERTa.
        device : str, optional
            "cpu" or "cuda". If None, defaults to "cpu" here.

        Returns
        -------
        ChembertaSearcher
        """
        base_dir = Path(base_dir)

        # 1) Global embeddings (NOT fully loaded into RAM, memmap only)
        embs = np.load(base_dir / "comps_embs.npy", mmap_mode="r")

        # 2) Global mappings
        with open(base_dir / "id_to_idx.pkl", "rb") as f:
            id_to_idx = pickle.load(f)
        with open(base_dir / "idx_to_id.pkl", "rb") as f:
            idx_to_id = pickle.load(f)

        # 3) Global SMILES dictionary
        with open(base_dir / "comps_smiles.pkl", "rb") as f:
            smiles_global = pickle.load(f)

        # 4) Ligand lists by organism
        with open(base_dir / "ligand_lists.pkl", "rb") as f:
            ligand_lists = pickle.load(f)

        organism_str = str(organism)
        if organism_str not in ligand_lists:
            raise KeyError(f"Organism {organism_str!r} not found in ligand_lists.pkl")

        ligand_ids_org = ligand_lists[organism_str]

        # 5) Global indices for this organism
        org_indices: List[int] = []
        for lig in ligand_ids_org:
            if lig not in id_to_idx:
                raise KeyError(
                    f"Ligand {lig!r} from organism {organism_str} not found in id_to_idx."
                )
            org_indices.append(id_to_idx[lig])
        org_indices_array = np.array(org_indices, dtype=np.int64)

        # 6) Restrict SMILES dictionary to this organism
        smiles_org = {
            lig: smiles_global[lig]
            for lig in ligand_ids_org
            if lig in smiles_global
        }

        # 7) Load ChemBERTa model and tokenizer
        if device is None:
            device = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()

        return cls(
            embs=embs,
            id_to_idx=id_to_idx,
            idx_to_id=idx_to_id,
            smiles_dict=smiles_org,
            organism=organism_str,
            org_indices=org_indices_array,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

    # --------- single-SMILES embedding ---------
    @torch.no_grad()
    def embed_smiles(
        self,
        smiles: str,
        use_amp: bool = False,
        pooling: str = "mean",
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Convert a SMILES string into a 1D ChemBERTa embedding on CPU.

        Parameters
        ----------
        smiles : str
            Input SMILES string.
        use_amp : bool
            Mixed precision (AMP) flag. Only relevant for GPU; ignored on CPU.
        pooling : str
            "mean" for masked mean pooling, or "cls" for first token.
        normalize : bool
            If True, return L2-normalized embedding.

        Returns
        -------
        np.ndarray
            1D embedding vector of shape (H,).
        """
        encoded = self.tokenizer(
            [smiles],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # For CPU we do not use autocast
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

        return vec.detach().cpu().numpy().astype(np.float32)

    # --------- cosine similarity search (within this organism) ---------
    def search(
        self,
        query_smiles: str,
        top_k: int = 10,
        chunk_size: int = 50_000,
    ) -> List[Dict[str, Any]]:
        """
        Perform cosine-similarity k-NN search for a query SMILES,
        restricted to the ligands of this organism.

        Parameters
        ----------
        query_smiles : str
            Query SMILES string.
        top_k : int
            Number of nearest neighbors to return.
        chunk_size : int
            Chunk size for streaming over embeddings to save RAM.

        Returns
        -------
        List[dict]
            Sorted list of neighbors with fields:
              - rank
              - idx_global
              - comp_id
              - score
              - smiles
              - organism
        """
        # 1) Query embedding
        query_vec = self.embed_smiles(query_smiles)

        org_idx = self.org_indices
        n_org = len(org_idx)

        best_scores = np.full(top_k, -1.0, dtype=np.float32)
        best_indices = np.full(top_k, -1, dtype=np.int64)

        # 2) Stream over organism indices in chunks
        for start in range(0, n_org, chunk_size):
            end = min(start + chunk_size, n_org)
            idx_chunk = org_idx[start:end]  # (M,)
            chunk_embs = np.asarray(self.embs[idx_chunk], dtype=np.float32)  # (M, H)

            # Cosine similarity = dot product if vectors are normalized
            sims = chunk_embs @ query_vec  # (M,)

            local_top_k = min(top_k, end - start)
            idx_local = np.argpartition(-sims, local_top_k - 1)[:local_top_k]
            scores_local = sims[idx_local]
            idx_global_local = idx_chunk[idx_local]

            all_scores = np.concatenate([best_scores, scores_local])
            all_indices = np.concatenate([best_indices, idx_global_local])

            k_eff = min(top_k, all_scores.size)
            new_idx = np.argpartition(-all_scores, k_eff - 1)[:k_eff]
            best_scores = all_scores[new_idx]
            best_indices = all_indices[new_idx]

        # 3) Final sort
        order = np.argsort(-best_scores)
        best_scores = best_scores[order]
        best_indices = best_indices[order]

        results: List[Dict[str, Any]] = []
        for rank, (idx_global, score) in enumerate(zip(best_indices, best_scores), start=1):
            if idx_global < 0:
                continue
            comp_id = self.idx_to_id[idx_global]
            smiles = self.smiles_dict.get(comp_id)
            results.append(
                {
                    "rank": rank,
                    "idx_global": int(idx_global),
                    "comp_id": comp_id,
                    "score": float(score),
                    "smiles": smiles,
                    "organism": self.organism,
                }
            )

        return results


class MorganTanimotoSearcher:
    """
    Similarity searcher based on Morgan fingerprints and Tanimoto coefficient
    for a single organism.
    """

    def __init__(
        self,
        fps: np.memmap,
        id_to_idx: Dict[str, int],
        idx_to_id: List[str],
        smiles_dict: Dict[str, str],
        organism: str,
        org_indices: np.ndarray,
        radius: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        fps : np.memmap
            Global fingerprint matrix of shape (N_total, fp_dim) with 0/1 values.
        id_to_idx : dict
            Global mapping from compound ID to global index.
        idx_to_id : list
            Global list mapping index -> compound ID.
        smiles_dict : dict
            Subset of global SMILES dictionary for this organism.
        organism : str
            Organism identifier (e.g. "1", "13", ...).
        org_indices : np.ndarray
            Global indices (into `fps`) that belong to this organism.
        radius : int
            Morgan fingerprint radius (default 2).
        """
        self.fps = fps
        self.id_to_idx = id_to_idx
        self.idx_to_id = idx_to_id
        self.smiles_dict = smiles_dict
        self.organism = organism
        self.org_indices = org_indices
        self.radius = radius

        self.N_total, self.fp_dim = fps.shape
        self.N_org = len(org_indices)

        # Precompute bit counts for organism fingerprints to speed up Tanimoto
        fps_org = np.asarray(self.fps[self.org_indices], dtype=np.uint8)
        self.bits_org = fps_org.sum(axis=1).astype(np.int32)

    # --------- convenience constructor ---------
    @classmethod
    def from_paths(
        cls,
        base_dir: str | Path,
        organism: str,
        radius: int = 2,
    ) -> "MorganTanimotoSearcher":
        """
        Build a MorganTanimotoSearcher for a single organism from disk.

        This method:
          - loads global FPS (memmap),
          - loads id_to_idx / idx_to_id,
          - loads global SMILES,
          - loads ligand_lists and extracts the ligands for this organism,
          - builds a restricted SMILES dict for the organism,
          - precomputes bit counts per-organism.

        Parameters
        ----------
        base_dir : str or Path
            Directory containing:
              - comps_fps.npy
              - id_to_idx.pkl
              - idx_to_id.pkl
              - comps_smiles.pkl
              - ligand_lists.pkl
        organism : str
            Organism identifier (e.g. "1", "13", ...).
        radius : int
            Morgan fingerprint radius to use.

        Returns
        -------
        MorganTanimotoSearcher
        """
        base_dir = Path(base_dir)

        # 1) Global FPS
        fps = np.load(base_dir / "comps_fps.npy", mmap_mode="r")

        # 2) Global mappings
        with open(base_dir / "id_to_idx.pkl", "rb") as f:
            id_to_idx = pickle.load(f)
        with open(base_dir / "idx_to_id.pkl", "rb") as f:
            idx_to_id = pickle.load(f)

        # 3) Global SMILES
        with open(base_dir / "comps_smiles.pkl", "rb") as f:
            smiles_global = pickle.load(f)

        # 4) Ligand lists by organism
        with open(base_dir / "ligand_lists.pkl", "rb") as f:
            ligand_lists = pickle.load(f)

        organism_str = str(organism)
        if organism_str not in ligand_lists:
            raise KeyError(f"Organism {organism_str!r} not found in ligand_lists.pkl")

        ligand_ids_org = ligand_lists[organism_str]

        # 5) Global indices for this organism
        org_indices: List[int] = []
        for lig in ligand_ids_org:
            if lig not in id_to_idx:
                raise KeyError(
                    f"Ligand {lig!r} from organism {organism_str} not found in id_to_idx."
                )
            org_indices.append(id_to_idx[lig])
        org_indices_array = np.array(org_indices, dtype=np.int64)

        # 6) Restrict SMILES dictionary to this organism
        smiles_org = {
            lig: smiles_global[lig]
            for lig in ligand_ids_org
            if lig in smiles_global
        }

        return cls(
            fps=fps,
            id_to_idx=id_to_idx,
            idx_to_id=idx_to_id,
            smiles_dict=smiles_org,
            organism=organism_str,
            org_indices=org_indices_array,
            radius=radius,
        )

    # --------- Morgan fingerprint from SMILES ---------
    def fp_from_smiles(self, smiles: str) -> np.ndarray:
        """
        Compute a binary Morgan fingerprint (0/1) for a query SMILES.

        Parameters
        ----------
        smiles : str
            Query SMILES.

        Returns
        -------
        np.ndarray
            1D binary fingerprint array of shape (fp_dim,).
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

    # --------- Tanimoto search within this organism ---------
    def search(
        self,
        query_smiles: str,
        top_k: int = 10,
        chunk_size: int = 50_000,
    ) -> List[Dict[str, Any]]:
        """
        Perform Tanimoto-based k-NN search for a query SMILES,
        restricted to the ligands of this organism.

        Parameters
        ----------
        query_smiles : str
            Query SMILES string.
        top_k : int
            Number of neighbors to return.
        chunk_size : int
            Chunk size for streaming over FPS to save RAM.

        Returns
        -------
        List[dict]
            Sorted list of neighbors with fields:
              - rank
              - idx_global
              - comp_id
              - score (Tanimoto)
              - smiles
              - organism
        """
        # 1) Query fingerprint
        q = self.fp_from_smiles(query_smiles).astype(np.uint8)
        bits_q = q.sum()

        org_idx = self.org_indices
        n_org = len(org_idx)
        bits_org = self.bits_org

        best_scores = np.full(top_k, -1.0, dtype=np.float32)
        best_indices = np.full(top_k, -1, dtype=np.int64)

        # 2) Stream over organism fingerprints
        for start in range(0, n_org, chunk_size):
            end = min(start + chunk_size, n_org)
            idx_chunk = org_idx[start:end]
            fps_chunk = np.asarray(self.fps[idx_chunk], dtype=np.uint8)  # (M, fp_dim)

            # Intersection = sum(fp & q)
            intersection = np.bitwise_and(fps_chunk, q).sum(axis=1).astype(np.float32)
            # Precomputed bits for organism
            bits_chunk = bits_org[start:end].astype(np.float32)
            union = bits_q + bits_chunk - intersection

            # Avoid division by zero
            denom = np.where(union > 0, union, 1.0)
            tanimoto = intersection / denom

            local_top_k = min(top_k, end - start)
            idx_local = np.argpartition(-tanimoto, local_top_k - 1)[:local_top_k]
            scores_local = tanimoto[idx_local]
            idx_global_local = idx_chunk[idx_local]

            all_scores = np.concatenate([best_scores, scores_local])
            all_indices = np.concatenate([best_indices, idx_global_local])

            k_eff = min(top_k, all_scores.size)
            new_idx = np.argpartition(-all_scores, k_eff - 1)[:k_eff]
            best_scores = all_scores[new_idx]
            best_indices = all_indices[new_idx]

        # 3) Final sort
        order = np.argsort(-best_scores)
        best_scores = best_scores[order]
        best_indices = best_indices[order]

        results: List[Dict[str, Any]] = []
        for rank, (idx_global, score) in enumerate(zip(best_indices, best_scores), start=1):
            if idx_global < 0:
                continue
            comp_id = self.idx_to_id[idx_global]
            smiles = self.smiles_dict.get(comp_id)
            results.append(
                {
                    "rank": rank,
                    "idx_global": int(idx_global),
                    "comp_id": comp_id,
                    "score": float(score),
                    "smiles": smiles,
                    "organism": self.organism,
                }
            )

        return results


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
    max_domain_ranks: int = 50,
    include_only_curated: bool = False,
    show_only_proteins_with_description: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build the final candidate proteins table from ligand-domain annotations.

    Final column order (per row):
      - rank
      - protein_id
      - protein_description
      - domain_id
      - domain_tag
      - reference_ligand_id
      - reference_ligand_score
      - reference_ligand_smiles

    Parameters
    ----------
    annotated_ligands : list of dict
        Output of attach_domains_to_ligands, containing ligand-domain mapping.
    base_dir : str or Path
        Directory containing:
          - fam_prot_dict.pkl
          - prot_descriptions.pkl
    organism : str or int
        Organism identifier used as key in fam_prot_dict.
    max_domain_ranks : int
        Maximum number of domain ranks to include. Domains sharing the same
        reference score share the same rank. All domains with rank <=
        max_domain_ranks are kept.
    include_only_curated : bool
        If True, only domains reached via 'curated' evidence are considered.
        If False, domains with 'possible' evidence are also included.
    show_only_proteins_with_description : bool
        If True, only proteins that have a description are included.
        If False, all proteins are included, with description possibly None.

    Returns
    -------
    List[dict]
        List of rows representing candidate proteins.
    """
    base_dir = Path(base_dir)
    organism_str = str(organism)

    # 1) Load family->protein mapping and protein descriptions
    with open(base_dir / "fam_prot_dict.pkl", "rb") as f:
        fam_prot_dict = pickle.load(f)

    if organism_str not in fam_prot_dict:
        raise KeyError(f"Organism {organism_str!r} not found in fam_prot_dict.pkl")

    fam_prot_org: Dict[str, List[str]] = fam_prot_dict[organism_str]

    prot_desc_path = base_dir / "prot_descriptions.pkl"
    if prot_desc_path.exists():
        with open(prot_desc_path, "rb") as f:
            prot_descriptions = pickle.load(f)
        # Expected to be a dict: protein_id -> description
        prot_descriptions = prot_descriptions['description']

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

    if max_domain_ranks > 0:
        domain_list = [d for d in domain_list if d["rank"] <= max_domain_ranks]

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