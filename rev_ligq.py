#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReverseLigQ - master CLI script (single-query or batch CSV)

Dataset layout (expected):

  databases/
    compound_data/pdb_chembl/
      ligands.parquet
      reps/
        morgan_1024_r2.dat
        morgan_1024_r2.meta.json
        chemberta_zinc_base_768.dat
        chemberta_zinc_base_768.meta.json
    rev_lq/
      ligand_lists.pkl
      ligs_fams_curated.pkl
      ligs_fams_possible.pkl
      fam_prot_dict.pkl
      prot_descriptions.pkl

Modes:
  1) Single query: provide --query-smiles and --organism
  2) Batch mode: provide --query-csv with columns: lig_id, smiles

Batch outputs:
  <out_dir>/<lig_id>/
      predicted_targets.csv
      similarity_search_results.csv

Notes:
- Similarity search uses a threshold (min_score) AND a cap (k_max_ligands).
  This prevents huge intermediate results and keeps memory usage bounded.
- In batch mode, the searcher is constructed once and reused for all queries
  (important for ChemBERTa model loading time).
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from search_tools.target_finding import (  # noqa: E402
    get_dataset,
    ChembertaSearcher,
    MorganTanimotoSearcher,
    attach_domains_to_ligands,
    build_candidate_proteins_table,
    build_ligand_summary_dataframe,
)


# -------------------------------------------------------------------------
# CLI helpers
# -------------------------------------------------------------------------
def _optional_int(x: str) -> Optional[int]:
    """
    Argparse helper: allow passing None from CLI.

    Accepted values for None: "none", "null", "" (case-insensitive).
    Otherwise parses int.
    """
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"none", "null", ""}:
        return None
    return int(s)


def _safe_dirname(name: str, fallback: str = "query") -> str:
    """
    Make a safe folder name from an arbitrary ligand id:
    - strips leading/trailing whitespace
    - replaces path separators and unsafe chars with '_'
    - avoids empty names
    """
    s = str(name).strip()
    if not s:
        return fallback
    # Replace path separators and any unsafe characters
    s = s.replace(os.sep, "_")
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    # Avoid "." or ".."
    if s in {".", ".."}:
        return fallback
    return s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ReverseLigQ target search. "
            "Runs ligand similarity search (Morgan+Tanimoto or ChemBERTa+cosine), "
            "then maps hit ligands to Pfam domains and candidate proteins."
        )
    )

    # Required organism (required in both modes)
    parser.add_argument(
    "--organism",
    type=str,
    required=True,
    help=(
        "Organism key. For built-in organisms, use one of:\n"
        "  1  Bartonella bacilliformis\n"
        "  2  Klebsiella pneumoniae\n"
        "  3  Mycobacterium tuberculosis\n"
        "  4  Trypanosoma cruzi\n"
        "  5  Staphylococcus aureus RF122\n"
        "  6  Streptococcus uberis 0140J\n"
        "  7  Enterococcus faecium\n"
        "  8  Escherichia coli MG1655\n"
        "  9  Streptococcus agalactiae NEM316\n"
        " 10  Pseudomonas syringae\n"
        " 11  DENV (Dengue virus)\n"
        " 12  SARS-CoV-2\n"
        " 13  Homo sapiens\n"
        "If --uploaded-organism is used, this must instead match the name of the "
        "uploaded organism (i.e. a subdirectory under <data_root>/local_organism_data)."
    ),
    )

    # Flag: organism uploaded via the upload pipeline
    parser.add_argument(
        "--uploaded-organism",
        action="store_true",
        help=(
            "Use an uploaded organism instead of the built-in ones. "
            "When set, --organism is interpreted as the name of the uploaded "
            "organism, and protein/domain data are read from "
            "<data_root>/local_organism_data/<organism> while ligand→domain data "
            "still come from --rev-dir."
        ),
    )

    # Query input: either a single SMILES or a CSV batch
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--query-smiles",
        type=str,
        default=None,
        help="Query molecule SMILES string (single-query mode).",
    )
    group.add_argument(
        "--query-csv",
        type=str,
        default=None,
        help=(
            "CSV file for batch mode. Must contain columns: lig_id, smiles. "
            "Each row is run as an independent query."
        ),
    )

    # Data directories
    parser.add_argument(
        "--compound-dir",
        type=str,
        default="databases/compound_data/pdb_chembl",
        help=(
            "Directory containing compound index + representations (LigandStore root): "
            "ligands.parquet and reps/<rep_name>.dat + reps/<rep_name>.meta.json. "
            "Default: databases/compound_data/pdb_chembl"
        ),
    )
    parser.add_argument(
        "--rev-dir",
        type=str,
        default="databases/rev_ligq",
        help=(
            "Directory containing ReverseLigQ metadata: ligand_lists.pkl, "
            "ligs_fams_curated.pkl, ligs_fams_possible.pkl, fam_prot_dict.pkl, "
            "prot_descriptions.pkl (optional), etc. "
            "Default: databases/rev_ligq"
        ),
    )

    # Output directory
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory where result tables will be written. Default: results",
    )

    # Search mode (default must be Tanimoto)
    parser.add_argument(
        "--search-type",
        type=str,
        default="morgan_fp_tanimoto",
        choices=["morgan_fp_tanimoto", "chemberta"],
        help=(
            "Ligand similarity model. Default: morgan_fp_tanimoto. "
            "Options: morgan_fp_tanimoto (Morgan+Tanimoto), chemberta (ChemBERTa+cosine)."
        ),
    )

    # Threshold (default 0.4 for Tanimoto; 0.8 for ChemBERTa)
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help=(
            "Similarity threshold. If omitted, defaults are applied by search type: "
            "0.4 for morgan_fp_tanimoto, 0.8 for chemberta."
        ),
    )

    # k_max (memory safety cap)
    parser.add_argument(
        "--k-max-ligands",
        type=int,
        default=1000,
        help=(
            "Maximum number of ligand hits to keep AFTER applying the threshold. "
            "Default: 1000 (memory-safe). "
            "Increase if you want more hits, but higher values can increase RAM usage "
            "(especially in downstream annotation steps)."
        ),
    )

    # Stream chunk size (optional but useful)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help=(
            "Chunk size (number of organism ligands processed per iteration) when streaming "
            "over the memmap. Default: 50000. "
            "Lower values use less RAM but can be slower."
        ),
    )

    # Downstream options
    parser.add_argument(
        "--max-domain-ranks",
        type=_optional_int,
        default=20,
        help=(
            "Maximum number of domain ranks to keep in the final protein table. "
            "Default: 20. "
            "Pass 'none' to keep ALL domain ranks (no truncation)."
        ),
    )
    parser.add_argument(
        "--include-only-curated",
        action="store_true",
        help="If set, only domains reached via 'curated' evidence are considered.",
    )
    parser.add_argument(
        "--only-proteins-with-description",
        action="store_true",
        help="If set, only proteins that have a description are kept.",
    )

    # ChemBERTa controls (only used when --search-type chemberta)
    parser.add_argument(
        "--chemberta-model",
        type=str,
        default="seyonec/ChemBERTa-zinc-base-v1",
        help="Hugging Face model name/path for ChemBERTa. Default: seyonec/ChemBERTa-zinc-base-v1",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device for ChemBERTa query embedding: 'cpu' or 'cuda'. "
            "If omitted, defaults to CUDA if available, else CPU."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max token length for ChemBERTa query SMILES. Default: 256.",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Dataset existence / download
# -------------------------------------------------------------------------
def _ensure_dataset(compound_dir: Path, rev_dir: Path) -> None:
    """
    If compound_dir or rev_dir is missing, download the dataset into the common parent
    directory using get_dataset(local_dir=...).

    This matches your get_dataset() implementation, which downloads a full dataset snapshot
    rooted at local_dir (typically 'databases').
    """
    if compound_dir.exists() and rev_dir.exists():
        return

    common_parent = Path(os.path.commonpath([str(compound_dir), str(rev_dir)]))
    common_parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Missing dataset folders. Downloading dataset into: {common_parent}")
    get_dataset(local_dir=str(common_parent))

    if not compound_dir.exists():
        raise FileNotFoundError(
            f"Compound dir still missing after download: {compound_dir} "
            f"(expected ligands.parquet and reps/ inside)."
        )
    if not rev_dir.exists():
        raise FileNotFoundError(
            f"ReverseLigQ dir still missing after download: {rev_dir} "
            f"(expected ligand_lists.pkl etc. inside)."
        )


# -------------------------------------------------------------------------
# Core pipeline primitives (build once; run many queries)
# -------------------------------------------------------------------------
def build_searcher(
    organism: str,
    compound_dir: Path,
    rev_dir: Path,
    search_type: str,
    min_score: Optional[float],
    k_max_ligands: int,
    chunk_size: int,
    chemberta_model: str,
    device: Optional[str],
    max_length: int,
):
    """
    Construct and return the appropriate searcher (MorganTanimotoSearcher or ChembertaSearcher)
    using your .from_defaults() APIs. The searcher caches representation + indices and can be
    reused across multiple queries.
    """
    if min_score is None:
        min_score = 0.4 if search_type == "morgan_fp_tanimoto" else 0.8

    ligand_lists_path = rev_dir / "ligand_lists.pkl"

    if search_type == "morgan_fp_tanimoto":
        return MorganTanimotoSearcher.from_defaults(
            organism=str(organism),
            store_root=compound_dir,
            ligand_lists_path=ligand_lists_path,
            rep_name="morgan_1024_r2",
            min_score=float(min_score),
            k_max=int(k_max_ligands),
            chunk_size=int(chunk_size),
        )

    if search_type == "chemberta":
        return ChembertaSearcher.from_defaults(
            organism=str(organism),
            store_root=compound_dir,
            ligand_lists_path=ligand_lists_path,
            rep_name="chemberta_zinc_base_768",
            model_name=chemberta_model,
            device=device,
            max_length=int(max_length),
            min_score=float(min_score),
            k_max=int(k_max_ligands),
            chunk_size=int(chunk_size),
        )

    raise ValueError(f"Unsupported search_type: {search_type!r}")


def run_one_query(
    searcher,
    query_smiles: str,
    organism: str,
    rev_dir: Path,
    proteins_base_dir: Path,
    max_domain_ranks: Optional[int],
    include_only_curated: bool,
    only_proteins_with_description: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full pipeline for a single query:
      1) ligand similarity search (threshold + k_max already handled in searcher)
      2) attach domains (curated/possible) using ReverseLigQ mapping
      3) build candidate proteins table using either:
           - base ReverseLigQ organism data (default), or
           - uploaded organism data (when --uploaded-organism is used)
      4) build ligand summary table
    """
    # 1) Ligand similarity search
    ligand_results = searcher.search(query_smiles)

    # 2) Attach domains using the ReverseLigQ base directory (rev_dir)
    annotated = attach_domains_to_ligands(
        ligand_results=ligand_results,
        base_dir=rev_dir,
    )

    # 3) Candidate proteins table
    #    - base_dir: either rev_dir (built-in organisms)
    #                or <data_root>/local_organism_data/<organism> (uploaded organisms)
    target_search_table = build_candidate_proteins_table(
        annotated_ligands=annotated,
        base_dir=proteins_base_dir,
        organism=organism,
        max_domain_ranks=max_domain_ranks,  # supports None
        include_only_curated=include_only_curated,
        show_only_proteins_with_description=only_proteins_with_description,
    )

    target_df = pd.DataFrame(target_search_table)
    if not target_df.empty:
        target_df = (
            target_df.sort_values(by=["rank", "domain_id"], ascending=[True, True])
            .reset_index(drop=True)
        )

    # 4) Ligand summary table
    lig_df = build_ligand_summary_dataframe(annotated)
    return target_df, lig_df


# -------------------------------------------------------------------------
# Batch input
# -------------------------------------------------------------------------
def load_batch_csv(path: str | Path) -> pd.DataFrame:
    """
    Read batch CSV with required columns: lig_id, smiles.

    Returns a DataFrame with both columns as string dtype, dropping rows where smiles is empty.
    """
    path = Path(path)
    df = pd.read_csv(path)

    required = {"lig_id", "smiles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Batch CSV is missing required columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )

    df = df.copy()
    df["lig_id"] = df["lig_id"].astype(str)
    df["smiles"] = df["smiles"].astype(str)

    # Drop empties / NaNs
    df["smiles"] = df["smiles"].fillna("").astype(str).str.strip()
    df = df[df["smiles"] != ""].reset_index(drop=True)

    if df.empty:
        raise ValueError("Batch CSV has no valid rows after filtering empty SMILES.")
    return df


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    compound_dir = Path(args.compound_dir)
    rev_dir = Path(args.rev_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_dataset(compound_dir=compound_dir, rev_dir=rev_dir)

    # Determine data_root (common parent of compound_dir and rev_dir)
    data_root = Path(os.path.commonpath([str(compound_dir), str(rev_dir)]))

    # Decide which base_dir to use for protein/domain data:
    #  - built-in organisms  -> rev_dir
    #  - uploaded organisms  -> <data_root>/local_organism_data/<organism>
    if args.uploaded_organism:
        local_org_base_dir = data_root / "local_organism_data"
        proteins_base_dir = local_org_base_dir / args.organism
        if not proteins_base_dir.exists():
            raise FileNotFoundError(
                f"Uploaded organism directory not found: {proteins_base_dir}. "
                "Make sure you ran the upload_proteome.py pipeline with the same "
                "--org-name, and that local_organism_data is located under the same "
                "data root as --compound-dir and --rev-dir."
            )
        print(
            f"[INFO] Using uploaded organism data for '{args.organism}' "
            f"from: {proteins_base_dir}"
        )
    else:
        proteins_base_dir = rev_dir

    # Build searcher once
    if args.uploaded_organism:
        rev_dir_for_searcher = proteins_base_dir
    else:
        rev_dir_for_searcher = rev_dir

    searcher = build_searcher(
        organism=args.organism,
        compound_dir=compound_dir,
        rev_dir=rev_dir_for_searcher,
        search_type=args.search_type,
        min_score=args.min_score,
        k_max_ligands=args.k_max_ligands,
        chunk_size=args.chunk_size,
        chemberta_model=args.chemberta_model,
        device=args.device,
        max_length=args.max_length,
    )

    if args.query_smiles is not None:
        # Single query mode
        target_df, lig_df = run_one_query(
            searcher=searcher,
            query_smiles=args.query_smiles,
            organism=args.organism,
            rev_dir=rev_dir,
            proteins_base_dir=proteins_base_dir,
            max_domain_ranks=args.max_domain_ranks,
            include_only_curated=args.include_only_curated,
            only_proteins_with_description=args.only_proteins_with_description,
        )

        target_csv = out_dir / "predicted_targets.csv"
        lig_csv = out_dir / "similarity_search_results.csv"
        target_df.to_csv(target_csv, index=False)
        lig_df.to_csv(lig_csv, index=False)

        print(f"[INFO] Wrote: {target_csv}")
        print(f"[INFO] Wrote: {lig_csv}")
        return

    # Batch mode
    batch_df = load_batch_csv(args.query_csv)
    n = len(batch_df)
    print(f"[INFO] Batch mode: {n} queries loaded from {args.query_csv}")

    # Optional: write a small run log
    log_rows: List[Dict[str, Any]] = []

    for i, row in batch_df.iterrows():
        lig_id = row["lig_id"]
        smi = row["smiles"]
        safe_id = _safe_dirname(lig_id, fallback=f"query_{i+1}")
        q_dir = out_dir / safe_id
        q_dir.mkdir(parents=True, exist_ok=True)

        try:
            target_df, lig_df = run_one_query(
                searcher=searcher,
                query_smiles=smi,
                organism=args.organism,
                rev_dir=rev_dir,
                proteins_base_dir=proteins_base_dir,
                max_domain_ranks=args.max_domain_ranks,
                include_only_curated=args.include_only_curated,
                only_proteins_with_description=args.only_proteins_with_description,
            )

            target_csv = q_dir / "predicted_targets.csv"
            lig_csv = q_dir / "similarity_search_results.csv"
            target_df.to_csv(target_csv, index=False)
            lig_df.to_csv(lig_csv, index=False)

            print(f"[{i+1}/{n}] lig_id={lig_id!r} -> OK ({safe_id})")
            log_rows.append(
                {
                    "lig_id": lig_id,
                    "smiles": smi,
                    "out_dir": str(q_dir),
                    "status": "ok",
                    "n_ligand_hits": int(len(lig_df)),
                    "n_target_rows": int(len(target_df)),
                }
            )

        except Exception as e:
            # Keep going
            err_path = q_dir / "error.txt"
            err_path.write_text(str(e), encoding="utf-8")
            print(f"[{i+1}/{n}] lig_id={lig_id!r} -> ERROR ({safe_id}): {e}")
            log_rows.append(
                {
                    "lig_id": lig_id,
                    "smiles": smi,
                    "out_dir": str(q_dir),
                    "status": "error",
                    "error": str(e),
                }
            )

    # Write batch log
    log_df = pd.DataFrame(log_rows)
    log_csv = out_dir / "batch_run_log.csv"
    log_df.to_csv(log_csv, index=False)
    print(f"[INFO] Batch log written: {log_csv}")


if __name__ == "__main__":
    main()
