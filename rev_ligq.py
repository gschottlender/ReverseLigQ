#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from search_tools.target_finding import (
    get_dataset,
    ChembertaSearcher,
    MorganTanimotoSearcher,
    attach_domains_to_ligands,
    build_candidate_proteins_table,
    build_ligand_summary_dataframe,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the target search pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run ligand similarity search and target (protein) prediction "
                    "using ReverseLigQ datasets."
    )

    parser.add_argument(
        "--query-smiles",
        type=str,
        required=True,
        help="Query molecule SMILES string.",
    )

    parser.add_argument(
        "--organism",
        type=str,
        required=True,
        help=""""Select organism number: 1- Bartonella bacilliformis, 2- Klebsiella pneumoniae, 3- Mycobacterium tuberculosis, 
        4- Trypanosoma cruzi, 5- Staphylococcus aureus RF122, 6- Streptococcus uberis 0140J, 7- Enterococcus faecium, 
        8- Escherichia coli MG1655, 9- Streptococcus agalactiae NEM316, 10- Pseudomonas syringae, 
        11- DENV (dengue virus), 12- SARS-CoV-2, 13- Homo sapiens""",
    )

    parser.add_argument(
        "--local-dir",
        type=str,
        default="data",
        help="Local directory where the ReverseLigQ dataset is stored or will be downloaded.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory where result tables will be written.",
    )

    parser.add_argument(
        "--search-type",
        type=str,
        default="morgan_fp_tanimoto",
        choices=["morgan_fp_tanimoto", "chemberta"],
        help="Type of similarity search to perform: "
             "'morgan_fp_tanimoto' (fingerprints + Tanimoto) or 'chemberta' (ChemBERTa embeddings).",
    )

    parser.add_argument(
        "--top-k-ligands",
        type=int,
        default=1000,
        help="Maximum number of ligand neighbors to retrieve in the similarity search.",
    )

    parser.add_argument(
        "--max-domain-ranks",
        type=int,
        default=10,
        help="Maximum number of domain ranks to keep in the final target (protein) table. "
             "Domains sharing the same reference score share the same rank.",
    )

    parser.add_argument(
        "--include-only-curated",
        action="store_true",
        help="If set, only domains reached via 'curated' evidence will be considered.",
    )

    parser.add_argument(
        "--only-proteins-with-description",
        action="store_true",
        help="If set, only proteins that have a description will be included in the final table.",
    )

    return parser.parse_args()


def target_search(
    query_smiles: str,
    organism: str,
    base_dir: str | Path,
    top_k_ligands: int = 1000,
    max_domain_ranks: int = 10,
    include_only_curated: bool = False,
    only_proteins_with_description: bool = False,
    search_type: str = "morgan_fp_tanimoto",
):
    """
    Run the full target search pipeline for a single query SMILES and organism.

    Steps:
      1. Build the appropriate searcher (Morgan+Tanimoto or ChemBERTa).
      2. Perform similarity search in the ligand space (top-k ligands).
      3. Attach curated/possible domains to each ligand.
      4. Build the final candidate proteins table (domain- and protein-level).
      5. Build the ligand-level similarity summary table.

    Parameters
    ----------
    query_smiles : str
        Query molecule in SMILES format.
    organism : str
        Organism identifier.
    base_dir : str or Path
        Directory containing ReverseLigQ data files
        (comps_fps.npy, comps_embs.npy, ligand_lists.pkl, etc.).
    top_k_ligands : int
        Number of nearest ligands to retrieve in the similarity search.
    max_domain_ranks : int
        Maximum number of domain ranks to keep in the final protein table.
    include_only_curated : bool
        If True, only domains with 'curated' evidence are considered.
    only_proteins_with_description : bool
        If True, only proteins with a description are included.
    search_type : str
        'morgan_fp_tanimoto' or 'chemberta'.

    Returns
    -------
    target_search_df : pandas.DataFrame
        Protein-level candidate targets table.
    similarity_search_df : pandas.DataFrame
        Ligand-level similarity search summary table.
    """
    base_dir = Path(base_dir)

    # Select searcher type
    if search_type == "morgan_fp_tanimoto":
        searcher = MorganTanimotoSearcher.from_paths(
            base_dir=base_dir,
            organism=organism,
        )
    elif search_type == "chemberta":
        searcher = ChembertaSearcher.from_paths(
            base_dir=base_dir,
            organism=organism,
            model_name="seyonec/ChemBERTa-zinc-base-v1",
            device="cpu",
        )
    else:
        raise ValueError(f"Unsupported search_type: {search_type!r}")

    # 1) Ligand similarity search
    ligand_results = searcher.search(query_smiles, top_k=top_k_ligands)

    # 2) Attach curated/possible domains to each ligand
    annotated = attach_domains_to_ligands(
        ligand_results=ligand_results,
        base_dir=base_dir,
    )

    # 3) Build protein-level candidate targets table
    target_search_table = build_candidate_proteins_table(
        annotated_ligands=annotated,
        base_dir=base_dir,
        organism=organism,
        max_domain_ranks=max_domain_ranks,
        include_only_curated=include_only_curated,
        show_only_proteins_with_description=only_proteins_with_description,
    )
    target_search_df = pd.DataFrame(target_search_table)
    target_search_df = target_search_df.sort_values(
                        by=["rank", "domain_id"],
                        ascending=[True, True]
                        ).reset_index(drop=True)
    # 4) Build ligand-level similarity summary table
    similarity_search_df = build_ligand_summary_dataframe(annotated)

    return target_search_df, similarity_search_df


def main() -> None:
    """
    Entry point for the command-line interface.

    - Ensures the dataset is available (downloads if missing).
    - Runs the target search pipeline.
    - Writes CSV files with predicted targets and similarity search results.
    """
    args = parse_args()

    local_dir = Path(args.local_dir)
    out_dir = Path(args.out_dir)

    # Ensure dataset is available: if local_dir does not exist, download it
    if not local_dir.exists():
        local_dir.mkdir(parents=True, exist_ok=True)
        # Download ReverseLigQ dataset into local_dir
        get_dataset(local_dir=str(local_dir))

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run the target search pipeline
    target_search_df, similarity_search_df = target_search(
        query_smiles=args.query_smiles,
        organism=args.organism,
        base_dir=local_dir,
        top_k_ligands=args.top_k_ligands,
        max_domain_ranks=args.max_domain_ranks,
        include_only_curated=args.include_only_curated,
        only_proteins_with_description=args.only_proteins_with_description,
        search_type=args.search_type,
    )

    # Save outputs as CSV
    target_csv_path = out_dir / "predicted_targets.csv"
    similarity_csv_path = out_dir / "similarity_search_results.csv"

    target_search_df.to_csv(target_csv_path, index=False)
    similarity_search_df.to_csv(similarity_csv_path, index=False)

    print(f"[INFO] Target search results written to: {target_csv_path}")
    print(f"[INFO] Similarity search results written to: {similarity_csv_path}")


if __name__ == "__main__":
    main()
