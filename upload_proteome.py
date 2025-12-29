#!/usr/bin/env python
"""
CLI tool to upload a new proteome and integrate it into the
molecular target prediction platform (ReverseLigQ ecosystem).

This script is a thin command-line front-end over
`update.add_organisms.prepare_local_organism_data`.

It will:
  - Ensure the ReverseLigQ dataset is available locally.
  - Scan the provided proteome FASTA against Pfam (via hmmscan).
  - Build domain → proteins and ligands mappings.
  - Store per-organism data as pickle files under data_dir/local_organism_data/{org_name}.

Example
-------
python upload_proteome.py \
    --org-name siniae \
    --fasta-path ./uniprotkb_taxonomy_id_1346_2025_12_29.fasta \
    --cpu 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from update.add_organisms import prepare_local_organism_data


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Upload a new organism proteome and integrate it into the "
            "molecular target prediction platform (ReverseLigQ)."
        )
    )

    parser.add_argument(
        "--org-name",
        required=True,
        help="Organism key/name (e.g. 'siniae').",
    )

    parser.add_argument(
        "--fasta-path",
        required=True,
        help="Path to the FASTA file containing the organism proteome.",
    )

    parser.add_argument(
        "--data-dir",
        default="databases",
        help=(
            "Base directory where the ReverseLigQ dataset is stored "
            "(merged_databases, compound_data, etc.). "
            "Default: 'databases'."
        ),
    )

    parser.add_argument(
        "--temp-base-dir",
        default="temp_data/new_proteomes",
        help=(
            "Base directory for temporary hmmscan outputs. "
            "Default: 'temp_data/new_proteomes'."
        ),
    )

    parser.add_argument(
        "--cpu",
        type=int,
        default=4,
        help="Number of CPUs to use in hmmscan. Default: 4.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the CLI.

    It parses command-line arguments, calls prepare_local_organism_data,
    and prints the paths of the generated pickle files.
    """
    args = parse_args()

    org_name = args.org_name
    fasta_path = Path(args.fasta_path)
    data_dir = Path(args.data_dir)
    temp_base_dir = Path(args.temp_base_dir)
    cpu = args.cpu

    print(
        f"[INFO] Uploading proteome for organism '{org_name}' "
        f"from FASTA: {fasta_path}"
    )
    print(f"[INFO] Using data_dir = {data_dir}")
    print(f"[INFO] Using temp_base_dir = {temp_base_dir}")
    print(f"[INFO] Using cpu = {cpu}")

    paths = prepare_local_organism_data(
        org_name=org_name,
        fasta_path=fasta_path,
        data_dir=data_dir,
        temp_base_dir=temp_base_dir,
        cpu=cpu,
    )

    print("[INFO] Done. Generated pickle files:")
    for key, path in paths.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
