#!/usr/bin/env python

import os
import json
import argparse
from datetime import date
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from update.pdb_db import (
    generate_pdb_database,
    update_pdb_database_from_dir,
)
from update.chembl_db import generate_chembl_database
from update.merge_pdb_chembl import merge_databases
from update.rev_ligq_db import generate_rev_ligq_databases

# HuggingFace dataset repo with the preprocessed databases and initial metadata
HF_DATASET_REPO_ID = "gschottlender/LigQ_2"


# ReverseLigQ dataset snapshot (contains compound_data/pdb_chembl and rev_ligq/)
REVERSE_LIGQ_DATASET_REPO_ID = "gschottlender/reverse_ligq"


def get_reverse_ligq_dataset(local_dir: str = "databases") -> None:
    """
    Download the ReverseLigQ dataset snapshot from Hugging Face into `local_dir`.

    This mirrors the behavior used by the search script: the dataset root will be
    created under `local_dir`, including:
      - compound_data/pdb_chembl/...
      - rev_ligq/...

    Parameters
    ----------
    local_dir : str
        Local directory where the dataset will be stored.
    """
    snapshot_download(
        repo_id=REVERSE_LIGQ_DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )


def _ensure_reverse_ligq_dataset(compound_dir: Path, rev_dir: Path) -> None:
    """
    If `compound_dir` or `rev_dir` is missing, download the ReverseLigQ snapshot
    into their common parent directory.

    This is the same pattern used in the search script: one snapshot download,
    then verify the expected folders exist.
    """
    if compound_dir.exists() and rev_dir.exists():
        return

    # Pick a shared parent so the snapshot lands in the right place.
    common_parent = Path(os.path.commonpath([str(compound_dir), str(rev_dir)]))
    common_parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Missing dataset folders. Downloading ReverseLigQ into: {common_parent}")
    get_reverse_ligq_dataset(local_dir=str(common_parent))

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


# ----------------------------------------------------------------------
# Command-line arguments
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full pipeline: PDB + ChEMBL → merged DB → UniProt sequences."
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="databases",
        help="Root directory where processed databases will be stored (PDB, ChEMBL, merged, rev_ligq).",
    )

    # ReverseLigQ dataset roots (downloaded from HF if missing)
    parser.add_argument(
        "--compound-dir",
        type=str,
        default="databases/compound_data/pdb_chembl",
        help=(
            "Directory containing compound index + representations (LigandStore root): "
            "ligands.parquet and reps/<rep_name>.dat + reps/<rep_name>.meta.json. "
            "If missing, the script will download the ReverseLigQ dataset snapshot. "
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
            "If missing, the script will download the ReverseLigQ dataset snapshot. "
            "Default: databases/rev_ligq"
        ),
    )


    parser.add_argument(
        "--temp-data-dir",
        default="temp_data",
        help="Directory for temporary files (e.g. ChEMBL SQLite tarball and extraction).",
    )

    parser.add_argument(
        "--chembl-version",
        type=int,
        default=36,
        help="ChEMBL version to use when regenerating the local database if needed.",
    )

    parser.add_argument(
        "--tanimoto-curation-threshold",
        type=float,
        default=0.35,
        help="Tanimoto threshold for curating 'possible' ChEMBL ligands when merging PDB–ChEMBL.",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def snapshot_dir(root_dir: str) -> dict:
    """
    Build a lightweight snapshot of a directory: for each file, store
    its relative path, size, and modification time.

    Used to detect whether an update step actually changed the PDB
    processed directory.
    """
    snapshot = {}
    if not os.path.isdir(root_dir):
        return snapshot

    for base, _, files in os.walk(root_dir):
        for fname in files:
            fpath = os.path.join(base, fname)
            rel = os.path.relpath(fpath, root_dir)
            try:
                stat = os.stat(fpath)
            except OSError:
                continue
            snapshot[rel] = (stat.st_size, int(stat.st_mtime))
    return snapshot


# ----------------------------------------------------------------------
# Download / sync base databases from HuggingFace
# ----------------------------------------------------------------------
def download_base_databases_from_huggingface(output_dir: str) -> dict:
    """
    Sync ONLY the essential preprocessed database files from the HuggingFace
    dataset repo into `output_dir`:

      - db_metadata.json
      - preprocessed PDB database under output_dir/pdb
      - preprocessed ChEMBL database under output_dir/chembl

    Each item is downloaded only if missing locally:
      - If output_dir/pdb does not exist or is empty → download all 'pdb/' files.
      - If output_dir/chembl does not exist or is empty → download all 'chembl/' files.
      - If db_metadata.json does not exist → download it.

    Returns a dict with booleans indicating whether each component was
    downloaded in this run:
      {
        "pdb_downloaded": bool,
        "chembl_downloaded": bool,
        "metadata_downloaded": bool,
      }
    """
    api = HfApi()
    os.makedirs(output_dir, exist_ok=True)

    pdb_dir = os.path.join(output_dir, "pdb")
    chembl_dir = os.path.join(output_dir, "chembl")
    metadata_path = os.path.join(output_dir, "db_metadata.json")

    # Determine which components are missing locally
    def is_missing_dir_or_empty(d: str) -> bool:
        return (not os.path.isdir(d)) or (len(os.listdir(d)) == 0)

    need_pdb = is_missing_dir_or_empty(pdb_dir)
    need_chembl = is_missing_dir_or_empty(chembl_dir)
    need_metadata = not os.path.exists(metadata_path)

    pdb_downloaded = False
    chembl_downloaded = False
    metadata_downloaded = False

    if not (need_pdb or need_chembl or need_metadata):
        print(
            "[INFO] Local PDB, ChEMBL and metadata already present. "
            "Skipping HuggingFace download."
        )
        return {
            "pdb_downloaded": False,
            "chembl_downloaded": False,
            "metadata_downloaded": False,
        }

    print(f"[INFO] Syncing required components from HF dataset: {HF_DATASET_REPO_ID}")
    files = api.list_repo_files(repo_id=HF_DATASET_REPO_ID, repo_type="dataset")

    # Ensure subdirectories exist if we are going to populate them
    if need_pdb:
        os.makedirs(pdb_dir, exist_ok=True)
    if need_chembl:
        os.makedirs(chembl_dir, exist_ok=True)

    for filename in files:
        # PDB subdirectory
        if need_pdb and filename.startswith("pdb/"):
            hf_hub_download(
                repo_id=HF_DATASET_REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
            pdb_downloaded = True

        # ChEMBL subdirectory
        elif need_chembl and filename.startswith("chembl/"):
            hf_hub_download(
                repo_id=HF_DATASET_REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
            chembl_downloaded = True

        # Metadata file
        elif need_metadata and filename == "db_metadata.json":
            hf_hub_download(
                repo_id=HF_DATASET_REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
            metadata_downloaded = True

    print(
        "[INFO] HuggingFace sync completed "
        f"(pdb_downloaded={pdb_downloaded}, "
        f"chembl_downloaded={chembl_downloaded}, "
        f"metadata_downloaded={metadata_downloaded})."
    )

    return {
        "pdb_downloaded": pdb_downloaded,
        "chembl_downloaded": chembl_downloaded,
        "metadata_downloaded": metadata_downloaded,
    }


def main():

    args = parse_args()


    # Ensure ReverseLigQ search dataset folders exist (compound representations + rev_ligq metadata)
    compound_dir = Path(args.compound_dir)
    rev_dir = Path(args.rev_dir)
    _ensure_reverse_ligq_dataset(compound_dir=compound_dir, rev_dir=rev_dir)

    output_dir = args.output_dir
    temp_data_dir = args.temp_data_dir
    chembl_version = args.chembl_version

    # Ensure root directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Sync required base data from HuggingFace (pdb, chembl, metadata)
    # ------------------------------------------------------------------
    sync_info = download_base_databases_from_huggingface(output_dir)
    pdb_downloaded = sync_info["pdb_downloaded"]
    chembl_downloaded = sync_info["chembl_downloaded"]
    # metadata_downloaded = sync_info["metadata_downloaded"]  # not strictly needed

    # ------------------------------------------------------------------
    # 2) Load metadata (which should now exist, or start from empty)
    # ------------------------------------------------------------------
    metadata_path = os.path.join(output_dir, "db_metadata.json")
    metadata = {}

    if os.path.exists(metadata_path):
        print(f"[INFO] Loading metadata from {metadata_path}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        print(
            "[WARN] No db_metadata.json found even after HF sync. "
            "Starting with empty metadata."
        )

    # ------------------------------------------------------------------
    # 3) Update / generate PDB database
    # ------------------------------------------------------------------
    pdb_db_dir = os.path.join(output_dir, "pdb")
    os.makedirs(pdb_db_dir, exist_ok=True)

    print("[INFO] Updating PDB database from local PDB directory...")
    pdb_was_updated = update_pdb_database_from_dir(
        pdb_db_dir,
        temp_dir=temp_data_dir,
    )

    # PDB is considered updated if:
    #   - We processed new PDB IDs in this run, OR
    #   - The PDB data was freshly downloaded from Hugging Face.
    pdb_updated = pdb_downloaded or pdb_was_updated

    if pdb_updated:
        metadata["pdb_last_update"] = date.today().isoformat()
    else:
        print("[INFO] No changes detected in PDB database (no new PDB IDs).")

    # ------------------------------------------------------------------
    # 4) Update / generate ChEMBL database
    # ------------------------------------------------------------------
    chembl_sql_dir = os.path.join(temp_data_dir, "chembl_sql")
    os.makedirs(chembl_sql_dir, exist_ok=True)

    current_chembl_in_metadata = metadata.get("chembl_version")

    # We only regenerate the ChEMBL database if the target version
    # differs from the one stored in metadata.
    need_chembl_update = (
        current_chembl_in_metadata is None
        or current_chembl_in_metadata != chembl_version
    )

    chembl_output_dir = os.path.join(output_dir, "chembl")
    os.makedirs(chembl_output_dir, exist_ok=True)

    chembl_updated = False

    if need_chembl_update:
        print(
            f"[INFO] Regenerating ChEMBL database. "
            f"Metadata version: {current_chembl_in_metadata}, "
            f"target version: {chembl_version}"
        )

        chembl_file = os.path.join(
            chembl_sql_dir, f"chembl_{chembl_version}_sqlite.tar.gz"
        )

        # Remove old tar.gz (e.g. truncated downloads from previous runs)
        if os.path.exists(chembl_file):
            print(f"[INFO] Removing previous tarball: {chembl_file}")
            os.remove(chembl_file)

        # Download ChEMBL SQLite for the requested version
        url = (
            "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
            f"chembl_{chembl_version}/chembl_{chembl_version}_sqlite.tar.gz"
        )

        print(f"[INFO] Downloading ChEMBL SQLite from: {url}")
        ret = os.system(f"wget -q -P {chembl_sql_dir} {url}")
        if ret != 0:
            raise RuntimeError(
                f"wget failed when downloading ChEMBL SQLite (exit code {ret}). "
                f"Check your network connection and the URL: {url}"
            )

        print("[INFO] Extracting ChEMBL SQLite tarball...")
        ret = os.system(f"tar -xvzf {chembl_file} -C {chembl_sql_dir}")
        if ret != 0:
            raise RuntimeError(
                f"tar extraction failed for {chembl_file} (exit code {ret})."
            )

        # Locate the .db file inside the extracted directory tree
        db_filename = f"chembl_{chembl_version}.db"
        chembl_db_path = None
        for root, _, files in os.walk(chembl_sql_dir):
            if db_filename in files:
                chembl_db_path = os.path.join(root, db_filename)
                break

        if chembl_db_path is None:
            raise FileNotFoundError(
                f"Could not find {db_filename} inside {chembl_sql_dir} "
                "after extraction."
            )

        print(f"[INFO] Generating local ChEMBL database from: {chembl_db_path}")

        generate_chembl_database(
            chembl_db_path=chembl_db_path,
            output_dir=chembl_output_dir,
        )

        metadata["chembl_version"] = chembl_version
        chembl_updated = True
    else:
        print(
            f"[INFO] ChEMBL database already at version {chembl_version}, "
            "no regeneration needed."
        )

    # If the ChEMBL directory was freshly downloaded from HF, that also
    # counts as an update that should trigger re-merging.
    chembl_updated = chembl_updated or chembl_downloaded

    # ------------------------------------------------------------------
    # 5) Merge PDB + ChEMBL (only if at least one changed)
    # ------------------------------------------------------------------
    if not (pdb_updated or chembl_updated):
        print(
            "[INFO] No PDB or ChEMBL updates detected. "
        )
        print(
            "[INFO] No new updates. Exiting "
        )

        return

    # Assumption: merge_databases() expects a directory that contains
    # subfolders "pdb" and "chembl" with the processed data.
    # This step also generates the vector database of compounds from PDB and ChEMBL.
    data_dir = output_dir

    print("[INFO] Merging PDB and ChEMBL databases...")
    (
        ligs_smiles_merged,
        binding_data_merged,
        uncurated_binding_data,
    ) = merge_databases(
        data_dir,
        tanimoto_curation_threshold=args.tanimoto_curation_threshold,
    )

    merged_dir = os.path.join(output_dir, "merged_databases")
    os.makedirs(merged_dir, exist_ok=True)

    ligs_smiles_merged_path = os.path.join(
        merged_dir, "ligs_smiles_merged.parquet"
    )
    binding_data_merged_path = os.path.join(
        merged_dir, "binding_data_merged.parquet"
    )
    uncurated_binding_data_path = os.path.join(
        merged_dir, "uncurated_binding_data.parquet"
    )

    print(f"[INFO] Saving merged ligands to {ligs_smiles_merged_path}")
    ligs_smiles_merged.to_parquet(ligs_smiles_merged_path, index=False)

    print(f"[INFO] Saving merged binding data to {binding_data_merged_path}")
    binding_data_merged.to_parquet(binding_data_merged_path, index=False)

    print(f"[INFO] Saving uncurated binding data to {uncurated_binding_data_path}")
    uncurated_binding_data.to_parquet(uncurated_binding_data_path, index=False)


    # ------------------------------------------------------------------
    # 7) Generate rev_ligq specific databases
    # ------------------------------------------------------------------

    print("[INFO] Generating ReverseLigQ databases)")
    generate_rev_ligq_databases(output_dir)
    print("Done")
    # ------------------------------------------------------------------
    # 8) Save updated metadata
    # ------------------------------------------------------------------
    print(f"[INFO] Saving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print("[INFO] Full pipeline finished successfully.")


if __name__ == "__main__":
    main()
