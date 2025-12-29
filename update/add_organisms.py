"""
Utility functions to add new organisms to the ReverseLigQ ecosystem.

This module:
  - Ensures the ReverseLigQ dataset snapshot is available locally.
  - Downloads and indexes Pfam-A HMMs (if needed).
  - Parses FASTA headers to extract stable protein IDs and descriptions.
  - Runs hmmscan against Pfam-A and parses domtblout.
  - Builds mappings from Pfam domains to proteins and protein descriptions.
  - Filters ReverseLigQ binding data by ligands present in ligands.parquet.
  - Stores per-organism data (domains, ligands, descriptions) as pickle files.
"""

from __future__ import annotations

import gzip
import pickle
import shutil
import subprocess
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from update_rev_ligq import get_reverse_ligq_dataset


# ----------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------

PFAM_A_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
)


# ----------------------------------------------------------------------
# Generic utilities
# ----------------------------------------------------------------------


def check_dependency(name: str) -> None:
    """
    Ensure that a required executable is available in the current PATH.

    Parameters
    ----------
    name : str
        Name of the executable to check (e.g. 'hmmscan', 'hmmpress').

    Raises
    ------
    RuntimeError
        If the executable is not found in PATH.
    """
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required program '{name}' not found in PATH. "
            f"Install it in your conda environment (e.g. "
            f"'conda install -c bioconda {name}') and try again."
        )


def run_command(cmd: List[str], cwd: Path | None = None) -> None:
    """
    Run an external command and raise a RuntimeError if it fails.

    Parameters
    ----------
    cmd : list of str
        Command and arguments to execute.
    cwd : Path, optional
        Working directory in which to run the command. If None, uses
        the current working directory.

    Raises
    ------
    RuntimeError
        If the command returns a non-zero exit code.
    """
    print(f"[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Print stdout/stderr to help debugging before raising
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory (and parents) if it does not exist and return it.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        The directory path as a Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ----------------------------------------------------------------------
# Block 0 – Complementary databases: Pfam
# ----------------------------------------------------------------------


def download_and_prepare_pfam_a(pfam_dir: Path, url: str = PFAM_A_URL) -> Path:
    """
    Download Pfam-A.hmm.gz (if needed), decompress it, and run hmmpress.

    This function ensures that Pfam-A.hmm and its indexed files are
    present in `pfam_dir`. If the compressed or uncompressed HMM file
    is already there, it skips the download/decompression step. If the
    hmmpress index files already exist, it skips re-indexing.

    Parameters
    ----------
    pfam_dir : Path
        Directory where Pfam HMM files will be stored.
    url : str
        URL for the Pfam-A HMM archive (gzipped).

    Returns
    -------
    Path
        Path to the uncompressed Pfam-A.hmm file.
    """
    pfam_dir.mkdir(parents=True, exist_ok=True)

    gz_path = pfam_dir / "Pfam-A.hmm.gz"
    hmm_path = pfam_dir / "Pfam-A.hmm"

    # 1) Download Pfam-A.hmm.gz if both compressed and uncompressed files are missing
    if not gz_path.exists() and not hmm_path.exists():
        print(f"[INFO] Downloading Pfam-A from {url}")
        urllib.request.urlretrieve(url, gz_path)
        print(f"[INFO] Downloaded to {gz_path}")
    else:
        print(
            "[INFO] Pfam-A compressed or uncompressed file already present, "
            "skipping download."
        )

    # 2) Decompress Pfam-A.hmm.gz if the uncompressed file is missing
    if not hmm_path.exists():
        print(f"[INFO] Decompressing {gz_path} -> {hmm_path}")
        with gzip.open(gz_path, "rb") as f_in, open(hmm_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print("[INFO] Decompression finished.")
    else:
        print("[INFO] Pfam-A.hmm already decompressed.")

    # 3) Run hmmpress to create index files if they are missing
    h3_files = list(pfam_dir.glob("Pfam-A.hmm.h3*"))
    if h3_files:
        print(
            "[INFO] Pfam-A HMM already indexed (h3 files found), "
            "skipping hmmpress."
        )
    else:
        check_dependency("hmmpress")
        print("[INFO] Indexing Pfam-A with hmmpress...")
        run_command(["hmmpress", hmm_path.name], cwd=pfam_dir)
        print("[INFO] hmmpress finished.")

    return hmm_path


def ensure_reverse_ligq_dataset(local_dir: str = "databases") -> Path:
    """
    Ensure that the ReverseLigQ dataset snapshot is available locally.

    If `local_dir` does not exist, it will be populated by downloading the
    dataset from Hugging Face using `get_reverse_ligq_dataset`. If `local_dir`
    already exists, no download is performed.

    Parameters
    ----------
    local_dir : str
        Local directory where the dataset should be stored.

    Returns
    -------
    Path
        Path to the local dataset root directory.
    """
    local_path = Path(local_dir)

    if local_path.exists():
        print(
            f"[INFO] Dataset directory already exists: {local_path}. "
            "Skipping download."
        )
    else:
        print(
            f"[INFO] Dataset directory not found: {local_path}. "
            "Downloading ReverseLigQ snapshot..."
        )
        get_reverse_ligq_dataset(local_dir=str(local_path))
        print("[INFO] ReverseLigQ dataset download completed.")

    return local_path


# ----------------------------------------------------------------------
# FASTA parsing
# ----------------------------------------------------------------------


def parse_fasta_metadata(fasta_path: str | Path) -> List[Dict[str, str]]:
    """
    Parse a FASTA file and extract metadata for each sequence.

    For each header line (starting with '>'), the function extracts:
      - id_original: header ID up to the first space (what HMMER sees).
      - id_final: cleaned ID using the following rule:
            * Split id_original by '|'.
            * If there are exactly 3 fields, use the second one
              (e.g. sp|Q9Y2Q5|RL21_HUMAN -> Q9Y2Q5).
            * Otherwise, keep id_original as-is.
      - description: the rest of the header after the first space (if any).

    Parameters
    ----------
    fasta_path : str or Path
        Path to the input FASTA file.

    Returns
    -------
    List[Dict[str, str]]
        One dictionary per sequence with keys:
        'id_original', 'id_final', and 'description'.
    """
    fasta_path = Path(fasta_path)
    records: List[Dict[str, str]] = []

    with fasta_path.open("r") as fh:
        for line in fh:
            line = line.rstrip("\n")

            # Skip non-header lines (sequence lines)
            if not line.startswith(">"):
                continue

            # Remove leading '>' and trim trailing/leading whitespace
            header = line[1:].strip()
            if not header:
                # Empty or malformed header; ignore it
                continue

            # Split by the first whitespace into ID block and description text
            parts = header.split(maxsplit=1)
            if len(parts) == 1:
                id_block = parts[0]
                description = ""
            else:
                id_block, description = parts[0], parts[1]

            id_original = id_block

            # Clean ID using the pipe rule
            pipe_parts = id_block.split("|")
            if len(pipe_parts) == 3:
                # UniProt-like: sp|ID|NAME or tr|ID|NAME
                id_final = pipe_parts[1]
            else:
                # If there are fewer or more than 3 fields, keep full ID
                id_final = id_original

            records.append(
                {
                    "id_original": id_original,
                    "id_final": id_final,
                    "description": description,
                }
            )

    return records


# ----------------------------------------------------------------------
# HMMER / Pfam scanning
# ----------------------------------------------------------------------


def run_hmmscan_pfam(
    fasta_path: str | Path,
    pfam_dir: str | Path,
    domtblout_path: str | Path,
    cpu: int = 4,
) -> Path:
    """
    Run hmmscan of the given FASTA against Pfam-A.hmm and write a domtblout.

    Parameters
    ----------
    fasta_path : str or Path
        FASTA file with protein sequences.
    pfam_dir : str or Path
        Directory where Pfam-A.hmm and its indices are (or will be) stored.
    domtblout_path : str or Path
        Path where the hmmscan --domtblout file will be written.
    cpu : int
        Number of CPUs to pass to hmmscan.

    Returns
    -------
    Path
        Path to the generated domtblout file.
    """
    fasta_path = Path(fasta_path)
    pfam_dir = Path(pfam_dir)
    domtblout_path = Path(domtblout_path)

    # Ensure Pfam-A is present and indexed
    pfam_hmm = download_and_prepare_pfam_a(pfam_dir)

    # Ensure hmmscan is available
    check_dependency("hmmscan")

    # Run hmmscan with domtblout output
    cmd = [
        "hmmscan",
        "--cpu",
        str(cpu),
        "--domtblout",
        str(domtblout_path),
        str(pfam_hmm),
        str(fasta_path),
    ]
    run_command(cmd, cwd=None)

    return domtblout_path


def parse_hmmer_domtbl(domtblout_path: str | Path) -> List[Dict[str, str]]:
    """
    Parse an HMMER3 domtblout file produced by hmmscan.

    The function extracts, for each domain hit:
      - domain_acc: Pfam accession without version, e.g. 'PF00067'.
      - domain_acc_full: Pfam accession with version, e.g. 'PF00067.20'.
      - domain_name: Pfam target name.
      - seq_id: query sequence ID (what HMMER sees as query name).

    Parameters
    ----------
    domtblout_path : str or Path
        Path to a domtblout file.

    Returns
    -------
    List[Dict[str, str]]
        One dictionary per domain–sequence hit.
    """
    domtblout_path = Path(domtblout_path)
    hits: List[Dict[str, str]] = []

    with domtblout_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                # Skip empty lines and comments
                continue

            cols = line.split()  # domtblout is whitespace-separated

            # HMMER3 hmmscan domtblout columns (relevant subset):
            #  0: target name    (Pfam name)
            #  1: target acc     (Pfam accession, e.g. PF00067.20)
            #  2: tlen
            #  3: query name     (sequence ID)
            #  4: query acc
            #  5: qlen
            # ...
            # 22: acc
            # 23+: description of target (if present)

            if len(cols) < 6:
                # Malformed line; ignore it
                continue

            domain_name = cols[0]
            domain_acc_full = cols[1]
            seq_id = cols[3]

            # Remove Pfam version suffix (PFxxxxx.xx -> PFxxxxx)
            domain_acc = domain_acc_full.split(".")[0]

            hits.append(
                {
                    "domain_acc": domain_acc,
                    "domain_acc_full": domain_acc_full,
                    "domain_name": domain_name,
                    "seq_id": seq_id,  # should match id_original
                }
            )

    return hits


def build_domain_to_proteins(
    domtblout_path: str | Path,
    fasta_metadata: List[Dict[str, str]],
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Build mappings from Pfam domains to proteins and protein descriptions.

    The function returns:
      1) domain_to_ids_list:
            {domain_acc: [id_final, ...]}
         Mapping from Pfam domain accession (without version) to a list of
         cleaned protein IDs ('id_final'), one list per domain.
      2) prot_descriptions:
            {id_final: description}
         Mapping from cleaned protein ID to its header description.

    Only proteins that appear at least once in the domtblout are included
    in these mappings.

    Parameters
    ----------
    domtblout_path : str or Path
        Path to the hmmscan domtblout file.
    fasta_metadata : List[Dict[str, str]]
        Output of parse_fasta_metadata(), containing id_original, id_final,
        and description for each sequence.

    Returns
    -------
    Tuple[Dict[str, List[str]], Dict[str, str]]
        domain_to_ids_list :
            Mapping from Pfam accession (without version) to a sorted list
            of cleaned protein IDs ('id_final').
        prot_descriptions :
            Mapping from cleaned protein ID ('id_final') to its description
            string, for proteins that have at least one Pfam hit.
    """
    # 1) Map id_original -> (id_final, description) for quick lookup
    id_map: Dict[str, Tuple[str, str]] = {
        row["id_original"]: (row["id_final"], row["description"])
        for row in fasta_metadata
    }

    # 2) Parse hmmscan hits
    hits = parse_hmmer_domtbl(domtblout_path)

    # 3) Build {domain: set(id_final)} and {id_final: description}
    domain_to_ids: Dict[str, Set[str]] = defaultdict(set)
    prot_descriptions: Dict[str, str] = {}

    for h in hits:
        domain = h["domain_acc"]
        seq_id_original = h["seq_id"]

        # Map original ID to (id_final, description);
        # if not found, fall back to original ID with empty description.
        if seq_id_original in id_map:
            id_final, desc = id_map[seq_id_original]
        else:
            id_final, desc = seq_id_original, ""

        # Associate protein (id_final) to domain
        domain_to_ids[domain].add(id_final)

        # Record description for that id_final (keep the first one encountered)
        if id_final not in prot_descriptions:
            prot_descriptions[id_final] = desc

    # 4) Convert sets to sorted lists for stable, clean output
    domain_to_ids_list: Dict[str, List[str]] = {
        dom: sorted(list(ids)) for dom, ids in domain_to_ids.items()
    }

    return domain_to_ids_list, prot_descriptions


def pfam_scan_group_by_domain(
    fasta_path: str | Path,
    pfam_dir: str | Path,
    work_dir: str | Path,
    cpu: int = 4,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    High-level wrapper to scan a proteome against Pfam and group by domain.

    Steps:
      1) Parse FASTA metadata (id_original, id_final, description).
      2) Run hmmscan against Pfam-A.hmm (downloading/indexing if needed).
      3) Parse domtblout.
      4) Build:
           - {domain_acc: [id_final, ...]}
           - {id_final: description}

    Parameters
    ----------
    fasta_path : str or Path
        Input FASTA file with protein sequences.
    pfam_dir : str or Path
        Directory to store Pfam-A files (Pfam-A.hmm and indices).
    work_dir : str or Path
        Directory to store hmmscan domtblout output.
    cpu : int
        Number of CPUs to use in hmmscan.

    Returns
    -------
    Tuple[Dict[str, List[str]], Dict[str, str]]
        domain_to_proteins :
            Mapping from Pfam domain accession (without version)
            to a sorted list of cleaned protein IDs ('id_final').
        prot_descriptions :
            Mapping from cleaned protein ID ('id_final') to description,
            only for proteins that have at least one Pfam hit.
    """
    fasta_path = Path(fasta_path)
    pfam_dir = ensure_dir(pfam_dir)
    work_dir = ensure_dir(work_dir)

    # 1) Parse FASTA metadata
    fasta_meta = parse_fasta_metadata(fasta_path)

    # 2) Run hmmscan
    domtblout_path = work_dir / "pfam_hmmscan.domtblout"
    run_hmmscan_pfam(
        fasta_path=fasta_path,
        pfam_dir=pfam_dir,
        domtblout_path=domtblout_path,
        cpu=cpu,
    )

    # 3) Build mappings: {domain: [id_final]} and {id_final: description}
    domain_to_proteins, prot_descriptions = build_domain_to_proteins(
        domtblout_path=domtblout_path,
        fasta_metadata=fasta_meta,
    )

    return domain_to_proteins, prot_descriptions


# ----------------------------------------------------------------------
# Persistence of per-organism data
# ----------------------------------------------------------------------


def save_local_organism_data(
    org_name: str,
    domain_to_proteins,
    lig_list,
    prot_descriptions,
    base_dir: str | Path = "databases/local_organism_data",
) -> Dict[str, Path]:
    """
    Save per-organism Pfam / ligand data into pickle files.

    This function writes three pickle files under:
        <base_dir>/<org_name>/

      - domain_to_proteins.pkl
      - lig_list.pkl
      - prot_descriptions.pkl

    Parameters
    ----------
    org_name : str
        Name/key of the organism (e.g. 'siniae').
    domain_to_proteins :
        Either a dict {org_name: {domain_acc: [id_final, ...]}}
        or directly {domain_acc: [id_final, ...]}, depending on the caller.
    lig_list :
        Either a dict {org_name: [chem_comp_id, ...]}
        or directly a list of ligand IDs.
    prot_descriptions :
        Either a dict {org_name: {id_final: description}}
        or directly {id_final: description}.
    base_dir : str or Path, optional
        Base directory where organism-specific data will be stored.
        Default is 'databases/local_organism_data'.

    Returns
    -------
    Dict[str, Path]
        A dictionary with the paths of the created pickle files, e.g.:
        {
            "fam_prot_dict": Path(...),
            "ligand_lists": Path(...),
            "prot_descriptions": Path(...),
        }
    """
    base_dir = Path(base_dir)
    org_dir = base_dir / org_name
    org_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    # 1) domain_to_proteins
    dtp_path = org_dir / "fam_prot_dict.pkl"
    with dtp_path.open("wb") as f:
        pickle.dump(domain_to_proteins, f, protocol=pickle.HIGHEST_PROTOCOL)
    paths["domain_to_proteins"] = dtp_path

    # 2) lig_list
    lig_path = org_dir / "ligand_lists.pkl"
    with lig_path.open("wb") as f:
        pickle.dump(lig_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    paths["lig_list"] = lig_path

    # 3) prot_descriptions
    desc_path = org_dir / "prot_descriptions.pkl"
    with desc_path.open("wb") as f:
        pickle.dump(prot_descriptions, f, protocol=pickle.HIGHEST_PROTOCOL)
    paths["prot_descriptions"] = desc_path

    print(f"[INFO] Saved local organism data for '{org_name}' in: {org_dir}")
    return paths


# ----------------------------------------------------------------------
# High-level orchestrator
# ----------------------------------------------------------------------


def prepare_local_organism_data(
    org_name: str,
    fasta_path: str | Path,
    data_dir: str | Path = "databases",
    temp_base_dir: str | Path = "temp_data/new_proteomes",
    cpu: int = 8,
) -> Dict[str, Path]:
    """
    High-level orchestrator to prepare and store per-organism data.

    Steps:
      1) Ensure the ReverseLigQ dataset snapshot is available in `data_dir`.
      2) Run Pfam/HMMER on a given organism FASTA to obtain:
           - domain_to_proteins (Pfam -> [id_final])
           - prot_descriptions (id_final -> description).
      3) Load ReverseLigQ binding data and concatenate:
           - binding_data_merged.parquet
           - uncurated_binding_data.parquet
      4) Filter binding data to keep only ligands present in:
           data_dir/compound_data/pdb_chembl/ligands.parquet
      5) Determine the list of ligands associated with the Pfam domains
         detected by HMMER.
      6) Wrap domain_to_proteins, lig_list, and prot_descriptions under
         the organism key (org_name).
      7) Save everything as pickle files under:
           data_dir/local_organism_data/{org_name}

    Parameters
    ----------
    org_name : str
        Organism key (e.g. 'siniae').
    fasta_path : str or Path
        Path to the FASTA file with the organism proteome.
    data_dir : str or Path, optional
        Base directory where the ReverseLigQ dataset lives
        (merged_databases, compound_data, etc.). Default is 'databases'.
    temp_base_dir : str or Path, optional
        Base directory for temporary hmmscan outputs.
        Default is 'temp_data/new_proteomes'.
    cpu : int, optional
        Number of CPUs to use in hmmscan. Default is 8.

    Returns
    -------
    Dict[str, Path]
        Paths to the pickle files created by save_local_organism_data(), e.g.:
        {
            "domain_to_proteins": Path(...),
            "lig_list": Path(...),
            "prot_descriptions": Path(...),
        }
    """
    data_dir = Path(data_dir)
    fasta_path = Path(fasta_path)
    temp_base_dir = Path(temp_base_dir)

    # 1) Ensure the ReverseLigQ dataset snapshot is present in data_dir
    ensure_reverse_ligq_dataset(local_dir=str(data_dir))

    # 2) Directories dependent on data_dir
    pfam_dir = data_dir / "complementary_databases" / "pfam"
    work_dir = temp_base_dir / org_name
    work_dir.mkdir(parents=True, exist_ok=True)

    # 3) Pfam / HMMER: obtain domain_to_proteins (by Pfam) and prot_descriptions
    domain_to_proteins, prot_descriptions = pfam_scan_group_by_domain(
        fasta_path=fasta_path,
        pfam_dir=pfam_dir,
        work_dir=work_dir,
        cpu=cpu,
    )

    # 4) Load binding data (curated + uncurated) from data_dir/merged_databases
    merged_db_dir = data_dir / "merged_databases"
    binding_curated_path = merged_db_dir / "binding_data_merged.parquet"
    binding_uncurated_path = merged_db_dir / "uncurated_binding_data.parquet"

    merged_db = pd.concat(
        [
            pd.read_parquet(binding_curated_path),
            pd.read_parquet(binding_uncurated_path),
        ],
        axis=0,
        ignore_index=True,
    )

    # 5) Load real ligands from data_dir/compound_data/pdb_chembl/ligands.parquet
    ligands_path = data_dir / "compound_data" / "pdb_chembl" / "ligands.parquet"
    true_ligands_db = pd.read_parquet(ligands_path)

    # Filter binding data to keep only ligands present in ligands.parquet
    merged_db = merged_db[
        merged_db["chem_comp_id"].isin(true_ligands_db["chem_comp_id"])
    ]

    # 6) Get the set of Pfam IDs detected by HMMER (keys of domain_to_proteins)
    dom_set = set(domain_to_proteins.keys())

    # Optionally normalize Pfam IDs in merged_db if they contain version suffixes
    # e.g. PF00067.20 -> PF00067
    # merged_db["pfam_id"] = merged_db["pfam_id"].str.split(".").str[0]

    # 7) Ligands associated with these Pfam IDs
    lig_list = list(
        merged_db[merged_db["pfam_id"].isin(dom_set)]["chem_comp_id"].unique()
    )

    # 8) Wrap everything under the organism key (following the existing pattern)
    domain_to_proteins_wrapped = {org_name: domain_to_proteins}
    lig_list_wrapped = {org_name: lig_list}
    prot_descriptions_wrapped = prot_descriptions

    # 9) Save as pickle files under data_dir/local_organism_data/{org_name}
    local_org_base_dir = data_dir / "local_organism_data"
    paths = save_local_organism_data(
        org_name=org_name,
        domain_to_proteins=domain_to_proteins_wrapped,
        lig_list=lig_list_wrapped,
        prot_descriptions=prot_descriptions_wrapped,
        base_dir=local_org_base_dir,
    )

    return paths
