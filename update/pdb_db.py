import os
import sys
import json
import time
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

# ----------------------------------------------------------------------
# Extract ligands from PDB and group by family (all)
# ----------------------------------------------------------------------

invalid_ligands = [
    # Solvents and organic additives
    "HOH", "DOD", "EDO", "EGL", "PGO", "PGR", "PDO", "BU1", "1BO", "HEZ",
    "MPD", "MRD", "IPA", "IOH", "EOH", "DMS", "DIO", "BME", "MES", "TRS",
    "MPO", "EPE", "BCN", "IMD", "ACT", "ACY", "ACE", "FMT", "OXM", "ACN",
    "URE", "GAI", "GOL", "PEG", "PGE", "PG4", "1PE", "2PE",
    # Simple sugars and amino acids
    "GLC", "BGC", "GAL", "GLA", "MAN", "BMA", "XYL", "XYP", "FUC", "FUL",
    "NAG", "NBG", "ALA", "DAL", "GLU", "DGL", "LYS", "DLY", "SER", "DSN", "GLY",
    # Metal ions and inorganic salts
    "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA",
    "ZN", "FE", "FE2", "MN", "MN3", "CO", "3CO", "NI", "3NI", "CU", "CU1", "CU3",
    "CR", "V", "4TI", "AL", "GA", "Y",
    "AG", "CD", "IR", "PT", "AU", "HG", "PB",
    "LA", "CE", "PR", "ND", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU",
    "CL", "BR", "F", "IOD", "OH", "CYN", "NO", "SO4", "SO3", "PO4", "PI", "2HP",
    "CO3", "BCT", "NO3", "SE4", "ART", "AST", "NCO",
    # Cofactors and metabolites
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "UTP", "UDP", "UMP", "CTP", "CDP", "CMP",
    "NAD", "NAI", "NAP", "NDP", "FAD", "FMN", "SAM", "SAH", "COA", "ACO",
    "PLP", "TPP", "BTN", "H4B", "THF", "LPA", "PQQ", "HEM", "HEA", "HEC", "HEB", "B12",
]

PDB_PFAM_MAPPING_URL = "https://ftp.ebi.ac.uk/pub/databases/Pfam/mappings/pdb_pfam_mapping.txt"
CCD_SMILES_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/Components-smiles-stereo-oe.smi"
PDBe_BINDING_SITES_URL = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/binding_sites/{pdb_id}"


def fetch_binding_sites_json(
    pdb_id: str,
    timeout: float = 15.0,
    cache_dir: str | None = None,
):
    """
    Fetch PDBe binding-site JSON for a single PDB ID.
    If cache_dir is provided, cache per-PDB JSON on disk.

    Returns:
        The list of binding sites for that PDB ID, or None if not available.
    """
    pdb_id = pdb_id.strip().lower()

    # 1) Try cache first
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{pdb_id}_binding_sites.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as fh:
                    data = json.load(fh)
                return data.get(pdb_id)
            except Exception:
                # If cache is corrupted, ignore it and fall back to the API
                pass

    # 2) If not cached, call the API
    url = PDBe_BINDING_SITES_URL.format(pdb_id=pdb_id)
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException:
        # Network issue, DNS failure, timeout, etc.
        return None

    if resp.status_code == 404:
        return None

    resp.raise_for_status()
    data = resp.json()
    if pdb_id not in data:
        return None

    # 3) Save to cache if applicable
    if cache_dir is not None:
        try:
            with open(cache_path, "w") as fh:
                json.dump(data, fh)
        except Exception:
            pass

    return data[pdb_id]


def binding_sites_to_df(pdb_id: str, sites_json: list[dict]) -> pd.DataFrame:
    """
    Convert PDBe binding-site JSON into a tidy DataFrame.

    Columns:
        - pdb_id
        - site_id
        - site_label
        - chem_comp_id
        - ligand_chain_id
        - ligand_res_seq
        - residue_chain_id
        - residue_number
        - residue_name
    """
    rows = []
    for site in sites_json:
        site_id = site.get("site_id") or site.get("site_number")
        site_label = site.get("description") or site.get("site_type")

        ligand_residues = site.get("ligand_residues") or []
        site_residues = site.get("site_residues") or []

        for lig in ligand_residues:
            chem_comp_id = lig.get("chem_comp_id") or lig.get("residue_name")
            ligand_chain_id = lig.get("chain_id") or lig.get("author_chain_id")
            ligand_res_seq = (
                lig.get("author_residue_number")
                or lig.get("residue_number")
            )

            for res in site_residues:
                residue_chain_id = res.get("chain_id") or res.get("author_chain_id")
                residue_number = (
                    res.get("author_residue_number")
                    or res.get("residue_number")
                )
                residue_name = res.get("residue_name")

                rows.append(
                    {
                        "pdb_id": pdb_id,
                        "site_id": site_id,
                        "site_label": site_label,
                        "chem_comp_id": chem_comp_id,
                        "ligand_chain_id": ligand_chain_id,
                        "ligand_res_seq": ligand_res_seq,
                        "residue_chain_id": residue_chain_id,
                        "residue_number": residue_number,
                        "residue_name": residue_name,
                    }
                )

    return pd.DataFrame(rows)


def binding_sites_for_entries(
    pdb_ids,
    timeout: float = 10.0,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """
    Fetch and flatten binding sites for multiple PDB entries.

    Uses an optional cache_dir to avoid repeated API calls.
    Returns a single concatenated DataFrame with binding-site information.
    """
    dfs = []

    for pdb_id in pdb_ids:
        js = fetch_binding_sites_json(
            pdb_id,
            timeout=timeout,
            cache_dir=cache_dir,
        )
        if js is None:
            continue

        df = binding_sites_to_df(pdb_id, js)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(
            columns=[
                "pdb_id",
                "site_id",
                "site_label",
                "chem_comp_id",
                "ligand_chain_id",
                "ligand_res_seq",
                "residue_chain_id",
                "residue_number",
                "residue_name",
            ]
        )

    return pd.concat(dfs, ignore_index=True)


def download_pdb_pfam_mapping(
    mapping_dir: str = "temp_data/pfam",
    filename: str = "pdb_pfam_mapping.txt",
    url: str = PDB_PFAM_MAPPING_URL,
) -> str:
    """
    Download pdb_pfam_mapping.txt to mapping_dir if not present.

    Returns:
        The local path to the mapping file.
    """
    os.makedirs(mapping_dir, exist_ok=True)
    local_path = os.path.join(mapping_dir, filename)

    if os.path.exists(local_path):
        print(f"[pdb_pfam_mapping] Using cached file: {local_path}")
        return local_path

    print(f"[pdb_pfam_mapping] Downloading from {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"[pdb_pfam_mapping] Saved to {local_path}")
    return local_path


def load_pdb_pfam_mapping(mapping_path: str) -> pd.DataFrame:
    """
    Load the pdb_pfam_mapping table (TSV with header) and
    add a lowercase 'pdb_id' column for merging with PDBe data.
    """
    df = pd.read_csv(
        mapping_path,
        sep="\t",
        dtype=str,
        comment="#",  # ignore the initial metadata line
    )

    # Create a standard column for merging with binding_sites (which uses 'pdb_id')
    df["pdb_id"] = df["PDB"].str.lower()

    # Convert AUTH positions to numeric
    for col in ["AUTH_PDBRES_START", "AUTH_PDBRES_END"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing ranges
    df = df.dropna(subset=["AUTH_PDBRES_START", "AUTH_PDBRES_END"]).copy()
    df["AUTH_PDBRES_START"] = df["AUTH_PDBRES_START"].astype(int)
    df["AUTH_PDBRES_END"] = df["AUTH_PDBRES_END"].astype(int)

    return df


def assign_pfam_to_binding_residues(
    df_binding: pd.DataFrame,
    df_pfam: pd.DataFrame
) -> pd.DataFrame:
    """
    Match binding-site residues (df_binding) with Pfam domains (df_pfam)
    based on AUTH residue ranges.

    df_binding columns:
        - pdb_id
        - site_id
        - chem_comp_id
        - residue_chain_id
        - residue_number

    df_pfam columns:
        - pdb_id
        - CHAIN
        - PFAM_ACCESSION
        - AUTH_PDBRES_START
        - AUTH_PDBRES_END

    Returns:
        A DataFrame with:
            pdb_id, site_id, chem_comp_id, residue_chain_id, residue_number, pfam_id
    """
    if df_binding.empty or df_pfam.empty:
        return pd.DataFrame(
            columns=[
                "pdb_id",
                "site_id",
                "chem_comp_id",
                "residue_chain_id",
                "residue_number",
                "pfam_id",
            ]
        )

    b = df_binding.copy()
    p = df_pfam.copy()

    # Ensure numeric types where needed
    b["residue_number"] = pd.to_numeric(b["residue_number"], errors="coerce")
    p["AUTH_PDBRES_START"] = pd.to_numeric(p["AUTH_PDBRES_START"], errors="coerce")
    p["AUTH_PDBRES_END"] = pd.to_numeric(p["AUTH_PDBRES_END"], errors="coerce")

    b = b.dropna(subset=["residue_number"])
    p = p.dropna(subset=["AUTH_PDBRES_START", "AUTH_PDBRES_END"])

    if b.empty or p.empty:
        return pd.DataFrame(
            columns=[
                "pdb_id",
                "site_id",
                "chem_comp_id",
                "residue_chain_id",
                "residue_number",
                "pfam_id",
            ]
        )

    b["residue_number"] = b["residue_number"].astype(int)
    p["AUTH_PDBRES_START"] = p["AUTH_PDBRES_START"].astype(int)
    p["AUTH_PDBRES_END"] = p["AUTH_PDBRES_END"].astype(int)

    # pdb_id is already lowercase in binding; enforce lowercase in Pfam as well
    b["pdb_id"] = b["pdb_id"].str.lower()
    p["pdb_id"] = p["pdb_id"].str.lower()

    # Merge by pdb_id + chain
    merged = b.merge(
        p[["pdb_id", "CHAIN", "PFAM_ACCESSION", "AUTH_PDBRES_START", "AUTH_PDBRES_END"]],
        left_on=["pdb_id", "residue_chain_id"],
        right_on=["pdb_id", "CHAIN"],
        how="left",
    )

    # Filter by residue ranges
    mask = (
        (merged["residue_number"] >= merged["AUTH_PDBRES_START"])
        & (merged["residue_number"] <= merged["AUTH_PDBRES_END"])
    )

    hits = merged[mask].copy()
    hits.rename(columns={"PFAM_ACCESSION": "pfam_id"}, inplace=True)

    return hits[
        ["pdb_id", "site_id", "chem_comp_id", "residue_chain_id", "residue_number", "pfam_id"]
    ]


def chunk_list(lst, size: int):
    """
    Yield successive chunks of length 'size' from a list.
    """
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def process_pdb_chunk(
    pdb_chunk,
    invalid_ligands_list,
    ligands_info_small: pd.DataFrame,
    df_pfam_all: pd.DataFrame,
    binding_cache_dir: str | None = None,
    binding_timeout: float = 10.0,
) -> pd.DataFrame:
    """
    Process a chunk of PDB IDs.

    For each PDB:
        1) Fetch binding sites (with optional cache).
        2) Clean ligand IDs (drop None and invalid ligands).
        3) Select the Pfam subset for that PDB.
        4) Assign Pfam domains to binding residues.
        5) Collapse to (pdb_id, chem_comp_id, pfam_id).
        6) Add uniprot_id from ligands_info_small.

    If a PDB fails, a warning is printed and the rest of the chunk is still processed.
    Only that PDB is skipped, not the entire chunk.
    """
    per_pdb_results = []

    # Pre-filter Pfam entries for the whole chunk (avoid scanning df_pfam_all for each PDB)
    lower_chunk = [p.lower() for p in pdb_chunk]
    df_pfam_chunk_all = df_pfam_all[df_pfam_all["pdb_id"].isin(lower_chunk)].copy()

    for pdb_id in pdb_chunk:
        pdb_lower = pdb_id.lower()
        try:
            # 1. Binding sites for this PDB
            df_sites = binding_sites_for_entries(
                [pdb_id],
                timeout=binding_timeout,
                cache_dir=binding_cache_dir,
            )

            if df_sites.empty:
                continue

            # 2. Clean ligands
            df_sites = df_sites[df_sites["chem_comp_id"].notna()]
            if df_sites.empty:
                continue

            df_sites["chem_comp_id"] = df_sites["chem_comp_id"].str.upper()

            if invalid_ligands_list:
                df_sites = df_sites[
                    ~df_sites["chem_comp_id"].isin(invalid_ligands_list)
                ]
            if df_sites.empty:
                continue

            # 3. Pfam subset only for this PDB
            df_pfam_pdb = df_pfam_chunk_all[df_pfam_chunk_all["pdb_id"] == pdb_lower]
            if df_pfam_pdb.empty:
                continue

            # 4. Assign Pfam to binding residues
            df_hits = assign_pfam_to_binding_residues(df_sites, df_pfam_pdb)
            if df_hits.empty:
                continue

            # 5. Collapse to ligand–domain pairs
            df_ld = (
                df_hits[["pdb_id", "chem_comp_id", "pfam_id"]]
                .dropna()
                .drop_duplicates()
            )
            if df_ld.empty:
                continue

            # 6. Add UniProt IDs
            df_ld = df_ld.merge(
                ligands_info_small,  # columns: pdb_id, uniprot_id
                on="pdb_id",
                how="left",
            ).drop_duplicates()

            if not df_ld.empty:
                per_pdb_results.append(
                    df_ld[["pdb_id", "chem_comp_id", "pfam_id", "uniprot_id"]]
                )

        except Exception as e:
            # Only this PDB is lost; the rest of the chunk still proceeds
            print(f"[WARN] Error processing PDB {pdb_id}: {e!r}")
            continue

    if not per_pdb_results:
        return pd.DataFrame(columns=["pdb_id", "chem_comp_id", "pfam_id", "uniprot_id"])

    # Concatenate valid results for this chunk
    return pd.concat(per_pdb_results, ignore_index=True).drop_duplicates()


def download_ccd_smiles(
    smiles_dir: str = "temp_data/ccd",
    filename: str = "Components-smiles-stereo-oe.smi",
    url: str = CCD_SMILES_URL,
) -> str:
    """
    Download the CCD SMILES table if it does not exist locally.

    Returns:
        The local path to the .smi file.
    """
    os.makedirs(smiles_dir, exist_ok=True)
    local_path = os.path.join(smiles_dir, filename)

    if not os.path.exists(local_path):
        print(f"Downloading CCD SMILES table to {local_path} ...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download finished.")
    else:
        print(f"Using cached CCD SMILES table at {local_path}")

    return local_path


def load_ccd_smiles_table(smiles_path: str) -> pd.DataFrame:
    """
    Load a .smi file where each line is:
        <smiles> <TAB> <chem_comp_id> <TAB> <name ...>

    Returns:
        A DataFrame with columns: smiles, chem_comp_id
    """
    df = pd.read_csv(
        smiles_path,
        sep="\t",       # TAB, not spaces
        header=None,
        dtype=str,
        comment="#",    # ignore comment lines
        engine="python",
    )

    # Ensure at least two columns are present
    if df.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 columns (smiles, chem_comp_id) in {smiles_path}, "
            f"but found only {df.shape[1]}."
        )

    # Keep only the first two columns: smiles and ID
    df = df.iloc[:, :2]
    df.columns = ["smiles", "chem_comp_id"]

    df["chem_comp_id"] = df["chem_comp_id"].str.upper()

    # If duplicates with different SMILES exist, keep the first occurrence
    df = df.dropna(subset=["smiles", "chem_comp_id"]).drop_duplicates(subset=["chem_comp_id"])

    return df


def fetch_smiles_from_rcsb(chem_comp_id: str, timeout: float = 10.0) -> str | None:
    """
    Query the RCSB API for SMILES corresponding to a chem_comp_id.

    Returns:
        The SMILES string if found, otherwise None.
    """
    chem_comp_id = chem_comp_id.upper().strip()
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{chem_comp_id}"

    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None

        data = resp.json()
        desc = data.get("rcsb_chem_comp_descriptor", {})

        smiles = (
            desc.get("smiles")
            or desc.get("smiles_stereo")
            or desc.get("smiles_canonical")
        )

        if isinstance(smiles, str) and smiles.strip():
            return smiles.strip()

        return None

    except Exception:
        return None


def _fetch_smiles_batch_rcsb(
    missing_ids: list[str],
    timeout: float = 10.0,
    max_workers: int = 8,
) -> Dict[str, str]:
    """
    Call fetch_smiles_from_rcsb in parallel for a list of IDs.

    Returns:
        A dictionary {chem_comp_id: smiles} only for successfully resolved IDs.
    """
    missing_ids = [cid.upper().strip() for cid in missing_ids if cid]
    results: Dict[str, str] = {}

    if not missing_ids:
        return results

    def worker(cid: str) -> tuple[str, str | None]:
        return cid, fetch_smiles_from_rcsb(cid, timeout=timeout)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, cid): cid for cid in missing_ids}
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                cid_res, smiles = fut.result()
                if smiles:
                    results[cid_res] = smiles
            except Exception:
                # Do not abort everything due to one failure
                continue

    return results


def build_ligand_smiles_table(
    df_ld_up_all: pd.DataFrame,
    ccd_smiles_dir: str = "temp_data/ccd",
    ccd_smiles_filename: str = "Components-smiles-stereo-oe.smi",
    use_rcsb: bool = True,
    rcsb_timeout: float = 10.0,
    rcsb_max_workers: int = 8,
) -> pd.DataFrame:
    """
    Build a ligand → SMILES table for PDB ligands present in df_ld_up_all.

    Returns:
        A DataFrame with columns:
            chem_comp_id | smiles | source ('pdb')
    """
    # 1) Load CCD SMILES table
    smiles_path = download_ccd_smiles(
        smiles_dir=ccd_smiles_dir,
        filename=ccd_smiles_filename,
    )
    df_ccd = load_ccd_smiles_table(smiles_path)

    # 2) Unique ligand IDs from the final table
    lig_ids = (
        df_ld_up_all["chem_comp_id"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )

    # 3) Extract SMILES from CCD
    df_ccd_sub = df_ccd[df_ccd["chem_comp_id"].isin(lig_ids)]

    lig_smiles = (
        df_ccd_sub[["chem_comp_id", "smiles"]]
        .dropna()
        .drop_duplicates("chem_comp_id")
        .set_index("chem_comp_id")["smiles"]
        .to_dict()
    )

    # 4) Use RCSB API to fill missing ligands
    if use_rcsb:
        missing = [cid for cid in lig_ids if cid not in lig_smiles]

        if missing:
            print(f"{len(missing)} ligands without SMILES in CCD. Querying RCSB...")
            fetched = _fetch_smiles_batch_rcsb(
                missing_ids=missing,
                timeout=rcsb_timeout,
                max_workers=rcsb_max_workers,
            )
            print(f"RCSB returned SMILES for {len(fetched)} ligands.")
            lig_smiles.update(fetched)

    # 5) Build final table: chem_comp_id | smiles | source='pdb'
    rows = []
    for cid, smi in lig_smiles.items():
        rows.append({"chem_comp_id": cid, "smiles": smi, "source": "pdb"})

    df_smiles = pd.DataFrame(rows).sort_values("chem_comp_id").reset_index(drop=True)

    return df_smiles


def generate_pdb_database(
    temp_dir: str = "temp_data",
    data_dir: str = "databases/pdb",
    pfam_mapping_filename: str = "pdb_pfam_mapping.txt",
):
    """
    Build the initial PDB-based database and save it into `data_dir`.

    This pipeline:
      1) Downloads interacting_chains_with_ligand_functions.tsv
      2) Loads and filters ligand information
      3) Loads pdb_pfam_mapping.txt
      4) Processes ALL PDB entries (binding sites + Pfam + UniProt)
      5) Builds ligand→SMILES table
      6) Saves:
            data_dir/pdb_binding_data.parquet
            data_dir/pdb_ligand_smiles.parquet
            data_dir/pdb_seen_ids.txt   <-- NEW
    """

    # Ensure directories exist
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    pfam_dir = os.path.join(temp_dir, "pfam")
    os.makedirs(pfam_dir, exist_ok=True)

    binding_cache_dir = os.path.join(temp_dir, "pdbe_binding_sites")
    os.makedirs(binding_cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Download interacting_chains_with_ligand_functions.tsv
    # ------------------------------------------------------------------
    interactions_url = (
        "https://ftp.ebi.ac.uk/pub/databases/msd/pdbechem_v2/additional_data/"
        "pdb_ligand_interactions/interacting_chains_with_ligand_functions.tsv"
    )
    interactions_local_path = os.path.join(
        temp_dir, "interacting_chains_with_ligand_functions.tsv"
    )

    print("[INFO] Downloading interacting_chains_with_ligand_functions.tsv ...")
    r = requests.get(interactions_url)
    r.raise_for_status()
    with open(interactions_local_path, "wb") as f:
        f.write(r.content)
    print("[INFO] Download complete.")

    # Load TSV
    ligands_info = pd.read_csv(interactions_local_path, sep="\t", dtype=str)

    # Filter invalid ligands
    global invalid_ligands
    ligands_info = ligands_info[~ligands_info["LigandID"].isin(invalid_ligands)]

    # ------------------------------------------------------------------
    # 2) Save SEEN PDB IDs (complete list after filtering)
    # ------------------------------------------------------------------
    pdb_ids = (
        ligands_info["PDBID"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )

    seen_file = os.path.join(data_dir, "pdb_seen_ids.txt")
    with open(seen_file, "w") as f:
        for pdb in sorted(pdb_ids):
            f.write(pdb + "\n")

    print(f"[INFO] Saved {len(pdb_ids)} seen PDB IDs to {seen_file}")

    # ------------------------------------------------------------------
    # 3) Load PDB → Pfam mapping
    # ------------------------------------------------------------------
    mapping_path = download_pdb_pfam_mapping(pfam_dir, pfam_mapping_filename)
    df_pfam_all = load_pdb_pfam_mapping(mapping_path)

    # Prepare PDB → UniProt mapping
    lig_small = (
        ligands_info[["PDBID", "BestUnpAccession"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"PDBID": "pdb_id", "BestUnpAccession": "uniprot_id"})
    )
    lig_small["pdb_id"] = lig_small["pdb_id"].str.lower()

    # ------------------------------------------------------------------
    # 4) Parallel processing of ALL PDB IDs
    # ------------------------------------------------------------------
    chunk_size = 10
    max_workers = os.cpu_count() or 4

    chunk_list_full = list(chunk_list(pdb_ids, chunk_size))
    total_chunks = len(chunk_list_full)

    print(f"[INFO] Processing {total_chunks} chunks using {max_workers} workers...")

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                process_pdb_chunk,
                chunk,
                invalid_ligands,
                lig_small,
                df_pfam_all,
                binding_cache_dir,
                10.0,
            ): idx
            for idx, chunk in enumerate(chunk_list_full, start=1)
        }

        for completed_i, fut in enumerate(as_completed(futures), start=1):
            idx = futures[fut]

            elapsed = time.time() - start_time
            avg = elapsed / completed_i
            eta = avg * (total_chunks - completed_i)

            try:
                df_res = fut.result()
                if df_res is not None and not df_res.empty:
                    results.append(df_res)
            except Exception as e:
                print(f"[WARN] Chunk {idx} failed: {e!r}")

            sys.stdout.write(
                f"\r[{completed_i}/{total_chunks}] Chunk {idx} | "
                f"Elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m"
            )
            sys.stdout.flush()

    if results:
        df_all = pd.concat(results, ignore_index=True).drop_duplicates()
    else:
        df_all = pd.DataFrame(
            columns=["pdb_id", "chem_comp_id", "pfam_id", "uniprot_id"]
        )

    df_all["source"] = "pdb"

    # ------------------------------------------------------------------
    # 5) Build SMILES table
    # ------------------------------------------------------------------
    df_smiles = build_ligand_smiles_table(df_all)

    df_smiles["chem_comp_id"] = df_smiles["chem_comp_id"].str.upper()
    df_all["chem_comp_id"] = df_all["chem_comp_id"].str.upper()

    df_all = df_all[df_all["chem_comp_id"].isin(df_smiles["chem_comp_id"])]

    # ------------------------------------------------------------------
    # 6) Save final outputs
    # ------------------------------------------------------------------
    out_rel = os.path.join(data_dir, "pdb_binding_data.parquet")
    out_smiles = os.path.join(data_dir, "pdb_ligand_smiles.parquet")

    df_all.to_parquet(out_rel, index=False)
    df_smiles.to_parquet(out_smiles, index=False)

    print("\n[INFO] PDB database successfully generated:")
    print(f"  · {out_rel}")
    print(f"  · {out_smiles}")
    print(f"  · {seen_file}  (seen IDs list)")


def update_pdb_database_from_dir(
    data_dir: str,
    temp_dir: str = "temp_data",
    pfam_mapping_filename: str = "pdb_pfam_mapping.txt",
) -> bool:
    """
    Incrementally update an existing PDB database stored in `data_dir`.

    Returns
    -------
    updated : bool
        True if new PDB entries were processed and the binding/SMILES
        tables were effectively updated. False if no new PDB IDs were
        found (only the seen-IDs list may have changed).
    """

    os.makedirs(temp_dir, exist_ok=True)

    pfam_dir = os.path.join(temp_dir, "pfam")
    os.makedirs(pfam_dir, exist_ok=True)

    binding_cache_dir = os.path.join(temp_dir, "pdbe_binding_sites")
    os.makedirs(binding_cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load existing DB
    # ------------------------------------------------------------------
    rel_path = os.path.join(data_dir, "pdb_binding_data.parquet")
    try:
        df_existing = pd.read_parquet(rel_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No pdb_binding_data.parquet in {data_dir}. "
            "Run generate_pdb_database() first."
        )

    existing_rel_ids = set(df_existing["pdb_id"].astype(str).str.lower().unique())
    print(f"[INFO] Existing DB contains {len(existing_rel_ids)} PDB with valid records.")

    # ------------------------------------------------------------------
    # 1b) Load SEEN PDB IDs list
    # ------------------------------------------------------------------
    seen_file = os.path.join(data_dir, "pdb_seen_ids.txt")
    if os.path.exists(seen_file):
        with open(seen_file) as f:
            seen_ids_before = {line.strip() for line in f}
        print(f"[INFO] Loaded {len(seen_ids_before)} previously seen PDB IDs.")
    else:
        seen_ids_before = set()
        print("[WARN] No pdb_seen_ids.txt found. Starting seen list empty.")

    # ------------------------------------------------------------------
    # 2) Download current interacting_chains_with_ligand_functions.tsv
    # ------------------------------------------------------------------
    interactions_url = (
        "https://ftp.ebi.ac.uk/pub/databases/msd/pdbechem_v2/additional_data/"
        "pdb_ligand_interactions/interacting_chains_with_ligand_functions.tsv"
    )
    interactions_local_path = os.path.join(
        temp_dir, "interacting_chains_with_ligand_functions.tsv"
    )

    print("[INFO] Downloading interacting_chains_with_ligand_functions.tsv ...")
    r = requests.get(interactions_url)
    r.raise_for_status()
    with open(interactions_local_path, "wb") as f:
        f.write(r.content)
    print("[INFO] Download complete.")

    ligands_info = pd.read_csv(interactions_local_path, sep="\t", dtype=str)

    global invalid_ligands
    ligands_info = ligands_info[~ligands_info["LigandID"].isin(invalid_ligands)]

    # Full candidate list
    pdb_ids_now = (
        ligands_info["PDBID"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )

    # ------------------------------------------------------------------
    # 3) Determine TRULY new PDB IDs
    # ------------------------------------------------------------------
    pdb_ids_to_process = [p for p in pdb_ids_now if p not in seen_ids_before]

    print(
        f"[INFO] Total PDB in TSV: {len(pdb_ids_now)} | "
        f"Previously seen: {len(seen_ids_before)} | "
        f"New (unseen): {len(pdb_ids_to_process)}"
    )

    if not pdb_ids_to_process:
        print("[INFO] No new PDBs to process. Updating seen list and exiting.")
        # Update seen list (in case TSV changed e.g. filtered ligands)
        updated_seen = seen_ids_before.union(pdb_ids_now)
        with open(seen_file, "w") as f:
            for pdb in sorted(updated_seen):
                f.write(pdb + "\n")
        # No logical update to binding/smiles tables
        return False

    # ------------------------------------------------------------------
    # 4) Prepare Pfam & UniProt linking
    # ------------------------------------------------------------------
    mapping_path = download_pdb_pfam_mapping(pfam_dir, pfam_mapping_filename)
    df_pfam_all = load_pdb_pfam_mapping(mapping_path)

    lig_small = (
        ligands_info[["PDBID", "BestUnpAccession"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"PDBID": "pdb_id", "BestUnpAccession": "uniprot_id"})
    )
    lig_small["pdb_id"] = lig_small["pdb_id"].str.lower()

    # ------------------------------------------------------------------
    # 5) Process ONLY NEW PDBs
    # ------------------------------------------------------------------
    chunk_size = 10
    max_workers = min(os.cpu_count() or 4, 16)

    chunk_list_full = list(chunk_list(pdb_ids_to_process, chunk_size))
    total_chunks = len(chunk_list_full)

    print(f"[INFO] Processing {total_chunks} new chunks using {max_workers} workers...")

    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                process_pdb_chunk,
                chunk,
                invalid_ligands,
                lig_small,
                df_pfam_all,
                binding_cache_dir,
                10.0,
            ): idx
            for idx, chunk in enumerate(chunk_list_full, start=1)
        }

        for completed, fut in enumerate(as_completed(futures), start=1):
            idx = futures[fut]

            elapsed = time.time() - start
            avg = elapsed / completed
            eta = avg * (total_chunks - completed)

            try:
                df_res = fut.result()
                if df_res is not None and not df_res.empty:
                    results.append(df_res)
            except Exception as e:
                print(f"[WARN] Chunk {idx} failed: {e!r}")

            sys.stdout.write(
                f"\r[{completed}/{total_chunks}] Chunk {idx} | Elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m"
            )
            sys.stdout.flush()

    if results:
        df_new = pd.concat(results, ignore_index=True).drop_duplicates()
    else:
        print("[WARN] No results from new PDBs.")
        df_new = pd.DataFrame(columns=df_existing.columns)

    # Si por alguna razón no se generaron filas nuevas, podemos considerar
    # que no hubo actualización lógica.
    if df_new.empty:
        print("[INFO] No new valid binding records from unseen PDBs.")
        # Aun así actualizamos la lista de IDs vistos:
        updated_seen = seen_ids_before.union(pdb_ids_now)
        with open(seen_file, "w") as f:
            for pdb in sorted(updated_seen):
                f.write(pdb + "\n")
        return False

    # ------------------------------------------------------------------
    # 6) Merge new results with existing
    # ------------------------------------------------------------------
    df_all = (
        pd.concat([df_existing, df_new], ignore_index=True)
        .drop_duplicates()
    )
    df_all["source"] = "pdb"

    # ------------------------------------------------------------------
    # 7) Rebuild SMILES table
    # ------------------------------------------------------------------
    df_smiles = build_ligand_smiles_table(df_all)

    df_smiles["chem_comp_id"] = df_smiles["chem_comp_id"].str.upper()
    df_all["chem_comp_id"] = df_all["chem_comp_id"].str.upper()

    df_all = df_all[df_all["chem_comp_id"].isin(df_smiles["chem_comp_id"])]

    # Save updated outputs
    rel_out = os.path.join(data_dir, "pdb_binding_data.parquet")
    smiles_out = os.path.join(data_dir, "pdb_ligand_smiles.parquet")

    df_all.to_parquet(rel_out, index=False)
    df_smiles.to_parquet(smiles_out, index=False)

    print("\n[INFO] PDB database successfully updated:")
    print(f"  · {rel_out}")
    print(f"  · {smiles_out}")

    # ------------------------------------------------------------------
    # 8) Update SEEN IDs list
    # ------------------------------------------------------------------
    updated_seen = seen_ids_before.union(pdb_ids_now)
    with open(seen_file, "w") as f:
        for pdb in sorted(updated_seen):
            f.write(pdb + "\n")

    print(f"[INFO] Updated PDB seen IDs list saved to {seen_file}")
    print("[INFO] Done.")

    # Hubo PDB nuevos con registros válidos → actualización real
    return True
