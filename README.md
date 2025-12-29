# ReverseLigQ — Ligand-Driven Protein Target Discovery

ReverseLigQ is a tool that retrieves candidate protein binding targets for a given query ligand across multiple pathogenic organisms — and now also supports human drug-target search. The method is based on an unsupervised chemical-similarity approach, identifying potential target proteins by leveraging compounds with known binding domains.

All underlying datasets are openly available as a Hugging Face dataset, with large binary files stored using Git LFS.

---

## Installation

Create the Conda environment from `environment.yml`:

```bash
git clone https://github.com/gschottlender/ReverseLigQ.git
cd ReverseLigQ
conda env create -n reverse_ligq -f environment.yml
conda activate reverse_ligq
```
---

## Downloading the dataset

### Automatic
On the first run both `rev_ligq.py` and `update_rev_ligq.py` will **auto-download** the dataset snapshot (via `snapshot_download`) if required folders are missing, placing it under the common parent (default: `databases/`).

---

## Repository layout (expected on disk)

After downloading the dataset snapshot, you should have:

```
databases/
  compound_data/
    pdb_chembl/
      ligands.parquet
      reps/
        chemberta_zinc_base_768.dat
        chemberta_zinc_base_768.meta.json
        morgan_1024_r2.dat
        morgan_1024_r2.meta.json
  rev_ligq/
    fam_prot_dict.pkl
    ligand_lists.pkl
    ligs_fams_curated.pkl
    ligs_fams_possible.pkl
    prot_descriptions.pkl
```

Two roots matter:

- `databases/compound_data/pdb_chembl/`  
  **LigandStore root**: compound index (`ligands.parquet`) + representations (`reps/`).

- `databases/rev_ligq/`  
  ReverseLigQ metadata: organism ligand lists, ligand→Pfam mappings, Pfam→protein mapping, protein descriptions.

---

## Organism-specific datasets

ReverseLigQ integrates multiple organisms, each identified by an integer key.

| Key | Organism |
|---:|---|
| 1 | *Bartonella bacilliformis* |
| 2 | *Klebsiella pneumoniae* |
| 3 | *Mycobacterium tuberculosis* |
| 4 | *Trypanosoma cruzi* |
| 5 | *Staphylococcus aureus* RF122 |
| 6 | *Streptococcus uberis* 0140J |
| 7 | *Enterococcus faecium* |
| 8 | *Escherichia coli* MG1655 |
| 9 | *Streptococcus agalactiae* NEM316 |
| 10 | *Pseudomonas syringae* |
| 11 | DENV (Dengue virus) |
| 12 | SARS‑CoV‑2 |
| 13 | *Homo sapiens* |

---

## Search script (rev_ligq.py)

### Single query (default: Morgan/Tanimoto)

- Default search type: **Morgan fingerprint + Tanimoto**
- Default threshold: **0.4**
- Default neighbor cap (`k_max`): **1000** (important to avoid excessive RAM usage)

```bash
python rev_ligq.py --organism 13 --query-smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --min-score 0.35 --out-dir results
```

### ChemBERTa (cosine)
- Default threshold: **0.8**
- Uses `seyonec/ChemBERTa-zinc-base-v1` to embed the query SMILES.

```bash
python rev_ligq.py --organism 13 --query-smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --search-type chemberta --min-score 0.7 --out-dir results
```

### Key arguments (most used)

- `--organism` *(required)*: organism key (see table above).
- `--query-smiles` *(required unless --query-csv)*: query molecule SMILES.
- `--query-csv` *(optional)*: batch mode CSV (see below).
- `--search-type`: `morgan_fp_tanimoto` (default) or `chemberta`.
- `--min-score`: threshold (default 0.4 for Tanimoto, 0.8 for ChemBERTa).
- `--k-max-ligands`: cap on returned neighbors after thresholding (default 1000).  
  This is mainly a **memory/time safety**. Increase carefully.
- `--max-domain-ranks`: how many domain ranks to keep in the final protein table.  
  Set to `None` to keep **all** discovered domains.
- `--compound-dir`: default `databases/compound_data/pdb_chembl`.
- `--rev-dir`: default `databases/rev_ligq`.
- `--out-dir`: output directory (default: `results/`).
- `--chunk-size`: streaming chunk size for memmap scanning (default 50,000).

---

## Batch mode (CSV input)

Instead of a single SMILES, you can pass a CSV with **two columns**:

- `lig_id`: your identifier for the query (used to name the output folder)
- `smiles`: the query SMILES

Example file (`queries.csv`):

```csv
lig_id,smiles
aspirin,CC(=O)OC1=CC=CC=C1C(=O)O
caffeine,CN1C(=O)N(C)c2ncn(C)c2C1=O
```

Run:

```bash
python rev_ligq.py --organism 13 --query-csv queries.csv --out-dir results
```

### Batch outputs

For each row in the CSV, results are written to:

```
results/<lig_id>/
  predicted_targets.csv
  similarity_search_results.csv
```

Batch runs may also write a summary log under `results/` (depending on your script options), typically containing per-query status and errors.

---

## Output files

Running the CLI produces:

### 1. `predicted_targets.csv`
Candidate protein targets with domain evidence and similarity scores.

### 2. `similarity_search_results.csv`
Ligand similarity ranking and associated domain summaries.

#### Interpretation of `domain_tag`

The column **`domain_tag`** in the predicted_targets.csv output file indicates the type of evidence supporting the ligand–domain association derived from the reference ligand:

- **`curated`**  
  Indicates that the **binding domain of the reference ligand is experimentally confirmed**.  
  These associations originate from ligands whose interaction with a specific Pfam domain has been validated in curated datasets.  
  As such, curated tags represent **high-confidence binding domain assignments**.

- **`possible`**  
  Indicates that the **ligand is known to bind a multidomain protein**, but the **exact binding domain is not experimentally resolved**.  
  In these cases, multiple domains are present in the protein, and although the ligand–protein interaction is supported by experimental data, the **precise domain-level binding site remains undetermined**.  
  Therefore, possible tags denote **putative binding domains** inferred from proteins with multiple domains rather than confirmed, domain-specific evidence.

---

## Updating / rebuilding the dataset (update_rev_ligq.py)

`update_rev_ligq.py` is the **dataset builder** used to regenerate the ReverseLigQ artifacts by querying upstream sources (**ChEMBL** and **PDB**) and rebuilding the derived files:

- merged ligand–domain evidence tables,
- organism-specific ligand lists,
- ligand → Pfam dictionaries (curated / possible),
- Pfam → protein mapping and protein descriptions,
- and refreshed compound-data directories (including rebuilding the ChemBERTa and Morgan representations if requested).

### When should I run this?

This script is intended to be run **infrequently**, typically:
- when a **new ChEMBL release** becomes available (e.g., upgrading from ChEMBL 36 → 37),
- or when you explicitly want to refresh the dataset against updated upstream databases.

### Typical usage

You usually need to specify which ChEMBL version you are targeting:

```bash
python update_rev_ligq.py --chembl-version 37 --output-dir databases
```

---

## Citation

If you use these tool or the datasets, please cite:

Schottlender G, Prieto JM, Palumbo MC, Castello FA, Serral F, Sosa EJ, Turjanski AG, Martí MA and Fernández Do Porto D (2022).  
*From drugs to targets: Reverse engineering the virtual screening process on a proteomic scale.* Front. Drug. Discov. 2:969983.  
doi: 10.3389/fddsv.2022.969983
