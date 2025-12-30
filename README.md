# ReverseLigQ — Ligand-Driven Protein Target Discovery

**ReverseLigQ** predicts **candidate protein targets** for a **query ligand** across multiple pathogenic organisms and also supports **human (Homo sapiens)** searches.  
It is based on **unsupervised chemical similarity**: it retrieves chemically similar compounds with known binding domain annotations and proposes compatible **protein/domain** candidates.

Large binaries are handled via **Git LFS**. Public datasets are available on **Hugging Face**.

---

## Table of Contents

- [Key ideas](#key-ideas)
- [Installation](#installation)
- [Dataset](#dataset)
- [Repository layout](#repository-layout)
- [Supported organisms](#supported-organisms)
- [Usage](#usage)
  - [Single query](#single-query)
  - [Batch mode (CSV)](#batch-mode-csv)
  - [Search types](#search-types)
  - [Common arguments](#common-arguments)
- [Outputs](#outputs)
  - [`domain_tag` meaning](#domain_tag-meaning)
- [Add a new organism proteome](#add-a-new-organism-proteome)
- [Search on uploaded organisms](#search-on-uploaded-organisms)
- [Updating / rebuilding the dataset](#updating--rebuilding-the-dataset)
- [Notes & best practices](#notes--best-practices)
- [Citation](#citation)

---

## Key ideas

- **Input**: SMILES (single or batch)
- **Core step**: nearest neighbors by similarity (Morgan/Tanimoto or ChemBERTa/cosine)
- **Evidence layer**: similar ligand → Pfam binding domain (curated/possible) → protein(s)
- **Output**: ranked candidate targets + similarity search details

---

## Installation

Clone the repository and create the conda environment from `environment.yml`:

```bash
git clone https://github.com/gschottlender/ReverseLigQ.git
cd ReverseLigQ
conda env create -n reverse_ligq -f environment.yml
conda activate reverse_ligq
```

> Tip: `mamba` is usually faster than `conda` for environment resolution.

---

## Dataset

### Automatic download (recommended)

On first run, `rev_ligq.py` and `update_rev_ligq.py` will **automatically download** the dataset snapshot (via `snapshot_download`) if required folders are missing.  
By default, it is placed under `databases/`.

---

## Repository layout

After downloading the snapshot, you should have:

```txt
databases/
  compound_data/
    pdb_chembl/
      ligands.parquet
      reps/
        chemberta_zinc_base_768.dat
        chemberta_zinc_base_768.meta.json
        morgan_1024_r2.dat
        morgan_1024_r2.meta.json
  merged_databases/
    binding_data_merged.parquet
    uncurated_binding_data.parquet
    ligs_smiles_merged.parquet
  rev_ligq/
    fam_prot_dict.pkl
    ligand_lists.pkl
    ligs_fams_curated.pkl
    ligs_fams_possible.pkl
    prot_descriptions.pkl
```

Two key roots:

- `databases/compound_data/pdb_chembl/`  
  **LigandStore root**: ligand index (`ligands.parquet`) + representations (`reps/`).

- `databases/rev_ligq/`  
  ReverseLigQ metadata: per-organism ligand lists, ligand→Pfam mappings, Pfam→protein mappings, and protein descriptions.

---

## Supported organisms

Each built-in organism is referenced by an integer **key**:

| Key | Organism |
|---:|:--|
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
| 12 | SARS-CoV-2 |
| 13 | *Homo sapiens* |

---

## Usage

### Single query

#### Default: Morgan fingerprint + Tanimoto
- Search type: `morgan_fp_tanimoto` (default)
- Default threshold: `0.4`
- Default neighbor cap: `k_max = 1000` (important to control RAM)

```bash
python rev_ligq.py \
  --organism 13 \
  --query-smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
  --min-score 0.35 \
  --out-dir results
```

---

### Search types

#### ChemBERTa (cosine)
- Default threshold: `0.8`
- Embeddings from `seyonec/ChemBERTa-zinc-base-v1`

```bash
python rev_ligq.py \
  --organism 13 \
  --query-smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
  --search-type chemberta \
  --min-score 0.7 \
  --out-dir results
```

---

### Batch mode (CSV)

Provide a CSV with **two columns**:

- `lig_id`: your query identifier (used as output folder name)
- `smiles`: ligand SMILES

Example (`queries.csv`):

```csv
lig_id,smiles
aspirin,CC(=O)OC1=CC=CC=C1C(=O)O
caffeine,CN1C(=O)N(C)c2ncn(C)c2C1=O
```

Run:

```bash
python rev_ligq.py --organism 13 --query-csv queries.csv --out-dir results
```

#### Batch outputs

For each row:

```txt
results/<lig_id>/
  predicted_targets.csv
  similarity_search_results.csv
```

---

### Common arguments

- `--organism` *(required)*: organism key (table above) or name (if uploaded).
- `--query-smiles` *(required unless --query-csv)*: single SMILES query.
- `--query-csv` *(optional)*: batch mode.
- `--search-type`: `morgan_fp_tanimoto` (default) or `chemberta`.
- `--min-score`: similarity threshold (default 0.4 Tanimoto, 0.8 ChemBERTa).
- `--k-max-ligands`: cap on neighbors passing the threshold (default 1000). **Increase with care**.
- `--max-domain-ranks`: how many ranked domains to keep in the final table. `None` = keep all.
- `--compound-dir`: default `databases/compound_data/pdb_chembl`.
- `--rev-dir`: default `databases/rev_ligq`.
- `--out-dir`: default `results/`.
- `--chunk-size`: streaming scan chunk size (default 50,000).

---

## Outputs

### 1) `predicted_targets.csv`
Candidate protein targets with domain evidence and similarity scores.

### 2) `similarity_search_results.csv`
Similarity ranking of retrieved ligands + associated domain summary.

---

### `domain_tag` meaning

The `domain_tag` column in `predicted_targets.csv` indicates the strength/type of ligand–domain evidence:

- **`curated`**  
  The binding domain for the reference ligand is **experimentally confirmed**.  
  Higher-confidence associations.

- **`possible`**  
  The ligand is known to bind a **multi-domain protein**, but the exact binding domain is not resolved.  
  Putative domain-level evidence.

---

## Add a new organism proteome

You can extend ReverseLigQ to a custom proteome (not included in the snapshot). High-level pipeline:

1. Parse FASTA
2. Run `hmmscan` against Pfam-A
3. Map protein → Pfam domains
4. Link Pfam → ligands (ReverseLigQ)
5. Persist locally for future searches

### Requirements

- Protein FASTA
- ReverseLigQ parquet databases available locally (auto-download on the first run)
- HMMER (`hmmscan`, `hmmpress`)
- Internet access on first use to download Pfam resources

### Command

```bash
python upload_proteome.py --org-name <organism_name> --fasta-path <proteome.fasta>
```

Example:

```bash
python upload_proteome.py \
  --org-name siniae \
  --fasta-path ./streptococcus_iniae_proteome.fasta \
  --cpu 4
```

### Output location

```txt
databases/local_organism_data/<org_name>/
  domain_to_proteins.pkl      # Pfam → proteins
  lig_list.pkl                # associated ligands
  prot_descriptions.pkl       # protein descriptions
```

> These files are **local only**.

---

## Search on uploaded organisms

To search an uploaded organism, add:

```txt
--uploaded-organism
```

and pass `--organism` with the **same name** used during upload.

### Single SMILES

```bash
python rev_ligq.py \
  --organism siniae \
  --uploaded-organism \
  --query-smiles "CCOC(=O)N1CCC(CC1)Oc2ccc(Cl)cc2"
```

### Batch CSV

```bash
python rev_ligq.py \
  --organism siniae \
  --uploaded-organism \
  --query-csv queries.csv
```

### How it works (data sources)

When using `--uploaded-organism`:

| Source | Purpose |
|---|---|
| `databases/rev_ligq/` | ligand → Pfam mappings |
| `databases/local_organism_data/<org_name>/` | Pfam → protein mappings |

---

## Updating / rebuilding the dataset

`update_rev_ligq.py` rebuilds artifacts from upstream sources (ChEMBL and PDB), including:

- ligand–domain tables (`curated` / `possible`)
- per-organism ligand lists
- ligand → Pfam dictionaries
- Pfam → protein dictionaries and descriptions
- `compound_data` refresh (including embeddings/fingerprints if requested)

### When to run this

Occasionally, e.g.:
- a new ChEMBL release is available (e.g., 36 → 37),
- you want to refresh against updated upstream data.

### Typical usage

```bash
python update_rev_ligq.py --chembl-version 37 --output-dir databases
```

---

## Notes & best practices

- For uploaded organisms, `--organism` must match the folder name exactly:  
  `siniae → databases/local_organism_data/siniae`
- Built-in organisms do **not** require `--uploaded-organism`.
- Increase `--k-max-ligands` carefully (time and memory impact).
- Typical defaults (overrideable):
  - `databases/`
  - `temp_data/new_proteomes/`
  - `cpu = 4`

---

## Citation

If you use this tool or its datasets, please cite:

Schottlender G, Prieto JM, Palumbo MC, Castello FA, Serral F, Sosa EJ, Turjanski AG, Martí MA and Fernández Do Porto D (2022).  
*From drugs to targets: Reverse engineering the virtual screening process on a proteomic scale.* Front. Drug. Discov. 2:969983.  
doi: 10.3389/fddsv.2022.969983

