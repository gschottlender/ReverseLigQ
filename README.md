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
- [Docker usage guide](#docker-usage-guide)
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

> **Note:** For `--org-name`, use a short identifier **without spaces or special characters** (preferably letters/numbers/underscores only).

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

# Docker Usage Guide

ReverseLigQ can be used entirely through Docker, without installing Conda on the host system.

You only need:

- Docker installed
- A local clone of this repository
- Enough disk space to build the Docker image (the scientific Conda environment is heavy)

All dataset files are stored in a persistent Docker volume so they are downloaded only once and automatically reused across runs.

---

## 1. Install Docker

Make sure Docker is installed and available on your system:

```bash
docker --version
```

---

## 2. Clone the repository

Clone the ReverseLigQ repository and move into its directory:

```bash
git clone https://github.com/gschottlender/ReverseLigQ.git
cd ReverseLigQ
```

All subsequent `docker build` and `docker run` commands assume you are in the root of this repository.

---

## 3. Build the Docker image locally

Build the image using the provided `Dockerfile`:

```bash
docker build -t gschottlender/reverseligq:latest .
```

This command:

- Uses the local repository as build context
- Creates a Conda environment inside the image
- Produces a local Docker image named `gschottlender/reverseligq:latest`

You do **not** need a Docker Hub account for this step; the tag is local.

---

## 4. Create a persistent database volume (one-time)

```bash
docker volume create reverse_ligq_db
```

This volume stores the ReverseLigQ databases and will be reused automatically by all containers.

---

## Single-ligand search (SMILES query)

From any working directory (it can be outside the repo):

```bash
mkdir -p results
```

Then run:

```bash
docker run --rm   -u $(id -u):$(id -g)   -v reverse_ligq_db:/app/databases   -v "$PWD/results":/app/results   -w /app   gschottlender/reverseligq:latest   --organism 13   --query-smiles "CC(=O)OC1=CC=CC=C1C(=O)O"   --min-score 0.35   --out-dir /app/results
```

Outputs are written to:

```
results/<ligand_id>/
```

with:

- `predicted_targets.csv`
- `similarity_search_results.csv`

> **Note:**  
> `-u $(id -u):$(id -g)` ensures files are owned by your user rather than `root`.

---

## Batch search using a CSV file

Create a CSV file:

```csv
lig_id,smiles
aspirin,CC(=O)OC1=CC=CC=C1C(=O)O
caffeine,CN1C(=O)N(C)c2ncn(C)c2C1=O
```

Save it as `queries.csv` in your working directory, then run:

```bash
mkdir -p results

docker run --rm   -u $(id -u):$(id -g)   -v reverse_ligq_db:/app/databases   -v "$PWD/results":/app/results   -v "$PWD/queries.csv":/app/queries.csv   -w /app   gschottlender/reverseligq:latest   --organism 13   --query-csv /app/queries.csv   --out-dir /app/results
```
Where:

- `$PWD` means **“your current directory”** in the terminal.
- `-v "$PWD/queries.csv":/app/queries.csv` tells Docker:  
  > “Use the `queries.csv` file from this folder as `/app/queries.csv` inside the container.”
- `-v "$PWD/results":/app/results` tells Docker:  
  > “Write all output files to the `results/` folder in this directory.”

For each `lig_id`, results appear under:

```
results/<lig_id>/
```

containing:

- `predicted_targets.csv`
- `similarity_search_results.csv`

---

## Using ChemBERTa similarity (optional)

ReverseLigQ can also use a ChemBERTa-based similarity search instead of Morgan–Tanimoto.

To avoid downloading the ChemBERTa model on every run, create a dedicated Docker volume for the Hugging Face cache (only once):

```bash
docker volume create reverse_ligq_hf_cache
```

Then run ChemBERTa searches like this:

```bash
mkdir -p results

docker run --rm \
  -u $(id -u):$(id -g) \
  -v reverse_ligq_db:/app/databases \
  -v reverse_ligq_hf_cache:/hf_cache \
  -v "$PWD/results":/app/results \
  -w /app \
  gschottlender/reverseligq:latest \
  --organism 13 \
  --query-smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
  --search-type chemberta \
  --min-score 0.7 \
  --out-dir /app/results
```

**Explanation:**

- `reverse_ligq_hf_cache` stores the ChemBERTa model files (Hugging Face cache) between runs.
- `/hf_cache` inside the container is configured as `HF_HOME`, so Transformers automatically uses it.
- The first ChemBERTa run will download the model into this volume.
- Subsequent runs reuse the cached model and **do not download it again**.

---

## Add a custom organism (upload proteome)

You can upload a proteome FASTA file and register a new organism.
This only needs to be done **once per organism** — the data are stored inside the persistent Docker volume `reverse_ligq_db`.

### Prepare your FASTA

Place your proteome file in any folder, e.g.:

```
streptococcus_iniae_proteome.fasta
```

Open a terminal **in that folder** and run:

```bash
docker run --rm   -u $(id -u):$(id -g)   -v reverse_ligq_db:/app/databases   -v "$PWD":/data   -w /app   --entrypoint python   gschottlender/reverseligq:latest   upload_proteome.py     --org-name siniae     --fasta-path /data/streptococcus_iniae_proteome.fasta     --cpu 4
```

### What this means

- `$PWD` = **your current directory**
- The FASTA file stays on your machine
- Docker sees it as `/data/streptococcus_iniae_proteome.fasta`
- The processed organism is saved permanently into:

```
databases/local_organism_data/siniae
```

**inside the volume** (not your local folder).

So after uploading, you never need to repeat this step again — just keep using the same Docker volume `reverse_ligq_db`.

---

## Search against a custom organism

Use the same volume and specify:

- `--organism <name>`
- `--uploaded-organism`

Example (single ligand):

```bash
mkdir -p results

docker run --rm   -u $(id -u):$(id -g)   -v reverse_ligq_db:/app/databases   -v "$PWD/results":/app/results   -w /app   gschottlender/reverseligq:latest   --organism siniae   --uploaded-organism   --query-smiles "CC(=O)OC1=CC=CC=C1C(=O)O"   --out-dir /app/results
```

or CSV mode:

```bash
docker run --rm   -u $(id -u):$(id -g)   -v reverse_ligq_db:/app/databases   -v "$PWD/results":/app/results   -v "$PWD/queries.csv":/app/queries.csv   -w /app   gschottlender/reverseligq:latest   --organism siniae   --uploaded-organism   --query-csv /app/queries.csv   --out-dir /app/results
```

The organism remains available for all future searches as long as you keep using the same `reverse_ligq_db` volume.

---

## Removing ReverseLigQ Docker Volumes

### Removing stored databases and cache (optional)

ReverseLigQ stores data in Docker **volumes**, so the downloaded databases and ChemBERTa cache persist across runs.

You only need to remove these volumes if you want to:

- free disk space
- reset all downloaded data
- start again from scratch

#### List existing volumes

```bash
docker volume ls
```

You should see entries like:

```
reverse_ligq_db
reverse_ligq_hf_cache   (only if using ChemBERTa)
```

#### Delete the ReverseLigQ databases

```bash
docker volume rm reverse_ligq_db
```

This permanently deletes:

```
/app/databases
```

including:

- PDB/ChEMBL ligand database
- precomputed fingerprints
- uploaded organisms

It will be re-created and re-downloaded automatically on the next run.

#### Delete the ChemBERTa model cache (optional)

Only needed if you used ChemBERTa search:

```bash
docker volume rm reverse_ligq_hf_cache
```

This removes the Hugging Face cached models.  
They will be downloaded again on the next ChemBERTa run.

---

#### ⚠️ Important notes

- Removing volumes **does not delete the Docker image**
- Your **results folders are NOT touched**, because they live on your local filesystem
- Once deleted, the data **cannot be recovered**

Confirm removal:

```bash
docker volume ls
```

---

## Notes

- The first run will download the dataset into the `reverse_ligq_db` Docker volume.
- All subsequent runs reuse the same data automatically.
- No Conda installation is required on the host; all dependencies live inside the Docker image.
- Works on Linux and macOS (Windows via WSL is recommended).

---

## Citation

If you use this tool or its datasets, please cite:

Schottlender G, Prieto JM, Palumbo MC, Castello FA, Serral F, Sosa EJ, Turjanski AG, Martí MA and Fernández Do Porto D (2022).  
*From drugs to targets: Reverse engineering the virtual screening process on a proteomic scale.* Front. Drug. Discov. 2:969983.  
doi: 10.3389/fddsv.2022.969983

