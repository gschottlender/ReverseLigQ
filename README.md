# ReverseLigQ

ReverseLigQ retrieves candidate binding target proteins from various pathogenic organisms (and human) for a query ligand using an unsupervised, similarity-based approach. Candidate targets are inferred from compounds with known binding domains, combining:

- Ligand similarity search (Morgan fingerprints + Tanimoto, or ChemBERTa embeddings).
- Mapping ligands to Pfam domains and proteins.

This repository provides a **command-line interface** (`rev_ligq.py`) to run the full pipeline for a single query SMILES and organism.  

---

## Table of Contents

- [Requirements](#requirements)  
- [Installation](#installation)  
- [Data and datasets](#data-and-datasets)  
- [Usage (command line)](#usage-command-line)  
  - [Supported organisms](#supported-organisms)  
  - [Main arguments](#main-arguments)  
  - [Choosing the similarity backend](#choosing-the-similarity-backend)  
- [Outputs](#outputs)  
- [Programmatic usage (Python API)](#programmatic-usage-python-api)  
- [Features](#features)  
- [Contributors and citation](#contributors-and-citation)  
- [License](#license)

---

## Requirements

- **Python**: 3.8+  
- Dependencies:
  - `conda`
  - `rdkit`
  - `numpy`, `pandas`
  - `torch`, `transformers`, `huggingface_hub`

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/gschottlender/ReverseLigQ.git
cd ReverseLigQ
```

### 2. Create environment

```bash
conda env create -f environment.yml -n reverse_ligq
conda activate reverse_ligq
```

### 3. Optional executable

```bash
chmod +x rev_ligq.py
```

---

## Data and datasets

ReverseLigQ uses a dataset stored in `data/` by default.  
If absent, it will be automatically downloaded from the HuggingFace repository on first run:

Dataset link:  
👉 **https://huggingface.co/datasets/gschottlender/reverse_ligq**

Contents include fingerprints/embeddings, SMILES dictionaries, ligand lists, domain mappings, and protein descriptions.

---

## Usage (command line)

### Basic example

```bash
python rev_ligq.py   --query-smiles "CCCCCOCCN"   --organism 2
```

### Example using ChemBERTa

```bash
python rev_ligq.py   --query-smiles "CCCCCOCCN"   --organism 2 --search-type chemberta
```

### Supported organisms

1. Bartonella bacilliformis  
2. Klebsiella pneumoniae  
3. Mycobacterium tuberculosis  
4. Trypanosoma cruzi  
5. Staphylococcus aureus RF122  
6. Streptococcus uberis 0140J  
7. Enterococcus faecium  
8. Escherichia coli MG1655  
9. Streptococcus agalactiae NEM316  
10. Pseudomonas syringae  
11. DENV  
12. SARS-CoV-2  
13. Homo sapiens  

### Main arguments

Below is a detailed description of the main parameters of `rev_ligq.py`:

-   **`--query-smiles`** *(required)*\
    The **SMILES string** of the query ligand.\
    This is the core input: the entire pipeline is built around
    comparing this ligand to all compounds in the dataset.

-   **`--organism`** *(required)*\
    Numeric ID of the organism (1--13).\
    Determines which organism's protein set will be used when mapping
    domains to candidate targets.\
    The full organism list is provided in the *Supported organisms*
    section.

-   **`--local-dir`** *(default: `data/`)*\
    Directory where the ReverseLigQ dataset is stored.\
    If it does not exist, the dataset will be automatically downloaded
    from HuggingFace into this folder.

-   **`--out-dir`** *(default: `results/`)*\
    Directory where output files will be saved, including:

    -   `predicted_targets.csv`\
    -   `similarity_search_results.csv`

-   **`--search-type`** *(default: `morgan_fp_tanimoto`)*\
    Method used for ligand similarity search:

    -   `morgan_fp_tanimoto`: fast, lightweight, RDKit-based fingerprint
        similarity.\
    -   `chemberta`: uses ChemBERTa transformer embeddings; slower but
        can capture richer chemical patterns.

-   **`--top-k-ligands`** *(default: `1000`)*\
    Number of most similar ligands retrieved during similarity search.\
    Higher values may increase recall of relevant domains but also
    increase computation time.

-   **`--max-domain-ranks`** *(default: `10`)*\
    Maximum number of **domain ranks** to include in the final protein
    table.\
    Ranks are based on similarity scores of the reference ligands;
    domains with equal scores share the same rank.

-   **`--include-only-curated`** *(flag)*\
    If set, only **curated** ligand--domain associations are used when
    inferring protein targets.\
    If not set, both curated and possible associations are included.

-   **`--only-proteins-with-description`** *(flag)*\
    If set, only proteins with an available functional description will
    be included in the final table.\
    If not set, all proteins are included regardless of annotation
    completeness.

### Choosing the similarity backend

- **Morgan + Tanimoto**: fastest, minimal dependencies  
- **ChemBERTa**: transformer-based, slower but richer

---

## Outputs

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

This distinction is crucial when interpreting predicted protein targets:  
**curated associations provide stronger domain-level evidence**, while **possible associations broaden the search space** by including biologically plausible but unconfirmed domain interactions.
---

## Programmatic usage (Python API)

```python
from pathlib import Path
from rev_ligq import target_search

targets_df, ligands_df = target_search(
    query_smiles="CCCCCOCCN",
    organism="2",
    base_dir=Path("data")
)
```

---

## Features

- Ligand similarity search (Morgan / ChemBERTa)  
- Mapping to Pfam domains (curated + possible)  
- Aggregation to protein-level targets  
- Human organism supported (ID 13)

---

## Contributors and citation

If you use this tool, please cite:

**Schottlender et al. (2022)**  
*From drugs to targets: reverse engineering the virtual screening process on a proteomic scale.*  
Front. Drug Discov. 2:969983. https://doi.org/10.3389/fddsv.2022.969983

---

## License

This project is licensed under the **MIT License**.

---

