# ReverseLigQ
Retrieves candidate binding targets of pathogenic organisms for a query ligand performing a chemical similarity search of compounds which their binding domains are known.

## Requirements
Anaconda is required to build the environment with the necessary packages in order to run the program.
 

## Installation
1. Clone the repository in the desired directory:
```sh
git clone https://gitlab.com/gustavoschottlender/reverse-ligq.git
```
2. In the directory where the repository is cloned, create the environment from the .yml file provided to run the program and activate it:

```sh
conda env create -f reverse_ligq.yml -n reverse_ligq
conda activate reverse_ligq
```
3. Add excecution permission to the script compound_test.py:
```sh
chmod +x compound_test.py
```


## General information
The query (input) ligand structure must be in SMILES format.

Databases of each pathogen are located in the Organisms folder.

The output consists of 3 files:

1. candidate_targets.csv: includes predicted target Pfam domains, their respective proteins (Uniprot ID), the names of the compounds that bind them and the Tanimoto Similarity index between the query compound and the candidate domain's most similar binder ligand.

2. candidate_targets_full_result.csv: similar to candidate_targets.csv, but also includes groups of candidate domains from which the correct one is unknown. They are denoted between parenthesis, for example (PF00905, PF00069). All their respective proteins are shown altogether in the corresponding row.

3. similar_compound_search.xlsx: shows 2D chemical structures, Tanimoto Similarity with the query ligand and known binding domains of the compounds obtained in the similarity search.

## Usage

### Graphic interface (using Streamlit)

Run with the following lines
```sh
conda activate reverse_ligq
streamlit run lq_rev.py
```

### Command line interface
Please use the help command first for details about the organisms included in the databases, their respective reference numbers and detailed info about other parameters:
```sh
./compound_test.py -h
```
Base example to search candidate targets for a query ligand with SMILES "CCCCCOCCN" in the organism Klebsiella pneumoniae (reference number 2) with default similarity threshold (0.4): 
```sh
./compound_test.py -org 2 -s "CCCCCOCCN"
```
Switching the Tanimoto Similarity threshold to a desired one (0.3 in this example):
```sh
./compound_test.py -org 2 -s "CCCCCOCCN" -t 0.3
```
Changing output files directory (must specify full path of existing directory):
```sh
./compound_test.py -org 2 -s "CCCCCOCCN" -o /home/username/example-directory/
```

### Upload new organisms

#### script 1_obtain_pfam_domains.py

Obtains protein domains in Pfam from a proteome and generates the corresponding protein database grouped by their domains.

Requirements: 
- The proteome of the bacterium.
- Local hmm database PfamA.
- Hmmer installed in the environment.

Preprocess PfamA database:
```sh
<pfam_directory>/hmmpress Pfam-A.hmm
```

Output:
- Protein database grouped by family in the fam_prot_dict.pkl file.

Usage:
```sh
1_obtain_pfam_domains.py --pfam_db <pfamA database directory>/Pfam-A.hmm -i ./<organism proteome directory>/organism_proteome.fasta -o ./organism_name
```

#### script 2_obtain_pdb_ligands.py

Obtains corresponding PDB ligands from Pfam families in the organism.

Requirements:
- Output from script 1_obtain_pfam_domains.py
- Moad.json database.

Usage: 

```sh
2_obtain_pdb_ligands.py --moad_db <moad database directory>/moad.json -o ./organism_name
```
#### script build_chembl_prot_pfam_db.py

Generates a Chembl targets database and their corresponding Pfam families (does not need to be used for each organism, only needs to be created once).

Requirements:
- Chembl database in sqlite format.

Usage:

```sh
build_chembl_prot_pfam_db.py -db_dir <directory to save database> -cdb <chembl database directory>/chembl_<version>.db
```

#### script 3_obtain_chembl_ligands.py

Obtains corresponding ChEMBL ligands from Pfam families in the organism.

Requirements:
- Output from script build_chembl_prot_pfam_db.py
- Output from script 1_obtain_pfam_domains.py 
- Local ChEMBL database in sqlite format (.db).

Usage:

```sh
3_obtain_chembl_ligands.py -cdb <local ChEMBL database directory>/chembl_32.db -tdb <directory to store ChEMBL targets database and Pfam families> -o ./organism_name
```

#### script 4_curate_chembl_domains.py

Cleans the ChEMBL database. For those ligands that bind to one of several possible domains, it searches for the correct domain through PDB ligands, which are known to bind to the exact domain.

Requirements:
- ChEMBL databases obtained from script 3_obtain_chembl_ligands.py
- PDB ligand databases.

Usage:

```sh
4_curate_chembl_domains.py -o ./organism_name
```

#### script 5_combine_dbs.py

Combines the curated PDB and ChEMBL databases.

Requirements:
- PDB and ChEMBL databases are built.

Usage:

```sh
5_combine_dbs.py -o organism_name
```

