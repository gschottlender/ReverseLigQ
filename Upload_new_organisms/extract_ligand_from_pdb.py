import argparse
from collections import defaultdict
import pandas as pd


def PDB_ID(input_file, skip_lines=1):
    """input_file: file with PDB entry list
    skip_lines: lines to skip(defualt=1, column name)
    returns pdb list"""
    pdb_ids = open(input_file)
    lines=pdb_ids.readlines()[skip_lines:]
    strip_pdb_ids = [line.lower().strip() for line in lines if not line.strip().startswith("#")]

    pdb_ids.close()

    return strip_pdb_ids

def pdb_ligands_mapping(mapping_handle):
    """reads a lig_pairs.lst file and builds a dict(key:pdb , value:ligand) 
       mapping_handle: file with pdb:ligand information  
       columns: 0=PDB_ID; 1=ligands """
    
    mapping_lines=mapping_handle.readlines()
    
    dic_out={}
    for line in mapping_lines:
        lines = line.split(':')

        key = lines[0].strip()
        value = [x.strip() for x in lines[1].strip().split(";") if x.strip()]
        if key in dic_out:
            dic_out[key] += value
        else:
          dic_out[key] = value
    return dic_out

def id_cross(PDB_data, mapping_data):
    """Cross IDs between inputs file"""
    ligand_pdb=[]
    ligand_pdb_key={}

    for pdb in PDB_data:
        if pdb.lower() in mapping_data:
            for ligand in mapping_data[pdb.lower()]:
                key= ligand + "|" + pdb
                if key not in ligand_pdb_key:
                    ligand_pdb.append([ligand,pdb])
                    ligand_pdb_key[key]=1
                    
    
    return pd.DataFrame(ligand_pdb, columns=["ligand", "pdb"])


def ligands_from_pdb(pdbs:list,ligand_pdb_mapping:dict):
    return id_cross(pdbs, ligand_pdb_mapping)

def parse_arguments():

    parser = argparse.ArgumentParser(description='Extract ligands from PDB entries')
    parser.add_argument("-p", '--PDB_list', default=None, help="Input file should have PDB ID list")
    parser.add_argument("-m", '--mapping_ligands_pdb',  default="lig_pairs.lst", help="Input file of ligands and PDB ID link: http://www.ebi.ac.uk/thornton-srv/databases/pdbsum/data/lig_pairs.lst")

    return parser

def main():

    parser=parse_arguments()
    args=parser.parse_args()
    with open(args.mapping_ligands_pdb) as mapping_handle:
        ligand_pdb = id_cross(PDB_ID(args.PDB_list), pdb_ligands_mapping(mapping_handle))
    for ligand,pdb in ligand_pdb.to_records(index=False):
        print(ligand + "\t"+ "|".join(pdb))
    return 0


if __name__=='__main__':
    main()
