import requests

from collections import defaultdict

from MOAD_PDBIND.filter_MOAD import filter_ligands
from ligand_from_pfam.domain_pdb_ligand import ligands_from_domain
from extracts.extract_ligand_from_pdb import ligands_from_pdb, pdb_ligands_mapping

from rdkit import Chem

from io import StringIO
import json
import pickle
import pandas as pd
import os

# Búsqueda de Smiles en pdb
def pdb_to_smiles(ligands):
    r = requests.post("https://www.ebi.ac.uk/pdbe/api/pdb/compound/summary/",data=",".join(ligands))
    if r.ok:
        data =  r.json()  
        #print(data)
        new_data = []
        for k,v in data.items():
            r = v[0]            
            r2 = {"pdb_ligand":k}
            if len(r["smiles"]):
                r2["smiles"] = r["smiles"][0]["name"]
            #if "chembl_id" in r:
             #   r2["chembl_id"] = r["chembl_id"]            
                new_data.append(r2)
       
        return new_data
    raise Exception(r.text)

# Obtener a partir de diccionario de familias pfam, todos los ligandos que interactuan con cada familia
def ligands_fams_dict(fams_prots_db,moad_json):	
	fams = list(fams_prots_db.keys())
	fams_chunks = [fams[i:i + 100] for i in range(0, len(fams), 100)]

	lig_est_pfam = []
	for chunk in fams_chunks:
		ligands_pfam = ligands_from_domain(chunk,StringIO(pdb_pfam))
		ligands_valid = filter_ligands(ligands_pfam[['ligand','pdb','domain']].to_records(index=False),moad_json)
		for lig in ligands_valid:
		    if len(lig[0])>0:
		        lig_est_pfam.append(lig)
		        
	dict_ligands_fams = defaultdict(list)
	for r in lig_est_pfam:
		if len(r[0])<5 and r[2] not in dict_ligands_fams[r[0]]:
			dict_ligands_fams[r[0]].append(r[2])
	
	return dict(dict_ligands_fams)

# Busqueda de smiles de todos los ligandos obtenidos en PDB
def smiles_search(total_ligand_list): 

	ligand_chunks = [total_ligand_list[i:i + 500] for i in range(0, len(total_ligand_list), 500)]
	smiles = defaultdict(list)
	missing = []

	for chunk in ligand_chunks:
		try:
		    smiles_list = pdb_to_smiles(chunk)
		    for r in smiles_list:    
		        smiles[r['pdb_ligand']] = r['smiles']
		except:
		    missing.append(chunk)

	if len(missing) >0:
		raise Warning("For some ligands the SMILES representation couldn't be obtained")    

	# 'Estandarizar' smiles con RDkit y quitar los que no pueden ser procesados
	smiles_rdkit = {smile:Chem.MolToSmiles(Chem.MolFromSmiles(smiles[smile])) for smile in smiles if Chem.MolFromSmiles(smiles[smile])!= None}
	return dict(smiles_rdkit)


if __name__ == '__main__':
	
	import argparse
	import sys
    
	parser = argparse.ArgumentParser()

	parser.add_argument("--moad_db", help="Moad json database path",default='./db/moad/moad.json',type=str)    
    # En lugar de usar este argumento, directamente le estoy pasando la carpeta donde se guardan todos los outputs del organismo, donde está el input para este script
    #parser.add_argument("-i","--pfam_proteins_input",help="Database of the organism's proteins grouped by pfam domains (output from script obtain_pfam_domains.py) file path",default = './db/fam_prot_dict.pkl',type=str)
	parser.add_argument("-org","--organism_folder",help='Folder where results for the same organism are saved',required=True,default='./db',type=str)
    #parser.add_argument("--evalue_limit",help='Higher e-value of each hit for assigning proteins to a Pfam family',default=1e-5,type=float)
	args = parser.parse_args()

	if not os.path.exists(args.organism_folder):
		raise Exception("Run first script obtain_pfam_domains.py")
	
	r = requests.get("http://ftp.ebi.ac.uk/pub/databases/Pfam/mappings/pdb_pfam_mapping.txt")
	if r.ok:
		pdb_pfam = r.text

	# Parametro 1: Base de datos moad en json   
	with open(args.moad_db) as j: 
		moad_json = json.load(j)
		
	# abrir diccionario de familias de mtb, armar lista de familias de mtb en bloques de 100 familias para buscar los ligandos asociados en cada estructura

	# Parametro 2: Base de datos de familias y porteinas (output del script 2)

	with open(args.organism_folder+'/fam_prot_dict.pkl','rb') as r:
		fams_prots_db = pickle.load(r)

	dict_ligands_fams = ligands_fams_dict(fams_prots_db,moad_json)
	# Output 1: diccionario de ligandos y familias que unen
	with open(args.organism_folder+'/ligands_fams_dict_pdb.pkl','wb') as u:
		pickle.dump(dict_ligands_fams,u)

	
	# Armar lista de ligandos totales
	total_ligands = list(set(dict_ligands_fams.keys()))
		        
	# Obtener smiles
	smiles_rdkit = smiles_search(total_ligands)
	# Output 2: diccionario de ligandos y smiles
	with open(args.organism_folder+'/ligands_smiles_dict_pdb.pkl','wb') as v:
		pickle.dump(smiles_rdkit,v)


