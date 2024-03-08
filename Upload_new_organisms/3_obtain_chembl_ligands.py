from collections import defaultdict
import pickle
import sqlite3

# Busqueda de ligandos asociados con proteínas del organismo estudiado en ChEMBL
# Filtros: sólo targets que son proteínas únicas (sin complejos proteicos), ensayos de unión,
# valor pchembl (actividad) >= 6.

# Revisar los nombres de las queries en sqlite3
def chembl_assay_dict(pfam_targets_dict,chembl_db):
    connection = sqlite3.connect(chembl_db)
    cursor = connection.cursor()
    fams_ligands_chembl = defaultdict(list)
    for fam in pfam_targets_dict:   
        for target in pfam_targets_dict[fam]:
            query = '''select assays.chembl_id as id_ensayo, assays.assay_type as tipo_de_ensayo, molecule_dictionary.chembl_id as id_ligando, target_dictionary.chembl_id as id_proteina
            from activities join assays on activities.assay_id = assays.assay_id
            join molecule_dictionary on activities.molregno = molecule_dictionary.molregno 
            join target_dictionary on assays.tid = target_dictionary.tid
            where tipo_de_ensayo = 'B' and activities.pchembl_value >= 6.0 and
            target_dictionary.target_type = 'SINGLE PROTEIN' and id_proteina = "'''+ target+'";'
            cursor.execute(query)
            result = cursor.fetchall()
            for item in result:
                if fam not in fams_ligands_chembl[item[2]]:
                    fams_ligands_chembl[item[2]].append(fam)    
    cursor.close()
    connection.close()
    return fams_ligands_chembl

# Agregar a la base de datos de ligandos y familias pfam de ensayos, los que provienen de mecanismos
def add_chembl_mech(pfam_targets_dict,ligand_pfam_dict,chembl_db):
    connection = sqlite3.connect(chembl_db)
    cursor = connection.cursor()
    for fam in pfam_targets_dict:
        for target in pfam_targets_dict[fam]:
            query = '''select target_dictionary.chembl_id as target, molecule_dictionary.chembl_id as ligando
            from target_dictionary join drug_mechanism on target_dictionary.tid = drug_mechanism.tid
            join molecule_dictionary on drug_mechanism.molregno = molecule_dictionary.molregno
            where target ="'''+target+'";'            
            cursor.execute(query)
            result = cursor.fetchall()
            for item in result:
                if fam not in ligand_pfam_dict[item[1]]:
                    ligand_pfam_dict[item[1]].append(fam)
    cursor.close()
    connection.close()

def smiles_chembl(chembl_ligands_list,db_chembl):
    # Armar diccionario de ligandos y smiles de CHEMBL
    connection = sqlite3.connect(db_chembl)
    cursor = connection.cursor()
    results = []
    for chembl_id in chembl_ligands_list:   
        query = '''select compound_structures.canonical_smiles as smiles, molecule_dictionary.chembl_id as chembl_id
        from compound_structures 
        join molecule_dictionary on compound_structures.molregno = molecule_dictionary.molregno
        where chembl_id="'''+chembl_id+'";'
        cursor.execute(query)
        result = cursor.fetchall()
        results.append(result)

    cursor.close()
    connection.close()
    ligand_ids_smiles_chembl = {results[i][0][1]:results[i][0][0] for i in range(len(results)) if len(results[i])>0}
    
    return ligand_ids_smiles_chembl

def filter_organism_domains_and_targets(domain_targets_chembl_db,fams_prots_db):
	fams_targets = defaultdict(list)
	for target in domain_targets_chembl_db:
		for fam in domain_targets_chembl_db[target]:
		    if fam in fams_prots_db and len(domain_targets_chembl_db[target]) == 1:
		        fams_targets[domain_targets_chembl_db[target][0]].append(target)
		    elif (fam in fams_prots_db and len(domain_targets_chembl_db[target]) >1) and (target not in fams_targets['('+', '.join(domain_targets_chembl_db[target])+')']):
		        fams_targets['('+', '.join(domain_targets_chembl_db[target])+')'].append(target)
	return dict(fams_targets)


if __name__ == '__main__':
	
	import argparse
	import sys
	import os
    
	parser = argparse.ArgumentParser()

	parser.add_argument("-o","--organism_folder",help='Folder where results for the same organism are saved',required=True,type=str)  
	parser.add_argument("-cdb","--chembl_database",help='Path where local ChEMBL db is located',default='./db/chembl/chembl_31.db',type=str)
	parser.add_argument("-tdb","--db_chembl_targets",help='Directory where ChEMBL target - pfam database is located',required = True,type=str)
	
	args = parser.parse_args()

	
	
	with open(args.db_chembl_targets+'/pfam_target_db_chembl.pkl','rb') as r:
		domain_targets_chembl_db = pickle.load(r)
	# Input 2: diccionario de familias y proteinas del organismo (estaría en la carpeta del organismo)
	with open(args.organism_folder+'/fam_prot_dict.pkl','rb') as s:
		fams_prots_db = pickle.load(s)

	# 2) i) Armo diccionario de familias y targets filtrados por dominios que tiene el organismo
	# (distinguiendo multidominios)

	fams_targets = filter_organism_domains_and_targets(domain_targets_chembl_db,fams_prots_db)
	# 3) Busco los ligandos que interactúan con familias Pfam del organismo por ensayos
	# Input 3: base de datos chembl en sqlite3
	chembl_db = args.chembl_database
	ligs_fams_dict = chembl_assay_dict(fams_targets,chembl_db)

	# Busco los ligandos que interactuan con familias Pfam del organismo por mecanismos
	add_chembl_mech(fams_targets,ligs_fams_dict,chembl_db)

	# 5) Busco Smiles de los ligandos
	ligand_list = list(ligs_fams_dict.keys())
	smiles_dict = smiles_chembl(ligand_list,chembl_db)

	# Output 1: diccionario de ligandos y familias a las que se unen
	with open(args.organism_folder+'/ligands_fams_dict_chembl_raw.pkl','wb') as t:
		pickle.dump(ligs_fams_dict,t)

	# Output 2: diccionario de ligandos y smiles
	with open(args.organism_folder+'ligands_smiles_dict_chembl.pkl','wb') as u:
		pickle.dump(smiles_dict,u)
