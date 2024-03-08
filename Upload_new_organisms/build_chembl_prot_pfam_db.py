# REVISAR LA EJECUCION DE ESTE SCRIPT EN EL PIPELINE, DEBERIA IR AL INICIO DE TODO

from collections import defaultdict
import sqlite3
import pickle

def generate_ligand_pfam_db(chembl_db):
    connection = sqlite3.connect(chembl_db)
    cursor = connection.cursor()

    query = '''select distinct target_dictionary.chembl_id, domains.source_domain_id
    from target_dictionary join
    target_components on target_dictionary.tid = target_components.tid join
    component_domains on target_components.component_id = component_domains.component_id join
    domains on component_domains.domain_id = domains.domain_id
    where target_dictionary.target_type = 'SINGLE PROTEIN';'''
    cursor.execute(query)
    result = cursor.fetchall()
    
    prot_pfam_dict = defaultdict(list)
    for r in result:
        if r[1] not in prot_pfam_dict[r[0]]:
            prot_pfam_dict[r[0]].append(r[1])
    
    return dict(prot_pfam_dict)

if __name__ == '__main__':
	
    import argparse
    import sys
    import os
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-db_dir","--database_directory",help='Path to save ChEMBL ligand-PFAM database',required = True,type=str)
    parser.add_argument("-cdb","--chembl_database",help='Path where local ChEMBL db is located',default='./db/ChEMBL/chembl_31.db',type=str)  

		    
    args = parser.parse_args()
    
    ligand_pfam_db = generate_ligand_pfam_db(args.chembl_database)
    
    
    with open(args.database_directory+'/pfam_target_db_chembl.pkl','wb') as r:
        pickle.dump(ligand_pfam_db,r)
