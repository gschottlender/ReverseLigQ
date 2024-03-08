from rdkit.Chem import MolFromSmiles
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from collections import defaultdict
import pickle

def par_to_list(l):
    return [f[1:-1].split(', ') if f.startswith('(') else [f] for f in l]
    
def list_to_fams(l):
    return [f[0] if len(f) == 1 else f'({", ".join(f)})' for f in l]

def int_l1_l2(l1,l2):
    ints = [list(set(l1).intersection(l2[i])) for i in range(len(l2)) if len(list(set(l1).intersection(l2[i])))>0]
    
    if len(ints) == 0:
        return l1
    else:
        return min(ints,key=len)

def reduce_possible_domains(result):    
    fams = par_to_list(result)
    f_curadas = []
    ints_evaluadas = []

    for f in fams:
        if len(f) > 1:
            ints = int_l1_l2(f,[f2 for f2 in fams if f2 != f])
            if ints not in f_curadas:
                f_curadas.append(ints)
        elif f not in f_curadas:
                f_curadas.append(f)
    return list_to_fams(f_curadas)

def curate_multidomains(target_fams,search_results):
    reduced_fams = []
    fs = par_to_list(target_fams)
    sr = par_to_list(search_results)
    for f in fs:
        f_red = int_l1_l2(f,sr)
        reduced_fams.append(f_red)
    return list_to_fams(reduced_fams)

def curate_chembl_db(ligs_fams_chembl,smiles_chembl,ligs_fams_pdb,smiles_pdb,threshold=0.4):
    
    curated_ligand_domains_db = defaultdict(list)
    
    compounds = list(smiles_chembl.keys())
    for compound in compounds:    
    
    # Buscar familias que se sabe que interactuan con el ligando:
        target_fams = ligs_fams_chembl[compound]


        multidomains = False
        for fam in target_fams:
            if fam.startswith('(PF'):
                multidomains = True

        if multidomains == False:
            curated_ligand_domains_db[compound] = target_fams
            continue
                
        search_results = []

        comp_fps = fps_chembl[compound]
        similarities ={c:DataStructs.FingerprintSimilarity(comp_fps,fps_pdb[c]) for c in fps_pdb}
        similarity_ranking = {k: similarities[k] for k in sorted(similarities, key=similarities.get, reverse=True) if similarities[k] > threshold}
        

        for comp in similarity_ranking:
        
            for fam in ligs_fams_pdb[comp]:
                if fam not in search_results:
                    search_results.append(fam)

        # Reemplazo los multidominios por el dominio correspondiente encontrado, si se puede


        curated_doms = reduce_possible_domains(reduce_possible_domains(curate_multidomains(target_fams,search_results)))                

        curated_ligand_domains_db[compound] = curated_doms
        


        
    return dict(curated_ligand_domains_db)
    
if __name__ == '__main__':
	
    import argparse
    import sys
        
    parser = argparse.ArgumentParser()

    parser.add_argument("-o","--organism_folder",help='Folder where results for the same organism are saved, and where ChEMBL and PDB databases from the organism are located',required=True,type=str)  
    parser.add_argument("-t","--threshold",help='Tanimoto Index threshold to search for similar compounds to curate binding domains',default=0.4,type=float)
		    
    args = parser.parse_args()
	    
	    # Uso el set de PDB para buscar el dominio correcto con el que interactuan los ligandos de ChEMBL
    with open(args.organism_folder+'/ligands_fams_dict_chembl_raw.pkl','rb') as r:
        ligs_fams_chembl = pickle.load(r)
        
    with open(args.organism_folder+'ligands_smiles_dict_chembl.pkl','rb') as s:
        smiles_chembl = pickle.load(s)

    with open(args.organism_folder+'/ligands_fams_dict_pdb.pkl','rb') as u:
        ligs_fams_pdb = pickle.load(u)

    with open(args.organism_folder+'ligands_smiles_dict_pdb.pkl','rb') as v:
        smiles_pdb = pickle.load(v)
        
    
    # Armar lista con fps e ids del set más grande (el que voy a usar para curar el set de ChEMBL)

    fps_pdb = {c:AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(smiles_pdb[c]),2,1024) for c in smiles_pdb}

# Armo lista de compuestos y diccionario de fps para el set de ChEMBL que quiero curar

    fps_chembl = {c:AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(smiles_chembl[c]),2,1024) for c in smiles_chembl}
    
    curated_db = curate_chembl_db(ligs_fams_chembl,smiles_chembl,ligs_fams_pdb,smiles_pdb,args.threshold)
    
    with open(args.organism_folder+'/ligands_fams_dict_chembl.pkl','wb') as t:
        pickle.dump(curated_db,t)


