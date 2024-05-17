import pickle
from collections import defaultdict
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem

def remove_repetitions(smiles_db,ligands_fams_db):
    smiles_list = []
    smiles_merged = defaultdict(list)
    ligands_fams_merged = defaultdict(list)
    
    for s in smiles_db:
        if smiles_db[s] not in smiles_list:
            smiles_list.append(smiles_db[s])
            smiles_merged[s] = smiles_db[s]
            ligands_fams_merged[s] = ligands_fams_db[s]
    
    return dict(smiles_merged),dict(ligands_fams_merged)
    

if __name__ == '__main__':
	
    import argparse
    import sys
        
    parser = argparse.ArgumentParser()

    parser.add_argument("-org","--organism_folder",help='Folder where results for the same organism are saved, and where ChEMBL and PDB databases from the organism are located',required=True,type=str)  

		    
    args = parser.parse_args()
    
    with open(args.organism_folder+'/ligands_fams_dict_chembl.pkl','rb') as r:
        ligs_fams_chembl = pickle.load(r)
        
    with open(args.organism_folder+'/ligands_smiles_dict_chembl.pkl','rb') as s:
        smiles_chembl = pickle.load(s)

    with open(args.organism_folder+'/ligands_fams_dict_pdb.pkl','rb') as u:
        ligs_fams_pdb = pickle.load(u)

    with open(args.organism_folder+'/ligands_smiles_dict_pdb.pkl','rb') as v:
        smiles_pdb = pickle.load(v)
    
    total_smiles = smiles_pdb | smiles_chembl
    total_ligands_fams = ligs_fams_pdb | ligs_fams_chembl
    
    smiles_merged,ligands_fams_merged = remove_repetitions(total_smiles,total_ligands_fams)
    
    fps_merged = {c:AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(smiles_merged[c]),2,1024) for c in smiles_merged}
    
    with open(args.organism_folder+'/ligands_fams_dict.pkl','wb') as w:
        pickle.dump(ligands_fams_merged,w)
    with open(args.organism_folder+'/ligands_smiles_dict.pkl','wb') as x:
        pickle.dump(smiles_merged,x)
    with open(args.organism_folder+'/ligands_fps_dict.pkl','wb') as y:
	    pickle.dump(fps_merged,y)
	

