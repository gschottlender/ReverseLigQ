#!/usr/bin/env python

from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
import pandas as pd
from collections import defaultdict
import pickle
import time
import argparse

def prot_from_multidomains(fam,dicc_familias):

    search_res = [dom for f in fam[1:-1].split(', ') for dom in dicc_familias[f]]  
    return list(set(search_res))


def binding_domain_search(query_compound,fps,dicc_ligandos_familias):
    comp_fps = AllChem.GetMorganFingerprintAsBitVect((MolFromSmiles(query_compound)),2,1024)

    # calculo similaridad de compuestos del set de datos con el de prueba, 
    # descarto aquellos con similaridad menor a 0.4
    ranking_comp_parecidos ={comp:DataStructs.FingerprintSimilarity(comp_fps,fps[comp]) for comp in fps if DataStructs.FingerprintSimilarity(comp_fps,fps[comp])>=threshold}
    # ordeno de mayor a menor
    ranking_comp_parecidos = {k:ranking_comp_parecidos[k] for k in sorted(ranking_comp_parecidos, key=ranking_comp_parecidos.get, reverse=True)}

    similaridades_fams_candidatas,compuestos_fams,compuestos_tanimoto = defaultdict(int),defaultdict(list),defaultdict(list)

    # Agregar el compuesto mas parecido que une a cada familia y agregar otra salida para resultados con multidominios
    # (pfam multiples + proteinas posibles)
    #results = pd.DataFrame(columns=['Compound name','Tanimoto Similarity with query','Known Binding Pfam domains','Smiles'])
    results = pd.DataFrame({'Compound name':['Query ligand'],'Tanimoto Similarity with query':['-'],'Known Binding Pfam domains':['-'],'Smiles':[comp_prueba]})
    #results = pd.concat([results, res], ignore_index=True)

    for comp in ranking_comp_parecidos:

        similaridad = ranking_comp_parecidos[comp]
        compuestos_tanimoto[comp] = similaridad

        familias_candidatas = dicc_ligandos_familias[comp]
        
        res = pd.DataFrame([{'Compound name':comp,'Tanimoto Similarity with query':round(similaridad,2),'Known Binding Pfam domains':', '.join(dicc_ligandos_familias[comp]),'Smiles':dicc_smiles[comp]}])
        results = pd.concat([results, res], ignore_index=True)

        for fam in familias_candidatas:


            if fam not in similaridades_fams_candidatas:
                similaridades_fams_candidatas[fam] = similaridad


            if comp not in compuestos_fams[fam]:
                compuestos_fams[fam].append(comp)
            
            
    similaridades_fams_candidatas = {k:similaridades_fams_candidatas[k] for k in sorted(similaridades_fams_candidatas, key=similaridades_fams_candidatas.get, reverse=True)}

    PandasTools.AddMoleculeColumnToFrame(results, smilesCol='Smiles')
    del(results['Smiles'])
    results = results.reindex(columns=['Compound name','ROMol','Tanimoto Similarity with query','Known Binding Pfam domains'])
    results.rename(columns={'ROMol':'Chemical structure'},inplace = True)

    return compuestos_fams, similaridades_fams_candidatas, results

def results_tables(similaridades_fams_candidatas,dicc_familias,compuestos_fams,protein_descriptions):
    
    prot_candidatas_ampliadas = pd.DataFrame(columns = ['Pfam','Protein (Uniprot ID)','Gene','Description','Most similar Binding-compound Tanimoto Similarity','Domain-binding similar ligands'])
    for f in similaridades_fams_candidatas:
        if f.startswith('('):
            fams = f[1:-1].split(', ')
        else:
            fams = [f]
        for fam in fams:
            for prot in list(dicc_familias.get(fam,'-')):
                candidatas = pd.DataFrame([{'Pfam':f,'Protein (Uniprot ID)':prot,'Gene':protein_descriptions['gene'].get(prot,'No data'),'Description':protein_descriptions['description'].get(prot,'-'),'Most similar Binding-compound Tanimoto Similarity':round(similaridades_fams_candidatas[f],2),'Domain-binding similar ligands':', '.join(compuestos_fams[f])}])
                prot_candidatas_ampliadas = pd.concat([prot_candidatas_ampliadas, candidatas], ignore_index=True)
        
    prot_candidatas = prot_candidatas_ampliadas.drop(prot_candidatas_ampliadas[prot_candidatas_ampliadas["Pfam"].str.startswith("(")].index)
    
    prot_candidatas.drop_duplicates(subset='Protein (Uniprot ID)', inplace=True)
    prot_candidatas_ampliadas.drop_duplicates(subset='Protein (Uniprot ID)',inplace=True)
    prot_candidatas_ampliadas = prot_candidatas_ampliadas[prot_candidatas_ampliadas['Protein (Uniprot ID)'] != "-"]

    return prot_candidatas, prot_candidatas_ampliadas
    
    
if __name__ == '__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-org","--organism", help="Select organism number: 1- Bartonella bacilliformis, 2- Klebsiella pneumoniae, 3- Mycobacterium tuberculosis, 4- Trypanosoma cruzi, 5- Staphylococcus aureus RF122, 6- Streptococcus uberis 0140J, 7- Enterococcus faecium, 8- Escherichia coli MG1655, 9- Streptococcus agalactiae NEM316, 10- Pseudomonas syringae, 11- DENV (dengue virus), 12- SARS-CoV-2, 13- Xanthomonas translucens",type=int,choices=range(1,15))
    parser.add_argument("--custom_organism",help="Directory with the databases of a new organism")
    parser.add_argument("-s","--smiles_query", help="Paste smiles of the query ligand",required = True,type=str)
    parser.add_argument("-t","--threshold",help='Select T.I. similar compound threshold for candidate domain search',type=float,default=0.4)
    parser.add_argument("-d","--db_dir",help='Full directory path of organism databases',type=str,default='./Organisms/')
    parser.add_argument("-o","--output_dir",help='Full directory path of output files',type=str,default='./')
    args = parser.parse_args()


    organismos = {1:'Bartonella bacilliformis',2:'Klebsiella pneumoniae',3:'Mycobacterium tuberculosis',4:'Trypanosoma cruzi',
                 5:'Staphylococcus aureus RF122',6:'Streptococcus uberis 0140J',7:'Enterococcus faecium',
                  8:'Escherichia coli MG1655',9:'Streptococcus agalactiae NEM316',10:'Pseudomonas syringae',
                 11: 'DENV (dengue virus)',12:'SARS-CoV-2',13:'Homo sapiens'}
    
    assert args.organism or args.custom_organism, "An organism to search is required"

    if args.custom_organism:
        directorio = args.custom_organism
    else:
        directorio = f'{args.db_dir}/{args.organism}'
        organismo = args.organism
    threshold = args.threshold
    comp_prueba = args.smiles_query
    outdir = args.output_dir
    
    ### Abrir diccionario ligandos - familias de union
    with open(directorio+'/ligands_fams_dict.pkl','rb') as r:
        dicc_ligandos_familias = pickle.load(r)

    # Abrir diccionario de familias y proteinas
    with open(directorio+'/fam_prot_dict.pkl','rb') as t:
        dicc_familias = pickle.load(t)

    # Armar fps del set de los compuestos, con ID, y de smiles para visualizar las moleculas
    with open(directorio+'/ligands_fps_dict.pkl','rb') as w:
        fps = pickle.load(w)

    with open(directorio+'/ligands_smiles_dict.pkl','rb') as x:
        dicc_smiles = pickle.load(x)

    # Cargar descripciones de proteinas
    with open('./Organisms/prot_descriptions.pkl','rb') as p:
        protein_descriptions = pickle.load(p)

    if args.custom_organism:
        print('Selected organism: Custom organism')
    else:
        print('Selected organism: '+organismos[organismo]+'\n')

    compuestos_fams, similaridades_fams_candidatas, results = binding_domain_search(comp_prueba,fps,dicc_ligandos_familias)
    
    prot_candidatas, prot_candidatas_ampliadas = results_tables(similaridades_fams_candidatas,dicc_familias,compuestos_fams,protein_descriptions)
    
    PandasTools.SaveXlsxFromFrame(results,outdir+'similar_compound_search.xlsx','Chemical structure')    
    prot_candidatas.to_csv(outdir+'candidate_targets.csv',index=False)
    prot_candidatas_ampliadas.to_csv(outdir+'candidate_targets_full_result.csv',index=False)
