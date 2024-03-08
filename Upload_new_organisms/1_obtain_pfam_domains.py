from collections import defaultdict
import pickle
import numpy as np
import pandas as pd
import os

# Generar base de datos de proteinas agrupadas por familia pfam

# Requiere hmmer instalado, proteoma del organismo, base de datos PfamA de HMMs

def pfam_prot_dict(hmmer_results,evalue_max=1e-05):
    dicc_pfam_prot = defaultdict(list)
    for linea in hmmer_results[3:]:
        if not linea.startswith('#'):
            resultado = linea.split()

            if  eval(resultado[4])<evalue_max and resultado[2] not in dicc_pfam_prot[resultado[1].split('.')[0]]:
                dicc_pfam_prot[resultado[1].split('.')[0]].append(resultado[2])

    return dict(dicc_pfam_prot)

if __name__ == '__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--pfam_db", help="Pfam database .hmm path",required = True,type=str)    
    parser.add_argument("-i","--proteome_input",help='Organism proteome .fasta file path',required = True,type=str)
    parser.add_argument("-o","--organism_folder",help='Folder where results for the same organism are saved',required=True,type=str)
    #parser.add_argument("--evalue_limit",help='Higher e-value of each hit for assigning proteins to a Pfam family',default=1e-5,type=float)
    args = parser.parse_args()

    if not os.path.exists(args.organism_folder):
        os.makedirs(args.organism_folder)

    db_pfam = args.pfam_db

    proteome = args.proteome_input
    
    # Correr Hmmer para agrupar proteinas por dominios de Pfam
    
    command = f'hmmscan --cut_ga --tblout {args.organism_folder}/hmmer_results.txt {db_pfam} {proteome}'

    os.system(command)


# Abrir resultado Hmmer
    with open(args.organism_folder+'/hmmer_results.txt') as handle:
        hmmer_results = handle.readlines()
    
# Filtro porque el e-value sea menor a 1e-5 (el filtro por gathering cutoff ya se hizo con Hmmer)

# Armo diccionario de proteinas por familia

    dict_pfam_prot = pfam_prot_dict(hmmer_results)

# Output: diccionario de familias y proteinas

        
    with open(args.organism_folder+'/fam_prot_dict.pkl','wb') as s:
        pickle.dump(dict_pfam_prot,s)

