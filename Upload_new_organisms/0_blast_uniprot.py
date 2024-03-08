from collections import defaultdict
import pickle
import time
import os

# Este script puede ser opcional, sirve para obtener las ids de uniprot de las proteínas del proteoma que está en la db de target

# Armar base de datos a partir de todas las proteínas del organismo descargadas de Uniprot

# Parametro 1
db_uniprot = './dbs/uniprot-mycobacterium+tuberculosis.fasta'
# Parametro 2
proteome = './dbs/proteoma_mtb_target.faa'
os.system(f'makeblastdb -in {db} -dbtype prot')

# Primer output: resultados de blast
output_name = 'blast_results.txt'
os.system(f'blastp -query {proteome} -db {db_uniprot} -outfmt 7 -out {output_name})


# Levantar resultado de blast

with open(output_name) as handle:
    resultados = handle.readlines()

i = 0
dicc_hits = defaultdict(list)
hits = False
for resultado in resultados:
    if resultado.endswith('hits found\n'):
        hits = True
        i+=1
    if resultado.endswith('2.9.0+\n'):
        hits = False
    if hits == True:
        if len(resultado.split('\t')) == 12:
            dicc_hits[i].append(resultado.split('\t'))
dicc_hits = dict(dicc_hits)  

# Elijo el resultados que superen 90% de identidad, me quedo con la ids de Uniprot
# En caso de haber un sp (swiss prot) elijo el de score mas alto,
# En caso de no haber sp, me quedo con el score más alto de Trembl.

genes_ids_uniprot = defaultdict(list)
ids_uniprot_genes = defaultdict(list)

for query in dicc_hits:
    resultados_sp = []
    sp = False
    for hit in dicc_hits[query]:
        
        
        if float(hit[2]) > 90.0:
            if hit[1].split('|')[0] == 'sp' and sp == False:
                genes_ids_uniprot[hit[0]] = hit[1].split('|')[1]
                ids_uniprot_genes[hit[1].split('|')[1]] = hit[0]
                sp = True
    if len(genes_ids_uniprot[hit[0]]) == 0:
        if float(dicc_hits[query][0][2]) > 90.0:
            genes_ids_uniprot[dicc_hits[query][0][0]] = dicc_hits[query][0][1].split('|')[1]
            ids_uniprot_genes[dicc_hits[query][0][1].split('|')[1]] = dicc_hits[query][0][0]
    
genes_ids_uniprot = dict(genes_ids_uniprot)

# Output: diccionario de conversion nomenclatura de target a Uniprot 
w = open('dicc_genes_uniprot_ids.pkl','wb')
pickle.dump(dict(genes_ids_uniprot),w)
w.close()

