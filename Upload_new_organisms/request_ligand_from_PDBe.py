import requests
import json
from json import JSONDecodeError
from requests.adapters import HTTPAdapter 
from requests.packages.urllib3.util.retry import Retry
sources = {x.split()[0]:x.split()[1] for x in """ 1	chembl
2	drugbank
3	pdb
4	gtopdb
5	pubchem_dotf
6	kegg_ligand
7	chebi
8	nih_ncc
9	zinc
10	emolecules
11	ibm
12	atlas
14	fdasrs
15	surechembl
17	pharmgkb
18	hmdb
20	selleck
21	pubchem_tpharma
22	pubchem
23	mcule
24	nmrshiftdb2
25	lincs
26	actor
27	recon
28	molport
29	nikkaji
31	bindingdb
32	comptox
33	lipidmaps
34	drugcentral
35	carotenoiddb
36	metabolights
37	brenda
38	rhea
39	chemicalbook
40	dailymed_old
41	swisslipids
45	dailymed
46	clinicaltrials
47	rxnorm""".split("\n")}

def pdb_ligand_data(ligands):
   retry_strategy = Retry(total=3, status_forcelist=[429, 500, 502, 503, 504], 
                       method_whitelist=["HEAD", "GET", "OPTIONS", "POST"])
   adapter = HTTPAdapter(max_retries=retry_strategy)
   http = requests.Session() 
   http.mount("https://", adapter) 
   http.mount("http://", adapter)
   r = http.post("https://www.ebi.ac.uk/pdbe/api/pdb/compound/summary/",data=",".join(ligands))
   if r.ok:
       data =  r.json()  
       new_data = []
       for k,v in data.items():
           r = v[0]            
           r2 = {"pdb_ligand":k}
           if len(r["smiles"]):
               r2["smiles"] = r["smiles"][0]["name"]
           if "chembl_id" in r:
               r2["chembl_id"] = r["chembl_id"]            
           new_data.append(r2)
       
       return new_data
   raise Exception(r.text)

def search_chembl(chembl_id):
   retry_strategy = Retry(total=3, status_forcelist=[429, 500, 502, 503, 504], 
                       method_whitelist=["HEAD", "GET", "OPTIONS", "POST"])
   adapter = HTTPAdapter(max_retries=retry_strategy)
   http = requests.Session() 
   http.mount("https://", adapter) 
   http.mount("http://", adapter)
   r = http.get(f"https://www.ebi.ac.uk/unichem/rest/src_compound_id_all/{chembl_id}/1")
   if r.ok:
       data =  r.json()
       for x in data:
           x["src_name"] = sources[x["src_id"]]
       return data
   raise Exception(r.text)

def pdb_ligand_data_batch(ligands,n=400):    
   ligs_chunks = [ligands[i:i + n] for i in range(0, len(ligands), n)]
   all_pdb_ligands = reduce(list.__add__, [pdb_ligand_data( chunk ) for chunk in tqdm(ligs_chunks)])    
   return all_pdb_ligands

def ligands_from_pdbs(pdbs):
    pdbs_count=len(pdbs)
    response_total={}
    pdbs_per_request=100
    retry_strategy = Retry(total=3, status_forcelist=[429, 500, 502, 503, 504], 
                       method_whitelist=["HEAD", "GET", "OPTIONS", "POST"])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session() 
    http.mount("https://", adapter) 
    http.mount("http://", adapter)
    for pdb_index in range(0,pdbs_count,pdbs_per_request):
        final=pdb_index+pdbs_per_request
        if final>pdbs_count:
            final=pdbs_count
        pdb_to_request=pdbs[pdb_index:final]
        pdb_to_request=",".join(pdb_to_request).lower()
        response = http.post('https://www.ebi.ac.uk/pdbe/api/pdb/entry/binding_sites/',data = pdb_to_request)
        try:
            response_total.update(response.json())
        except JSONDecodeError:
            print(pdb_to_request)
            print(response.text)
            raise 
    return response_total
#def ligands_from_pdbs(pdbs):
#    pdbs_count=len(pdbs)
#    response_total={}
#    pdbs_per_request=100
#    for pdb_index in range(0,pdbs_count,pdbs_per_request):
#        final=pdb_index+pdbs_per_request
#        if final>pdbs_count:
#            final=pdbs_count
#        pdb_to_request=pdbs[pdb_index:final]
#        pdb_to_request=",".join(pdb_to_request).lower()
#        response = requests.post('https://www.ebi.ac.uk/pdbe/api/pdb/entry/binding_sites/',data = pdb_to_request)
#        try:         
#            response_total.update(response.json())
#        except JSONDecodeError:
#          print(pdb_to_request)
#          print(response.text)
#          raise 
#    return response_total
