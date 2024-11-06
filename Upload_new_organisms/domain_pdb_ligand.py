from ligand_from_pfam.request_ligand_from_PDBe import ligands_from_pdbs
import argparse
import sys
import pandas as pd

def pfam_mapping(pfam_pdb_mapping_handle):
    lines=pfam_pdb_mapping_handle.readlines()
    pfam_pdbs_dictionary={}
    for line in lines[1:]:
        line=line.split("\t")
        pdb=line[0]
        chain=line[1]
        # Saco las letras si es que las hay en la posicion
        position=''.join(i for i in line[2] if i.isdigit())+","+''.join(i for i in line[3] if i.isdigit())
        pfam=line[4].split(".")[0]
        if pfam not in pfam_pdbs_dictionary.keys():
            pfam_pdbs_dictionary[pfam]=[(pdb,chain,position)]
        else:
            list_aux=pfam_pdbs_dictionary[pfam]
            list_aux.append((pdb,chain,position))
            pfam_pdbs_dictionary[pfam]=list_aux

    return pfam_pdbs_dictionary

def request(pfam_entry, pfam_pdbs_dictionary):
    all_pdbs_of_pfams=[]
    for pfam in pfam_entry:
        try:
            pdbs_of_pfam_list=pfam_pdbs_dictionary[pfam]
        except:
            sys.stderr.write("Warning! pfam id not in pdb_pfam_mapping: "+pfam+"\n")
            continue
        all_pdbs_of_pfams=all_pdbs_of_pfams+pdbs_of_pfam_list
    pdbs_of_pfam=[pdb[0] for pdb in all_pdbs_of_pfams]
    sys.stderr.write("Warning! Number of pdbs to request: "+str(len(pdbs_of_pfam))+"\n")
    pdbs_of_pfam=list(set(pdbs_of_pfam))
    sys.stderr.write("Warning! Number of uniques pdbs to request: "+str(len(pdbs_of_pfam))+"\n")
    PDBe_dic=(ligands_from_pdbs(pdbs_of_pfam))

    return PDBe_dic

def pfam_pdb_ligand(pfam_entry, PDBe_dic, pfam_pdbs_dictionary):
    """
    pfam_entry: list of pfam domains
    PDBe_dic: binding PDB/API dict
    pfam_pdbs_dictionary: pfam domains = key , pdb list = value
    """
    ligands_list=[]
    for pfam in pfam_entry:
        try:
            pdbs_of_pfam_list=pfam_pdbs_dictionary[pfam]
        except:
            continue
        for pdb in pdbs_of_pfam_list:
            pdb_pfam_id=pdb[0].lower()
            pdb_pfam_chain=pdb[1]
            pdb_pfam_position=pdb[2]
            pdb_pfam_position_inicio=int(pdb_pfam_position.split(",")[0])
            pdb_pfam_position_final=int(pdb_pfam_position.split(",")[1])

            if (pdb_pfam_id not in PDBe_dic) or (len(PDBe_dic[pdb_pfam_id])==0):
            # Si esta en el PDBe pero esta vacio, es decir no tiene informacion sobre los ligandos del pdb
                sys.stderr.write("Warning! PDB with no data in PDBe: "+pdb_pfam_id+"\n")
                pass
            
            else:
                pdb_pdbe=PDBe_dic[pdb_pfam_id]
                for residues in pdb_pdbe:
                    site_residues=residues["site_residues"]
                    
                    author_insertion_code=residues.get("author_insertion_code","None")
                    # Si el details viene vacio. me  salteo la busqueda de ese pdb
                    if residues["details"]==None or residues["details"].split(" ")[0].lower()!="binding":
                        pass
                    else:
                        
                        #Example1: binding site for residue PO4 A 503
                        #Example2: binding site for Ligand residues SEP A 59 through GLY A 60 bound to SER A 58                        
                        #Example 1bkx: BINDING SITE FOR RESIDUE A A 351 --> uses 3
                        
                        pdb_pdbe_details=residues["details"].replace("Ligand", "").split(" ")
                        
                        if len(pdb_pdbe_details) < 5:
                            sys.stderr.write("Warning! PDB with wrong biding site: "+pdb_pfam_id+"\n")
                            
                            continue

                        else:
                             pdb_pdbe_details = pdb_pdbe_details[4]
                                
                                
                        #Example 4hpu: BINDING SITE FOR CHAIN I OF CAMP-DEPENDENT PROTEIN KINASE INHIBITOR ALPHA                                               
                        #Example 4ib5: BINDING SITE FOR CHAIN F OF CK2BETA-DERIVED CYCLIC PEPTIDE
                        if residues["details"].split(" ")[3] == "CHAIN" or residues["details"].split(" ")[4] == "Ligands" or residues["details"].split(" ")[4] == "residues"  or residues["details"].split(" ")[4] == "RESIDUES" or residues["details"].split(" ")[4] == "SUC" or residues["details"].split(" ")[4] == "SUP" or residues["details"].split(" ")[4] == "TRE" or residues["details"].split(" ")[4] == "CBI" or residues["details"].split(" ")[4] == "chain":
                            continue
                                                
                        if len(residues["details"].split(" ")) == 11:
                        # Puede haber dos ligandos por detail. Cuando pasa eso Tomo los dos
                            two_inOneDetail=True
                            two_inOneDetail_data=residues["details"].split(" ")[8]
                        else:
                            two_inOneDetail=False
                        
                        for site in site_residues:
                            # Tiene letras el ligando? Las saco si las hay
                            posicion_ligando=str(site["author_residue_number"])
                            
                            posicion_ligando=int(''.join(i for i in posicion_ligando if i.isdigit()))
                            
                                                   
                            if (site["chain_id"] == pdb_pfam_chain and posicion_ligando>=pdb_pfam_position_inicio and
                                                   posicion_ligando<=pdb_pfam_position_final):
                                
                                out=[pfam,pdb_pfam_chain,str(pdb_pfam_position_inicio),str(pdb_pfam_position_final),pdb_pfam_id,pdb_pdbe_details,site["chain_id"],str(posicion_ligando),author_insertion_code]
                                ligands_list.append(out)
#                                 sys.stderr.write(out+"\n")
#                                 ligands_list.append(pdb_pdbe_details)
                                if two_inOneDetail:
                                    out=[pfam,pdb_pfam_chain,str(pdb_pfam_position_inicio),str(pdb_pfam_position_final),pdb_pfam_id,two_inOneDetail_data,site["chain_id"],str(posicion_ligando),author_insertion_code]
                                    ligands_list.append(out)
#                                     sys.stderr.write(out+"\n")
#                                     ligands_list.append(two_inOneDetail_data)

        
    return pd.DataFrame(ligands_list,columns=["domain", "chain" ,"resid_start" ,"resid_end" , "pdb", "ligand", "ligand_chain", "ligand_resid", "insertion_code"])

def pfam_entry_handly(file):
    input=open(file,"r")
    lines=input.readlines()
    output=[]
    for line in lines:
        line=line.rstrip()
        if line[0]!="#":
            output.append(line)
    return output

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-f", '--pdb_pfam_mapping', default='pdb_pfam_mapping.txt', help="")
    parser.add_argument('-i','--pfam_input', default='pfam_entry.txt', help="")

    return parser

def ligands_from_domain(pfams, pfam_pdb_mapping_handle):
    pfam_pdbs_dictionary=pfam_mapping(pfam_pdb_mapping_handle)
    PDBe_dic=request(pfams,pfam_pdbs_dictionary)
    return pfam_pdb_ligand(pfams, PDBe_dic, pfam_pdbs_dictionary) 
    

def main():
    parser=parse_arguments()
    args=parser.parse_args()
    pfams=pfam_entry_handly(args.pfam_input)
    with open(args.pdb_pfam_mapping) as pdb_pfam_handle:
        ligands = ligands_from_domain(pfams, pdb_pfam_handle)
    print(ligands.to_csv())           
    
    
if __name__=='__main__':
    main()
