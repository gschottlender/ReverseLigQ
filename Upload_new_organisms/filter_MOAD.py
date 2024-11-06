import json
import argparse

peptide = ["ALA","LYS","VAL","ARG","GLY","CYS","TYR", "PHE","LEU","ILE","MET",
 "SER","PRO","THR","HIS","ASN","GLN","GLU","ASP","TRP"]
family_nad = ["NAD","NAI","NDC","TAP","ADJ","NAJ","NDO","ZID","CAN","NAP","NDP",
 "CND","NAQ","NHD","DND","NAX","NHO","NAC","NDB","ODP","NAE","NBP","PAD","NAH","NDA","SND"]
family_fad = ["FAD","FMN","6FA","FNS","FAA","MGD","FAB","RFL","FAE","FAS","FDA",
  "FMA"]
nucl =["AMP","GDP","UDP","ADP","GNP","UTP","ANP", "GTP","PSU","ATP","2GP",
 "CMP","TMP","CDP","TDP","CTP","TTP","GMP","UMP"]
mol_junk = ["UNK", "UNX" "BMA","FUC","MAN","POP","BOG","GAL","MES","PYR","C8E","GLC","MPD",
 "SPM","CIT","GOL","MYR","TRS","CRY","HED","NAG","XYS","DTT","LDA","NGA","EPE","LI1","PEG","F6P","MAL","PG4"]

def moad_db(input_file):
    with open(input_file) as a:
         MOAD = json.load(a)
    return MOAD

def invalid_list (moad_dict, limit=0.2):
    invalids = []
    for ligand,data in moad_dict.items():
        invalid,valid = 0,0
        for pdb in data ["pdbs"]:
            for residue in pdb["residues"]:
                if residue["status"]== "valid":
                    valid = valid +1
                else:
                    invalid = invalid +1
        if valid/(valid + invalid ) <= limit:
            invalids.append(ligand)
    return invalids


def ligands(input_file):
    ligands_list= open(input_file)
    read_ligands=ligands_list.readlines()
    ligands = [line.strip().split("\t") for line in read_ligands if line.strip()]
    ligands_list.close()
    return ligands

def true_ligands(MOAD,ligands,limit,aa,nad,fad,nucleotides,junk):
    invalids=invalid_list(MOAD, limit) + aa + nad + fad + nucleotides + junk
    valid_ligands_key={}
    valid_ligands =[]
    for ligand_tuple in ligands:
        key= "_".join(ligand_tuple)
        if key not in valid_ligands_key:
         valid_ligands_key[key]=1
         if ligand_tuple[0].strip() not in invalids:
             valid_ligands.append(ligand_tuple)
    return valid_ligands


def parse_arguments():

    parser = argparse.ArgumentParser(description='Extract valid ligands from MOAD DATABASE')
    parser.add_argument("-l", '--ligands_list', default=None, help="Input file should have ligands ID list")
    parser.add_argument("-db", '--moad_database',  default="MOAD.json")
    parser.add_argument("-f1", '--filter_aa', action="store_false")
    parser.add_argument("-f2", '--filter_nad', action="store_true")
    parser.add_argument("-f3", '--filter_fad', action="store_true")
    parser.add_argument("-f4", '--filter_nucl', action="store_true")
    parser.add_argument("-f5", '--filter_junk', action="store_true")
    parser.add_argument('--limit', default = 0.2)

    return parser

def filter_ligands(ligands,moad_json,limit=0.2,aa=False,nad=False,fad=False,nucleotides=False,junk=True):
    aa = peptide if aa else []
    nad = family_nad if nad else []
    fad = family_fad if fad else []
    nucleotides = nucl if nucl else []
    junk = mol_junk if junk else []
    return true_ligands (moad_json,ligands,limit,aa,nad,fad,nucleotides,junk)
    

def main():
   
    parser=parse_arguments()
    args=parser.parse_args()
    aa = peptide if args.filter_aa else []
    nad = family_nad if args.filter_nad else []
    fad = family_fad if args.filter_fad else []
    nucleotides = nucl if args.filter_nucl else []
    junk = mol_junk if args.filter_junk else []

    unique_valid_ligands = true_ligands(moad_db(args.moad_database), ligands(args.ligands_list), args.limit,aa,nad,fad,nucleotides,junk)
    for ligand_tuple in unique_valid_ligands:
        print("\t".join(ligand_tuple))
    return 0


if __name__=='__main__':
    main()
