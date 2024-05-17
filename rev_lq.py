import base64
import io

import streamlit as st
import subprocess
import pandas as pd

import rdkit
import rdkit.Chem
import rdkit.Chem.Draw

def smi_to_png(smi: str) -> str:
    """Returns molecular image as data URI."""
    mol = rdkit.Chem.MolFromSmiles(smi)
    pil_image = rdkit.Chem.Draw.MolToImage(mol)

    with io.BytesIO() as buffer:
        pil_image.save(buffer, "png")
        data = base64.encodebytes(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{data}"

# Options for arguments
organisms_dict = {'Bartonella bacilliformis':1,'Klebsiella pneumoniae':2,'Mycobacterium tuberculosis':3,'Trypanosoma cruzi':4,
                 'Staphylococcus aureus RF122':5,'Streptococcus uberis 0140J':6,'Enterococcus faecium':7,
                  'Escherichia coli MG1655':8,'Streptococcus agalactiae NEM316':9,'Pseudomonas syringae':10,
                 'DENV (dengue virus)':11,'SARS-CoV-2':12,'Homo sapiens':13}

organism_options = list(organisms_dict.keys())

# Streamlit UI elements
st.title('Reverse LigQ')
st.subheader('Unsupervised learning tool for Compound Target Prediction based on chemical similarity')
st.caption('Provides potential target proteins for a given query compound, prioritized based on their similarity (measured using the Tanimoto Index) to compounds with known binding sites.')
# Dropdowns for selecting arguments
st.image('./Scheme.jpg')

organism = organisms_dict[st.selectbox('Select Organism', organism_options)]
smiles_inp = st.text_input('Enter query compound in SMILES format', 'CCCCO')
threshold = st.number_input('Define the Tanimoto Index threshold (from 0.2 to 1.0, higher values correspond to more similar compounds)', value=0.5,min_value=0.2,max_value=1.0)
only_description = st.selectbox("Only show proteins with description:", ("Yes", "No"))

# Button to trigger the Python script
if st.button('Run Candidate Target Search'):
    # Call the Python script with selected arguments
    command = ['python3', 'compound_test.py', '-org', str(organism), '-s', smiles_inp,'-t',str(threshold)]

    result = subprocess.run(command, capture_output=True, text=True)
    
    # Open result
    result_df = pd.read_csv('candidate_targets_full_result.csv')
    if only_description == 'Yes':
        result_df = result_df.loc[result_df['Description'] != '-']
    
    sim_search = pd.read_excel('similar_compound_search.xlsx', engine='openpyxl')
    sim_search = sim_search.drop(['Unnamed: 0'],axis=1)
    sim_search["Smiles"] = sim_search["Smiles"].apply(smi_to_png)
    sim_search = sim_search.rename(columns={'Smiles':'Structure (click to enlarge)'})
    
    tab1,tab2 = st.tabs(["Candidate Target Results", "Compound Similarity Results"])
    tab1.dataframe(result_df,sep=',')
    tab2.dataframe(sim_search,column_config={"Structure (click to enlarge)": st.column_config.ImageColumn()})
st.caption("Algorithm detailed information and evaluation are provided in the article:")
st.markdown(f" [From drugs to targets: Reverse engineering the virtual screening process on a proteomic scale](https://www.frontiersin.org/articles/10.3389/fddsv.2022.969983): Schottlender G, Prieto JM, Palumbo MC, Castello FA, Serral F, Sosa EJ, Turjanski AG, Martì MA and Fernández Do Porto D")
st.markdown(f"[GitHub site](https://github.com/gschottlender/ReverseLigQ)")