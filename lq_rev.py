import streamlit as st
import subprocess
import pandas as pd

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
# Info, link al paper
st.caption("The algorithm information and evaluation are provided in the article. Please cite if applicable:")
st.link_button('"From drugs to targets: Reverse engineering the virtual screening process on a proteomic scale"', "https://www.frontiersin.org/articles/10.3389/fddsv.2022.969983")

organism = organisms_dict[st.selectbox('Select Organism', organism_options)]
smiles_inp = st.text_input('Enter query compound in SMILES format', 'CCCCCCOCCCCO')
threshold = st.number_input('Define the Tanimoto Index threshold', value=0.4)

# Button to trigger the Python script
if st.button('Run Similarity Search'):
    # Call the Python script with selected arguments
    command = ['python3', 'compound_test.py', '-org', str(organism), '-s', smiles_inp,'-t',str(threshold)]

    result = subprocess.run(command, capture_output=True, text=True)
    
    # Open result
    result_df = pd.read_csv('candidate_targets_full_result.csv')
    sim_search = pd.read_excel('similar_compound_search.xlsx', engine='openpyxl')
    sim_search = sim_search.drop(['Unnamed: 0'],axis=1)
    # Display the DataFrame
    #st.write(result_df)

    tab1,tab2 = st.tabs(["Candidate Target Results", "Compound Similarity Results"])
    tab1.write(result_df)
    tab2.write(sim_search)

st.page_link("https://github.com/gschottlender/ReverseLigQ",label=':blue[Link to GitHub repository]')