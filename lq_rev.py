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
# Dropdowns for selecting arguments
st.image('./Scheme.jpg')

organism = organisms_dict[st.selectbox('Select Organism', organism_options)]
smiles_inp = st.text_input('Enter compound in SMILES format', 'CCCCCCOCCCCO')
threshold = st.number_input('Define the Tanimoto Index threshold', value=0.4)
# Info
st.caption("Citation:") 
st.caption('''Schottlender, G., Prieto, J. M., Palumbo, M. C., Castello, F. A., Serral, F., Sosa, E. J., et al. (2022). From drugs to targets: reverse engineering the virtual screening process on a proteomic scale. Front. Drug Discov. 2:969983. doi: 10.3389/fddsv.2022.969983''')
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
