# Import necessary libraries
from cProfile import label
import streamlit as st
import pubchempy as pcp
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import *
from rdkit import Chem
from rdkit.Chem import AllChem

# Interface and User Input
st.write("Test App For Tyrosine Kinase Screening")
st.write("The current performance of this screening app can be seen below")

# Read and Draw the Metrics Plot
df_50 = pd.read_csv('RTK_RF_50_Report.csv')
df_50 = df_50.astype({'ChemblID':'category', 'prot_name':'category'})

sns.set_theme(style='whitegrid')
fig, ax = plt.subplots(figsize = (6,15))
b = sns.barplot(y='prot_name', x='F1', data=df_50)
b.set_xlabel("Algorithm F1 Score", fontsize = 20)
b.set_ylabel("Protein Names", fontsize = 20)
#ax.set(title='Prediction Metrics',ylabel="Protein Names", xlabel="Algorithm F1 Score")
sns.despine(left=True, bottom=True)
st.pyplot(fig)

# Search Interface
st.write("Please input the compound you would like to search")
cmpd_name = st.text_input(label='Compound Name', value="Glucose") # Default compound is glucose
if cmpd_name is not None:
    search_id = cmpd_name
else:
    st.stop()
c = pcp.get_compounds(search_id, 'name')
name = [x.synonyms[0] for x in c]
option = st.selectbox('Please pick one of the results from below', name)
'You selected: ', option

# Analysis button
if st.button("Run Analysis"):
    # Search back the option to produce the Canonical Smiles
    c = pcp.get_compounds(option, 'name')
    can_smi = [x.canonical_smiles for x in c]
    can_smi

    # Convert the canonical smiles to ECFP6 Fingerprints
    # Custom function to convert Mol from Smiles
    def molsmile (smiles):
        moldata = []
        for m in smiles: # iterate through iterables in smiles
            mol = Chem.MolFromSmiles(m) # convert smiles to mols
            moldata.append(mol)
        return moldata
    mol_list = molsmile(can_smi)
    bit_num = 1024
    fingerprint_ECFP6 = [AllChem.GetMorganFingerprintAsBitVect(x,3, nBits = bit_num) for x in mol_list] # Create efcp6 fingerprints
    fingerprint_ECFP6_lists = [list(l) for l in fingerprint_ECFP6] # turn fingerprints to list
    ecfp6_name = [f'ecfp_bit_{i}' for i in range(bit_num)] # create bit name
    fingerprint_ECFP6_df = pd.DataFrame(fingerprint_ECFP6_lists, columns=ecfp6_name) # create efcp dataframe

    # Initiate the machine learning algorithm and model list
    algorithm_id = 'rf'
    IC50_threshold = 50
    df_target = pd.read_csv(f'RTK_RF_{IC50_threshold}_Report.csv')
    target_list = df_target.ChemblID.unique()
    target_list = target_list.tolist()

    # Retrieve the dictionary of protein names
    df_prot_names = pd.read_csv('RTK_RF_50_Report.csv')
    names = dict(zip(df_prot_names['ChemblID'], df_prot_names['prot_name']))
    #names

    # Search the models
    name_dict = [{'Compound Name':option}]
    df_names = pd.DataFrame.from_dict(name_dict)
    predict_summary = df_names
    for n in target_list:
        target_id = n
        model = load_model(f'models/{algorithm_id}_classifier_{IC50_threshold}_high/{target_id}_{algorithm_id}/{algorithm_id}_pipeline_{target_id}')
        df_prediction = predict_model(model, raw_score= True, data=fingerprint_ECFP6_df)
        df_prediction = df_prediction.rename(columns={'Score_1':target_id})
        predict_summary = pd.concat([predict_summary, df_prediction[target_id]], axis=1)
    predict_summary = predict_summary.rename(columns=names)
    predict_summary = predict_summary.melt(id_vars=["Compound Name"], var_name='Protein Name', value_name='Inhibitor Probability')

    # Results
    st.write("The table and chart below outline the probability of the searched compound functioning as an inhibitor for the tyrosine kinases panel")
    st.table(predict_summary) # Draw the result table

    # Draw the plot
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize = (6,15))
    b = sns.barplot(y='Protein Name', x='Inhibitor Probability', data=predict_summary)
    b.axes.set_title(f'{option}', fontsize = 15)
    b.set_xlabel("Compound's Probability of Inhibition", fontsize = 10)
    b.set_ylabel("Tyrosin Kinase Names", fontsize = 10)
    #ax.set(title='Prediction Metrics',ylabel="Protein Names", xlabel="Algorithm F1 Score")
    sns.despine(left=True, bottom=True)
    st.pyplot(fig)

else:
    st.stop()

