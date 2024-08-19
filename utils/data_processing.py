import pandas as pd
import numpy as np
import streamlit as st
import requests
import json

from sklearn.feature_selection import VarianceThreshold




def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    input['pIC50'] = input['pIC50'].astype(float) 
    x = input.drop(columns = 'standard_value_norm')

    return x



def norm_value(input):
    
    input = input[input['standard_value'].astype(float) != 0]
    
    norm = []

    for j in input['standard_value']:
        i = float(j)
        if i > 100000000:
          i = 100000000
        norm.append(i)

    input['standard_value_norm'] = norm
    x = input.drop(columns = 'standard_value')

    return x

def clean_smiles(df):
    df_no_smiles = df.drop(columns = 'canonical_smiles')
    smiles = []

    for i in df.canonical_smiles.tolist():

        cpd = str(i).split('.')
        cpd_longest = max(cpd, key=len)
        smiles.append(cpd_longest)

    smiles = pd.Series(smiles, name = 'canonical_smiles', index=df_no_smiles)
    df_clean_smiles = pd.concat([df_no_smiles, smiles], axis=1)
    return df_clean_smiles


def label_bioactivity(df_selected):
    
    bioactivity_threshold = []
        
    for i in df_selected.standard_value:
        if float(i) >= 10000:
            bioactivity_threshold.append("inativo")
        elif float(i) <= 1000:
            bioactivity_threshold.append("ativo")
        else:
            bioactivity_threshold.append("intermediário")
    bioactivity_class = pd.Series(bioactivity_threshold, name='class', index = df_selected.index)
    
    
    
    df_labeled = pd.concat([df_selected, bioactivity_class], axis=1)
    return df_labeled


def remove_low_variance(input_data, threshold=0.1):
    
    try:
        selection = VarianceThreshold(threshold)
        selection.fit(input_data)
        return input_data[input_data.columns[selection.get_support(indices=True)]]
    except Exception as e:
        st.error('Erro na remoção de baixa variância: {e}')
        return pd.DataFrame()
    

def convert_ugml_nm(df):
   
    converted_values = []
    df_ugml = df[df['standard_units'] == 'ug.mL-1']

    df = df.drop(df_ugml.index)

    for index, row in df_ugml.iterrows():
        
        concentration_g_l = float(row['standard_value']) * 1e-3
        molarity_m = concentration_g_l / float(row['MW'])        
        concentration_nm = molarity_m * 1e9
        converted_values.append(concentration_nm)

    df_ugml.drop('standard_value', axis=1)
    df_ugml.loc[:, 'standard_value'] = converted_values
    df_ugml['standard_value'] = df_ugml['standard_value'].astype(object)

    df = pd.concat([df, df_ugml], axis=0)
    

    return df

def classify_compound(df):

    try:
    
        classes = []
        for smiles in df.canonical_smiles:
            response = requests.get(f'https://npclassifier.gnps2.org/classify?smiles={smiles}', timeout=50)
            
            if response.status_code != 200:
                st.error(f'Falha no API request - Status: {response.status_code}')
            
            
            data_json = response.json()
            
            class_name = data_json['class_results']
            
            classes.append(class_name)

        classes_series = pd.Series(classes, name = 'compound_class', index=df)
        df_combined = pd.concat([df, classes_series], axis=1)
        return df_combined
    except Exception as e:
        st.error(f'Erro na classificação das moléculas: {e}')
        return df

