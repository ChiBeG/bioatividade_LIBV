import streamlit as st 
import pandas as pd
from chembl_webresource_client.new_client import new_client


def search_target(search):
    try:
        target = new_client.target
        target_query = target.search(search)
        targets = pd.DataFrame.from_dict(target_query)
        return targets
    except Exception as e:
        st.error(f'Erro na busca do alvo: {e}')
        return pd.DataFrame()
    

def select_target(selected_index, targets):
    try:
        selected_index = int(selected_index)
        selected_target = targets.loc[selected_index]['target_chembl_id']
        activity = new_client.activity
        res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
        df = pd.DataFrame.from_dict(res)

        st.header("Dados das moléculas")
        st.write(df)
        medidas = df['standard_units'].value_counts()
        st.write("Medidas:")
        st.write(medidas)
        
        ## Temporário: filtrar apenas as entradas com medidas em nM

        df = nm_filter(df)

        st.write(df)
        medidas = df['standard_units'].value_counts()
        st.write("Medidas:")
        st.write(medidas)


        df_clean = df[df.standard_value.notna()]
        df[df.canonical_smiles.notna()]
        df_clean = df_clean.drop_duplicates(subset=['canonical_smiles'])

        selection = ['molecule_chembl_id','canonical_smiles','standard_value']
        df_selected = df_clean[selection]

        return df_selected
    
    except Exception as e:
        st.error(f'Erro na seleção do alvo: {e}')
        return pd.DataFrame()


def nm_filter(df):
    filtered_df = df[df['standard_units'] == 'nM']
    return filtered_df
