import streamlit as st
import pandas as pd
from PIL import Image
import os
import altair as alt
from io import StringIO


from utils.data_search import search_target, select_target
from utils.data_processing import pIC50, norm_value, label_bioactivity, convert_ugml_nm, classify_compound
from utils.descriptors import lipinski, desc_calc
from utils.model import build_model, model_generation
from utils.visualization import molecules_graph_analysis, mannwhitney
from utils.admet_evaluation import evaluate_admet, summarize_results
from utils.mol_draw import get_molecular_image, image_to_base64

if not os.path.isdir("models"):
        os.mkdir("models")

if not os.path.isdir("descriptor_lists"):
    os.mkdir("descriptor_lists")

st.set_page_config(layout="wide")

image = Image.open('logo.png')
selected_index = ''

st.image(image, use_column_width=True)

# Page title
st.markdown("""
# Aplicação de Predição de Bioatividade
            
Essa aplicação permite preparar modelos de Machine Learning utilizando o algoritmo Random Forest, de modo a realizar cálculo de predição de bioatividade em relação a alvos terapêuticos. 

**Créditos**
- Aplicativo originalmente desenvolvido em Python + Streamlit por [Chanin Nantasenamat](https://medium.com/@chanin.nantasenamat) (aka [Data Professor](http://youtube.com/dataprofessor))
- Descritores calculados utilizando [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) [[Leia o Paper]](https://doi.org/10.1002/jcc.21707).
---
""")

st.header("Criação de modelo (Dados da plataforma CHEMBL)")


col1, col2 = st.columns([3,1])
if 'targets' not in st.session_state:
    st.session_state['targets'] = pd.DataFrame()
with col1:
    search = st.text_input("Alvo")
with col2:
    if st.button('Buscar'):
        selected_index = ''
        with st.spinner("Buscando..."):
            st.session_state['targets'] = search_target(search)


targets = st.session_state['targets']
molecules_processed = pd.DataFrame()


# TODO: Alinhar input e botão

container1 = st.container()
with container1:
    if not targets.empty:
        st.write(targets)


if not targets.empty:
    selected_index = st.text_input("Índice do alvo selecionado")
    

    if selected_index:
        with st.spinner("Selecionando base de dados de moléculas: "):
            selected_molecules = select_target(selected_index, targets)

        if not selected_molecules.empty:

            with st.spinner("Processando base de dados: "):

                df_lipinski = lipinski(selected_molecules)
                df_combined = pd.concat([selected_molecules, df_lipinski], axis=1)
                df_converted = convert_ugml_nm(df_combined)
                df_labeled = label_bioactivity(df_converted)            
                df_norm = norm_value(df_labeled)
                molecules_processed = pIC50(df_norm)
                st.header("Moléculas Processadas")
                st.write(molecules_processed)
                st.write(molecules_processed.shape)


        if not molecules_processed.empty:

            classify = st.checkbox("Classificar moléculas")

            if (classify):
                with st.spinner("Classificando moléculas: "):
                    molecules_processed = classify_compound(molecules_processed)
                    st.header("Moléculas classificadas")
                    st.write(molecules_processed)

            if st.button("Realizar análise gráfica", key="btn_analise_grafica"):
                st.header("Análise Gráfica")
                molecules_graph_analysis(molecules_processed)
                st.header("Teste de Mann-Whitney")
                df_mannwhitney = mannwhitney(molecules_processed)
                st.write(df_mannwhitney)

            if st.button("Realizar análise gráfica por classe", key="btn_analise_grafica_classe", disabled= not classify):
                pass

            if st.button("Realizar avaliação ADMET", key="btn_avaliacao_admet"):
                st.header("Avaliação ADMET")             
                st.subheader("Regras e Limiares da Avaliação ADMET")
                
                rules_data = {
                    'Regra': ['Lipinski', 'Pfizer', 'GSK', 'Golden Triangle', 'PAINS'],
                    'Descrição': [
                        'Regra dos 5 de Lipinski',
                        'Regra de toxicidade da Pfizer',
                        'Regra da GSK',
                        'Regra do Triângulo Dourado',
                        'Filtro de Pan-Assay Interference Compounds'
                    ],
                    'Critérios': [
                        'MW ≤ 500; LogP ≤ 5; HBA ≤ 10; HBD ≤ 5',
                        'LogP > 3 e TPSA < 75',
                        'MW ≤ 400; LogP ≤ 4',
                        '200 ≤ MW ≤ 500; -2 ≤ LogP ≤ 5',
                        'Presença de subestruturas problemáticas'
                    ],
                    'Interpretação': [
                        'Excelente: < 2 violações; Pobre: ≥ 2 violações',
                        'Excelente: não atende ambos os critérios; Pobre: atende ambos os critérios',
                        'Excelente: 0 violações; Pobre: ≥ 1 violação',
                        'Excelente: 0 violações; Pobre: ≥ 1 violação',
                        'Aceito: sem alertas; Não aceito: com alertas'
                    ]
                }
                
                rules_df = pd.DataFrame(rules_data)
                st.table(rules_df)
                with st.spinner("Realizando avaliação ADMET..."):
                    smiles_list = molecules_processed["canonical_smiles"].tolist()
                    admet_results = evaluate_admet(smiles_list)

                    if not admet_results.empty:
                        st.write("Resultados ADMET detalhados:")
                        st.write(admet_results)

                        summary = summarize_results(admet_results)
                        st.write("Resumo da avaliação ADMET:")
                        for rule, result in summary.items():
                            st.write(f"{rule}: {result}")

                        st.subheader("Distribuição das propriedades")
                        for prop in ["MW", "LogP", "HBD", "HBA", "TPSA"]:
                            st.write(f"Distribuição de {prop}")
                            chart = (
                                alt.Chart(admet_results)
                                .mark_bar()
                                .encode(
                                    alt.X(prop, bin=True),
                                    y="count()",
                                )
                                .properties(width=600, height=300)
                            )
                            st.altair_chart(chart, use_container_width=True)

                        st.subheader("Violações das regras")
                        for rule in ["Lipinski", "GSK", "GoldenTriangle"]:
                            st.write(f"Violações da regra {rule}")
                            st.bar_chart(admet_results[f"{rule}_violations"].value_counts())

                        st.subheader("PAINS alerts")
                        st.write("Número de compostos com alertas PAINS")
                        st.bar_chart(admet_results["PAINS"].value_counts())
                    else:
                        st.error(
                            "Não foi possível realizar a avaliação ADMET. Verifique os dados de entrada e tente novamente."
                        )

            model_col1, model_col2 = st.columns([0.5, 0.5])
            with model_col1:
                variance_input = st.number_input(
                    "Limite de variância:", min_value=0.0, value=0.1
                )
            with model_col2:
                estimators_input = st.number_input(
                    "Número de estimadores:", min_value=1, value=500
                )
            model_name = st.text_input("Nome para salvamento do modelo: ")
            if st.button("Gerar modelo"):
                with st.spinner("Gerando modelo"):
                    model_generation(
                        molecules_processed, variance_input, estimators_input, model_name
                    )
            if st.button("Gerar modelos separados por classe", disabled= not classify):
                pass


models = os.listdir('models')

if len(os.listdir('models')) != 0:
    with st.sidebar.header('1. Selecione o modelo a ser utilizado (alvo): '):
        selected_model_name = st.sidebar.selectbox("Modelo", models).removesuffix(".pkl")
        selected_model = f'models/{selected_model_name}.pkl'

    with st.sidebar.header('2. Faça upload dos dados em CSV:'):
        uploaded_file = st.sidebar.file_uploader("Faça upload do arquivo de entrada", type=['csv'])
        st.sidebar.markdown("""
    [Exemplo de arquivo de entrada](https://raw.githubusercontent.com/ChiBeG/bioatividade_LIBV/main/exemplo.csv)
    """)

    if st.sidebar.button('Predizer'):
        st.header('Cálculo de predição:')
        # Modificar a leitura do arquivo CSV
        load_data = pd.read_csv(uploaded_file, sep=';', header=None, names=['SMILES', 'ID', 'Name'])

        # Criar o arquivo .smi apenas com SMILES e ID
        load_data[['SMILES', 'ID']].to_csv('molecule.smi', sep=' ', header=False, index=False)

        st.header('**Dados originais de entrada**')
        st.write(load_data)

        with st.spinner("Calculando descritores..."):
            desc_calc()

        st.header('**Cálculo de descritores moleculares realizado**')
        desc = pd.read_csv('descriptors_output.csv')
        st.write(desc)
        st.write(desc.shape)

        # Read descriptor list used in previously built model
        st.header('**Subconjunto de descritores do modelo selecionado**')
        Xlist = list(pd.read_csv(f'descriptor_lists/{selected_model_name}_descriptor_list.csv').columns)
        desc_subset = desc[Xlist]
        st.write(desc_subset)
        st.write(desc_subset.shape)

        # Apply trained model to make prediction on query compounds
        df_result = build_model(
            desc_subset, load_data, selected_model, selected_model_name
        )

        if not df_result.empty:
            result_lipinski = lipinski(load_data)
            df_final = pd.concat([df_result, result_lipinski], axis=1)
            st.write(df_final)

            # Adicionar avaliação ADMET para as moléculas preditas
            st.header("Avaliação ADMET das Moléculas Preditas")
            with st.spinner("Realizando avaliação ADMET..."):
                smiles_list = load_data['SMILES'].tolist()
                admet_results = evaluate_admet(smiles_list)

                if not admet_results.empty:
                    st.write("Resultados ADMET detalhados:")
                    st.write(admet_results)

                    summary = summarize_results(admet_results)
                    st.write("Resumo da avaliação ADMET:")
                    for rule, result in summary.items():
                        st.write(f"{rule}: {result}")

                    st.subheader("Distribuição das propriedades")
                    for prop in ["MW", "LogP", "HBD", "HBA", "TPSA"]:
                        st.write(f"Distribuição de {prop}")
                        chart = (
                            alt.Chart(admet_results)
                            .mark_bar()
                            .encode(
                                alt.X(prop, bin=True),
                                y="count()",
                            )
                            .properties(width=600, height=300)
                        )
                        st.altair_chart(chart, use_container_width=True)

                    st.subheader("Violações das regras")
                    for rule in ["Lipinski", "GSK", "GoldenTriangle"]:
                        st.write(f"Violações da regra {rule}")
                        st.bar_chart(admet_results[f"{rule}_violations"].value_counts())

                    st.subheader("PAINS alerts")
                    st.write("Número de compostos com alertas PAINS")
                    st.bar_chart(admet_results["PAINS"].value_counts())

                    # Combinar resultados de predição com avaliação ADMET
                    admet_columns_to_keep = [col for col in admet_results.columns if col not in df_final.columns and col != 'smiles']
                    df_final_with_admet = pd.concat([df_final, admet_results[admet_columns_to_keep]], axis=1)
                    
                    st.header("Resultados Finais (Predição + ADMET)")
                    
                    # Gerar imagens das estruturas moleculares
                    with st.spinner("Gerando imagens das estruturas moleculares..."):
                        molecular_images = [get_molecular_image(smiles) for smiles in smiles_list]
                    
                    # Exibir informações de depuração
                    st.subheader("Informações de Depuração")
                    st.write(f"Número de moléculas: {len(smiles_list)}")
                    st.write(f"Número de imagens geradas: {len([img for img in molecular_images if img is not None])}")
                    
                    # Adicionar uma coluna com as imagens ao DataFrame
                    df_final_with_admet.insert(1, 'Estrutura Molecular', [image_to_base64(img) for img in molecular_images])
                    
                    # Exibir a tabela final com as imagens
                    st.subheader("Tabela de Resultados com Estruturas Moleculares")
                    st.write(df_final_with_admet.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                else:
                    st.error(
                        "Não foi possível realizar a avaliação ADMET. Verifique os dados de entrada e tente novamente."
                    )
        else:
            st.error("Não foi possível gerar predições. Verifique os dados de entrada e o modelo selecionado.")

    else:
        st.info('Utilize a barra lateral para selecionar o modelo e realizar o upload dos dados de entrada!')

else:
    with st.sidebar.header('Não há modelos disponíveis para predição'):
        pass
