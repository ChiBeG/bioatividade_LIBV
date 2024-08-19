import streamlit as st
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from numpy.random import seed
from numpy.random import randn

def model_graph_analysis(Y, Y_pred, mse, r2):
    try:
        st.header("Análise do modelo")
        plt.clf()
        plt.figure(figsize=(5,5))
        plt.scatter(x=Y, y=Y_pred, c="#7CAE00", alpha=0.3)
        z = np.polyfit(Y, Y_pred, 1)
        p = np.poly1d(z)

        plt.plot(Y, p(Y),"#F8766D")
        plt.ylabel('pIC50 predito')
        plt.xlabel('pIC50 experimental')
        st.pyplot(plt)
        st.write(f'Mean squared error: {mse}')
        st.write(f'Coeficiente de determinação: {r2}')
    except Exception as e:
        st.error(f'Erro na análise do modelo: {e}')


def molecules_graph_analysis(molecules_processed):
    
    try:
        graph1, graph2 = st.columns([0.5, 0.5])
        
        sns.set(style='ticks')
        
        plt.figure(figsize=(5.5, 5.5))
        ax = sns.countplot(x='class', data=molecules_processed, edgecolor='black')
        plt.xlabel('Classe de Bioatividade', fontsize=14, fontweight='bold')
        plt.ylabel('Frequência', fontsize=14, fontweight='bold')

        for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() + 0.5), 
                ha='center', va='baseline', fontsize=12, color='black', fontweight='bold')

        
        with graph1:
            st.write("Frequências")
            with st.spinner("Gerando gráfico de frequências"):
                st.pyplot(plt)

        
        plt.clf()
        sns.scatterplot(x='MW', y='LogP', data=molecules_processed, hue='class', size='pIC50', edgecolor='black', alpha=0.7)

        plt.xlabel('MW', fontsize=14, fontweight='bold')
        plt.ylabel('LogP', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
        with graph2:
            st.write("MW x LogP")
            with st.spinner("Gerando gráfico MW x LogP"):
                st.pyplot(plt)


        st.header("Classes x Descritores de Lipinski")
        graph3, graph4, graph5, graph6, graph7 = st.columns([0.20, 0.20, 0.20, 0.20, 0.20])
        
        plt.clf()
        sns.boxplot(x = 'class', y = 'pIC50', data = molecules_processed)
        plt.xlabel('Classe de Bioatividade', fontsize=14, fontweight='bold')
        plt.ylabel('Valor pIC50', fontsize=14, fontweight='bold')
        
        with graph3:
            st.write("Classe x pIC50")
            with st.spinner("Gerando gráfico de Classe x pIC50"):
                st.pyplot(plt)

        plt.clf()
        sns.boxplot(x = 'class', y = 'LogP', data = molecules_processed)
        plt.xlabel('Classe de Bioatividade', fontsize=14, fontweight='bold')
        plt.ylabel('LogP', fontsize=14, fontweight='bold')

        with graph4:
            st.write("Classe x LogP")
            with st.spinner("Gerando gráfico de Classe x LogP"):
                st.pyplot(plt)

        
        plt.clf()
        sns.boxplot(x = 'class', y = 'MW', data = molecules_processed)
        plt.xlabel('Classe de Bioatividade', fontsize=14, fontweight='bold')
        plt.ylabel('MW', fontsize=14, fontweight='bold')

        with graph5:
            st.write("Classe x MW")
            with st.spinner("Gerando gráfico de Classe x MW"):
                st.pyplot(plt)

        
        plt.clf()
        sns.boxplot(x = 'class', y = 'NumHDonors', data = molecules_processed)
        plt.xlabel('Classe de Bioatividade', fontsize=14, fontweight='bold')
        plt.ylabel('Doadores H', fontsize=14, fontweight='bold')

        with graph6:
            st.write("Classe x Doadores H")
            with st.spinner("Gerando gráfico de Classe x Doadores H"):
                st.pyplot(plt)

        
        plt.clf()
        sns.boxplot(x = 'class', y = 'NumHAcceptors', data = molecules_processed)
        plt.xlabel('Classe de Bioatividade', fontsize=14, fontweight='bold')
        plt.ylabel('Aceptores H', fontsize=14, fontweight='bold')

        with graph7:
            st.write("Classe x Aceptores H")
            with st.spinner("Gerando gráfico de Classe x Aceptores H"):
                st.pyplot(plt)
        
    except Exception as e:
            st.error(f'Erro na criação dos gráficos: {e}')

 

def mannwhitney(df, verbose=False):
    
    try:

        descriptors = ['pIC50', 'LogP', 'MW', 'NumHDonors', 'NumHAcceptors']
        columns_names = ['Descritor', 'Estatistica', 'p valor', 'alpha', 'Interpretação']
        resultado = pd.DataFrame(columns= columns_names)
        
        
        for descriptor in descriptors:
        
            seed(1)

            selection = [descriptor, 'class']
            df_mannwhitney = df[selection]
            
            active = df_mannwhitney[df_mannwhitney['class'] == 'ativo']
            active = active[descriptor]

            selection = [descriptor, 'class']
            df_mannwhitney = df[selection]
            inactive = df_mannwhitney[df_mannwhitney['class'] == 'inativo']
            inactive = inactive[descriptor]

            stat, p = mannwhitneyu(active, inactive)

            alpha = 0.05
            if p > alpha:
                interpretation = 'Mesma distribuição (falha em rejeitar H0)'
            else:
                interpretation = 'Distribuição diferente (rejeita H0)'

            new_row = [descriptor, stat, p, alpha, interpretation]
            new_row_df = pd.DataFrame([new_row], columns=columns_names)
            new_row_df = new_row_df.dropna()
            resultado = pd.concat([resultado, new_row_df], ignore_index=True)

        return resultado
    
    except Exception as e:
        st.error(f'Falha no teste Mann-Whitney: {e}')
        return pd.DataFrame()