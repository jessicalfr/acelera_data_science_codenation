import streamlit as st
import pandas as pd
from scipy.spatial import distance

# carrega a base com as empresas do mercado e atributos
mercado = pd.read_csv('./data/dados_mercado_final.csv')

# define a função que calcula as distancias
@st.cache(suppress_st_warning=True, show_spinner=False)
def euclidian(d, p):
    return d.apply(lambda row: distance.euclidean(p, row), axis=1)

st.title('Projeto Final - AceleraDev Data Science')
st.markdown('Autora: Jéssica Ramos')
st.markdown('*Versão 1 - Jun/2020*')

st.subheader('Lista de clientes da empresa:')
file = st.file_uploader('Insira o arquivo .csv:', type='csv')
if file is not None:
    # lê os clientes
    df = pd.read_csv(file)
    clientes = df.loc[:, 'id']
    # coloca no display o número de empresas clientes
    st.markdown('A lista contém '+str(len(clientes))+' clientes.')

    # calcula o perfil médio das empresas do cliente
    atr_clientes = mercado.loc[mercado['id'].isin(clientes), :]
    perfil = atr_clientes.iloc[:, 2:].mean()

    # separa apenas as empresas que não são clientes
    mercado_possivel = mercado.loc[~mercado['id'].isin(clientes), :]

    # calcula a similaridade com cada empresa possível
    distancia = pd.DataFrame({'id': mercado_possivel['id']})
    distancia['dist'] = euclidian(mercado_possivel.iloc[:, 2:], perfil)

    # ordena o dataset pelas distâncias
    distancia.sort_values(by='dist', inplace=True)

    # devolve as top n recomendações
    st.subheader('Leads mais aderentes:')
    n = st.slider('Escolha o número de leads:', min_value=1, max_value=100,value=10)
    top_n = distancia.iloc[0:n, 0].reset_index(drop=True)
    st.table(top_n)
