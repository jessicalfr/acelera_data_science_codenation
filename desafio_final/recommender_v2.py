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
st.markdown('*Versão 2 - Jun/2020*')

st.subheader('Lista de clientes da empresa:')
file = st.file_uploader('Insira o arquivo .csv:', type='csv')
if file is not None:
    # lê os clientes
    df = pd.read_csv(file)
    clientes = df.loc[:, 'id']
    # coloca no display o número de empresas clientes
    st.markdown('A lista contém '+str(len(clientes))+' clientes.')

    # subset de variáveis utilizadas para o cálculo de perfil
    variaveis = ['id','sg_uf_AC', 'sg_uf_AM', 'sg_uf_MA', 'sg_uf_PI', 'sg_uf_RN', 'sg_uf_RO',
                 'natureza_juridica_macro_ADMINISTRACAO PUBLICA', 'natureza_juridica_macro_CARGO POLITICO',
                 'natureza_juridica_macro_ENTIDADES EMPRESARIAIS','natureza_juridica_macro_ENTIDADES SEM FINS LUCRATIVOS',
                 'natureza_juridica_macro_INSTITUICOES EXTRATERRITORIAIS', 'natureza_juridica_macro_PESSOAS FISICAS',
                 'setor_AGROPECUARIA', 'setor_COMERCIO', 'setor_CONSTRUÇÃO CIVIL', 'setor_INDUSTRIA',
                 'setor_SEM INFORMACAO','setor_SERVIÇO', 'fl_rm_SIM', 'idade_empresa_anos',
                 'log_faturamento_estimado_aux','log_faturamento_estimado_grupo_aux', 'filiais', 'uf_mesma_matriz']

    # calcula o perfil médio das empresas do cliente
    atr_clientes = mercado.loc[mercado['id'].isin(clientes), variaveis]
    perfil = atr_clientes.iloc[:, 1:].mean()

    # separa apenas as empresas que não são clientes
    mercado_possivel = mercado.loc[~mercado['id'].isin(clientes), variaveis]

    # calcula a similaridade com cada empresa possível
    distancia = pd.DataFrame({'id': mercado_possivel['id']})
    distancia['dist'] = euclidian(mercado_possivel.iloc[:, 1:], perfil)

    # ordena o dataset pelas distâncias
    distancia.sort_values(by='dist', inplace=True)

    # devolve as top n recomendações
    st.subheader('Leads mais aderentes:')
    n = st.slider('Escolha o número de leads:', min_value=1, max_value=100,value=10)
    top_n = distancia.iloc[0:n, 0].reset_index(drop=True)
    st.table(top_n)