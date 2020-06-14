import streamlit as st
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import KMeans

# carrega a base com as empresas do mercado e atributos
mercado = pd.read_csv('./data/dados_mercado_final.csv')


# define a função que calcula as distancias
@st.cache(suppress_st_warning=True, show_spinner=False)
def euclidian(d, p):
    return d.apply(lambda row: distance.euclidean(p, row), axis=1)


# define a função que devolve os centroides dos clusters
@st.cache(suppress_st_warning=True, show_spinner=False)
def cluster(data, n_clusters):
    clstr = KMeans(n_clusters=n_clusters)
    resultado = clstr.fit(data)
    return resultado


###

st.title('Projeto Final - AceleraDev Data Science')
st.markdown('Autora: Jéssica Ramos')
st.markdown('*Versão 3 - Jun/2020*')

st.subheader('Lista de clientes da empresa:')
file = st.file_uploader('Insira o arquivo .csv:', type='csv')
if file is not None:
    # lê os clientes
    df = pd.read_csv(file)
    clientes = df.loc[:, 'id']
    # coloca no display o número de empresas clientes
    st.markdown('A lista contém ' + str(len(clientes)) + ' clientes.')

    # subset de variáveis utilizadas para o cálculo de perfil (cluster)
    variaveis = ['id', 'sg_uf_AC', 'sg_uf_AM', 'sg_uf_MA', 'sg_uf_PI', 'sg_uf_RN', 'sg_uf_RO',
                 'natureza_juridica_macro_ADMINISTRACAO PUBLICA', 'natureza_juridica_macro_CARGO POLITICO',
                 'natureza_juridica_macro_ENTIDADES EMPRESARIAIS',
                 'natureza_juridica_macro_ENTIDADES SEM FINS LUCRATIVOS',
                 'natureza_juridica_macro_INSTITUICOES EXTRATERRITORIAIS', 'natureza_juridica_macro_PESSOAS FISICAS',
                 'setor_AGROPECUARIA', 'setor_COMERCIO', 'setor_CONSTRUÇÃO CIVIL', 'setor_INDUSTRIA',
                 'setor_SEM INFORMACAO', 'setor_SERVIÇO', 'fl_rm_SIM', 'idade_empresa_anos',
                 'log_faturamento_estimado_aux', 'log_faturamento_estimado_grupo_aux', 'filiais', 'uf_mesma_matriz']

    # calcula os centroides dos clusters de clientes
    atr_clientes = mercado.loc[mercado['id'].isin(clientes), variaveis]
    ajuste_clusters = cluster(atr_clientes.iloc[:, 1:], 25)
    centro_clusters = ajuste_clusters.cluster_centers_

    # separa apenas as empresas que não são clientes
    mercado_possivel = mercado.loc[~mercado['id'].isin(clientes), variaveis]

    # calcula a similaridade com cada centroide de cluster
    distancia = pd.DataFrame({'id': mercado_possivel['id']})

    for i in range(0, 25):
        distancia[i + 1] = euclidian(mercado_possivel.iloc[:, 1:], centro_clusters[i, :])

    # calcula o tamanho de cada cluster
    clusters_clientes = ajuste_clusters.predict(atr_clientes.iloc[:, 1:])
    tam_clusters = pd.Series(clusters_clientes).value_counts()

    # calcula o fator multiplicador para cada cluster
    p = len(clientes) / 25
    fator_clusters = p / tam_clusters

    # multiplica as distâncias pelo fator
    for i in range(0, 25):
        distancia[i + 1] = distancia[i + 1] * fator_clusters[i]

    # calcula a distância final
    distancia['dist'] = distancia.iloc[:, 1:].apply(lambda row: min(row), axis=1)

    # ordena o dataset pelas distâncias
    distancia.sort_values(by='dist', inplace=True)

    # devolve as top n recomendações
    st.subheader('Leads mais aderentes:')
    n = st.slider('Escolha o número de leads:', min_value=1, max_value=100, value=10)
    top_n = distancia.iloc[0:n, 0].reset_index(drop=True)
    st.table(top_n)
