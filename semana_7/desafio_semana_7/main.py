#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

# from IPython.core.pylabtools import figsize


# figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# Primeiro o ajuste do *dataset*. A primeira coisa a ser feita é corrigir as colunas de strings que deveriam ter sido lidas como numéricas.

# In[5]:


# verifica o tipo das variaveis
countries.dtypes


# In[6]:


# variáveis que deveriam ser numéricas
var_numericas = ['Pop_density','Coastline_ratio','Net_migration','Infant_mortality','Literacy','Phones_per_1000','Arable','Crops','Other','Climate','Birthrate','Deathrate','Agriculture','Industry','Service']

# ajusta o separador e converte para float
countries[var_numericas] = countries[var_numericas].apply(lambda x: x.str.replace(',','.')).astype(float)

# verifica os datatypes novamente
countries.dtypes


# Em segundo, é feito o ajuste de espaços em branco nas variáveis `Country` e `Region` com o método `str.strip()`. 

# In[7]:


# ajusta as variáveis Country e Region
countries[['Country','Region']] = countries[['Country','Region']].apply(lambda x: x.str.strip())


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[8]:


def q1():
    # pega os valores únicos da coluna
    unicos = countries['Region'].unique().tolist()
    
    # ordena a lista
    unicos.sort()
    
    # retorna a lista
    return unicos


# In[9]:


# chama a função
q1()


# In[10]:


len(q1())


# In[11]:


countries['Region'].value_counts()


# No *dataset* existem 11 regiões diferentes, sendo que a região com maior número de países é a Áfria Subsaariana e a com menor número de países sendo os Países Bálticos.

# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[12]:


def q2():
    # cria os bins na variável
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    discretizer.fit(countries[['Pop_density']])
    pop_dens_bins = discretizer.transform(countries[['Pop_density']])
    
    # retorna o número de casos no último quantil
    return int(sum(pop_dens_bins==9))


# In[13]:


# chama a função
q2()


# In[14]:


# percentil 90
np.percentile(countries['Pop_density'], 90)


# Segundo os resultados acima, 23 países têm densidade populacional maior do que 396,74.

# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# Usando *one-hot encoding*, cria-se um novo atributo para cada categoria da variável.

# In[15]:


def q3():
    # conta o número de únicos na variável Region
    region_uniques = len(countries['Region'].unique())
    
    # conta o número de únicos na variável Climate
    climate_uniques = len(countries['Climate'].unique())
    
    # retorna a soma
    return region_uniques + climate_uniques


# In[16]:


# chama a função
q3()


# Já vimos na questão 1 que a variável `Region` tem 11 valores distintos. Portanto, a variável `Climate` tem 7 valores distintos.

# In[17]:


countries['Climate'].unique()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[18]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[19]:


def q4():
    # cria o pipeline
    pipe_country = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy="median")),
        ('padronizer', StandardScaler())
    ])
    
    # separa as variáveis numéricas
    var_num = countries.columns.tolist()
    var_num.remove('Country')
    var_num.remove('Region')
    
    # aplica o pipeline aos dados
    pipe_country.fit(countries[var_num])
    
    # tranforma os dados de teste num data frame
    test_df = pd.DataFrame([test_country], columns=countries.columns)
    
    # aplica o pipeline ao dado de teste
    result_test = pipe_country.transform(test_df[var_num])
    
    # retorna o valor na variável Arable
    return round(result_test[0][9],3)


# In[20]:


# chama a função
q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[21]:


def q5():
    # calcula os limites para outliers
    q1 = np.percentile(countries['Net_migration'].dropna(), 25)
    q3 = np.percentile(countries['Net_migration'].dropna(), 75)
    iqr = q3-q1
    
    lim_inf = q1 - 1.5*iqr
    lim_sup = q3 + 1.5*iqr
    
    # conta o número de outliers abaixo
    abaixo = sum(countries['Net_migration']<lim_inf)
    
    # conta o número de outliers acima
    acima = sum(countries['Net_migration']>lim_sup)
    
    # retorna a tupla com o resultado
    return (abaixo, acima, False)


# In[22]:


# chama a função
q5()


# Precisamos, antes que decidir sobre remover os outliers, verificar se não há erros de medição e se eles não impactam a análise final. Somente com esse método não posso afirmar que posso remover esses casos.

# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[23]:


# carrega o dataset
categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[24]:


def q6():
    # ajusta o contador de palavras
    count_vec = CountVectorizer()
    newsgroup_counts = count_vec.fit_transform(newsgroup.data)
    
    # retorna a contagem
    return int(newsgroup_counts[:, count_vec.vocabulary_['phone']].sum())


# In[25]:


# chama a função
q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[26]:


def q7():    
    # ajusta o TF-IDF    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(newsgroup.data)
    newsgroup_tfidf = tfidf_vectorizer.transform(newsgroup.data)
    
    # retorna o TF-IDF
    return float(newsgroup_tfidf[:, tfidf_vectorizer.vocabulary_['phone']].sum().round(3))


# In[27]:


# chama a função
q7()

