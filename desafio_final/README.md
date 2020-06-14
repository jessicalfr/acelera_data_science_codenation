## Projeto Final - AceleraDev Data Science
## Documentação

*Autora: Jéssica Ramos*

*Junho/2020*

O projeto tinha como objetivo identificar os leads mais aderentes para uma empresa dado o seu portfólio e os atributos de todos os possíveis clientes no mercado. Esse problema pode ser resolvido com um sistema de recomendação baseado em conteúdo.

### Primeira versão

A primeira versão do recomendador, cujo app pode ser encontrado em `recommender_v1.py`, se trata de uma versão simples de um sistema de recomendação *content based*. O passo a passo do algoritmo é o seguinte:

- Dado um portfólio, o código calcula o perfil de cliente para aquela empresa. Isso é feito calculando a média de cada coluna no dataset de atributos, filtrando apenas os clientes no portfólio da empresa.
- Daí, se calcula a distância (no caso, a distância euclidiana) entre o perfil e cada possível cliente no dataset do mercado, ou seja, cada empresa no mercado que ainda não faz parte do portfólio.
- Em seguida, os possíveis clientes são ordenados da menor distância para a maior. É exibido na tela o Top N de clientes possíveis, sendo o N entre 1 e 100 e escolhido pelo ususário.

A base de dados utilizada contém as colunas pré-selecionadas na análise exploratória do banco de dados, que foi realizada no notebook `analise_exploratoria_dados.ipynb`. Foram utilizadas 28 variáveis. Após o *one-hot encoding* das variáveis categóricas e a normalização (min-max) das variáveis numéricas, a base continha 54 colunas de atributos.


### Segunda versão

A segunda versão do recomendador, cujo app pode ser encontrado em `recommender_v2.py`, modifica apenas o número de variáveis utilizadas na análise. Ao invés das 54 colunas foram utilizadas apenas 24 que correspondiam a variáveis que concluí serem mais relacionadas aos motivos de uma empresa estar no portfólio da empresa em questão.

As variáveis selecionadas foram:

- `sg_uf`: Sigla do estado em que a empresa se encontra.
- `natureza_juridica_macro`: Categorias macro para a natureza jurídica.
- `setor`: Setor econômico da empresa.
- `fl_rm`: Indica se a empresa está na região metropolitana do estado.
- `idade_empresa_anos`: Idade da empresa em anos.
- `log_faturamento_estimado_aux`: Transformação logarítimica (base 10) da variável `vl_faturamento_estimado_aux`, que é o valor estimado do faturamento da empresa (apenas esse CNPJ).
- `log_faturamento_estimado_grupo_aux`: Transformação logarítimica (base 10) da variável `vl_faturamento_estimado_grupo_aux`, que é o valor estimado do faturamento da matriz e das filiais da empresa.
- `filiais`: Indica se a empresa tem filiais. Construída com base na variável `qt_filiais`.
- `uf_mesma_matriz`: Indica se a UF da matriz é a mesma da empresa. Construída com base nas variáveis `sg_uf` e `sg_uf_matriz`.

As 24 colunas são as indicadoras das variáveis categóricas acima (obtidas por *one-hot encoding*) e as variáveis numéricas padronizadas.

### Terceira versão

A terceira versão do recomendados, cujo app pode ser encontraod em `recommender_v3.py`, tem um raciocínio um pouco mais complexo. Nela, ao calcular o perfil de cliente é feita um ajuste de 25 clusters dentre os clientes do portfólio, utilizando as 24 colunas da versão 2. O número 25 foi escolhido com base na análise de cluster exploratória feita em `analise_cluster_exploratoria.ipynb`. Em resumo, o número ótimo de clusters dentre os clientes das empresas 1, 2 e 3 (disponibilizadas pelo desafio) ficou entre 20 e 30. O algoritmo utilizado foi o k-médias e o método para determinar o número de clusters foi o *elbow method*.

Após ajustar os clusters, são extraídas as coordenadas do centroide de cada cluster. Para cada possível cliente, é calculada a distância euclidiana entre o cliente e cada possível cluster. Dessa forma, eram obtidas 25 distâncias para cada possível cliente.

Depois, é calculado um fator multiplicador para cada cluster. O objetivo dele é diminuir o valor da distância para clusters maiores e aumentar a distância para clusters menores. O raciocínio aqui foi de que distância 1 de um cluster grande é mais relevante o que distância 1 para um cluster pequeno. Um cluster grande significa que a empresa tem muitos clientes com aquele perfil.

O fator multiplicador é calculado em três etapas:

1) Calcular o número esperado de clientes por cluster se todos os clusters tivessem o mesmo tamanho. No caso, é o número de clientes no portfólio dividido pelo número de clusters. Vamos chamar esse número de P.
2) Calcular o número de clientes no portfólio em cada cluster.
3) Dividir o número P calculado na etapa 1 pelo núemro de clientes em cada portfólio. Dessa forma, o fator multiplicador do cluster 1 será P dividido pelo número de clientes do portfólio que são do cluster 1, por exemplo.

Nesse cálculo, distâncias a clusters grandes serão diminuídas e distâncias a clusters menores serão aumentadas. Para cada possível cliente, multiplicamos o valor da distância a cada cluster pelo respectivo fator multiplicador e pegamos o menor valor restante. Vamos chamar esse valor de distância sumarizada.

Por fim, ordenamos todos os possíveis clientes pela distância sumarizada, da menor para a maior. O Top N de possíveis clientes então é exibido na tela.

Esse algoritmo, por exigir o cálculo de mais distâncias, demora um pouco mais para rodar.

### Validação e resultado final

A validação das 3 versões está registrada em `validacao_recomendadores.ipynb`. Foram avaliadas as métricas *Recall@10* e *Recall@20*. A versão 1 teve resultados bons, considerando a simplicidade do método. A versão 3 também tem resultados interessantes, porém leva mais tempo para rodar, o que pode acabar não compensando, dependendo da utilização.

Caso tivesse mais tempo para explorar opções, tentaria fazer modificações na versão 3 para tentar obter um resultado ainda melhor. Entre as opções que considerei estão:

1) Tentar modificar o fator multiplicador utilizado em cada cluster;
2) Tentar otimizar melhor o número de clusters utilizado;
3) Experimentar outros métodos de análise de cluster;
4) Tentar outras combinações de variáveis utilizadas.

### Considerações finais

Foi uma experiência muito interessante realizar esse trabalho. Ainda não tinha estudado com mais atenção sistemas de recomendação e foi bastante enriquecedor poder desenvolver um projeto do início ao fim, desde o entendimento do problema, aos estudos e finalmente a implementação de resultados. Acredito fortemente que poderia desenvolver resultados ainda mais interessantes caso tivesse mais tempo para me dedicar ao projeto, mas estou feliz com o resultado!