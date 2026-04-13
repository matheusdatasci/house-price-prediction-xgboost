# California House Pricing — Previsão de Preços com XGBoost

Dataset utilizado: California Housing Prices, presente no livro *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn e TensorFlow* (Aurélien Géron).

---

## Ferramentas

- **Pré-processamento e limpeza:** pandas, numpy  
- **Visualização:** Matplotlib, Seaborn  
- **Algoritmos de aprendizado de máquina:** XGBoost, scikit-learn

---

## Metodologia

### 1. Filtragem inicial dos dados

O dataset possui um limite artificial: todas as casas com valor acima de $500.000 foram registradas como exatamente $500.000, o que cria um acúmulo artificial nesse ponto. Manter essas observações contaminaria o modelo, pois elas não representam o preço real. Por isso, o dataset foi filtrado para conter apenas amostras com `median_house_value < 500.000`.

### 2. Análise exploratória

A análise inicial revelou três pontos importantes:

- **Alta multicolinearidade** entre `total_rooms`, `total_bedrooms`, `population` e `households`. Essas variáveis medem totais por bloco, não por domicílio, o que as torna redundantes entre si e pouco informativas individualmente.
- **Correlação relevante** entre `median_income` e `median_house_value`, a variável alvo, confirmando que a renda é o preditor mais forte do dataset.
- **Distribuição assimétrica** da variável alvo (assimetria ≈ 0.6), indicando presença de outliers à direita que precisariam ser tratados.

### 3. Tratamento de dados faltantes

A coluna `total_bedrooms` apresentou aproximadamente 200 valores nulos (~1% dos dados). A escolha utilizada foi o preenchimento com a mediana, uma escolha conservadora que não distorce a distribuição e é robusta a outliers.

### 4. Engenharia de features

Para não dependermos da multicolinearidade das variáveis apresentadas acima, foram criadas duas features derivadas:

- `comodos_por_casas`: razão entre `total_rooms` e `households`, representando a média de cômodos por domicílio.
- `populacao_por_casas`: razão entre `population` e `households`, representando a densidade de moradores por domicílio.

Essas features capturam a informação relevante das variáveis originais de forma per capita, tornando-as comparáveis entre blocos de tamanhos diferentes.

### 5. Clusterização geográfica com K-Means

A variável `ocean_proximity` apresentou limitação prática: com exceção da categoria ISLAND, todas as demais classes possuem distribuições de preço com alta variância e amplitude similar. Isso significa que a categoria em si não discrimina bem o preço, uma casa NEAR OCEAN pode custar o mesmo que uma INLAND. Usar a variável diretamente geraria pouco ganho preditivo.

A alternativa foi aplicar **K-Means com 10 clusters** sobre as coordenadas geográficas (`longitude` e `latitude`). O K-Means agrupa os imóveis em regiões geograficamente, capturando padrões de preço locais que a variável `ocean_proximity` não conseguiria representar. Os clusters foram então transformados em variáveis dummy e o `ocean_proximity` foi removido para evitar redundância.

### 6. Transformação logarítmica na variável dependente

Antes do treinamento, foi aplicada a transformação `log1p` na variável alvo (`median_house_value`) no conjunto de treino. Os motivos são dois:

1. **Assimetria:** a distribuição do preço é positivamente assimétrica, e a transformação logarítmica aproxima a distribuição da normalidade, o que favorece o aprendizado do modelo.
2. **Compressão de outliers:** casas com preços muito elevados exercem influência desproporcional no erro quadrático, justamente por serem elevados ao quadrado. O log comprime valores maiores, o que permite usarmos a função de perda `squarederror` mesmo com a presença de outliers.

As previsões foram revertidas para a escala original com `expm1` antes de verficar as métricas do modelo, pois é adequado que estejam na mesma escala.

### 7. Treinamento e ajuste do modelo

O modelo escolhido foi o **XGBRegressor**. Os hiperparâmetros foram definidos utilizando GridSearch do Scikit-learn (não está no notebook):

- `objective='reg:squarederror'`: penaliza erros grandes de forma quadrática, incentivando o modelo a generalizar além da média, sendo viável aqui justamente porque o log na variável alvo já atenuou o impacto dos outliers.
- `n_estimators=2000` com `learning_rate=0.01`: muitas árvores com passos pequenos para convergência estável.
- `subsample=0.8` e `colsample_bytree=0.7`: subamostragem de dados e features por árvore para reduzir overfitting.
- `reg_lambda=2` e `reg_alpha=0.1`: regularização L2 e L1 para penalizar modelos excessivamente complexos.
- `min_child_weight=15` e `max_depth=8`: controle da profundidade e do tamanho mínimo das folhas para evitar que o modelo memorize ruído.

---

## Métricas

| Métrica | Valor |
|---|---|
| MAE (treino) | $21.343 |
| MAE (teste) | $27.733 |
| RMSE (teste) | $42.745 |
| R² (teste) | 80,53% |
| MAPE (teste) | 15,64% |

### Onde o modelo vai bem

O modelo performa de forma satisfatória na faixa de preços médios e medianos do dataset, errando em média 15% para mais ou para menos. A diferença pequena entre o MAE de treino e de teste indica que não houve overfitting forte, o R² de 80,53% demonstra que o modelo captura a maior parte da variância dos preços.

### Onde o modelo peca

A visualização dos resíduos expõe um padrão sistemático: **o modelo subestima casas de maior valor**. Erros positivos acima de $100.000 ocorrem com frequência nessa faixa, enquanto superestimações são mais incomuns.

O modelo viu poucos exemplo de medianas mais caras, pois existem poucas amostras delas, então quando encontra uma, é natural que preveja valores mais baixos, pois foi o que ele mais viu.

A censura que esse conjunto de dados possui em $500.000 é a limitação principal. Qualquer modelo treinado com esses dados terá dificuldade em generalizar para imóveis acima dessa faixa, independentemente do algoritmo ou dos hiperparâmetros escolhidos.
