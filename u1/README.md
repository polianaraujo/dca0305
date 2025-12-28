# Projeto 1 - Problema de regressão

Este repositório contém um pipeline completo para análise de dados, pré-processamento, tratamento de outliers e treinamento de modelos de regressão utilizando PyTorch. O objetivo principal foi comparar o impacto de diferentes técnicas de normalização e otimizadores no desempenho do modelo para obter o cenário para o melhor resultado possível (menor erro).

## Desenvolvimento

1. Pré-processamento e Outliers
Foi aplicada a técnica de Intervalo Interquartil (IQR) para identificar e remover valores discrepantes que poderiam prejudicar a convergência da rede neural.

    - Antes: Presença de caudas longas e valores extremos.

    ![violinplots_antes_iqr.png](https://github.com/polianaraujo/dca0305/blob/main/u1/images/violinplots_antes_iqr.png)

    - Depois: Distribuição de dados mais robusta e centrada, facilitando o aprendizado do otimizador.

    ![violinplots_apos_iqr.png](https://github.com/polianaraujo/dca0305/blob/main/u1/images/violinplots_apos_iqr.png)

2. Arquitetura do Modelo
O modelo consiste em uma rede neural sequencial (MLP) construída em PyTorch, projetada para tarefas de regressão, utilizando:

    - Camadas lineares (nn.Linear).

    - Funções de ativação para introduzir não-linearidade.

    - Cálculo de erro via MSE (Mean Squared Error) para otimização.

3. Experimentos Realizados
Os testes compararam as seguintes variáveis:

    - Normalização: Min-Max Scaling vs. Z-Score (Standardization).

    - Otimizadores: SGD, Adam, entre outros.

    - Hiperparâmetros: Taxa de aprendizado (LR), Épocas e Tamanho do Batch.

## Melhores Resultados

Todos os resultados podem ser visualizados no csv *resultados_experimentos_sequencial_1n*, e os melhores resultados estão em *resultados_experimentos_sequencial_teste2.csv* e pode ser visualizado abaixo.

|Normalização|Otimizador|MAE|RMSE|R²|
|-|-|-|-|-|
|Min-Max|SGD|112.22|158.98|0.373|
|Z-Score|SGD|112.43|159.51|0.369|

## Sua própria execução do projeto

1. Clone este repositório.
```bash
git clone <link_ssh_repositorio>
```

2. Certifique-se de ter as bibliotecas instaladas:

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn
```

3. Abra o notebook.ipynb em um ambiente Jupyter ou Google Colab.

4. Execute as células sequencialmente para reproduzir o tratamento de dados e o treinamento.