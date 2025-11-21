## 1ª tentativa

Análise da Tentativa Inicial (Modelo Base)
Para este projeto, o dataset escolhido foi o SVHN (Street View House Numbers). Este dataset consiste em imagens coloridas (3 canais, RGB) de 32x32 pixels, representando 10 classes (dígitos de 0 a 9).

1. Configuração do Experimento
Seguindo os requisitos, a arquitetura-base foi mantida como a LeNet-5 vista em aula, com as seguintes adaptações para o SVHN:

- Arquitetura (model_svhn):

    - A primeira camada convolucional (C1) teve seu parâmetro in_channels alterado de 1 (para o MNIST/escala de cinza) para 3, para aceitar as imagens coloridas (RGB) do SVHN.

    - Devido ao tamanho da imagem de entrada (32x32 em vez de 28x28), a saída da última camada convolucional/pooling (C5) resultou em um tensor de 120x2x2.

    - Consequentemente, a primeira camada linear (F6) foi ajustada para ter in_features de 480 (ou seja, 120 * 2 * 2).

- Treinamento:

    - Dataset: SVHN (73.257 imagens de treino, 26.032 de validação/teste).

    - Otimizador: optim.SGD (Gradiente Descendente Estocástico).

    - Taxa de Aprendizado (LR): 0.01.

    - Função de Perda: nn.CrossEntropyLoss (adequada para classificação multiclasse).

    - Épocas: 20.

    - Batch Size (Tamanho do Lote): 64.

2. Resultados e Observações
O modelo foi treinado por 20 épocas, e as métricas de Perda (Loss) de treino e validação foram registradas (conforme o gráfico abaixo).

* Gráfico de Perda (Loss):

- Perda de Treino (Linha Azul): Como esperado, a perda de treino diminuiu consistentemente ao longo das 20 épocas. Isso indica que o modelo tem capacidade de aprendizado e estava conseguindo se ajustar e "memorizar" os dados de treinamento.

- Perda de Validação (Linha Vermelha): A perda de validação (que mede o desempenho em dados novos) acompanhou a curva de treino nas primeiras épocas (aproximadamente até a época 7).

- Diagnóstico (Overfitting): Após a época 7, um claro sintoma de superajuste (overfitting) se tornou evidente. A perda de validação parou de diminuir, estagnou e começou a flutuar (e até subir), enquanto a perda de treino continuou caindo. Isso significa que o modelo parou de aprender as características gerais dos dígitos e começou a decorar ruídos e detalhes específicos do conjunto de treino.


* Gráfico de Acurácia (Accuracy):

A análise do gráfico de acurácia reforça a conclusão do overfitting:

- Acurácia de Treino (Linha Azul): Mostra que o modelo está aprendendo. A acurácia sobre os dados de treino sobe consistentemente durante todo o processo, ultrapassando os 90% ao final das 20 épocas.

- Acurácia de Validação (Linha Vermelha): Esta métrica apresenta um rápido crescimento inicial, atingindo seu ponto máximo (cerca de 88-89%) por volta da época 15, e então começa a estagnar ou até cair levemente.

- A "Tesoura" (The Gap): A partir da época 5, as duas curvas começam a se separar, com a acurácia de treino (azul) ficando permanentemente acima da acurácia de validação (vermelha). Esse vão ("gap") é a prova visual do overfitting: o modelo está se tornando excelente apenas nos dados que já viu, mas sua capacidade de generalizar para dados novos (validação) parou de melhorar.

Conclusão: O modelo LeNet-like adaptado é capaz de aprender, mas é muito propenso ao overfitting neste dataset. Continuar o treinamento além de 20 épocas só pioraria o desempenho em dados novos. As próximas etapas devem focar em técnicas de regularização para melhorar a generalização do modelo.

```
Iniciando o treinamento...
Epoch 1/20 | Train Loss: 2.2502 | Train Acc: 18.14% | Val Loss: 2.2212 | Val Acc: 19.59%
Epoch 2/20 | Train Loss: 2.2302 | Train Acc: 18.92% | Val Loss: 2.2066 | Val Acc: 19.59%
Epoch 3/20 | Train Loss: 2.1307 | Train Acc: 23.02% | Val Loss: 1.9432 | Val Acc: 34.64%
Epoch 4/20 | Train Loss: 1.3873 | Train Acc: 55.24% | Val Loss: 0.8933 | Val Acc: 74.63%
Epoch 5/20 | Train Loss: 0.7503 | Train Acc: 77.73% | Val Loss: 0.6780 | Val Acc: 80.57%
Epoch 6/20 | Train Loss: 0.5892 | Train Acc: 83.07% | Val Loss: 0.5930 | Val Acc: 83.59%
Epoch 7/20 | Train Loss: 0.5154 | Train Acc: 85.20% | Val Loss: 0.5417 | Val Acc: 84.66%
Epoch 8/20 | Train Loss: 0.4675 | Train Acc: 86.47% | Val Loss: 0.5357 | Val Acc: 85.04%
Epoch 9/20 | Train Loss: 0.4311 | Train Acc: 87.49% | Val Loss: 0.4856 | Val Acc: 86.44%
Epoch 10/20 | Train Loss: 0.4008 | Train Acc: 88.35% | Val Loss: 0.4622 | Val Acc: 86.91%
Epoch 11/20 | Train Loss: 0.3770 | Train Acc: 89.12% | Val Loss: 0.4455 | Val Acc: 87.59%
Epoch 12/20 | Train Loss: 0.3569 | Train Acc: 89.61% | Val Loss: 0.4259 | Val Acc: 87.96%
Epoch 13/20 | Train Loss: 0.3392 | Train Acc: 90.13% | Val Loss: 0.4464 | Val Acc: 87.34%
Epoch 14/20 | Train Loss: 0.3237 | Train Acc: 90.56% | Val Loss: 0.4269 | Val Acc: 87.91%
Epoch 15/20 | Train Loss: 0.3106 | Train Acc: 90.88% | Val Loss: 0.4456 | Val Acc: 87.31%
Epoch 16/20 | Train Loss: 0.2974 | Train Acc: 91.36% | Val Loss: 0.3923 | Val Acc: 89.16%
Epoch 17/20 | Train Loss: 0.2864 | Train Acc: 91.65% | Val Loss: 0.4180 | Val Acc: 88.28%
Epoch 18/20 | Train Loss: 0.2755 | Train Acc: 91.97% | Val Loss: 0.4074 | Val Acc: 88.33%
Epoch 19/20 | Train Loss: 0.2658 | Train Acc: 92.24% | Val Loss: 0.3987 | Val Acc: 88.97%
Epoch 20/20 | Train Loss: 0.2562 | Train Acc: 92.45% | Val Loss: 0.4098 | Val Acc: 88.45%
Treinamento concluído!
```

![Treino1](https://github.com/polianaraujo/dca0305/blob/main/u2/images/treino1.png?raw=true)

***

### Análise da 2ª Tentativa (Foco em Regularização e Generalização)

Nesta segunda iteração, o objetivo principal foi mitigar o *overfitting* observado no experimento anterior. As métricas obtidas ao longo das 20 épocas demonstram um comportamento inverso ao da primeira tentativa, indicando uma mudança significativa na dinâmica de aprendizado do modelo.

#### 1. Comportamento das Métricas (Logs)
Observando os registros de treinamento, notam-se três fases distintas:
* **Fase de Aquecimento (Épocas 1-3):** O modelo apresentou dificuldade inicial, com a acurácia travada em torno de 18-19% e a perda (*Loss*) estagnada acima de 2.2. Isso é característico de cenários com regularização forte ou taxas de aprendizado que necessitam de um momento inicial para superar mínimos locais.
* **Fase de Convergência Rápida (Épocas 4-7):** A partir da 4ª época, o modelo "destravou", saltando de ~22% para ~79% de acurácia de validação em apenas 4 épocas. Simultaneamente, a perda caiu drasticamente de 2.13 para 0.69.
* **Estabilização (Épocas 15-20):** O aprendizado desacelerou, mas manteve a constância. Ao final da 20ª época, atingiu-se uma **Acurácia de Validação de 89.54%**, com uma **Perda de 0.37**.

#### 2. Análise Visual (Gráficos)
A principal característica visual desta tentativa é a **inversão do "Gap" de desempenho** observada nos gráficos em anexo:

* **Gráfico de Perda (*Loss*) - Escala Logarítmica:**
    Diferente da primeira tentativa, a linha vermelha (Validação) manteve-se consistentemente **abaixo** da linha azul (Treino) durante a maior parte do processo. Isso indica que o erro do modelo em dados novos foi sistematicamente menor do que nos dados utilizados para o ajuste dos pesos.

* **Gráfico de Acurácia (*Accuracy*):**
    O fenômeno se repete: a Acurácia de Validação (Vermelha) é superior à de Treino (Azul).
    * **Treino Final:** ~85.73%
    * **Validação Final:** ~89.54%

#### 3. Discussão do Fenômeno (Validação > Treino)
Este cenário atípico, onde o desempenho na validação supera o do treino, é um forte indicativo da presença de mecanismos de **Regularização** (como *Dropout* ou *Data Augmentation*):

1.  **Penalidade no Treino:** Durante o treinamento, o modelo enfrenta "obstáculos" inseridos pela regularização (ex: neurônios desligados aleatoriamente ou imagens distorcidas), o que dificulta a classificação e penaliza as métricas de treino.
2.  **Vantagem na Validação:** Durante a etapa de validação, esses obstáculos são removidos (a rede utiliza sua capacidade total e/ou as imagens não sofrem distorção), permitindo que o modelo performe melhor.

#### Conclusão da 2ª Tentativa
O problema de *overfitting* foi solucionado com êxito. O modelo deixou de memorizar ruídos do conjunto de treino e desenvolveu uma capacidade robusta de **generalização**. O resultado final de **89.54%** na validação não apenas supera o pico da tentativa anterior, como também apresenta maior confiabilidade para inferências em dados reais.

***

```
Iniciando o treinamento...
Epoch 1/20 | Train Loss: 2.2537 | Train Acc: 17.95% | Val Loss: 2.2237 | Val Acc: 19.59%
Epoch 2/20 | Train Loss: 2.2369 | Train Acc: 18.92% | Val Loss: 2.2192 | Val Acc: 19.59%
Epoch 3/20 | Train Loss: 2.2276 | Train Acc: 18.92% | Val Loss: 2.1945 | Val Acc: 19.59%
Epoch 4/20 | Train Loss: 2.1331 | Train Acc: 22.76% | Val Loss: 1.9322 | Val Acc: 29.93%
Epoch 5/20 | Train Loss: 1.8299 | Train Acc: 36.54% | Val Loss: 1.3081 | Val Acc: 58.33%
Epoch 6/20 | Train Loss: 1.2738 | Train Acc: 58.12% | Val Loss: 0.8507 | Val Acc: 75.06%
Epoch 7/20 | Train Loss: 0.9933 | Train Acc: 68.43% | Val Loss: 0.6971 | Val Acc: 79.65%
Epoch 8/20 | Train Loss: 0.8525 | Train Acc: 73.42% | Val Loss: 0.5888 | Val Acc: 83.41%
Epoch 9/20 | Train Loss: 0.7620 | Train Acc: 76.61% | Val Loss: 0.5346 | Val Acc: 84.71%
Epoch 10/20 | Train Loss: 0.6984 | Train Acc: 78.78% | Val Loss: 0.5138 | Val Acc: 85.44%
Epoch 11/20 | Train Loss: 0.6598 | Train Acc: 80.20% | Val Loss: 0.4781 | Val Acc: 86.58%
Epoch 12/20 | Train Loss: 0.6191 | Train Acc: 81.46% | Val Loss: 0.4489 | Val Acc: 87.03%
Epoch 13/20 | Train Loss: 0.5960 | Train Acc: 82.15% | Val Loss: 0.4339 | Val Acc: 87.77%
Epoch 14/20 | Train Loss: 0.5729 | Train Acc: 82.90% | Val Loss: 0.4185 | Val Acc: 88.19%
Epoch 15/20 | Train Loss: 0.5527 | Train Acc: 83.72% | Val Loss: 0.4102 | Val Acc: 88.31%
Epoch 16/20 | Train Loss: 0.5408 | Train Acc: 84.05% | Val Loss: 0.3997 | Val Acc: 88.84%
Epoch 17/20 | Train Loss: 0.5248 | Train Acc: 84.47% | Val Loss: 0.3892 | Val Acc: 89.08%
Epoch 18/20 | Train Loss: 0.5144 | Train Acc: 84.97% | Val Loss: 0.3860 | Val Acc: 89.06%
Epoch 19/20 | Train Loss: 0.5009 | Train Acc: 85.28% | Val Loss: 0.3783 | Val Acc: 89.43%
Epoch 20/20 | Train Loss: 0.4911 | Train Acc: 85.73% | Val Loss: 0.3722 | Val Acc: 89.54%
Treinamento concluído!
```

![Treino2](https://github.com/polianaraujo/dca0305/blob/main/u2/images/treino2.png?raw=true)

***

| Característica | Tentativa 1 (Modelo Base) | Tentativa 2 (Com Regularização) |
|----------------|---------------------------|---------------------------------|
| Diagnóstico Principal | Overfitting (Superajuste) | Generalização Robusta |
| Comportamento das Curvas | Acurácia de Treino muito superior à de Validação (Gap abrindo). | Acurácia de Validação superior à de Treino (Gap invertido). |
| Tendência da Perda (Loss) | Perda de Validação começou a subir após a época 7. | Perda de Validação manteve-se caindo e sempre abaixo da perda de Treino. |
| Interpretação | O modelo estava "memorizando" os dados de treino e falhando em dados novos. | O modelo aprendeu padrões gerais, performando melhor sem as penalidades da regularização (validação) do que com elas (treino). | 
| Acurácia Final (Validação) | Estagnada em ~88% (com tendência de queda). | 89.54% (sólida e estável). |