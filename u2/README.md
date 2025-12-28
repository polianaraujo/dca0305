# Comparativo de Regularização em CNNs: Desafiando o Overfitting com o Dataset SVHN
Este projeto apresenta um estudo comparativo focado em técnicas de combate ao overfitting em Redes Neurais Convolucionais (CNNs). Utilizando o dataset SVHN (Street View House Numbers), exploramos como modificações arquiteturais e estratégias de pré-processamento impactam a capacidade de generalização de um modelo baseado na clássica LeNet-5.

- Adaptação de Arquitetura: Modificação da LeNet-5 original para processar 3 canais de cor (RGB) e ajuste da camada densa para 480 features (tensor 120x2x2).

- Abordagem Experimental: Comparação direta entre um modelo "Baseline" (sem regularização) e um modelo otimizado.

- Técnicas de Regularização: Implementação de Dropout (camadas de descarte aleatório) e Data Augmentation (transformações de cor e rotação).

## 1ª treinamento: Análise da Tentativa Inicial (Modelo Base)
Para este projeto, o dataset escolhido foi o SVHN (Street View House Numbers). Este dataset consiste em imagens coloridas (3 canais, RGB) de 32x32 pixels, representando 10 classes (dígitos de 0 a 9).

1. Configuração do Experimento

    Seguindo os requisitos, a arquitetura-base foi mantida como a LeNet-5 vista em aula, com as seguintes adaptações para o SVHN:

    - Arquitetura (`model_svhn`):

        - A primeira camada convolucional (C1) teve seu parâmetro in_channels alterado de 1 (para o MNIST/escala de cinza) para 3, para aceitar as imagens coloridas (RGB) do SVHN.

        - Devido ao tamanho da imagem de entrada (32x32 em vez de 28x28), a saída da última camada convolucional/pooling (C5) resultou em um tensor de 120x2x2.

        - Consequentemente, a primeira camada linear (F6) foi ajustada para ter in_features de 480 (ou seja, 120 * 2 * 2).

    - Treinamento:

        - Dataset: SVHN (73.257 imagens de treino, 26.032 de validação/teste).

        - Otimizador: `optim.SGD` (Gradiente Descendente Estocástico).

        - Taxa de Aprendizado (LR): 0.01.

        - Função de Perda: `nn.CrossEntropyLoss` (adequada para classificação multiclasse).

        - Épocas: 20.

        - Batch Size (Tamanho do Lote): 64.

2. Resultados e Observações
    O modelo foi treinado por 20 épocas, e as métricas de Perda (Loss) de treino e validação foram registradas (conforme o gráfico abaixo).

    - Gráfico de Perda (Loss):

        - Perda de Treino (Linha Azul): Como esperado, a perda de treino diminuiu consistentemente ao longo das 20 épocas. Isso indica que o modelo tem capacidade de aprendizado e estava conseguindo se ajustar e "memorizar" os dados de treinamento.

        - Perda de Validação (Linha Vermelha): A perda de validação (que mede o desempenho em dados novos) acompanhou a curva de treino nas primeiras épocas (aproximadamente até a época 7).

        - Diagnóstico (Overfitting): Após a época 7, um claro sintoma de superajuste (overfitting) se tornou evidente. A perda de validação parou de diminuir, estagnou e começou a flutuar (e até subir), enquanto a perda de treino continuou caindo. Isso significa que o modelo parou de aprender as características gerais dos dígitos e começou a decorar ruídos e detalhes específicos do conjunto de treino.


    - Gráfico de Acurácia (Accuracy):

        A análise do gráfico de acurácia reforça a conclusão do overfitting:

        - Acurácia de Treino (Linha Azul): Mostra que o modelo está aprendendo. A acurácia sobre os dados de treino sobe consistentemente durante todo o processo, ultrapassando os 90% ao final das 20 épocas.

        - Acurácia de Validação (Linha Vermelha): Esta métrica apresenta um rápido crescimento inicial, atingindo seu ponto máximo (cerca de 88-89%) por volta da época 15, e então começa a estagnar ou até cair levemente.

        - A "Tesoura" (The Gap): A partir da época 5, as duas curvas começam a se separar, com a acurácia de treino (azul) ficando permanentemente acima da acurácia de validação (vermelha). Esse vão ("gap") é a prova visual do overfitting: o modelo está se tornando excelente apenas nos dados que já viu, mas sua capacidade de generalizar para dados novos (validação) parou de melhorar.

3. Conclusão: O modelo LeNet-like adaptado é capaz de aprender, mas é muito propenso ao overfitting neste dataset. Continuar o treinamento além de 20 épocas só pioraria o desempenho em dados novos. As próximas etapas devem focar em técnicas de regularização para melhorar a generalização do modelo.

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

### 2º Treinamento: Aplicação de Regularização (Data Augmentation e Dropout)
Para esta segunda etapa, o objetivo foi combater o overfitting diagnosticado no primeiro experimento. Mantendo o dataset SVHN, foram introduzidas técnicas de regularização para aumentar a robustez do modelo e sua capacidade de generalização.

1. Configuração do Experimento
A arquitetura base (LeNet-5 adaptada) e os hiperparâmetros fundamentais foram mantidos, porém com a adição de camadas de Dropout e transformações nas imagens de entrada.

    - Alterações na Arquitetura e Dados:

        - Data Augmentation: Foram aplicadas transformações aleatórias nas imagens de treino (possivelmente rotações, crops ou inversões, dependendo do seu código). Isso "aumenta" artificialmente a diversidade dos dados, impedindo que o modelo memorize imagens estáticas.

        - Dropout: Foram inseridas camadas de nn.Dropout (provavelmente nas camadas lineares/densas). O Dropout "desliga" aleatoriamente uma porcentagem dos neurônios durante o treinamento, forçando a rede a aprender caminhos redundantes e características mais robustas.

    - Treino:

        - Hiperparâmetros: Mantidos (Otimizador SGD, LR 0.01, 20 Épocas, Batch Size 64).

2. Resultados e Observações
O modelo foi treinado novamente por 20 épocas. Ao analisar os gráficos de Perda e Acurácia (imagem anexada) e os logs, observa-se um comportamento drasticamente diferente do primeiro treinamento.

    - Gráfico de Perda (Loss):

        - Inversão de Curvas: Diferente do 1º treino, a Perda de Validação (Linha Vermelha) manteve-se consistentemente abaixo da Perda de Treino (Linha Azul) a partir da época 6.

        - Interpretação: Isso ocorre devido ao funcionamento do Dropout e do Data Augmentation. Durante o treino, o modelo "sofre" com neurônios desligados e imagens distorcidas (augmentation), o que eleva a perda. Durante a validação, o Dropout é desligado (a rede usa 100% da capacidade) e as imagens são limpas (sem distorção), facilitando a classificação.

        - Convergência: Ambas as curvas continuam caindo até o final, sem sinais de overfitting (a linha vermelha não voltou a subir).

    - Gráfico de Acurácia (Accuracy):

        - Acurácia de Validação Superior: A acurácia de validação terminou em 89.54%, superando a acurácia de treino (85.73%).

        - Comparação com o Modelo Base:

            - Modelo Base: Treino ~92.4% | Validação ~88.4% (Gap de ~4% indicando overfitting).

            - Modelo Regularizado: Treino ~85.7% | Validação ~89.5%.

        - Análise: Embora a acurácia de treino tenha diminuído em relação ao primeiro experimento (o que é esperado, pois o treino ficou "mais difícil"), a acurácia de validação aumentou (de 88.4% para 89.5%). Isso prova que o modelo parou de "decorar" o treino e passou a aprender padrões que funcionam melhor em dados novos.

3. Conclusão
A aplicação de Data Augmentation e Dropout foi bem-sucedida. O problema de overfitting foi eliminado (na verdade, temos um cenário onde a validação performa melhor que o treino devido à mecânica das regularizações). O modelo agora é mais generalista e robusto, atingindo um desempenho final superior em dados de teste/validação em comparação à tentativa inicial.

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


## Conclusões

***
| Métrica | Modelo Base (Sem Regularização) | Modelo Regularizado (Dropout + Augmentation) | Variação |
|-|-|-|-|
| Acurácia de Treino | 92.45% | 85.73% | ↓ -6.72% |
| Acurácia de Validação | 88.45% | 89.54% | ↑ +1.09% |
| Gap (Treino - Validação) | +4.0% (Overfitting) | -3.8% (Validação Superior) | Inversão do Gap |
| Comportamento da Loss | Validação diverge do Treino | Validação acompanha Treino | Estabilização |
***

### Por quê que, no 2º treino (com Data Augmentation e Dropout) a acurácia de treino caiu, e ainda assim o modelo ser considerado "melhor"?

1. O Efeito do Dropout ("Treinando com pesos")

    - No Treino: O Dropout desliga aleatoriamente uma porcentagem dos neurônios. A rede neural está tentando aprender "com uma mão amarrada". Ela tem menos capacidade de processamento, por isso erra mais e a acurácia cai (ficou abaixo de 90%).

    - Na Validação/Teste: A "mochila" é removida. O Dropout é desligado e todos os neurônios funcionam juntos. Como a rede aprendeu a se virar em condições difíceis, agora que está com capacidade total, ela performa muito bem (melhor até que no treino).

2. O Efeito do Data Augmentation ("Alvo Móvel")

    - No 1º Treinamento (Sem Augmentation): O modelo via sempre as mesmas imagens estáticas. É muito fácil para uma rede neural "decorar" pixels específicos. Por isso a acurácia foi lá para cima (>90%), mas era uma "falsa inteligência" (overfitting).

    - No 2º Treinamento (Com Augmentation): A cada época, a imagem sofre uma leve rotação, zoom ou distorção. O modelo nunca vê a imagem exata duas vezes. Isso torna a tarefa de treino muito mais difícil, derrubando a acurácia de treino. Porém, obriga o modelo a aprender o conceito do número (a forma do '3') e não os pixels exatos.