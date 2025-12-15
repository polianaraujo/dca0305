Aqui está uma proposta de **README.md** objetivo e profissional para o seu repositório no GitHub, estruturado com base nas informações do seu artigo.

---

#Otimizadores Adaptativos e Learning Rate Schedulers (SVHN)Este projeto apresenta uma análise comparativa aprofundada sobre algoritmos de otimização e estratégias de agendamento de taxa de aprendizado (Learning Rate Schedulers) em Deep Learning, aplicados ao dataset **SVHN (Street View House Numbers)**.

O estudo explora desde a matemática da suavização de gradientes até o impacto prático na convergência de redes neurais convolucionais (CNNs).

##📋 Tópicos Abordados* **EWMA (Exponentially Weighted Moving Averages):** Análise da suavização de gradientes e comparação com médias simples (SMA).
* **Otimizadores:**
* **Adam:** Impacto dos hiperparâmetros (\beta_1, \beta_2, \epsilon).
* **SGD:** Comparação entre SGD Puro, Momentum e Nesterov.


* **Schedulers:** Comparação de performance entre `StepLR` e `MultiStepLR`.
* **Visualização Interna:** Inspeção de gradientes crus vs. adaptados durante o treino.

##📊 Principais ResultadosCom base nos experimentos realizados:

1. **Estabilidade:** A aplicação de EWMA nos gradientes é crítica para reduzir a variância e estabilizar o treino.
2. **Performance:** O **MultiStepLR** obteve o melhor desempenho (92.41% de acurácia) comparado ao StepLR, demonstrando melhor refinamento na descida do gradiente.
3. **Convergência:** O **Adam** domina na velocidade de convergência inicial, ideal para prototipagem, enquanto o **SGD + Momentum** (bem ajustado) tende a oferecer melhor generalização final.

##🛠️ Instalação e ExecuçãoPara reproduzir os experimentos:

1. Clone o repositório:
```bash
git clone https://github.com/polianaraujo/dca0305
cd dca0305/u3

```


2. Instale as dependências:
```bash
pip install -r requirements.txt

```


3. Execute o notebook principal:
```bash
jupyter notebook notebooks/02_comparacao_otimizadores.ipynb

```



##📚 Referências Principais* *Adam: A method for stochastic optimization* (Kingma & Ba, 2014)
* *A disciplined approach to neural network hyper-parameters* (Smith, 2018)
* *On the importance of initialization and momentum in deep learning* (Sutskever et al., 2013)

##👥 AutoresProjeto desenvolvido como parte da disciplina de **Projetos de Sistemas Baseados em Machine Learning (DCA0305 - UFRN)**.

* **Poliana Ellen de Araújo** - [polianaellena123@gmail.com](mailto:polianaellena123@gmail.com)
* **Vitor Gabriel da Silva Alves** - [vitorg135@gmail.com](mailto:vitorg135@gmail.com)