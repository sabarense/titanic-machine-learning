# 🚢 Titanic Machine Learning

## 📊 Sobre o Projeto

Análise dos dados do Titanic utilizando Python e bibliotecas de ciência de dados. O objetivo é identificar padrões de sobrevivência dos passageiros, aplicando técnicas de pré-processamento, classificação, clusterização, regras de associação e visualizações.

## 🛠️ Tecnologias Utilizadas

- 🐍 Python 3
- 🐼 pandas, numpy
- 🤖 scikit-learn
- 📈 matplotlib, seaborn
- 🔗 mlxtend

## 🔎 Etapas do Projeto

1. **Pré-processamento dos Dados** 🧹
   - Limpeza, tratamento de valores nulos e engenharia de features.
   - Padronização de variáveis numéricas e codificação de variáveis categóricas.

2. **Modelagem de Classificação** 🌲🌳
   - Random Forest e Árvore de Decisão.
   - Avaliação dos modelos com acurácia, precisão, recall e F1-score.

3. **Clusterização** 🧩
   - K-Means para agrupar passageiros com perfis semelhantes.
   - Visualização dos clusters com PCA.

4. **Regras de Associação** 📋
   - Algoritmo Apriori para extrair padrões de sobrevivência.
   - Interpretação das principais regras encontradas.

5. **Visualizações** 📊
   - Gráficos para análise exploratória e apresentação dos resultados.

6. **Geração de Submissão** 📝
   - Criação de arquivo para submissão das previsões.

## ▶️ Como Executar

1. Clone este repositório:
````
git clone https://github.com/sabarense/titanic-machine-learning.git
````
2. Instale as dependências:
````
pip install -r requirements.txt
````

3. Execute `pipe.py` conforme instruções no projeto.

## 🏆 Resultados

- O Random Forest obteve o melhor desempenho na previsão de sobrevivência.
- A clusterização revelou grupos distintos, como mulheres jovens da 1ª classe com alta taxa de sobrevivência.
- As regras de associação reforçaram a influência do sexo, classe e título na sobrevivência.

## 💡 Conclusão

O projeto mostra como diferentes técnicas de machine learning podem ser usadas em conjunto para extrair insights relevantes de dados históricos, destacando fatores determinantes na sobrevivência dos passageiros do Titanic.
