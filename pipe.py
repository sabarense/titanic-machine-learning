# ==============================
# 1. Importação de Bibliotecas
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ==============================
# 2. Funções de Pré-processamento
# ==============================
def load_data():
    """Carrega dados do diretório especificado"""
    train = pd.read_csv('content/sample_data/train.csv')
    test = pd.read_csv('content/sample_data/test.csv')
    return train, test

def preprocess_data(df, fit_preprocessor=None):
    """Pipeline completo de pré-processamento"""
    df = df.copy()

    # Engenharia de features
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Tratamento de nulos
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Remover colunas
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'])

    # Codificação categórica
    categorical_features = ['Sex', 'Embarked', 'Title', 'Pclass']
    numerical_features = ['Age', 'Fare', 'FamilySize']

    if fit_preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])
        processed_data = preprocessor.fit_transform(df)
        return processed_data, preprocessor
    else:
        processed_data = fit_preprocessor.transform(df)
        return processed_data

# ==============================
# 3. Pipeline Principal
# ==============================
# Carregar dados
train_raw, test_raw = load_data()

# Pré-processar dados de treino
X_train_processed, preprocessor = preprocess_data(train_raw.drop(columns='Survived'))
y_train = train_raw['Survived'].values

# Pré-processar dados de teste
X_test_processed = preprocess_data(test_raw, fit_preprocessor=preprocessor)

# Recriar DataFrames com colunas corretas
cat_encoder = preprocessor.named_transformers_['cat']
categorical_features = ['Sex', 'Embarked', 'Title', 'Pclass']
cat_columns = cat_encoder.get_feature_names_out(categorical_features)
numerical_features = ['Age', 'Fare', 'FamilySize']
all_columns = numerical_features + list(cat_columns)

train_processed = pd.DataFrame(X_train_processed, columns=all_columns)
train_processed['Survived'] = y_train

test_processed = pd.DataFrame(X_test_processed, columns=all_columns)
test_processed['Survived'] = np.nan  # Mantém a coluna

# ==============================
# 4. Modelagem Preditiva
# ==============================
# Divisão treino/validação
X = train_processed.drop(columns=['Survived'])
y = train_processed['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo 1: Random Forest
model_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_split=12,
    random_state=42
)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_val)

print(f"\n{'=' * 40}\nAvaliação do Modelo - Random Forest\n{'=' * 40}")
print(f"Acurácia: {accuracy_score(y_val, y_pred_rf):.2%}")
print(classification_report(y_val, y_pred_rf))

# Modelo 2: Árvore de Decisão
model_dt = DecisionTreeClassifier(max_depth=6, random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_val)

print(f"\n{'=' * 40}\nAvaliação do Modelo - Decision Tree\n{'=' * 40}")
print(f"Acurácia: {accuracy_score(y_val, y_pred_dt):.2%}")
print(classification_report(y_val, y_pred_dt))

# Comparação dos modelos
print("\nComparação dos Modelos:")
print(f"Random Forest - Acurácia: {accuracy_score(y_val, y_pred_rf):.2%}")
print(f"Decision Tree - Acurácia: {accuracy_score(y_val, y_pred_dt):.2%}")

# ==============================
# 5. Clusterização (K-Means + PCA)
# ==============================
# Usando dados completos de treino para clusterização
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', alpha=0.8)
plt.title('Clusterização de Passageiros (K-Means + PCA)', fontsize=14)
plt.xlabel('Componente Principal 1', fontsize=12)
plt.ylabel('Componente Principal 2', fontsize=12)
plt.grid(True)
plt.show()

# Interpretação dos clusters
train_processed['Cluster'] = clusters
print("\nResumo dos clusters:")
print(train_processed.groupby('Cluster').mean(numeric_only=True)[['Survived', 'Age', 'Fare', 'FamilySize']])

print("\nInterpretação dos clusters:")
for cluster in range(3):
    grupo = train_processed[train_processed['Cluster'] == cluster]
    print(f"Cluster {cluster}:")
    print(f"  Sobreviventes: {grupo['Survived'].mean():.2%}")
    print(f"  Idade média: {grupo['Age'].mean():.2f}")
    print(f"  Tamanho médio da família: {grupo['FamilySize'].mean():.2f}")
    print(f"  Tarifa média: {grupo['Fare'].mean():.2f}")
    print("  ---")

# ==============================
# 6. Regras de Associação
# ==============================
df_rules = train_processed.copy()
df_rules['Survived'] = df_rules['Survived'].map({0: 'Não', 1: 'Sim'})

# Converter para transações
transactions = []
for _, row in df_rules.iterrows():
    transaction = []
    for col in df_rules.columns:
        if col == 'Survived':
            transaction.append(f"Survived={row[col]}")
        elif row[col] == 1 and col.startswith('Sex_'):
            transaction.append(col.split('_')[1])
        elif row[col] == 1 and col.startswith('Embarked_'):
            transaction.append(f"Embarked={col.split('_')[1]}")
        elif row[col] == 1 and col.startswith('Title_'):
            transaction.append(f"Title={col.split('_')[1]}")
        elif row[col] == 1 and col.startswith('Pclass_'):
            transaction.append(f"Pclass={col.split('_')[1]}")
    transactions.append(transaction)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

print(f"\n{'=' * 40}\nTop 5 Regras de Associação\n{'=' * 40}")
top_rules = rules.sort_values(by='lift', ascending=False).head(5)
print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Interpretação de 3 regras
print("\nInterpretação das 3 principais regras de associação:")
for idx, rule in top_rules.head(3).iterrows():
    antecedents = ', '.join([str(a) for a in rule['antecedents']])
    consequents = ', '.join([str(c) for c in rule['consequents']])
    print(f"Regra: SE [{antecedents}] ENTÃO [{consequents}]")
    print(f"  Suporte: {rule['support']:.2%}")
    print(f"  Confiança: {rule['confidence']:.2%}")
    print(f"  Lift: {rule['lift']:.2f}")
    # Exemplo de interpretação:
    if 'female' in antecedents and 'Pclass=1' in antecedents and 'Survived=Sim' in consequents:
        print("  Interpretação: Mulheres da 1ª classe têm alta probabilidade de sobreviver.")
    elif 'male' in antecedents and 'Survived=Não' in consequents:
        print("  Interpretação: Homens têm maior chance de não sobreviver.")
    elif 'Title=Miss' in antecedents and 'Survived=Sim' in consequents:
        print("  Interpretação: Passageiras com título 'Miss' têm probabilidade aumentada de sobrevivência.")
    print("  ---")

# ==============================
# 7. Visualizações Chave
# ==============================
plt.figure(figsize=(15, 5))

# Sobrevivência por Sexo
plt.subplot(1, 3, 1)
sns.barplot(x='Sex_male', y='Survived', data=train_processed)
plt.title('Sobrevivência por Sexo')
plt.xticks([0, 1], ['Feminino', 'Masculino'])

# Distribuição de Idades
plt.subplot(1, 3, 2)
sns.histplot(data=train_processed, x='Age', hue='Survived', kde=True, bins=30)
plt.title('Distribuição de Idades')

# Importância das Features
plt.subplot(1, 3, 3)
feature_imp = pd.Series(model_rf.feature_importances_, index=X.columns).sort_values()
feature_imp.plot(kind='barh', title='Importância das Features')
plt.tight_layout()
plt.show()

# ==============================
# 8. Geração de Submissão
# ==============================
test_pred = model_rf.predict(test_processed.drop(columns=['Survived']))
submission = pd.DataFrame({
    'PassengerId': test_raw['PassengerId'],
    'Survived': test_pred.astype(int)
})
submission.to_csv('content/sample_data/titanic_submission.csv', index=False)
print("\nArquivo de submissão salvo em: content/sample_data/titanic_submission.csv")

# ==============================
# 9. Conclusão Final
# ==============================
print("""
Conclusão:
- O modelo Random Forest apresentou desempenho superior ao Decision Tree, com melhor acurácia e F1-score.
- A clusterização revelou grupos distintos: um cluster com alta taxa de sobrevivência composto majoritariamente por mulheres jovens e famílias pequenas.
- As regras de associação confirmaram que ser mulher, estar na 1ª classe e ter o título 'Miss' aumentam significativamente a chance de sobrevivência.
- O perfil socioeconômico e o gênero são determinantes importantes para a sobrevivência no Titanic.
""")
