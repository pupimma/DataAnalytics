# gerar_modelo.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("🔄 Iniciando geração do modelo...")

# 1. Carga
try:
    df = pd.read_csv('Obesity.csv')
except FileNotFoundError:
    print("❌ Erro: Obesity.csv não encontrado.")
    exit()

# 2. Limpeza e Arredondamento
cols_to_round = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in cols_to_round:
    df[col] = df[col].round().astype(int)

# 3. Tratamento de Dados (Para Inglês - O modelo treina em inglês)
# Mapeamento Manual
mapping_dict = {
    'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3,
    'yes': 1, 'no': 0,
    'Female': 0, 'Male': 1
}

cols_map = ['family_history', 'FAVC', 'SMOKE', 'SCC', 'CAEC', 'CALC', 'Gender']

df_processed = df.copy()
for col in cols_map:
    # Garante que só mapeia o que existe no dicionário
    df_processed[col] = df_processed[col].map(mapping_dict).fillna(0)

# One-Hot Encoding
df_processed = pd.get_dummies(df_processed, columns=['MTRANS'], drop_first=True)

# Target
le = LabelEncoder()
df_processed['Obesity_Encoded'] = le.fit_transform(df_processed['Obesity'])

# Separação
X = df_processed.drop(['Obesity', 'Obesity_Encoded'], axis=1)
y = df_processed['Obesity_Encoded']

# Treino
print("🧠 Treinando Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# 4. Salvar
artifacts = {
    "model": rf_model,
    "label_encoder": le,
    "features": X.columns.tolist()
}

joblib.dump(artifacts, 'modelo_obesidade.pkl')
print("✅ Sucesso! Novo arquivo 'modelo_obesidade.pkl' gerado.")