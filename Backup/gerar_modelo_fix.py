# Arquivo: gerar_modelo_fix.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("1. Lendo arquivo CSV...")
try:
    # Tenta ler o CSV (confirme se o nome é Obesity.csv ou obesity.csv)
    df = pd.read_csv('Obesity.csv')
except FileNotFoundError:
    print("ERRO: O arquivo 'Obesity.csv' não está na pasta. Verifique o nome.")
    exit()

print("2. Tratando dados...")
# Arredondamento (Correção do ruído)
cols_to_round = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in cols_to_round:
    df[col] = df[col].round().astype(int)

# Mapeamento Manual (Para garantir compatibilidade com o App em Inglês/Português)
mapping_dict = {
    'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3,
    'yes': 1, 'no': 0,
    'Female': 0, 'Male': 1
}

cols_map = ['family_history', 'FAVC', 'SMOKE', 'SCC', 'CAEC', 'CALC', 'Gender']

df_processed = df.copy()
for col in cols_map:
    df_processed[col] = df_processed[col].map(mapping_dict).fillna(0)

# One-Hot Encoding
df_processed = pd.get_dummies(df_processed, columns=['MTRANS'], drop_first=True)

# Target
le = LabelEncoder()
df_processed['Obesity_Encoded'] = le.fit_transform(df_processed['Obesity'])

# Separação X e y
X = df_processed.drop(['Obesity', 'Obesity_Encoded'], axis=1)
y = df_processed['Obesity_Encoded']

print("3. Treinando o modelo (Random Forest)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Artefatos para salvar
artifacts = {
    "model": rf_model,
    "label_encoder": le,
    "features": X.columns.tolist()
}

print("4. Salvando novo arquivo .pkl...")
joblib.dump(artifacts, 'modelo_obesidade.pkl')
print("✅ SUCESSO! Arquivo 'modelo_obesidade.pkl' foi gerado limpo.")