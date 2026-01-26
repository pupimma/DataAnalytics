import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configuração da página
st.set_page_config(page_title="Predição de Obesidade", page_icon="🏥", layout="wide")

# Função de carregamento com cache
@st.cache_resource
def load_model():
    try:
        # Caminho absoluto para evitar erros de diretório no deploy
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelo_obesidade.pkl')
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

# Inicialização
data_artifacts = load_model()

if data_artifacts is None:
    st.warning("Arquivo 'modelo_obesidade.pkl' não encontrado. Verifique o diretório.")
    st.stop()

model = data_artifacts["model"]
le = data_artifacts["label_encoder"]
features = data_artifacts["features"]

# Interface Principal
st.title("🏥 Sistema de Triagem de Obesidade")
st.markdown("Preencha os dados do paciente para realizar a análise preditiva de risco.")

st.sidebar.header("Ficha do Paciente")

def get_user_input():
    # Dados Demográficos
    gender = st.sidebar.selectbox("Gênero", ["Masculino", "Feminino"])
    age = st.sidebar.number_input("Idade", 14, 100, 25)
    height = st.sidebar.number_input("Altura (m)", 1.00, 2.50, 1.70)
    weight = st.sidebar.number_input("Peso (kg)", 30.0, 200.0, 70.0)
    
    st.sidebar.markdown("---")
    
    # Histórico e Hábitos
    fam_hist = st.sidebar.selectbox("Histórico Familiar de Obesidade?", ["Não", "Sim"])
    favc = st.sidebar.selectbox("Consome alimentos calóricos frequente?", ["Não", "Sim"])
    fcvc = st.sidebar.slider("Frequência de Vegetais (1=Nunca, 3=Sempre)", 1, 3, 2)
    ncp = st.sidebar.slider("Refeições principais por dia", 1, 4, 3)
    
    # Mapeamento de Frequência (CAEC/CALC)
    freq_map = {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}
    
    caec = st.sidebar.selectbox("Come entre refeições?", list(freq_map.keys()))
    smoke = st.sidebar.selectbox("Fumante?", ["Não", "Sim"])
    ch2o = st.sidebar.slider("Consumo de Água (1=Pouco, 3=Muito)", 1, 3, 2)
    scc = st.sidebar.selectbox("Monitora calorias?", ["Não", "Sim"])
    faf = st.sidebar.slider("Atividade Física Semanal (0=Nenhuma, 3=Alta)", 0, 3, 1)
    tue = st.sidebar.slider("Tempo em Dispositivos (0=Baixo, 2=Alto)", 0, 2, 1)
    calc = st.sidebar.selectbox("Consumo de Álcool", list(freq_map.keys()))
    
    # Transporte (Mapeamento para Inglês para OneHotEncoding posterior)
    trans_map = {
        "Transporte Público": "Public_Transportation",
        "Caminhada": "Walking",
        "Automóvel": "Automobile",
        "Motocicleta": "Motorbike",
        "Bicicleta": "Bike"
    }
    mtrans = st.sidebar.selectbox("Meio de Transporte", list(trans_map.keys()))

    # Construção do Dicionário (Já aplicando conversão binária/ordinal)
    user_data = {
        'Gender': 1 if gender == "Masculino" else 0,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history': 1 if fam_hist == "Sim" else 0,
        'FAVC': 1 if favc == "Sim" else 0,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': freq_map[caec],
        'SMOKE': 1 if smoke == "Sim" else 0,
        'CH2O': ch2o,
        'SCC': 1 if scc == "Sim" else 0,
        'FAF': faf,
        'TUE': tue,
        'CALC': freq_map[calc],
        'MTRANS': trans_map[mtrans] # Mantém string para get_dummies
    }
    
    return pd.DataFrame(user_data, index=[0])

# Processamento
df_input = get_user_input()

# Tratamento de variáveis categóricas (Dummy Variables)
df_processed = pd.get_dummies(df_input, columns=['MTRANS'])

# Garante alinhamento de colunas com o modelo treinado (preenche ausentes com 0)
df_processed = df_processed.reindex(columns=features, fill_value=0)

# Botão de Ação
if st.button("Realizar Diagnóstico"):
    try:
        # Inferência
        prediction = model.predict(df_processed)
        proba = model.predict_proba(df_processed)
        
        # Decodificação
        class_name = le.inverse_transform(prediction)[0]
        confidence = np.max(proba) * 100
        
        # Dicionário de Tradução Visual
        labels_pt = {
            'Insufficient_Weight': 'Abaixo do Peso',
            'Normal_Weight': 'Peso Normal',
            'Overweight_Level_I': 'Sobrepeso Nível I',
            'Overweight_Level_II': 'Sobrepeso Nível II',
            'Obesity_Type_I': 'Obesidade Tipo I',
            'Obesity_Type_II': 'Obesidade Tipo II',
            'Obesity_Type_III': 'Obesidade Tipo III (Mórbida)'
        }
        
        result_text = labels_pt.get(class_name, class_name)
        
        # Exibição
        st.subheader("Resultado")
        
        if "Obesity" in class_name:
            st.error(f"Diagnóstico: {result_text}")
        elif "Overweight" in class_name:
            st.warning(f"Diagnóstico: {result_text}")
        else:
            st.success(f"Diagnóstico: {result_text}")
            
        st.info(f"Probabilidade estimada: {confidence:.2f}%")
        
        # Visualização Gráfica
        st.divider()
        st.subheader("Probabilidades por Classe")
        
        cols_pt = [labels_pt.get(c, c) for c in le.classes_]
        df_proba = pd.DataFrame(proba, columns=cols_pt)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=df_proba.columns, y=df_proba.iloc[0].values, palette="viridis", ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Probabilidade")
        plt.xlabel("")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Erro no processamento: {e}")

