import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🏥",
    layout="wide"
)

# --- CARREGAMENTO DO MODELO (COM CAMINHO ABSOLUTO) ---
@st.cache_resource
def load_model():
    try:
        # Pega o diretório onde o arquivo app.py está rodando
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        
        # Monta o caminho para o arquivo .pkl
        caminho_modelo = os.path.join(diretorio_atual, 'modelo_obesidade.pkl')
        
        # Carrega o modelo
        return joblib.load(caminho_modelo)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Erro inesperado ao carregar modelo: {e}")
        return None

# Carrega os artefatos
artifacts = load_model()

# --- VERIFICAÇÃO DE SEGURANÇA ---
if artifacts is None:
    st.error("❌ Erro Crítico: O arquivo 'modelo_obesidade.pkl' não foi encontrado.")
    st.warning("Certifique-se de que o arquivo .pkl está na mesma pasta que este app.py e que você fez o upload dele para o GitHub.")
    st.stop()

model = artifacts["model"]
le = artifacts["label_encoder"]
feature_columns = artifacts["features"]

# --- INTERFACE: TÍTULO ---
st.title("🏥 Sistema de Triagem de Obesidade")
st.markdown("""
Este sistema utiliza Inteligência Artificial para analisar fatores de risco e prever o diagnóstico 
de obesidade. **Preencha os dados na barra lateral.**
""")

# --- BARRA LATERAL (INPUTS) ---
st.sidebar.header("📋 Dados do Paciente")

def get_user_data():
    # 1. Dados Pessoais
    genero = st.sidebar.selectbox("Gênero", ["Masculino", "Feminino"])
    idade = st.sidebar.number_input("Idade", 14, 100, 25)
    altura = st.sidebar.number_input("Altura (m)", 1.00, 2.50, 1.70)
    peso = st.sidebar.number_input("Peso (kg)", 30.0, 200.0, 70.0)
    
    st.sidebar.markdown("---")
    
    # 2. Histórico e Hábitos (Interface PT -> Valor Interno)
    
    # Histórico Familiar (yes/no)
    hist_fam = st.sidebar.selectbox("Histórico Familiar de Obesidade?", ["Sim", "Não"])
    family_history = 1 if hist_fam == "Sim" else 0
    
    # Alimentos Calóricos (FAVC) (yes/no)
    favc_input = st.sidebar.selectbox("Consome alimentos calóricos com frequência?", ["Sim", "Não"])
    favc = 1 if favc_input == "Sim" else 0
    
    # Vegetais (FCVC) (1-3)
    fcvc = st.sidebar.slider("Frequência de consumo de vegetais (1=Nunca, 3=Sempre)", 1, 3, 2)
    
    # Refeições (NCP) (1-4)
    ncp = st.sidebar.slider("Número de refeições principais por dia", 1, 4, 3)
    
    # Beliscar (CAEC) (Scale)
    mapa_caec = {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}
    caec_label = st.sidebar.selectbox("Come entre as refeições?", list(mapa_caec.keys()))
    caec = mapa_caec[caec_label]
    
    # Fumante (SMOKE) (yes/no)
    smoke_input = st.sidebar.selectbox("Fumante?", ["Sim", "Não"])
    smoke = 1 if smoke_input == "Sim" else 0
    
    # Água (CH2O) (1-3)
    ch2o = st.sidebar.slider("Consumo diário de água (1=Pouco, 3=Muito)", 1, 3, 2)
    
    # Monitora Calorias (SCC) (yes/no)
    scc_input = st.sidebar.selectbox("Monitora calorias ingeridas?", ["Sim", "Não"])
    scc = 1 if scc_input == "Sim" else 0
    
    # Atividade Física (FAF) (0-3)
    faf = st.sidebar.slider("Frequência de atividade física semanal (0=Nenhuma, 3=Muita)", 0, 3, 1)
    
    # Eletrônicos (TUE) (0-2)
    tue = st.sidebar.slider("Tempo usando dispositivos eletrônicos (0=Pouco, 2=Muito)", 0, 2, 1)
    
    # Álcool (CALC) (Scale)
    mapa_calc = {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}
    calc_label = st.sidebar.selectbox("Consumo de álcool", list(mapa_calc.keys()))
    calc = mapa_calc[calc_label]
    
    # Transporte (MTRANS) -> Mapeia para Inglês para o OneHotEncoding funcionar
    mapa_transporte = {
        "Transporte Público": "Public_Transportation",
        "Caminhada": "Walking",
        "Automóvel": "Automobile",
        "Motocicleta": "Motorbike",
        "Bicicleta": "Bike"
    }
    transporte_label = st.sidebar.selectbox("Meio de transporte principal", list(mapa_transporte.keys()))
    mtrans = mapa_transporte[transporte_label]
    
    # Retorna dicionário com os dados brutos
    # O Gênero precisa ser convertido aqui: Male=1, Female=0
    gender_val = 1 if genero == "Masculino" else 0
    
    user_data = {
        'Gender': gender_val,
        'Age': idade,
        'Height': altura,
        'Weight': peso,
        'family_history': family_history,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans # Ainda é string, será tratado abaixo
    }
    
    return pd.DataFrame(user_data, index=[0])

# Pega os dados do usuário
input_df = get_user_data()

# --- PROCESSAMENTO DOS DADOS ---
# Aplica OneHotEncoding na coluna de transporte
df_processed = pd.get_dummies(input_df, columns=['MTRANS'])

# REINDEX: Garante que as colunas sejam EXATAMENTE as que o modelo aprendeu
# Se faltar alguma coluna (ex: usuário escolheu Carro, mas modelo tem coluna Moto), preenche com 0
df_processed = df_processed.reindex(columns=feature_columns, fill_value=0)

# --- BOTÃO DE PREDIÇÃO ---
if st.button("🔍 Realizar Diagnóstico"):
    try:
        # Faz a predição
        prediction = model.predict(df_processed)
        prediction_proba = model.predict_proba(df_processed)
        
        # Pega o nome da classe original (Inglês)
        classe_original = le.inverse_transform(prediction)[0]
        
        # Dicionário de tradução para exibição
        traducoes = {
            'Insufficient_Weight': 'Abaixo do Peso',
            'Normal_Weight': 'Peso Normal',
            'Overweight_Level_I': 'Sobrepeso Nível I',
            'Overweight_Level_II': 'Sobrepeso Nível II',
            'Obesity_Type_I': 'Obesidade Tipo I',
            'Obesity_Type_II': 'Obesidade Tipo II',
            'Obesity_Type_III': 'Obesidade Tipo III'
        }
        
        resultado_pt = traducoes.get(classe_original, classe_original)
        
        # Confiança
        confianca = np.max(prediction_proba) * 100
        
        # --- EXIBIÇÃO DOS RESULTADOS ---
        st.subheader("Resultado da Análise:")
        
        if "Obesity" in classe_original:
            st.error(f"⚠️ Diagnóstico: **{resultado_pt}**")
        elif "Overweight" in classe_original:
            st.warning(f"⚠️ Diagnóstico: **{resultado_pt}**")
        else:
            st.success(f"✅ Diagnóstico: **{resultado_pt}**")
            
        st.info(f"🎯 Nível de Confiança do Modelo: **{confianca:.2f}%**")
        
        # --- GRÁFICO ---
        st.markdown("---")
        st.subheader("📊 Probabilidades Detalhadas")
        
        # Cria dataframe para o gráfico com nomes traduzidos
        colunas_traduzidas = [traducoes.get(c, c) for c in le.classes_]
        proba_df = pd.DataFrame(prediction_proba, columns=colunas_traduzidas)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=proba_df.columns, y=proba_df.iloc[0].values, palette="viridis", ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Probabilidade")
        plt.title("Análise de Risco por Categoria")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Erro ao processar predição: {e}")

# --- RODAPÉ ---
st.markdown("---")
st.markdown("**Tech Challenge Fase 4** | Sistema de Apoio à Decisão Médica")