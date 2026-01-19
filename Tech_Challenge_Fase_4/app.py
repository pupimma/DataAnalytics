import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuração da Página
st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🏥",
    layout="wide"
)

# --- CARREGAMENTO DO MODELO ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('modelo_obesidade.pkl')
    except FileNotFoundError:
        return None

artifacts = load_model()

# --- TÍTULO E INTRODUÇÃO ---
st.title("🏥 Sistema de Triagem de Obesidade")
st.markdown("""
Este sistema utiliza Inteligência Artificial para analisar fatores de risco e prever o diagnóstico 
de obesidade. Preencha os dados do paciente na barra lateral para obter o resultado.
""")

if artifacts is None:
    st.error("Erro: O arquivo 'modelo_obesidade.pkl' não foi encontrado.")
    st.stop()

model = artifacts["model"]
le = artifacts["label_encoder"]
feature_columns = artifacts["features"]

# --- BARRA LATERAL (Entrada de Dados em Português) ---
st.sidebar.header("📋 Dados do Paciente")

def user_input_features():
    # Dados Fisiológicos
    genero = st.sidebar.selectbox("Gênero", ["Masculino", "Feminino"])
    idade = st.sidebar.number_input("Idade", 14, 100, 25)
    altura = st.sidebar.number_input("Altura (m)", 1.00, 2.50, 1.70)
    peso = st.sidebar.number_input("Peso (kg)", 30.0, 200.0, 70.0)

    st.sidebar.markdown("---")
    
    # Histórico e Hábitos
    hist_familiar = st.sidebar.selectbox("Histórico Familiar de Obesidade?", ["Sim", "Não"])
    calorico = st.sidebar.selectbox("Consome alimentos calóricos com frequência?", ["Sim", "Não"])
    vegetais = st.sidebar.slider("Frequência de consumo de vegetais (1=Nunca, 3=Sempre)", 1, 3, 2)
    refeicoes = st.sidebar.slider("Número de refeições principais por dia", 1, 4, 3)
    beliscar = st.sidebar.selectbox("Come entre as refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
    fumante = st.sidebar.selectbox("Fumante?", ["Sim", "Não"])
    agua = st.sidebar.slider("Consumo diário de água (1=Pouco, 3=Muito)", 1, 3, 2)
    monitora = st.sidebar.selectbox("Monitora calorias ingeridas?", ["Sim", "Não"])
    fisico = st.sidebar.slider("Frequência de atividade física semanal (0=Nenhuma, 3=Muita)", 0, 3, 1)
    eletronicos = st.sidebar.slider("Tempo usando dispositivos eletrônicos (0=Pouco, 2=Muito)", 0, 2, 1)
    alcool = st.sidebar.selectbox("Consumo de álcool", ["Não", "Às vezes", "Frequentemente", "Sempre"])
    transporte = st.sidebar.selectbox("Meio de transporte principal", 
                                      ["Transporte Público", "Caminhada", "Automóvel", "Motocicleta", "Bicicleta"])

    # Criando o DataFrame com os nomes das colunas já traduzidos (igual ao notebook)
    data = {
        'Gênero': genero, 'Idade': idade, 'Altura': altura, 'Peso': peso,
        'Histórico_Familiar': hist_familiar, 'Consumo_Calórico': calorico, 
        'Consumo_Vegetais': vegetais, 'Refeições_Dia': refeicoes, 
        'Comer_Entre_Refeições': beliscar, 'Fumante': fumante, 
        'Consumo_Água': agua, 'Monitora_Calorias': monitora, 
        'Atividade_Física': fisico, 'Tempo_Eletrônicos': eletronicos, 
        'Consumo_Álcool': alcool, 'Transporte': transporte
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- PRÉ-PROCESSAMENTO ---
# O dicionário de mapeamento deve coincidir com o usado no treinamento
mapa_ordinais = {
    'Não': 0, 'Às vezes': 1, 'Frequentemente': 2, 'Sempre': 3,
    'Sim': 1, 'Não': 0,
    'Feminino': 0, 'Masculino': 1
}

colunas_map = ['Histórico_Familiar', 'Consumo_Calórico', 'Fumante', 'Monitora_Calorias', 
               'Comer_Entre_Refeições', 'Consumo_Álcool', 'Gênero']

df_processed = input_df.copy()

# Aplicando mapeamento manual
for col in colunas_map:
    df_processed[col] = df_processed[col].map(mapa_ordinais)

# One-Hot Encoding para Transporte
df_processed = pd.get_dummies(df_processed, columns=['Transporte'])

# Reindexar colunas para garantir compatibilidade com o modelo treinado
# Isso garante que todas as colunas de "Transporte" existam, mesmo que não selecionadas
df_processed = df_processed.reindex(columns=feature_columns, fill_value=0)

# --- PREDIÇÃO E RESULTADOS ---
if st.button("🔍 Realizar Diagnóstico"):
    
    # Predição da Classe
    prediction = model.predict(df_processed)
    # Predição das Probabilidades
    prediction_proba = model.predict_proba(df_processed)
    
    # Recuperando o nome da classe (já em Português)
    resultado_texto = le.inverse_transform(prediction)[0]
    
    # Calculando a confiança (maior probabilidade * 100)
    confianca = np.max(prediction_proba) * 100

    # Exibição do Texto
    st.subheader("Resultado da Análise:")
    
    if "Obesidade" in resultado_texto:
        st.error(f"⚠️ Diagnóstico: **{resultado_texto}**")
    elif "Sobrepeso" in resultado_texto:
        st.warning(f"⚠️ Diagnóstico: **{resultado_texto}**")
    else:
        st.success(f"✅ Diagnóstico: **{resultado_texto}**")
        
    # --- LINHA DE CONFIANÇA SOLICITADA ---
    st.info(f"🎯 Nível de Confiança do Modelo: **{confianca:.2f}%**")

    # --- GRÁFICO DE PROBABILIDADES ---
    st.markdown("---")
    st.subheader("📊 Probabilidades Detalhadas")
    
    proba_df = pd.DataFrame(prediction_proba, columns=le.classes_)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=proba_df.columns, y=proba_df.iloc[0].values, palette="viridis", ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Probabilidade (0-1)")
    plt.title("Análise de Risco por Categoria")
    st.pyplot(fig)

# --- RODAPÉ ---
st.markdown("---")
st.markdown("**Tech Challenge Fase 4** | Sistema de Apoio à Decisão Médica")