import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configuração global
st.set_page_config(page_title="Predição de Obesidade", page_icon="🏥", layout="wide")

# Funções Utilitárias
def calc_imc(peso, altura):
    return peso / (altura ** 2)

def get_insights(row):
    """Gera alertas baseados em regras de negócio simples."""
    alerts = []
    
    # Regra 1: Consumo de Água
    if row['CH2O'].values[0] < 2:
        alerts.append("💧 **Hidratação:** Consumo de água abaixo do ideal. Recomendado: > 2L/dia.")
    
    # Regra 2: Consumo de Vegetais
    if row['FCVC'].values[0] < 2:
        alerts.append("🥦 **Nutrição:** Baixo consumo de vegetais reportado.")
    
    # Regra 3: Sedentarismo
    if row['FAF'].values[0] == 0:
        alerts.append("🏃 **Atividade Física:** Nenhuma atividade física registrada. Risco de sedentarismo.")
        
    # Regra 4: Tecnologia
    if row['TUE'].values[0] > 1:
        alerts.append("📱 **Tempo de Tela:** Uso elevado de dispositivos eletrônicos.")

    return alerts

# Carga de Modelo e Artefatos
@st.cache_resource
def load_artifacts():
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelo_obesidade.pkl')
        return joblib.load(path)
    except Exception as e:
        st.error(f"Falha ao carregar modelo: {e}")
        return None

artifacts = load_artifacts()

if not artifacts:
    st.stop()

model = artifacts["model"]
le = artifacts["label_encoder"]
features = artifacts["features"]

# Layout Principal
st.title("🏥 Sistema de Triagem de Obesidade")
st.markdown("---")

# Sidebar - Coleta de Dados
st.sidebar.header("Dados do Paciente")

def get_data():
    # Dados Fisiológicos
    sexo = st.sidebar.selectbox("Gênero", ["Masculino", "Feminino"])
    idade = st.sidebar.number_input("Idade", 14, 100, 25)
    altura = st.sidebar.number_input("Altura (m)", 1.00, 2.50, 1.70)
    peso = st.sidebar.number_input("Peso (kg)", 30.0, 200.0, 70.0)
    
    st.sidebar.markdown("---")
    
    # Histórico
    hist_fam = st.sidebar.selectbox("Histórico Familiar Obesidade?", ["Não", "Sim"])
    
    # Hábitos Alimentares
    favc = st.sidebar.selectbox("Consome calóricos frequentemente?", ["Não", "Sim"])
    fcvc = st.sidebar.slider("Frequência de Vegetais (1-3)", 1, 3, 2)
    ncp = st.sidebar.slider("Refeições ao dia", 1, 4, 3)
    
    # Hábitos Gerais
    map_freq = {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}
    
    caec = st.sidebar.selectbox("Belisca entre refeições?", list(map_freq.keys()))
    smoke = st.sidebar.selectbox("Fumante?", ["Não", "Sim"])
    ch2o = st.sidebar.slider("Consumo de Água (1-3)", 1, 3, 2)
    scc = st.sidebar.selectbox("Monitora Calorias?", ["Não", "Sim"])
    faf = st.sidebar.slider("Atividade Física (0-3)", 0, 3, 1)
    tue = st.sidebar.slider("Tempo em Telas (0-2)", 0, 2, 1)
    calc = st.sidebar.selectbox("Álcool", list(map_freq.keys()))
    
    map_trans = {
        "Transporte Público": "Public_Transportation",
        "Caminhada": "Walking",
        "Automóvel": "Automobile",
        "Motocicleta": "Motorbike",
        "Bicicleta": "Bike"
    }
    mtrans = st.sidebar.selectbox("Transporte Principal", list(map_trans.keys()))

    # Construção do DataFrame
    data = {
        'Gender': 1 if sexo == "Masculino" else 0,
        'Age': idade,
        'Height': altura,
        'Weight': peso,
        'family_history': 1 if hist_fam == "Sim" else 0,
        'FAVC': 1 if favc == "Sim" else 0,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': map_freq[caec],
        'SMOKE': 1 if smoke == "Sim" else 0,
        'CH2O': ch2o,
        'SCC': 1 if scc == "Sim" else 0,
        'FAF': faf,
        'TUE': tue,
        'CALC': map_freq[calc],
        'MTRANS': map_trans[mtrans]
    }
    return pd.DataFrame(data, index=[0])

df = get_data()

# Pré-processamento
df_proc = pd.get_dummies(df, columns=['MTRANS'])
df_proc = df_proc.reindex(columns=features, fill_value=0)

# Botão de Execução
if st.button("Executar Análise"):
    # Inferência
    pred = model.predict(df_proc)
    proba = model.predict_proba(df_proc)
    
    # Tratamento de Resultados
    classe_raw = le.inverse_transform(pred)[0]
    confianca = np.max(proba) * 100
    imc = calc_imc(df['Weight'].values[0], df['Height'].values[0])
    
    map_labels = {
        'Insufficient_Weight': 'Abaixo do Peso',
        'Normal_Weight': 'Peso Normal',
        'Overweight_Level_I': 'Sobrepeso Nível I',
        'Overweight_Level_II': 'Sobrepeso Nível II',
        'Obesity_Type_I': 'Obesidade Tipo I',
        'Obesity_Type_II': 'Obesidade Tipo II',
        'Obesity_Type_III': 'Obesidade Tipo III (Mórbida)'
    }
    classe_pt = map_labels.get(classe_raw, classe_raw)

    # Exibição - KPIs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classificação do Modelo")
        if "Obesity" in classe_raw:
            st.error(f"🔴 {classe_pt}")
        elif "Overweight" in classe_raw:
            st.warning(f"🟡 {classe_pt}")
        else:
            st.success(f"🟢 {classe_pt}")
        st.caption(f"Probabilidade: {confianca:.1f}%")

    with col2:
        st.subheader("Métrica Fisiológica")
        st.metric("IMC Calculado", f"{imc:.2f}")

    # Exibição - Insights
    st.divider()
    st.subheader("Análise de Hábitos")
    
    alerts = get_insights(df)
    if alerts:
        for alert in alerts:
            st.info(alert)
    else:
        st.success("✅ Nenhum hábito de risco crítico identificado.")

    # Exibição - Gráfico (Agora Fixo)
    st.divider()
    st.subheader("Distribuição de Probabilidades")
    
    labels_clean = [map_labels.get(c, c) for c in le.classes_]
    df_chart = pd.DataFrame(proba, columns=labels_clean)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.barplot(x=df_chart.columns, y=df_chart.iloc[0].values, palette="viridis", ax=ax)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.ylabel("Score")
    plt.xlabel("")
    st.pyplot(fig)
