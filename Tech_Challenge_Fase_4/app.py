import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configuração inicial da página
st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🏥",
    layout="wide"
)

# Função para carregar o modelo de forma segura
# Usa o caminho absoluto para garantir que funcione no Streamlit Cloud e localmente
@st.cache_resource
def load_model():
    try:
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        caminho_modelo = os.path.join(diretorio_atual, 'modelo_obesidade.pkl')
        return joblib.load(caminho_modelo)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Carrega o modelo e os artefatos
artifacts = load_model()

# Validação se o modelo foi carregado corretamente
if artifacts is None:
    st.error("Erro Crítico: O arquivo 'modelo_obesidade.pkl' não foi encontrado.")
    st.warning("Verifique se o arquivo está na mesma pasta do script.")
    st.stop()

model = artifacts["model"]
le = artifacts["label_encoder"]
feature_columns = artifacts["features"]

# Título e descrição do app
st.title("🏥 Sistema de Triagem de Obesidade")
st.markdown("""
Este sistema utiliza Inteligência Artificial para analisar fatores de risco e prever o diagnóstico.
Preencha os dados do paciente na barra lateral para iniciar.
""")

# Barra lateral para entrada de dados
st.sidebar.header("Dados do Paciente")

def get_user_data():
    # Dados Pessoais
    genero = st.sidebar.selectbox("Gênero", ["Masculino", "Feminino"])
    idade = st.sidebar.number_input("Idade", 14, 100, 25)
    altura = st.sidebar.number_input("Altura (m)", 1.00, 2.50, 1.70)
    peso = st.sidebar.number_input("Peso (kg)", 30.0, 200.0, 70.0)
    
    st.sidebar.markdown("---") # Separador visual
    
    # Histórico e Hábitos
    # Convertendo inputs visuais (PT) para valores numéricos que o modelo entende
    hist_fam = st.sidebar.selectbox("Histórico Familiar de Obesidade?", ["Sim", "Não"])
    family_history = 1 if hist_fam == "Sim" else 0
    
    favc_input = st.sidebar.selectbox("Consome alimentos calóricos com frequência?", ["Sim", "Não"])
    favc = 1 if favc_input == "Sim" else 0
    
    fcvc = st.sidebar.slider("Frequência de consumo de vegetais (1=Nunca, 3=Sempre)", 1, 3, 2)
    ncp = st.sidebar.slider("Número de refeições principais por dia", 1, 4, 3)
    
    # Mapeamento para variáveis ordinais
    mapa_caec = {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}
    caec_label = st.sidebar.selectbox("Come entre as refeições?", list(mapa_caec.keys()))
    caec = mapa_caec[caec_label]
    
    smoke_input = st.sidebar.selectbox("Fumante?", ["Sim", "Não"])
    smoke = 1 if smoke_input == "Sim" else 0
    
    ch2o = st.sidebar.slider("Consumo diário de água (1=Pouco, 3=Muito)", 1, 3, 2)
    
    scc_input = st.sidebar.selectbox("Monitora calorias ingeridas?", ["Sim", "Não"])
    scc = 1 if scc_input == "Sim" else 0
    
    faf = st.sidebar.slider("Frequência de atividade física semanal (0=Nenhuma, 3=Muita)", 0, 3, 1)
    tue = st.sidebar.slider("Tempo usando dispositivos eletrônicos (0=Pouco, 2=Muito)", 0, 2, 1)
    
    mapa_calc = {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}
    calc_label = st.sidebar.selectbox("Consumo de álcool", list(mapa_calc.keys()))
    calc = mapa_calc[calc_label]
    
    # Mapeamento do transporte para inglês (necessário para o OneHotEncoding)
    mapa_transporte = {
        "Transporte Público": "Public_Transportation",
        "Caminhada": "Walking",
        "Automóvel": "Automobile",
        "Motocicleta": "Motorbike",
        "Bicicleta": "Bike"
    }
    transporte_label = st.sidebar.selectbox("Meio de transporte principal", list(mapa_transporte.keys()))
    mtrans = mapa_transporte[transporte_label]
    
    # Conversão do Gênero
    gender_val = 1 if genero == "Masculino" else 0
    
    # Cria o dicionário com os dados
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
        'MTRANS': mtrans
    }
    
    return pd.DataFrame(user_data, index=[0])

# Captura os dados do usuário
input_df = get_user_data()

# Processamento dos dados
# Aplica OneHotEncoding e garante que as colunas sejam iguais ao treinamento
df_processed = pd.get_dummies(input_df, columns=['MTRANS'])
df_processed = df_processed.reindex(columns=feature_columns, fill_value=0)

# Botão de ação
if st.button("🔍 Realizar Diagnóstico"):
    try:
        # Predição
        prediction = model.predict(df_processed)
        prediction_proba = model.predict_proba(df_processed)
        
        # Recupera o label original em inglês
        classe_original = le.inverse_transform(prediction)[0]
        
        # Dicionário de tradução
        traducoes = {
            'Insufficient_Weight': 'Abaixo do Peso',
            'Normal_Weight': 'Peso Normal',
            'Overweight_Level_I': 'Sobrepeso Nível I',
            'Overweight_Level_II': 'Sobrepeso Nível II',
            'Obesity_Type_I': 'Obesidade Tipo I',
            'Obesity_Type_II': 'Obesidade Tipo II',
            'Obesity_Type_III': 'Obesidade Tipo III (Mórbida)'
        }
        
        # Traduz o resultado
        resultado_pt = traducoes.get(classe_original, classe_original)
        
        # Calcula a confiança
        confianca = np.max(prediction_proba) * 100
        
        # Exibe o resultado com cores apropriadas
        st.subheader("Resultado da Análise:")
        
        if "Obesity" in classe_original:
            st.error(f"⚠️ Diagnóstico: **{resultado_pt}**")
        elif "Overweight" in classe_original:
            st.warning(f"⚠️ Diagnóstico: **{resultado_pt}**")
        else:
            st.success(f"✅ Diagnóstico: **{resultado_pt}**")
            
        st.info(f"🎯 Nível de Confiança do Modelo: **{confianca:.2f}%**")
        
        # Gráfico de probabilidades
        st.markdown("---")
        st.subheader("📊 Probabilidades Detalhadas")
        
        # Cria um DataFrame para o gráfico, traduzindo as colunas
        colunas_traduzidas = [traducoes.get(c, c) for c in le.classes_]
        proba_df = pd.DataFrame(prediction_proba, columns=colunas_traduzidas)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=proba_df.columns, y=proba_df.iloc[0].values, palette="viridis", ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Probabilidade")
        plt.xlabel("Categoria")
        plt.title("Análise de Risco por Categoria")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Erro ao processar predição: {e}")

# Rodapé simples
st.markdown("---")
st.markdown("**Tech Challenge Fase 4** | Sistema de Apoio à Decisão Médica")