# Sistema de Triagem de Obesidade - Tech Challenge (Fase 4)

**Pós-Tech Data Analytics | FIAP**
**Autor:** Mauro Pedro Pupim Jr (RM 365239)

---

## Sobre o Projeto

Este projeto consiste no desenvolvimento de uma solução completa de Machine Learning para auxiliar profissionais de saúde na **triagem de níveis de obesidade**.

A partir de dados históricos de pacientes (hábitos alimentares, atividade física e histórico familiar), treinamos um modelo preditivo capaz de classificar o paciente em 7 categorias de peso, desde "Abaixo do Peso" até "Obesidade Mórbida (Tipo III)".

O diferencial da entrega é a integração do modelo em uma aplicação web interativa (**Streamlit**), permitindo o uso prático e imediato em ambiente clínico.

---

## 🏗️ Arquitetura da Solução

O diagrama abaixo ilustra o fluxo de dados automatizado do projeto:

```mermaid
graph TD
    User((Pacientes/Médicos)) -->|Input de Dados| UI[Interface Web Streamlit]
    
    subgraph "Aplicação (Frontend)"
        UI -->|Coleta Variáveis| Pandas[Tratamento de Dados]
        Pandas -->|One-Hot Encoding| Feat[Engenharia de Features]
    end
    
    subgraph "Núcleo de Inteligência (Backend)"
        Feat -->|Dados Processados| Model{Modelo Random Forest}
        Model -->|Inferência| Pred[Previsão de Classe]
        Model -->|Probabilidade| Prob[Cálculo de Confiança]
    end
    
    subgraph "Camada de Negócio"
        Pred -->|Resultado| Rules[Regras de Insights]
        Rules -->|Alertas de Saúde| Output[Dashboard Final]
    end
    
    Output -->|Diagnóstico + Recomendações| User
    
    style User fill:#f9f,stroke:#333,stroke-width:2px
    style UI fill:#bbf,stroke:#333,stroke-width:2px
    style Model fill:#bfb,stroke:#333,stroke-width:2px
    style Output fill:#f96,stroke:#333,stroke-width:2px
---

## Funcionalidades da Aplicação

* **Diagnóstico em Tempo Real:** Predição instantânea da classe de obesidade com base no formulário preenchido.
* **Interface Intuitiva:** Formulário lateral para inserção de dados fisiológicos e comportamentais.
* **Probabilidade de Risco:** Gráfico visual mostrando a confiança do modelo e a probabilidade para outras classes.
* **Segurança de Dados:** O modelo roda localmente ou em nuvem, sem armazenar dados sensíveis do paciente.

---

## Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Análise de Dados:** Pandas, Numpy
* **Visualização:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (Random Forest Classifier)
* **Deploy/Web App:** Streamlit
* **Serialização:** Joblib

---

## Performance do Modelo

O algoritmo escolhido foi o **Random Forest Classifier** devido à sua robustez para lidar com dados tabulares complexos e relações não-lineares.

* **Acurácia Global:** > 90%
* **Principais Preditores:**
    1.  Peso (Weight)
    2.  Histórico Familiar (Family History)
    3.  Consumo de Vegetais (FCVC)

*A análise exploratória completa e a justificativa técnica encontram-se no arquivo `Tech_Challenge_Fase_4.ipynb`.*

---

## Estrutura do Repositório

```text
/
├── app.py                  # Código principal da interface Web (Streamlit)
├── Tech_Challenge_Fase_4.ipynb # Notebook com EDA, Tratamento e Treinamento
├── modelo_obesidade.pkl    # Modelo treinado e artefatos
├── Obesity.csv             # Base de dados original (UCI)
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação do projeto


---

## Como Executar o Projeto Localmente ##
Siga os passos abaixo para rodar a aplicação na sua máquina:

1. Instalar Dependências
No terminal, navegue até a pasta do projeto e instale as bibliotecas necessárias:

pip install -r requirements.txt

2. Executar a Aplicação
Para iniciar o sistema, utilize o comando do Streamlit:

streamlit run app.py

3. Acessar no Navegador
O sistema abrirá automaticamente no seu navegador padrão no endereço:

http://localhost:8501

4. (Opcional) Retreinar o Modelo
Caso queira gerar um novo arquivo .pkl a partir do zero:

Abra o arquivo Tech_Challenge_Fase_4.ipynb no VS Code ou Jupyter.

Execute todas as células ("Run All").

O novo modelo será salvo na pasta automaticamente.

Deploy em Nuvem (Streamlit Cloud)
A aplicação encontra-se implantada e acessível publicamente através do Streamlit Cloud.

🔗 Link de Acesso: https://obbesidade-ml.streamlit.app/

FIAP - Tech Challenge Fase 4


[def]: image.png




