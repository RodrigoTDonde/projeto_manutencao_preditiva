

-----

# 🔧 Manutenção Preditiva com Dados de IoT

Projeto final do Bootcamp **CDIA – SENAI SC** focado em prever modos de falha em máquinas industriais utilizando dados de sensores (IoT). A abordagem utiliza um modelo de **classificação multirrótulo**, onde cada um dos cinco modos de falha possíveis (`FDF`, `FDC`, `FP`, `FTE`, `FA`) é previsto por um classificador **Random Forest** dedicado, tudo encapsulado em um pipeline de Machine Learning robusto e reprodutível.

-----

## 📋 Índice

  - [📌 Funcionalidades Principais](https://www.google.com/search?q=%23-funcionalidades-principais)
  - [📂 Estrutura do Projeto](https://www.google.com/search?q=%23-estrutura-do-projeto)
  - [🛠️ Tecnologias Utilizadas](https://www.google.com/search?q=%23%EF%B8%8F-tecnologias-utilizadas)
  - [⚙️ Pré-requisitos e Instalação](https://www.google.com/search?q=%23%EF%B8%8F-pr%C3%A9-requisitos-e-instala%C3%A7%C3%A3o)
  - [🚀 Como Executar](https://www.google.com/search?q=%23-como-executar)
      - [1. Treinamento do Modelo](https://www.google.com/search?q=%231-treinamento-do-modelo)
      - [2. Dashboard Interativo](https://www.google.com/search?q=%232-dashboard-interativo)
      - [3. Avaliação via API](https://www.google.com/search?q=%233-avalia%C3%A7%C3%A3o-via-api)
  - [🧪 Metodologia de Modelagem](https://www.google.com/search?q=%23-metodologia-de-modelagem)
  - [🔭 Análise Exploratória de Dados (EDA)](https://www.google.com/search?q=%23-an%C3%A1lise-explorat%C3%B3ria-de-dados-eda)
  - [🧾 Resultados](https://www.google.com/search?q=%23-resultados)
  - [📈 Melhorias Futuras](https://www.google.com/search?q=%23-melhorias-futuras)
  - [🔐 Boas Práticas de Segurança](https://www.google.com/search?q=%23-boas-pr%C3%A1ticas-de-seguran%C3%A7a)
  - [💻 Ambiente de Desenvolvimento](https://www.google.com/search?q=%23-ambiente-de-desenvolvimento)
  - [🧷 Comandos Rápidos](https://www.google.com/search?q=%23-comandos-r%C3%A1pidos)
  - [✍️ Autor](https://www.google.com/search?q=%23%EF%B8%8F-autor)

-----

## 📌 Funcionalidades Principais

  - 📥 **Carregamento de Dados:** Lê os arquivos CSV de treino e teste.
  - 🧼 **Pré-processamento:** Executa imputação de dados faltantes e codificação *one-hot* para variáveis categóricas.
  - 🔎 **Análise Exploratória:** Gera e salva visualizações como histogramas, boxplots e distribuições.
  - 🧠 **Treinamento Multirrótulo:** Treina um modelo `RandomForestClassifier` para cada rótulo de falha usando `MultiOutputClassifier`.
  - 🎚️ **Otimização de Threshold:** Ajusta um *threshold* de decisão global para maximizar a métrica **F1-macro** na validação.
  - 📤 **Geração de Submissões:** Cria arquivos de previsão em formato curto (para análise) e longo (compatível com a API).
  - 🌐 **Avaliação Automatizada:** Um script dedicado envia as previsões para a API oficial, recupera e salva as métricas de desempenho.
  - 📊 **Dashboard Interativo:** Uma aplicação **Streamlit** para explorar os dados e testar o modelo com novos arquivos.
  - 💾 **Versionamento de Artefatos:** Salva o modelo treinado, o *threshold* ótimo, métricas e gráficos na pasta `results/`.

-----

## 📂 Estrutura do Projeto

```
projeto_manutencao_preditiva/
├── data/                             → Dados de treino e teste (não versionados)
├── results/                          → Artefatos gerados (modelo, métricas, gráficos)
│   ├── modelo_multilabel_rf.joblib
│   ├── best_threshold.txt
│   ├── bootcamp_submission.csv
│   ├── bootcamp_submission_long.csv
│   ├── api_metrics_last.json
│   └── plots/                        → Gráficos da EDA e avaliação
├── notebooks/
│   └── Projeto_Manutencao_Preditiva.ipynb → Notebook principal com a análise completa
├── src/                              → (Opcional) Módulos e funções utilitárias
├── app.py                            → Dashboard interativo com Streamlit
├── evaluate_api.py                   → Script CLI para avaliação na API
├── main.py                           → Script principal para treino e predição
├── requirements.txt                  → Dependências do projeto
└── README.md                         → Este documento
```

-----

## 🛠️ Tecnologias Utilizadas

  - **Linguagem:** Python 3.10+
  - **Bibliotecas Principais:**
      - `pandas` e `numpy` para manipulação de dados
      - `scikit-learn` para o pipeline de Machine Learning
      - `matplotlib` e `seaborn` para visualização de dados
      - `streamlit` para o dashboard interativo
      - `joblib` para serialização do modelo
      - `requests` para comunicação com a API
  - **IDE:** Visual Studio Code (com Jupyter) e Google Colab
  - **Controle de Versão:** Git & GitHub

-----

## ⚙️ Pré-requisitos e Instalação

Antes de começar, garanta que você tem o **Python 3.10+** e o **Git** instalados.

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/RodrigoTDonde/projeto_manutencao_preditiva.git
    cd projeto_manutencao_preditiva
    ```

2.  **Crie e ative um ambiente virtual:**

    ```bash
    # Windows (PowerShell)
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Adicione os dados:**
    Crie uma pasta `data/` na raiz do projeto e coloque os arquivos `bootcamp_train.csv` e `bootcamp_test.csv` dentro dela.

-----

## 🚀 Como Executar

### 1\. Treinamento do Modelo

Execute o script `main.py` para treinar o modelo, otimizar o threshold e gerar os arquivos de submissão.

```bash
# Windows (PowerShell)
.\.venv\Scripts\python.exe main.py --train data\bootcamp_train.csv --test data\bootcamp_test.csv
```

As saídas principais serão salvas no diretório `results/`, incluindo o modelo (`.joblib`), o threshold (`.txt`) e as submissões (`.csv`).

### 2\. Dashboard Interativo

Inicie a aplicação Streamlit para explorar os dados e fazer predições de forma interativa.

```bash
# Windows (PowerShell)
.\.venv\Scripts\python.exe -m streamlit run app.py
```

O dashboard estará disponível em `http://localhost:8501` e possui duas abas:

  - **Exploração:** Visualiza os gráficos gerados e as últimas métricas da API.
  - **Predição:** Permite o upload de um arquivo CSV para obter predições em tempo real.

### 3\. Avaliação via API

Para avaliar o modelo nos dados de teste oficiais, utilize o script `evaluate_api.py`.

1.  **Obtenha seu token de autenticação** na [documentação da API](http://34.193.187.218:5000/docs) ou no [dashboard auxiliar](http://34.193.187.218:8501).

2.  **Defina o token como uma variável de ambiente** para segurança:

    ```powershell
    $env:BOOTCAMP_API_TOKEN="SEU_TOKEN_AQUI"
    ```

3.  **Execute o script de avaliação:**

    ```bash
    .\.venv\Scripts\python.exe evaluate_api.py --csv .\results\bootcamp_submission_long.csv --threshold 0.30 --save-plot
    ```

      - O script salva as métricas em `results/api_metrics_cli.json` e o gráfico de F1-Score em `results/plots/`.

-----

## 🧪 Metodologia de Modelagem

  - **Pré-processamento:**

      - **Variáveis Numéricas:** Imputação de valores faltantes com a **mediana**.
      - **Variáveis Categóricas:** Imputação com o valor **mais frequente** e aplicação de **One-Hot Encoding**.
      - **Pipeline:** Todo o processo é encapsulado em um `Pipeline` do Scikit-learn para garantir reprodutibilidade.

  - **Modelo:**

      - Um `RandomForestClassifier` é treinado para cada um dos cinco rótulos de falha.
      - O `MultiOutputClassifier` do Scikit-learn gerencia o treinamento e a predição dos múltiplos modelos.
      - Hiperparâmetros base: `n_estimators=150`, `min_samples_leaf=5`, `class_weight='balanced'`.

  - **Validação & Otimização do Threshold:**

      - Os dados de treino foram divididos em **80/20 (hold-out)**.
      - Um grid search foi realizado em um intervalo de thresholds (`0.10` a `0.90`) para encontrar o valor que **maximiza o F1-macro** no conjunto de validação.

-----

## 🔭 Análise Exploratória de Dados (EDA)

As visualizações geradas durante a execução do `main.py` são salvas em `results/plots/`.

| Distribuição de Variáveis Numéricas                                 | Distribuição do Tipo de Máquina                           | F1-Score por Classe (API)                                     |
| ------------------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------- |
|  |  |  |

-----

## 🧾 Resultados

### Avaliação Local (Hold-out)

  - **Threshold Ótimo:** `0.30`
  - **F1-Macro (Validação):** `~0.04`
  - **Interpretação:** O baixo F1-Score é esperado devido ao severo desbalanceamento das classes de falha. O modelo apresentou melhor desempenho na classe `FTE` (Falha por Tensão Elétrica), que possui mais exemplos positivos.

### Avaliação na API (Dados de Teste Ocultos)

| Métrica         | Valor    |
| --------------- | -------- |
| Macro Accuracy  | 0.9912   |
| Macro ROC AUC   | 0.6093   |
| **F1-Score (FTE)** | **0.2772** |
| F1-Score (outras) | 0.0000   |

*Resultados completos salvos em `results/api_metrics_last.json`.*

-----

## 📈 Melhorias Futuras

  - [ ] Implementar técnicas avançadas de balanceamento de dados (e.g., SMOTE).
  - [ ] Testar outros modelos, como `XGBoost` ou `LightGBM`.
  - [ ] Otimizar o *threshold* de decisão de forma individual para cada classe de falha.
  - [ ] Realizar `Feature Engineering` para criar variáveis mais informativas.
  - [ ] Implementar validação cruzada estratificada para uma avaliação mais robusta.

-----

## 🔐 Boas Práticas de Segurança

  - **Dados e Tokens:** Nunca versione dados sensíveis ou tokens de API. O `.gitignore` está configurado para ignorar a pasta `data/` e artefatos binários.
  - **Variáveis de Ambiente:** Utilize variáveis de ambiente para armazenar chaves de API, como `BOOTCAMP_API_TOKEN`.
    ```powershell
    # Exemplo para definir a variável no PowerShell
    $env:BOOTCAMP_API_TOKEN="SEU_TOKEN_AQUI"
    ```

-----

## 💻 Ambiente de Desenvolvimento

### VS Code

1.  Abra a pasta do projeto.
2.  Ative o ambiente virtual: `.\.venv\Scripts\activate`.
3.  Selecione o interpretador Python correto (`Ctrl+Shift+P` \> *Python: Select Interpreter*).
4.  Abra o notebook `notebooks/Projeto_Manutencao_Preditiva.ipynb` para ver a análise completa.

### Google Colab

O notebook foi projetado para ser compatível com o Colab. Ele detecta o ambiente e realiza as instalações e configurações necessárias automaticamente. Basta fazer o upload, montar seu Google Drive e executar as células.

-----

## 🧷 Comandos Rápidos

```bash
# Ativar ambiente virtual (Windows)
.\.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Treinar modelo e gerar submissões
.\.venv\Scripts\python.exe main.py --train data\bootcamp_train.csv --test data\bootcamp_test.csv

# Iniciar dashboard
.\.venv\Scripts\python.exe -m streamlit run app.py

# Avaliar na API
$env:BOOTCAMP_API_TOKEN="SEU_TOKEN"
.\.venv\Scripts\python.exe evaluate_api.py --csv .\results\bootcamp_submission_long.csv
```

-----

## ✍️ Autor

  - **Rodrigo Teles Dondé**
  - Projeto final do Bootcamp CDIA – SENAI SC