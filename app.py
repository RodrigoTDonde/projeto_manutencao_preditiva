import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
from joblib import load

st.set_page_config(page_title="Manutenção Preditiva - Demo", layout="wide")

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
MODEL_PATH  = os.path.join(RESULTS_DIR, "modelo_multilabel_rf.joblib")
TH_PATH     = os.path.join(RESULTS_DIR, "best_threshold.txt")

# Colunas de entrada (as mesmas do main)
X_COLS = [
    "tipo",
    "temperatura_ar",
    "temperatura_processo",
    "umidade_relativa",
    "velocidade_rotacional",
    "torque",
    "desgaste_da_ferramenta",
]
LABELS = ["FDF", "FDC", "FP", "FTE", "FA"]
LONG_MAP = {
    'FDF': 'FDF (Falha Desgaste Ferramenta)',
    'FDC': 'FDC (Falha Dissipacao Calor)',
    'FP' : 'FP (Falha Potencia)',
    'FTE': 'FTE (Falha Tensao Excessiva)',
    'FA' : 'FA (Falha Aleatoria)'
}

def load_threshold(default=0.30):
    try:
        with open(TH_PATH, "r", encoding="utf-8") as f:
            return float(f.read().strip())
    except Exception:
        return default

def load_model():
    if os.path.exists(MODEL_PATH):
        return load(MODEL_PATH)
    return None

def predict_proba_df(model, df_input: pd.DataFrame) -> pd.DataFrame:
    # garante colunas esperadas
    for c in X_COLS:
        if c not in df_input.columns:
            df_input[c] = np.nan
    X = df_input[X_COLS].copy()
    # predict_proba retorna lista de arrays (um por rótulo)
    probs_list = model.predict_proba(X)
    # empilha colunas
    out = {}
    for i, lab in enumerate(LABELS):
        # alguns classificadores retornam shape (n,2); pega prob de classe 1
        p = probs_list[i]
        if p.ndim == 2 and p.shape[1] == 2:
            out[lab] = p[:, 1]
        else:
            out[lab] = p.ravel()
    return pd.DataFrame(out, index=df_input.index)

def to_long_cols(df_short: pd.DataFrame) -> pd.DataFrame:
    return df_short.rename(columns=LONG_MAP)

# Sidebar
st.sidebar.title("Configurações")
th = st.sidebar.slider("Limiar (threshold)", 0.0, 1.0, load_threshold(), 0.01)
st.sidebar.caption("Abaixo/igual ao limiar → 0 | Acima → 1")

st.title("Manutenção Preditiva — EDA e Predições")
tabs = st.tabs(["Exploração", "Predição"])

# === Tab 1: Exploração ===
with tabs[0]:
    st.subheader("Gráficos salvos")
    col1, col2 = st.columns(2)

    def show_png(path, title):
        if os.path.exists(path):
            st.image(path, caption=title, use_column_width=True)
        else:
            st.info(f"Arquivo não encontrado: {os.path.relpath(path, ROOT_DIR)}")

    with col1:
        show_png(os.path.join(PLOTS_DIR, "distribuicoes_numericas.png"), "Distribuições numéricas")
        show_png(os.path.join(PLOTS_DIR, "distribuicao_tipo.png"), "Distribuição de 'tipo'")
    with col2:
        # tenta mostrar alguns boxplots (se existirem)
        for name in os.listdir(PLOTS_DIR) if os.path.isdir(PLOTS_DIR) else []:
            if name.startswith("box_") and name.endswith(".png"):
                show_png(os.path.join(PLOTS_DIR, name), f"Boxplot - {name[4:-4]}")

    st.divider()
    st.subheader("Resumo numérico (se disponível)")
    resumo_csv = os.path.join(RESULTS_DIR, "resumo_numericas.csv")
    if os.path.exists(resumo_csv):
        st.dataframe(pd.read_csv(resumo_csv), use_container_width=True)
    else:
        st.info("Resumo não encontrado (results/resumo_numericas.csv).")

    st.divider()
    st.subheader("Métricas da API (se disponível)")
    api_json = os.path.join(RESULTS_DIR, "api_metrics_last.json")
    if os.path.exists(api_json):
        with open(api_json, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.json(metrics)
        # gráfico de F1 por classe salvo
        show_png(os.path.join(PLOTS_DIR, "api_f1_por_classe.png"), "API — F1 por classe")
    else:
        st.info("Ainda não há métricas salvas da API.")

# === Tab 2: Predição ===
with tabs[1]:
    st.subheader("Carregue um CSV para gerar predições")
    st.caption("Estrutura esperada: colunas 'id' (opcional), 'tipo' (L/M/H) e variáveis de entrada.")
    uploaded = st.file_uploader("Arquivo CSV", type=["csv"])
    model = load_model()

    if model is None:
        st.error("Modelo não encontrado (results/modelo_multilabel_rf.joblib). Treine/salve antes.")
    else:
        st.success("Modelo carregado com sucesso.")

    if uploaded and model is not None:
        df_in = pd.read_csv(uploaded)
        if "id" not in df_in.columns:
            df_in["id"] = np.arange(1, len(df_in) + 1)

        probs = predict_proba_df(model, df_in)
        preds = (probs.values > th).astype(int)
        df_preds = pd.DataFrame(preds, columns=LABELS, index=df_in.index)
        sub_curta = pd.concat([df_in[["id"]], probs[["FDF","FDC","FP","FTE","FA"]]], axis=1)
        sub_longa = pd.concat([df_in[["id"]], to_long_cols(probs)], axis=1)

        st.markdown("**Prévia das probabilidades (curto):**")
        st.dataframe(sub_curta.head(10), use_container_width=True)

        # botões de download
        def to_csv_bytes(df):
            bio = BytesIO()
            df.to_csv(bio, index=False, encoding="utf-8")
            return bio.getvalue()

        st.download_button("Baixar CSV (nomes curtos)", to_csv_bytes(sub_curta),
                           file_name="predicoes_curto.csv", mime="text/csv")
        st.download_button("Baixar CSV (nomes longos)", to_csv_bytes(sub_longa),
                           file_name="predicoes_longos.csv", mime="text/csv")

        st.markdown("**Threshold escolhido:** {:.2f}".format(th))
        st.markdown("**Contagem de positivos por rótulo (com threshold):**")
        st.write(df_preds.sum().to_frame("positivos"))
