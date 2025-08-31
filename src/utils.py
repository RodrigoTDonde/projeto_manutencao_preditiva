import numpy as np
import pandas as pd

LABELS = ['FDF','FDC','FP','FTE','FA']
LONG_MAP = {
    'FDF': 'FDF (Falha Desgaste Ferramenta)',
    'FDC': 'FDC (Falha Dissipacao Calor)',
    'FP':  'FP (Falha Potencia)',
    'FTE': 'FTE (Falha Tensao Excessiva)',
    'FA':  'FA (Falha Aleatoria)'
}

def rename_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_dict = {v: k for k, v in LONG_MAP.items()}
    return df.rename(columns=rename_dict)

def to_binary(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').fillna(0)
    return s.clip(lower=0, upper=1).astype(int)

def corrigir_negativos(df: pd.DataFrame, excluir=None) -> pd.DataFrame:
    df = df.copy()
    excluir = set(excluir or [])
    for c in df.select_dtypes(include=[np.number]).columns:
        if c in excluir:
            continue
        mask = df[c] < 0
        if mask.any():
            mediana = df.loc[~mask, c].median() if (~mask).any() else 0
            df.loc[mask, c] = mediana
    return df

def predict_proba_multilabel(pipeline, X, n_labels: int) -> np.ndarray:
    moc = pipeline.named_steps['clf']
    X_trans = pipeline.named_steps['prep'].transform(X)
    probas = []
    for est in moc.estimators_:
        probas.append(est.predict_proba(X_trans)[:, 1])
    return np.vstack(probas).T  # (n_amostras, n_labels)

def to_long_columns(df_subm: pd.DataFrame) -> pd.DataFrame:
    return df_subm.rename(columns=LONG_MAP)
