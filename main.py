import os
import argparse
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report

from src.utils import (
    LABELS, rename_label_columns, to_binary, corrigir_negativos,
    predict_proba_multilabel, to_long_columns
)

# Tentar importar XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def ensure_dirs(results_dir="results", plots_dir="results/plots"):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return results_dir, plots_dir

def assert_required_columns(df: pd.DataFrame, cols: list, df_name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            "[%s] Faltam as colunas: %s\nColunas encontradas: %s"
            % (df_name, missing, df.columns.tolist())
        )

def carregar_treino(train_path: str) -> pd.DataFrame:
    df = pd.read_csv(train_path)
    df = rename_label_columns(df)  # rótulos longos -> curtos
    assert_required_columns(df, LABELS, "train")
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    for c in LABELS:
        df[c] = to_binary(df[c])
    if "falha_maquina" in df.columns:
        df["falha_maquina"] = (df[LABELS].sum(axis=1) > 0).astype(int)
    df = corrigir_negativos(df, excluir=LABELS)
    return df

def build_pipeline(cat_cols, num_cols, model_name="rf", profile="fast"):
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop"
    )

    if model_name == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost não instalado. Rode: pip install xgboost==2.0.3")
        # perfis para XGB
        if profile == "full":
            base = XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.9,
                objective="binary:logistic", tree_method="hist",
                reg_lambda=1.0, n_jobs=-1, random_state=42, eval_metric="logloss"
            )
        else:  # fast
            base = XGBClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", tree_method="hist",
                reg_lambda=1.0, n_jobs=-1, random_state=42, eval_metric="logloss"
            )
    else:
        # perfis para RF
        if profile == "full":
            base = RandomForestClassifier(
                n_estimators=400, min_samples_leaf=1,
                class_weight="balanced", n_jobs=-1, random_state=42
            )
        else:  # fast
            base = RandomForestClassifier(
                n_estimators=150, min_samples_leaf=5,
                class_weight="balanced", n_jobs=-1, random_state=42
            )

    clf = Pipeline([("prep", preprocess), ("clf", MultiOutputClassifier(base))])
    return clf

def choose_best_threshold(y_true: np.ndarray, y_proba: np.ndarray):
    thresholds = np.arange(0.30, 0.71, 0.05)
    best_th, best_f1 = 0.30, -1.0
    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1_mac > best_f1:
            best_th, best_f1 = float(th), float(f1_mac)
    return best_th, best_f1

def main(args):
    results_dir, _ = ensure_dirs()
    print("[1/7] Carregando treino...")
    df = carregar_treino(args.train)

    # por padrão NÃO usamos id_produto (alta cardinalidade). habilite com --use_id se quiser.
    X_cols = [
        "tipo",
        "temperatura_ar",
        "temperatura_processo",
        "umidade_relativa",
        "velocidade_rotacional",
        "torque",
        "desgaste_da_ferramenta",
    ]
    if args.use_id:
        X_cols = ["id_produto"] + X_cols

    assert_required_columns(df, X_cols, "train")
    X = df[X_cols].copy()
    Y = df[LABELS].copy()

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    print("[2/7] Split treino/val...")
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print("[3/7] Montando pipeline (%s | %s) e treinando..." % (args.model, args.profile))
    clf = build_pipeline(cat_cols, num_cols, model_name=args.model, profile=args.profile)
    clf.fit(X_train, Y_train)

    print("[4/7] Ajustando limiar por F1-macro...")
    Y_valid_proba = predict_proba_multilabel(clf, X_valid, len(LABELS))
    best_th, best_f1 = choose_best_threshold(Y_valid.values, Y_valid_proba)
    Y_pred_best = (Y_valid_proba >= best_th).astype(int)
    report = classification_report(
        Y_valid.values, Y_pred_best, target_names=LABELS, zero_division=0
    )
    print("Melhor threshold: %.2f | F1-macro: %.4f" % (best_th, best_f1))
    print("\n=== Classification Report (val) ===\n" + report)

    print("[5/7] Re-treinando com 100%% do treino...")
    clf.fit(X, Y)
    dump(clf, os.path.join(results_dir, "modelo_multilabel_rf.joblib"))  # nome mantido para simplicidade
    with open(os.path.join(results_dir, "best_threshold.txt"), "w") as f:
        f.write(str(float(best_th)))

    print("[6/7] Gerando submissões (teste)...")
    df_test = pd.read_csv(args.test)
    assert_required_columns(df_test, ["id"] + X_cols, "test")
    X_test = df_test[X_cols].copy()
    proba_test = predict_proba_multilabel(clf, X_test, len(LABELS))

    subm = pd.DataFrame(proba_test, columns=LABELS)
    subm.insert(0, "id", df_test["id"].values)
    for c in LABELS:
        subm[c] = subm[c].round(6)
    subm.to_csv(os.path.join(results_dir, "bootcamp_submission.csv"), index=False)

    subm_long = to_long_columns(subm)
    subm_long.to_csv(os.path.join(results_dir, "bootcamp_submission_long.csv"), index=False)

    print("[7/7] Salvando relatório...")
    with open(os.path.join(results_dir, "relatorio_classificacao.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print("\n Pronto! Artefatos em 'results/'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/bootcamp_train.csv")
    parser.add_argument("--test",  default="data/bootcamp_test.csv")
    parser.add_argument("--model", default="rf", choices=["rf","xgb"], help="Modelo base")
    parser.add_argument("--profile", default="fast", choices=["fast","full"], help="Perfil de treino")
    parser.add_argument("--use_id", action="store_true", help="Inclui id_produto como feature (não recomendado)")
    args = parser.parse_args()
    main(args)
