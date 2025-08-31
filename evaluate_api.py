# evaluate_api.py
# Avaliação do CSV de submissão na API do Bootcamp CDIA.
# - Lê token via --token ou variável de ambiente BOOTCAMP_API_TOKEN
# - Aceita CSV com nomes longos (recomendado) ou curtos (com --auto-map)
# - Salva métricas em JSON e (opcional) gera gráfico de F1 por classe

import os
import sys
import json
import argparse
import tempfile
from typing import Tuple, Optional

import requests

# Dependências opcionais só usadas quando necessário
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

API_BASE = "http://34.193.187.218:5000"

SHORT2LONG = {
    "FDF": "FDF (Falha Desgaste Ferramenta)",
    "FDC": "FDC (Falha Dissipacao Calor)",
    "FP":  "FP (Falha Potencia)",
    "FTE": "FTE (Falha Tensao Excessiva)",
    "FA":  "FA (Falha Aleatoria)",
}
LABELS = ["FDF", "FDC", "FP", "FTE", "FA"]
LABELS_LONG = [SHORT2LONG[k] for k in LABELS]


def _ensure_dirs(path: str):
    """Cria diretório pai do arquivo, se necessário."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _prepare_csv_for_upload(csv_path: str, auto_map: bool) -> Tuple[str, Optional[str]]:
    """
    Garante que o arquivo a ser enviado tenha as colunas longas.
    Se auto_map=True e o CSV estiver com colunas curtas, cria um arquivo temporário renomeado.
    Retorna (path_para_upload, path_temp_ou_None).
    """
    if not auto_map:
        return csv_path, None

    if pd is None:
        print("[aviso] pandas não disponível; --auto-map ignorado.", file=sys.stderr)
        return csv_path, None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[erro] Falha ao ler CSV para auto-map: {e}", file=sys.stderr)
        return csv_path, None

    cols = list(df.columns)
    has_short = any(c in LABELS for c in cols)
    has_long = any(c in LABELS_LONG for c in cols)

    if not has_short:
        # já está com longos (ou outro formato); não mexe
        return csv_path, None

    # renomeia curtos -> longos
    df2 = df.rename(columns=SHORT2LONG)

    # cria arquivo temporário com o mapeamento aplicado
    fd, tmp_path = tempfile.mkstemp(prefix="subm_long_", suffix=".csv")
    os.close(fd)
    df2.to_csv(tmp_path, index=False, encoding="utf-8")
    print(f"[info] CSV temporário com nomes longos criado em: {tmp_path}")
    return tmp_path, tmp_path


def _post_evaluate(csv_path: str, threshold: float, token: str, timeout: int = 90) -> requests.Response:
    """Envia o CSV para o endpoint de avaliação."""
    url = f"{API_BASE}/evaluate/multilabel_metrics"
    headers = {"X-API-Key": token}
    params = {"threshold": float(threshold)}

    with open(csv_path, "rb") as f:
        files = {"file": (os.path.basename(csv_path), f, "text/csv")}
        resp = requests.post(url, headers=headers, params=params, files=files, timeout=timeout)
    return resp


def _save_metrics_json(metrics: dict, out_path: str) -> None:
    _ensure_dirs(out_path)
    with open(out_path, "w", encoding="utf-8") as w:
        json.dump(metrics, w, indent=2, ensure_ascii=False)
    print(f"[ok] Métricas salvas em: {out_path}")


def _save_f1_plot(metrics: dict, out_png: str) -> None:
    if plt is None:
        print("[aviso] matplotlib não disponível; --save-plot ignorado.", file=sys.stderr)
        return

    f1 = metrics.get("f1_score")
    if not isinstance(f1, list) or len(f1) != 5:
        print("[aviso] não encontrei vetor f1_score esperado no JSON; gráfico não gerado.", file=sys.stderr)
        return

    _ensure_dirs(out_png)
    plt.figure(figsize=(6, 4))
    plt.bar(LABELS, f1)
    plt.title("F1-score por classe (API)")
    plt.ylim(0, 1)
    plt.xlabel("Classe")
    plt.ylabel("F1-score")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] Gráfico salvo em: {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Avaliar submissão na API do Bootcamp CDIA.")
    ap.add_argument("--csv", required=True, help="Caminho do CSV de submissão (preferir nomes longos).")
    ap.add_argument("--threshold", type=float, default=0.30, help="Limiar de decisão (default: 0.30).")
    ap.add_argument("--token", default=os.getenv("BOOTCAMP_API_TOKEN", ""), help="Token da API (ou defina BOOTCAMP_API_TOKEN).")
    ap.add_argument("--auto-map", action="store_true",
                    help="Se seu CSV tiver colunas curtas (FDF/FDC/...), renomeia para longas em um arquivo temporário.")
    ap.add_argument("--save-json", default=os.path.join("results", "api_metrics_cli.json"),
                    help="Caminho para salvar o JSON de métricas (default: results/api_metrics_cli.json).")
    ap.add_argument("--save-plot", action="store_true",
                    help="Gera gráfico de F1 por classe em results/plots/api_f1_por_classe_cli.png.")
    args = ap.parse_args()

    if not args.token:
        print("Defina o token via --token ou variável de ambiente BOOTCAMP_API_TOKEN.", file=sys.stderr)
        sys.exit(1)

    # Prepara CSV (auto-map se necessário)
    path_to_send, tmp_to_cleanup = _prepare_csv_for_upload(args.csv, args.auto_map)

    try:
        resp = _post_evaluate(path_to_send, args.threshold, args.token)
    finally:
        # limpa arquivo temporário, se criado
        if tmp_to_cleanup and os.path.exists(tmp_to_cleanup):
            try:
                os.remove(tmp_to_cleanup)
            except Exception:
                pass

    print("Status:", resp.status_code)

    if resp.status_code != 200:
        # mostra erro e sai com 1
        try:
            print(resp.json())
        except Exception:
            print(resp.text)
        sys.exit(1)

    # Sucesso → parse do JSON
    try:
        metrics = resp.json()
    except Exception:
        print(resp.text)
        sys.exit(1)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Salvar JSON
    _save_metrics_json(metrics, args.save_json)

    # Salvar gráfico (opcional)
    if args.save_plot:
        out_png = os.path.join("results", "plots", "api_f1_por_classe_cli.png")
        _save_f1_plot(metrics, out_png)

    sys.exit(0)


if __name__ == "__main__":
    main()
