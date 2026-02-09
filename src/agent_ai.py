import numpy as np
import pandas as pd

def iqr_outliers(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 8:
        return {"count": 0, "low": None, "high": None}
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((s < low) | (s > high)).sum())
    return {"count": count, "low": float(low), "high": float(high)}

def detect_drift(series: pd.Series, last_n: int = 3):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < last_n + 5:
        return {"drift": False, "reason": "amostra pequena"}
    a = s.iloc[:-last_n]
    b = s.iloc[-last_n:]
    mean_change = float((b.mean() - a.mean()) / (a.mean() + 1e-9))
    var_change = float((b.var() - a.var()) / (a.var() + 1e-9))
    drift = abs(mean_change) > 0.20 or abs(var_change) > 0.50
    return {"drift": drift, "mean_change_pct": mean_change, "var_change_pct": var_change}

def run_agent(df: pd.DataFrame, year_col: str = "Ano", target: str = "despesa_total", results_df: pd.DataFrame | None = None):
    """Agente simples para validação e monitorização do dataset e recomendações.
    - Qualidade: missing, duplicados, anos em falta
    - Outliers: regra IQR
    - Drift: mudança estrutural (últimos N anos vs anteriores)
    - Recomendação: melhor modelo por RMSE (se results_df fornecido)
    - Report: relatório em Markdown (report_md)
    """
    d = df.sort_values(year_col).copy()

    missing_pct = (d.isna().mean() * 100).sort_values(ascending=False)
    dup_years = int(d[year_col].duplicated().sum()) if year_col in d.columns else None

    years = d[year_col].dropna().astype(int).tolist() if year_col in d.columns else []
    years_missing = []
    if years:
        y_min, y_max = min(years), max(years)
        years_missing = [y for y in range(y_min, y_max + 1) if y not in set(years)]

    out = iqr_outliers(d[target]) if target in d.columns else {"count": None, "low": None, "high": None}
    drift = detect_drift(d[target]) if target in d.columns else {"drift": None}

    best_model = None
    if results_df is not None and "RMSE" in results_df.columns:
        best_model = results_df.sort_values("RMSE").iloc[0]["Modelo"]

    report = []
    report.append("# Agent AI Report")
    report.append("")
    report.append("## Qualidade dos dados")
    report.append(f"- Duplicados por ano: **{dup_years}**")
    report.append(f"- Anos em falta: **{years_missing if years_missing else 'nenhum'}**")
    report.append("")
    report.append("### Missing values (% por coluna) — top 10")
    report.append(missing_pct.head(10).to_string())
    report.append("")
    report.append("## Outliers (IQR) — despesa_total")
    report.append(f"- Quantidade de outliers: **{out.get('count')}**")
    report.append(f"- Limites IQR: **[{out.get('low')}, {out.get('high')}]**")
    report.append("")
    report.append("## Drift (mudança estrutural recente)")
    report.append(f"- Drift detectado? **{drift.get('drift')}**")
    if "mean_change_pct" in drift:
        report.append(f"- Variação média (últimos 3 anos vs anteriores): **{drift['mean_change_pct']*100:.1f}%**")
        report.append(f"- Variação da variância: **{drift['var_change_pct']*100:.1f}%**")
    report.append("")
    report.append("## Recomendação")
    if best_model:
        report.append(f"- Melhor modelo (por RMSE): **{best_model}**")
        report.append("- Sugestão: revalidar anualmente e re-treinar se o drift for persistente.")
    else:
        report.append("- Sem resultados de ML fornecidos ao agente (results_df=None).")

    report_md = "\n".join(report)

    return {
        "missing_pct": missing_pct,
        "dup_years": dup_years,
        "years_missing": years_missing,
        "outliers": out,
        "drift": drift,
        "best_model": best_model,
        "report_md": report_md,
    }
