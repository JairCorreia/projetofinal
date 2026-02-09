# app/streamlit_app.py
# ============================================================
# DASHBOARD  (alinhado com o notebook)
# - sem boxplot
# - elasticidade aproximada (correlação de variações %)
# - previsão SARIMAX(1,1,1) com exógenas e projeção até ano alvo
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# IMPORTS DA TUA CAMADA src/
from src.agent_ai import run_agent
from src.ml import compare_models, time_series_cv_rmse


# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Projeto Final G3 — Dashboard", layout="wide")


# ----------------------------
# UI helpers
# ----------------------------
def apply_css():
    st.markdown(
        """
        <style>
          .kpi-card{
            padding:14px 16px;
            border-radius:16px;
            border:1px solid rgba(0,0,0,0.08);
            background: rgba(255,255,255,0.6);
          }
          .kpi-title{ font-size:13px; color: #6b7280; margin-bottom:6px;}
          .kpi-value{ font-size:24px; font-weight:800; color:#111827; line-height:1.1;}
          .kpi-sub{ font-size:12px; color:#6b7280; margin-top:6px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_line_plot(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


# ----------------------------
# Data helpers
# ----------------------------
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    # normaliza coluna de ano
    if "Ano" not in df.columns:
        for c in df.columns:
            if c.lower() in ("ano", "year"):
                df = df.rename(columns={c: "Ano"})
                break

    if "Ano" not in df.columns:
        raise ValueError("Dataset não tem coluna 'Ano'.")

    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Ano"]).sort_values("Ano").reset_index(drop=True)
    return df


def detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_rubricas_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("despesa_")]


def calc_cagr(series: pd.Series) -> float | None:
    s = series.dropna()
    if len(s) < 2:
        return None
    vi = float(s.iloc[0])
    vf = float(s.iloc[-1])
    if vi <= 0 or vf <= 0:
        return None
    n = len(s)
    return (vf / vi) ** (1 / (n - 1)) - 1


def mean_yoy_growth(series: pd.Series) -> float | None:
    s = series.dropna()
    if len(s) < 2:
        return None
    yoy = s.pct_change().dropna()
    if yoy.empty:
        return None
    return float(yoy.mean())


def top_n_weights(df_ref: pd.DataFrame, rubricas: list[str], total_col: str, n: int = 10) -> pd.DataFrame:
    if df_ref.empty or total_col not in df_ref.columns:
        return pd.DataFrame()
    total = float(df_ref[total_col].iloc[0]) if pd.notna(df_ref[total_col].iloc[0]) else 0.0
    rows = []
    for c in rubricas:
        val = float(df_ref[c].iloc[0]) if c in df_ref.columns and pd.notna(df_ref[c].iloc[0]) else 0.0
        w = (val / total) if total else np.nan
        rows.append({"Rubrica": c, "Valor": val, "Peso": w})
    out = pd.DataFrame(rows).sort_values("Peso", ascending=False).head(n)
    return out


# --- Elasticidade aproximada (IGUAL ao notebook: correlação de variações percentuais)
def elasticidade_aproximada_notebook(
    df_in: pd.DataFrame,
    variavel_alvo: str = "despesa_total",
    col_ano: str = "Ano",
    top_n: int = 5,
) -> pd.DataFrame:
    d = df_in.copy()

    # só numéricas, exclui ano e alvo
    variaveis = [
        v for v in d.columns
        if v not in [col_ano, variavel_alvo]
        and pd.api.types.is_numeric_dtype(d[v])
    ]

    elasticidade = []
    for var in variaveis:
        tmp = d[[variavel_alvo, var]].dropna().copy()

        tmp[f"{var}_pct"] = pd.to_numeric(tmp[var], errors="coerce").pct_change()
        tmp[f"{variavel_alvo}_pct"] = pd.to_numeric(tmp[variavel_alvo], errors="coerce").pct_change()

        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

        if len(tmp) > 2:
            corr = tmp[f"{var}_pct"].corr(tmp[f"{variavel_alvo}_pct"])
            elasticidade.append({"Variável": var, "Elasticidade_aprox": float(corr)})

    if not elasticidade:
        return pd.DataFrame()

    out = (
        pd.DataFrame(elasticidade)
        .sort_values("Elasticidade_aprox", key=lambda s: s.abs(), ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return out


def save_report_md(report_md: str, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md or "", encoding="utf-8")


# --- Previsão SARIMAX (igual ao notebook)
def build_exog_future_from_mean_growth(df_hist: pd.DataFrame, exog_cols: list[str], steps: int) -> pd.DataFrame:
    """
    Projeta exógenas para o futuro como no notebook:
    exog_future[var] = last_value * (1 + mean_growth)**i
    onde mean_growth é a média do pct_change histórico.
    """
    d = df_hist[exog_cols].copy()
    d = d.apply(pd.to_numeric, errors="coerce")
    growth = d.pct_change().mean(numeric_only=True)

    last = d.iloc[-1]
    fut = {}
    for c in exog_cols:
        g = float(growth.get(c, 0.0)) if pd.notna(growth.get(c, np.nan)) else 0.0
        lv = float(last.get(c, np.nan))
        if pd.isna(lv):
            # se faltar, mantém NaN (o SARIMAX vai falhar -> tratamos antes)
            fut[c] = [np.nan] * steps
        else:
            fut[c] = [lv * (1 + g) ** i for i in range(1, steps + 1)]

    return pd.DataFrame(fut)


@st.cache_resource
def fit_sarimax(y: np.ndarray, exog: np.ndarray):
    # Igual ao notebook: SARIMAX com order(1,1,1) e sem restrições
    model = SARIMAX(
        y,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    return res


# ----------------------------
# UI init
# ----------------------------
apply_css()

# Sidebar branding
logo_path = ROOT_DIR / "figs" / "logo.png"
if logo_path.exists():
    st.sidebar.image(str(logo_path), use_container_width=True)
st.sidebar.markdown("### Projeto Final G3")
st.sidebar.caption("Dashboard • ML • Agent AI")


# ----------------------------
# Load dataset
# ----------------------------
st.title("📊 Dashboard")

default_csv = str(ROOT_DIR / "data" / "processed" / "dataset_merge_wb_gdp.csv")
csv_path = st.sidebar.text_input("Caminho do CSV (data/processed)", value=default_csv)

try:
    df = load_data(csv_path)
except Exception as e:
    st.error(f"Erro ao carregar dataset: {e}")
    st.stop()

# Detect columns (mantém o teu)
col_despesa_total = detect_col(df, ["despesa_total", "Despesa_Total", "total_despesa", "Total_Despesa"])
col_segurados = detect_col(df, ["Segurados", "segurados"])
col_beneficiarios = detect_col(df, ["Beneficiarios", "Beneficiários", "beneficiarios", "beneficiários"])
col_pop = detect_col(df, ["Populacao", "População", "populacao", "população"])
col_gdp = detect_col(df, ["PIB", "NY.GDP.MKTP.CD", "gdp", "GDP"])
col_infl = detect_col(df, ["Inflacao", "Inflação", "inflacao", "inflação"])

# Colunas específicas do notebook (para SARIMAX exógeno)
col_pop_emp = detect_col(df, ["População_empregada", "Populacao_empregada", "populacao_empregada"])
col_pensionista_inps = detect_col(df, ["Pensionista_INPS", "pensionista_inps"])

rubricas_cols_all = get_rubricas_cols(df)

years_all = sorted(df["Ano"].dropna().astype(int).unique())
if not years_all:
    st.warning("Sem anos válidos no dataset.")
    st.stop()

# Filters
st.sidebar.header("Filtros")
year_ref = st.sidebar.selectbox("Ano de referência", years_all, index=len(years_all) - 1)
periodo = st.sidebar.multiselect("Período (anos)", years_all, default=years_all)

df_periodo = df[df["Ano"].astype(int).isin(periodo)].copy()
df_ref = df[df["Ano"].astype(int) == int(year_ref)].copy()

# Top KPIs
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Ano referência", str(year_ref), "Para Top N e resumo")
with c2:
    kpi("Registos (total)", str(len(df)), f"Selecionado: {len(df_periodo)}")
with c3:
    kpi("Anos", str(len(years_all)), f"{years_all[0]}–{years_all[-1]}")
with c4:
    kpi("Rubricas despesa_*", str(len(rubricas_cols_all)), "Detetadas no dataset")

st.divider()

# Menu dinâmico
pages = ["Visão Geral"]

if col_despesa_total is not None:
    pages += ["Despesas", "Peso das despesas (Top N)", "Elasticidades (Top 5)"]

if col_segurados is not None:
    pages += ["Segurados"]
if col_beneficiarios is not None:
    pages += ["Beneficiários"]

if col_despesa_total is not None:
    pages += ["Previsão SARIMAX (igual ao notebook)"]

pages += ["ML (executar)", "Agent AI (executar)"]

menu = st.sidebar.radio("Menu", pages)


# ----------------------------
# Pages
# ----------------------------
if menu == "Visão Geral":
    st.subheader("Resumo do dataset")
    st.write(f"Registos: **{len(df)}** | Anos: **{len(years_all)}** ({years_all[0]}–{years_all[-1]})")
    st.dataframe(df.head(30), use_container_width=True)

    st.subheader("Colunas detectadas")
    st.write(
        {
            "despesa_total": col_despesa_total,
            "segurados": col_segurados,
            "beneficiarios": col_beneficiarios,
            "populacao": col_pop,
            "PIB": col_gdp,
            "inflacao": col_infl,
            "População_empregada (notebook)": col_pop_emp,
            "Pensionista_INPS (notebook)": col_pensionista_inps,
            "rubricas_despesa_*": len(rubricas_cols_all),
        }
    )

elif menu == "Despesas":
    st.header("Despesas")
    if col_despesa_total is None:
        st.warning("Não encontrei coluna de despesa_total no dataset.")
        st.stop()

    ser = df_periodo.set_index("Ano")[col_despesa_total].dropna()
    safe_line_plot(ser.index, ser.values, "Evolução — Despesa total", "Ano", "Despesa")

    c1, c2 = st.columns(2)
    cagr = calc_cagr(ser)
    yoy = mean_yoy_growth(ser)
    with c1:
        kpi("CAGR (período)", f"{cagr*100:.2f}%" if pd.notna(cagr) else "n/d")
    with c2:
        kpi("Média YoY (período)", f"{yoy*100:.2f}%" if pd.notna(yoy) else "n/d")

    if rubricas_cols_all:
        st.subheader("Rubricas (despesa_*) — selecione poucas para ficar leve")
        default_sel = rubricas_cols_all[: min(3, len(rubricas_cols_all))]
        rubricas_sel = st.multiselect("Rubricas", rubricas_cols_all, default=default_sel)

        if len(rubricas_sel) > 8:
            st.warning("Muitas rubricas selecionadas pode ficar lento. Tenta reduzir para 3–6.")

        for c in rubricas_sel:
            ser_r = df_periodo.set_index("Ano")[c].dropna()
            if ser_r.empty:
                continue
            st.markdown(f"### {c}")
            safe_line_plot(ser_r.index, ser_r.values, f"Evolução — {c}", "Ano", c)

elif menu == "Peso das despesas (Top N)":
    st.header("Peso das despesas — Top N (ano de referência)")
    if df_ref.empty:
        st.warning("Ano de referência não encontrado no dataset.")
        st.stop()
    if col_despesa_total is None:
        st.warning("Sem despesa_total.")
        st.stop()
    if not rubricas_cols_all:
        st.warning("Não há rubricas despesa_* no dataset.")
        st.stop()

    topn = st.slider("Top N", 3, 30, 10)
    out = top_n_weights(df_ref, rubricas_cols_all, col_despesa_total, n=topn)
    st.dataframe(out, use_container_width=True)

elif menu == "Elasticidades (Top 5)":
    st.header("Elasticidades (Top 5) — Elasticidade aproximada (igual ao notebook)")

    if col_despesa_total is None:
        st.warning("Sem despesa_total.")
        st.stop()

    # notebook calcula elasticidade aproximada para TODAS numéricas (exceto ano e alvo)
    base_cols = ["Ano", col_despesa_total]
    d = df_periodo[base_cols + [c for c in df_periodo.columns if c not in base_cols]].copy()
    d = d.rename(columns={col_despesa_total: "despesa_total"})

    out = elasticidade_aproximada_notebook(d, variavel_alvo="despesa_total", col_ano="Ano", top_n=5)
    if out.empty:
        st.info("Não foi possível calcular elasticidades (dados insuficientes / variação nula).")
    else:
        st.dataframe(out, use_container_width=True)

        # pequeno gráfico horizontal (sem boxplot)
        fig, ax = plt.subplots()
        ax.barh(out["Variável"], out["Elasticidade_aprox"])
        ax.axvline(0, linestyle="--", alpha=0.3)
        ax.set_xlabel("Elasticidade aproximada (correlação de %Δ)")
        ax.set_title("Top 5 — Elasticidade aproximada (notebook)")
        ax.grid(True, axis="x", alpha=0.2)
        st.pyplot(fig)

elif menu == "Segurados":
    st.header("Segurados")
    if col_segurados is None:
        st.warning("Sem coluna de Segurados.")
        st.stop()

    ser = df_periodo.set_index("Ano")[col_segurados].dropna()
    safe_line_plot(ser.index, ser.values, "Evolução — Segurados", "Ano", "Segurados")

elif menu == "Beneficiários":
    st.header("Beneficiários")
    if col_beneficiarios is None:
        st.warning("Sem coluna de Beneficiários.")
        st.stop()

    ser = df_periodo.set_index("Ano")[col_beneficiarios].dropna()
    safe_line_plot(ser.index, ser.values, "Evolução — Beneficiários", "Ano", "Beneficiários")

elif menu == "Previsão SARIMAX (igual ao notebook)":
    st.header("Previsão de Despesa Total com SARIMAX (igual ao notebook)")

    if col_despesa_total is None:
        st.warning("Sem despesa_total.")
        st.stop()

    # Exógenas do notebook (precisa das 3)
    exog_cols = []
    if col_segurados is not None:
        exog_cols.append(col_segurados)
    if col_pop_emp is not None:
        exog_cols.append(col_pop_emp)
    if col_pensionista_inps is not None:
        exog_cols.append(col_pensionista_inps)

    if len(exog_cols) < 3:
        st.error(
            "Para ficar IGUAL ao notebook, o dataset precisa destas colunas:\n"
            "- Segurados\n- População_empregada\n- Pensionista_INPS\n\n"
            f"Detectadas agora: {exog_cols}"
        )
        st.stop()

    # dataset do período (ordenado)
    d = df_periodo[["Ano", col_despesa_total] + exog_cols].dropna().sort_values("Ano").copy()
    d = d.rename(columns={col_despesa_total: "despesa_total"})

    if len(d) < 8:
        st.info("Série curta para SARIMAX. Selecione mais anos no filtro de período.")
        st.stop()

    ultimo_ano = int(d["Ano"].max())
    ano_fim = st.slider("Prever até ao ano", min_value=ultimo_ano + 1, max_value=ultimo_ano + 20, value=max(ultimo_ano + 6, ultimo_ano + 1))
    steps = int(ano_fim - ultimo_ano)

    st.caption(f"Ajuste: SARIMAX(1,1,1) com exógenas {exog_cols} | Base: {d['Ano'].min()}–{ultimo_ano} | Passos: {steps}")

    run_btn = st.button("Gerar previsão SARIMAX")
    if not run_btn:
        st.info("Clique em **Gerar previsão SARIMAX** para executar (mantém o dashboard leve).")
        st.stop()

    y = pd.to_numeric(d["despesa_total"], errors="coerce").values
    exog_hist = d[exog_cols].apply(pd.to_numeric, errors="coerce").values

    if np.isnan(y).any() or np.isnan(exog_hist).any():
        st.error("Há NaN nas séries usadas (despesa_total ou exógenas). Verifica o CSV/periodo.")
        st.stop()

    # Fit
    try:
        res = fit_sarimax(y, exog_hist)
    except Exception as e:
        st.error(f"Erro ao ajustar SARIMAX: {e}")
        st.stop()

    # Exógenas futuras (igual notebook: crescimento médio histórico)
    exog_future_df = build_exog_future_from_mean_growth(d, exog_cols, steps=steps)
    if exog_future_df.isna().any().any():
        st.error("Falha ao projetar exógenas futuras (há NaN). Verifica valores finais das colunas exógenas.")
        st.stop()

    # Forecast + IC
    try:
        fc = res.get_forecast(steps=steps, exog=exog_future_df.values)
        fc_mean = pd.Series(fc.predicted_mean)
        fc_ci = fc.conf_int()
    except Exception as e:
        st.error(f"Erro ao prever com SARIMAX: {e}")
        st.stop()

    anos_previstos = list(range(ultimo_ano + 1, ultimo_ano + steps + 1))

    st.subheader("Tabela de previsão (com IC 95%)")
    out = pd.DataFrame(
        {
            "Ano": anos_previstos,
            "Previsão": fc_mean.values,
            "IC_95_inf": fc_ci.iloc[:, 0].values,
            "IC_95_sup": fc_ci.iloc[:, 1].values,
        }
    )
    st.dataframe(out, use_container_width=True)

    st.subheader("Gráfico (histórico + previsão + IC)")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d["Ano"], d["despesa_total"], marker="o", label="Observado")
    ax.plot(anos_previstos, fc_mean.values, marker="o", linestyle="--", label="Previsto (SARIMAX)")
    ax.fill_between(
        anos_previstos,
        fc_ci.iloc[:, 0].values,
        fc_ci.iloc[:, 1].values,
        alpha=0.2,
        label="IC 95%",
    )
    ax.set_xlabel("Ano")
    ax.set_ylabel("Despesa Total")
    ax.set_title("Previsão de Despesa Total com SARIMAX(1,1,1) — alinhado com notebook")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

elif menu == "ML (executar)":
    st.header("ML — comparação de modelos (executar)")

    if col_despesa_total is None:
        st.warning("Sem despesa_total.")
        st.stop()

    feature_candidates = [col_pop, col_gdp, col_infl, col_segurados]
    feature_cols = [c for c in feature_candidates if c is not None]
    if not feature_cols:
        st.info("Sem variáveis (Pop/PIB/Inflação/Segurados) para treinar modelos.")
        st.stop()

    test_years = st.slider("Anos no teste (últimos N anos)", 1, 6, 3)

    run = st.button("Rodar comparação (ML)")
    if not run:
        st.info("Clique em **Rodar comparação (ML)** para executar (evita pesar o PC).")
        st.stop()

    results, _test_df_out, _preds, _test_set = compare_models(
        df_periodo,
        feature_cols=feature_cols,
        target_col=col_despesa_total,
        year_col="Ano",
        test_years=test_years,
    )

    st.subheader("Resultados (holdout temporal)")
    st.dataframe(results, use_container_width=True)

    st.subheader("Resultados CV (TimeSeriesSplit)")
    cv = time_series_cv_rmse(
        df_periodo,
        feature_cols=feature_cols,
        target_col=col_despesa_total,
        year_col="Ano",
        n_splits=min(4, max(2, len(df_periodo) - 2)),
    )
    st.dataframe(cv, use_container_width=True)

    st.subheader("Agent AI (com base nos resultados ML)")
    base = df_periodo[["Ano", col_despesa_total]].rename(columns={col_despesa_total: "despesa_total"}).copy()

    agent_out = run_agent(base, results_df=results)
    report_md = agent_out.get("report_md", "")

    report_path = ROOT_DIR / "reports" / "agent_report.md"
    save_report_md(report_md, report_path)

    st.markdown(report_md)

    st.download_button(
        "Baixar relatório (.md)",
        data=report_md.encode("utf-8"),
        file_name="agent_report.md",
        mime="text/markdown",
    )

elif menu == "Agent AI (executar)":
    st.header("Agent AI — Diagnóstico Automático (executar)")

    if col_despesa_total is None:
        st.warning("Sem despesa_total para analisar.")
        st.stop()

    base = df_periodo[["Ano", col_despesa_total]].rename(columns={col_despesa_total: "despesa_total"}).copy()
    report_path = ROOT_DIR / "reports" / "agent_report.md"

    use_cached = st.checkbox("Mostrar último relatório salvo (sem recalcular)", value=True)

    if use_cached and report_path.exists():
        st.info("A mostrar o último relatório salvo.")
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        run_agent_btn = st.button("Gerar relatório do Agent")
        if not run_agent_btn:
            st.info("Clique em **Gerar relatório do Agent** para executar.")
            st.stop()

        agent_out = run_agent(base, results_df=None)
        report_md = agent_out.get("report_md", "")

        save_report_md(report_md, report_path)

        st.success("Relatório gerado e guardado em reports/agent_report.md")
        st.markdown(report_md)

        st.download_button(
            "Baixar relatório (.md)",
            data=report_md.encode("utf-8"),
            file_name="agent_report.md",
            mime="text/markdown",
        )
