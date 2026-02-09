# app/streamlit_app.py
# ============================================================
# DASHBOARD (baseado no teu código) — Ajustes pedidos:
# ✅ Remover BOX do "Segurados" (fica só a linha)
# ✅ Elasticidades: TOP 5 fixo (igual ao notebook)
# ✅ Elasticidade "aproximada" (extra) calculada também por regressão
#    em log-diferenças (mais estável), MAS mantendo a elasticidade do notebook
#    como "Elasticidade_média (pct/pct)".
# ============================================================

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# IMPORTS DA TUA CAMADA src/
from src.agent_ai import run_agent
from src.time_series import train_test_split_time, forecast_arima, forecast_ets
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


def safe_box_plot(values, title, ylabel):
    fig, ax = plt.subplots()
    ax.boxplot(values, vert=True)
    ax.set_title(title)
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
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return None
    vi = float(s.iloc[0])
    vf = float(s.iloc[-1])
    n = len(s)
    if vi <= 0 or vf <= 0:
        return None
    return (vf / vi) ** (1 / (n - 1)) - 1


def mean_yoy_growth(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return None
    yoy = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
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


# --- Elasticidades (igual notebook) + elasticidade aproximada ---
def _elasticidade_pct_pct(d: pd.DataFrame, target_col: str, feature_col: str) -> tuple[float | None, int]:
    """
    Igual ao notebook: elasticidade média ~ mean( Δ%Y / Δ%X )
    """
    t = pd.to_numeric(d[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = pd.to_numeric(d[feature_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    t_pct = t.pct_change().replace([np.inf, -np.inf], np.nan)
    x_pct = x.pct_change().replace([np.inf, -np.inf], np.nan)

    ratio = (t_pct / x_pct).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return None, 0
    return float(ratio.mean()), int(ratio.shape[0])


def _elasticidade_aprox_logdiff(d: pd.DataFrame, target_col: str, feature_col: str) -> tuple[float | None, int]:
    """
    Elasticidade aproximada (mais estável):
    beta ~ regressão simples em log-diferenças:
      Δlog(Y) = beta * Δlog(X) + erro
    """
    y = pd.to_numeric(d[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = pd.to_numeric(d[feature_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # precisa de valores positivos para log
    y = y.where(y > 0)
    x = x.where(x > 0)

    dy = np.log(y).diff()
    dx = np.log(x).diff()

    m = pd.concat([dx, dy], axis=1).dropna()
    if m.shape[0] < 3:
        return None, int(m.shape[0])

    dxv = m.iloc[:, 0].values
    dyv = m.iloc[:, 1].values

    # beta = cov(dx,dy)/var(dx)
    var = float(np.var(dxv))
    if var == 0.0:
        return None, int(m.shape[0])

    beta = float(np.cov(dxv, dyv, bias=True)[0, 1] / var)
    return beta, int(m.shape[0])


def compute_elasticities(df_periodo: pd.DataFrame, target_col: str, feature_cols: list[str], top_n: int = 5):
    """
    Retorna TOP N por |Elasticidade_média| (igual notebook),
    mas inclui também Elasticidade_aprox (log-diff) como coluna extra.
    """
    d = df_periodo.sort_values("Ano").copy()
    if len(d) < 4:
        return pd.DataFrame()

    rows = []
    for f in feature_cols:
        if f not in d.columns:
            continue

        e_mean, n1 = _elasticidade_pct_pct(d, target_col, f)
        e_apx, n2 = _elasticidade_aprox_logdiff(d, target_col, f)

        if e_mean is None and e_apx is None:
            continue

        rows.append(
            {
                "Variável": f,
                "Elasticidade_média (pct/pct)": e_mean,
                "N (pct/pct)": n1,
                "Elasticidade_aprox (log-diff)": e_apx,
                "N (log-diff)": n2,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # ranking igual notebook: por abs(elasticidade média pct/pct)
    # se pct/pct for None, usa aprox só para preencher, mas ranking mantém a lógica original
    rank_series = out["Elasticidade_média (pct/pct)"].copy()
    rank_series = rank_series.where(rank_series.notna(), out["Elasticidade_aprox (log-diff)"])
    out["_rank_abs"] = rank_series.abs()

    out = out.sort_values("_rank_abs", ascending=False).drop(columns=["_rank_abs"]).head(top_n)

    # deixar mais "limpo"
    return out


def save_report_md(report_md: str, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md or "", encoding="utf-8")


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
st.title("📊  Dashboard ")

default_csv = str(ROOT_DIR / "data" / "processed" / "dataset_merge_wb_gdp.csv")
csv_path = st.sidebar.text_input("Caminho do CSV (data/processed)", value=default_csv)

try:
    df = load_data(csv_path)
except Exception as e:
    st.error(f"Erro ao carregar dataset: {e}")
    st.stop()

# Detect columns
col_despesa_total = detect_col(df, ["despesa_total", "Despesa_Total", "total_despesa", "Total_Despesa"])
col_segurados = detect_col(df, ["Segurados", "segurados"])
col_beneficiarios = detect_col(df, ["Beneficiarios", "Beneficiários", "beneficiarios", "beneficiários"])
col_pop = detect_col(df, ["Populacao", "População", "populacao", "população"])
col_gdp = detect_col(df, ["PIB", "NY.GDP.MKTP.CD", "gdp", "GDP"])
col_infl = detect_col(df, ["Inflacao", "Inflação", "inflacao", "inflação"])

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

has_sdo = "despesa_SDO" in df_periodo.columns
has_mat = "despesa_MATERNIDADE" in df_periodo.columns
has_any_series = (col_despesa_total is not None) or (has_sdo and has_mat)
if has_any_series:
    pages += ["Previsões (executar)"]

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
    safe_box_plot(ser.values, "Box — Despesa total (período)", "Despesa")

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
    st.header("Elasticidades — TOP 5 (igual notebook) + Aproximada")

    if col_despesa_total is None:
        st.warning("Sem despesa_total.")
        st.stop()

    # Igual ao teu código: usa Pop/PIB/Inflação/Segurados se existirem
    feature_cols = [c for c in [col_pop, col_gdp, col_infl, col_segurados] if c is not None]
    if not feature_cols:
        st.info("Sem variáveis (Pop/PIB/Inflação/Segurados) para calcular elasticidades.")
        st.stop()

    # TOP 5 fixo
    out = compute_elasticities(df_periodo, col_despesa_total, feature_cols, top_n=5)

    if out.empty:
        st.info("Não foi possível calcular elasticidades (dados insuficientes / variação nula).")
    else:
        st.caption(
            "• **Elasticidade_média (pct/pct)** = média de (Δ% despesa / Δ% variável) — igual notebook.\n"
            "• **Elasticidade_aprox (log-diff)** = regressão em log-diferenças (mais estável quando há ruído)."
        )
        st.dataframe(out, use_container_width=True)

elif menu == "Segurados":
    st.header("Segurados")
    ser = df_periodo.set_index("Ano")[col_segurados].dropna()
    safe_line_plot(ser.index, ser.values, "Evolução — Segurados", "Ano", "Segurados")

    # ✅ pedido: remover BOX no Segurados (não mostrar)
    st.info("Boxplot removido conforme solicitado (fica apenas a evolução por ano).")

elif menu == "Beneficiários":
    st.header("Beneficiários")
    ser = df_periodo.set_index("Ano")[col_beneficiarios].dropna()
    safe_line_plot(ser.index, ser.values, "Evolução — Beneficiários", "Ano", "Beneficiários")
    safe_box_plot(ser.values, "Box — Beneficiários (período)", "Beneficiários")

elif menu == "Previsões (executar)":
    st.header("Previsões (executar)")

    serie_name, y = None, None
    if has_sdo and has_mat:
        serie_name = "Ramo Doença e Maternidade (SDO + MATERNIDADE)"
        y = (
            df_periodo.set_index("Ano")[["despesa_SDO", "despesa_MATERNIDADE"]]
            .fillna(0)
            .sum(axis=1)
            .dropna()
        )
    elif col_despesa_total is not None:
        serie_name = "Despesa total (despesa_total)"
        y = df_periodo.set_index("Ano")[col_despesa_total].dropna()
    else:
        st.warning("Não há série disponível para previsão.")
        st.stop()

    st.subheader(f"Série usada: {serie_name}")

    if len(y) < 6:
        st.info("Série curta: recomenda-se pelo menos 6 anos. Selecione mais anos no período.")
        st.stop()

    horizon = st.slider("Horizonte (anos)", 1, 10, 3)
    method = st.radio("Modelo", ["ARIMA", "ETS"], horizontal=True)

    safe_line_plot(y.index.tolist(), y.values.tolist(), "Histórico", "Ano", "Valor")

    run_forecast = st.button("Gerar previsão")
    if not run_forecast:
        st.info("Clique em **Gerar previsão** para executar (mantém o dashboard leve).")
        st.stop()

    ydf = y.reset_index()
    ydf.columns = ["Ano", "despesa_total"]

    try:
        _train_df, _test_df, y_train, _y_test = train_test_split_time(
            ydf,
            year_col="Ano",
            target="despesa_total",
            test_years=1,
        )
    except TypeError:
        st.error(
            "A função train_test_split_time do teu src/time_series.py tem assinatura diferente. "
            "Ajusta para year_col/target/test_years."
        )
        st.stop()

    if method == "ARIMA":
        yhat, _meta = forecast_arima(y_train, steps=horizon)
    else:
        yhat, _meta = forecast_ets(y_train, steps=horizon)

    future_years = list(range(int(y.index.max()) + 1, int(y.index.max()) + horizon + 1))
    yhat_vals = np.array(yhat).reshape(-1)

    fig, ax = plt.subplots()
    ax.plot(y.index, y.values, marker="o", label="Histórico")
    ax.plot(future_years, yhat_vals, marker="o", linestyle="--", label="Projeção")
    ax.set_title(f"Previsão ({method}) — projeção destacada")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Valor")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Tabela de projeção")
    st.dataframe(pd.DataFrame({"Ano": future_years, "Projecao": yhat_vals}), use_container_width=True)

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
