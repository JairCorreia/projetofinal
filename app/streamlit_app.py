# app/streamlit_app.py
# ============================================================
# DASHBOARD (alinhado ao notebook Trabalho_final_PAGD_G3)
# - Lê data/processed/dataset_merge_wb_gdp.csv
# - Usa log_PIB_WB_USD = np.log(PIB_WB_USD) (igual notebook)
# - ML igual notebook: baseline (lag1) + Ridge/Lasso/RF com GridSearchCV + TimeSeriesSplit
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Agent (src/)
from src.agent_ai import run_agent


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
# Data helpers (alinhado notebook)
# ----------------------------
def normalize_year_column(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    for c in ["Ano", "ANO", "ano", "Year", "YEAR", "year"]:
        if c in d.columns:
            if c != "Ano":
                d = d.rename(columns={c: "Ano"})
            break
    if "Ano" not in d.columns:
        raise KeyError("Não encontrei coluna de ano (Ano/ANO/ano/Year).")

    d["Ano"] = pd.to_numeric(d["Ano"], errors="coerce")
    d = d.dropna(subset=["Ano"]).copy()
    d["Ano"] = d["Ano"].astype(int)
    d = d.sort_values("Ano").reset_index(drop=True)
    return d


@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = normalize_year_column(df)

    # Garantir colunas essenciais no formato esperado
    if "PIB_WB_USD" in df.columns and "log_PIB_WB_USD" not in df.columns:
        # notebook faz np.log (não log1p)
        df["log_PIB_WB_USD"] = np.log(df["PIB_WB_USD"])

    return df


def get_rubricas_cols(df: pd.DataFrame) -> list[str]:
    # notebook gera despesa_sdo, despesa_maternidade, etc (lowercase)
    return [c for c in df.columns if str(c).startswith("despesa_") and c != "despesa_total"]


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
    return pd.DataFrame(rows).sort_values("Peso", ascending=False).head(n)


def compute_elasticities(df_periodo: pd.DataFrame, target_col: str, feature_cols: list[str], top_n: int = 5):
    d = df_periodo.sort_values("Ano").copy()
    if len(d) < 4:
        return pd.DataFrame()

    t = pd.to_numeric(d[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    t_pct = t.pct_change().replace([np.inf, -np.inf], np.nan)

    rows = []
    for f in feature_cols:
        if f not in d.columns:
            continue
        x = pd.to_numeric(d[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        x_pct = x.pct_change().replace([np.inf, -np.inf], np.nan)
        ratio = (t_pct / x_pct).replace([np.inf, -np.inf], np.nan).dropna()
        if ratio.empty:
            continue
        rows.append({"Variável": f, "Elasticidade_média": float(ratio.mean()), "N": int(ratio.shape[0])})

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("Elasticidade_média", key=lambda s: s.abs(), ascending=False)
        .head(top_n)
    )


# ----------------------------
# ML helpers (IGUAL AO NOTEBOOK)
# ----------------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def add_time_features(df: pd.DataFrame, target: str = "despesa_total", year_col: str = "Ano") -> pd.DataFrame:
    d = df.sort_values(year_col).copy()
    d[f"{target}_lag1"] = d[target].shift(1)
    d[f"{target}_roll3"] = d[target].rolling(3).mean()
    d[f"{target}_growth"] = d[target].pct_change()
    return d


def run_ml_like_notebook(df_periodo: pd.DataFrame, target_col: str = "despesa_total", year_col: str = "Ano", test_years: int = 3):
    if target_col not in df_periodo.columns:
        raise KeyError(f"Não existe a coluna '{target_col}' no dataset.")

    df_ml = add_time_features(df_periodo, target=target_col, year_col=year_col).dropna(subset=[target_col])
    num_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in (year_col, target_col)]

    # Igual ao teu streamlit anterior: evitar Segurados e Beneficiários juntos (colinearidade alta)
    if "Segurados" in feature_cols and "Beneficiários" in feature_cols:
        feature_cols.remove("Beneficiários")

    years = sorted(df_ml[year_col].dropna().astype(int).unique())
    if len(years) <= test_years + 2:
        raise ValueError("Poucos anos para treino/teste. Reduz test_years ou aumenta o período.")

    cut = years[-test_years]
    train = df_ml[df_ml[year_col] < cut].copy()
    test = df_ml[df_ml[year_col] >= cut].copy()

    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    baseline_pred = test[f"{target_col}_lag1"].values
    results = [{"Modelo": "Baseline (Lag1)", **evaluate(y_test.values, baseline_pred)}]

    pip_lin = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", LinearRegression())])

    pip_ridge = Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler()),
                          ("model", Ridge())])

    pip_lasso = Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler()),
                          ("model", Lasso(max_iter=5000))])

    rf = Pipeline([("imputer", SimpleImputer(strategy="median")),
                   ("model", RandomForestRegressor(random_state=42))])

    tscv = TimeSeriesSplit(n_splits=4)

    ridge_cv = GridSearchCV(
        pip_ridge, {"model__alpha": [0.1, 1, 10, 50, 100]},
        cv=tscv, scoring="neg_root_mean_squared_error"
    )
    lasso_cv = GridSearchCV(
        pip_lasso, {"model__alpha": [0.001, 0.01, 0.1, 1]},
        cv=tscv, scoring="neg_root_mean_squared_error"
    )
    rf_cv = GridSearchCV(
        rf,
        {"model__n_estimators": [200, 500],
         "model__max_depth": [None, 4, 6],
         "model__min_samples_leaf": [1, 2, 4]},
        cv=tscv, scoring="neg_root_mean_squared_error"
    )

    models = {
        "LinearRegression": pip_lin,
        "Ridge (CV)": ridge_cv,
        "Lasso (CV)": lasso_cv,
        "RandomForest (CV)": rf_cv,
    }

    preds = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        pred = mdl.predict(X_test)
        preds[name] = pred
        results.append({"Modelo": name, **evaluate(y_test.values, pred)})

    res_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best = res_df.iloc[0]["Modelo"]

    if best != "Baseline (Lag1)":
        best_model = models[best]
        best_model.fit(X_train, y_train)
        pred_best = best_model.predict(X_test)
    else:
        best_model = None
        pred_best = baseline_pred

    return {
        "df_ml": df_ml,
        "feature_cols": feature_cols,
        "train": train,
        "test": test,
        "res_df": res_df,
        "best_name": best,
        "best_model": best_model,
        "pred_best": pred_best,
        "y_test": y_test.values,
    }


# ----------------------------
# UI init
# ----------------------------
apply_css()

# Sidebar branding
logo_path = ROOT_DIR / "figs" / "logo.png"
if logo_path.exists():
    st.sidebar.image(str(logo_path), use_container_width=True)
st.sidebar.markdown("### Projeto Final G3")
st.sidebar.caption("Dashboard • Notebook-aligned • ML • Agent AI")

st.title("📊 Dashboard (igual ao Notebook)")


# ----------------------------
# Load dataset
# ----------------------------
default_csv = str(ROOT_DIR / "data" / "processed" / "dataset_merge_wb_gdp.csv")
csv_path = st.sidebar.text_input("Caminho do CSV (data/processed)", value=default_csv)

try:
    df = load_data(csv_path)
except Exception as e:
    st.error(f"Erro ao carregar dataset: {e}")
    st.stop()

if "despesa_total" not in df.columns:
    st.error("Não existe a coluna 'despesa_total' no dataset. Confirma se exportaste o CSV do notebook corretamente.")
    st.stop()

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

# KPIs
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Ano referência", str(year_ref), "Para Top N e resumo")
with c2:
    kpi("Registos (total)", str(len(df)), f"Selecionado: {len(df_periodo)}")
with c3:
    kpi("Anos", str(len(years_all)), f"{years_all[0]}–{years_all[-1]}")
with c4:
    kpi("Rubricas (despesa_*)", str(len(rubricas_cols_all)), "Detetadas no dataset")

st.divider()

# Menu
pages = ["Visão Geral", "Despesas", "Peso das despesas (Top N)", "Elasticidades", "ML (igual notebook)", "Agent AI"]
menu = st.sidebar.radio("Menu", pages)


# ----------------------------
# Pages
# ----------------------------
if menu == "Visão Geral":
    st.subheader("Resumo do dataset")
    st.write(f"Registos: **{len(df)}** | Anos: **{len(years_all)}** ({years_all[0]}–{years_all[-1]})")
    st.dataframe(df.head(30), use_container_width=True)

    st.subheader("Colunas-chave (esperadas do notebook)")
    show_cols = [c for c in ["PIB_WB_USD", "log_PIB_WB_USD", "Inflação", "Segurados", "Beneficiários"] if c in df.columns]
    st.write({"Encontradas": show_cols, "Rubricas": len(rubricas_cols_all)})

elif menu == "Despesas":
    st.header("Despesas")
    ser = df_periodo.set_index("Ano")["despesa_total"].dropna()
    safe_line_plot(ser.index, ser.values, "Evolução — despesa_total", "Ano", "Despesa")
    safe_box_plot(ser.values, "Box — despesa_total (período)", "Despesa")

    c1, c2 = st.columns(2)
    cagr = calc_cagr(ser)
    yoy = mean_yoy_growth(ser)
    with c1:
        kpi("CAGR (período)", f"{cagr*100:.2f}%" if cagr is not None else "n/d")
    with c2:
        kpi("Média YoY (período)", f"{yoy*100:.2f}%" if yoy is not None else "n/d")

    if rubricas_cols_all:
        st.subheader("Rubricas (despesa_*) — selecione poucas")
        default_sel = rubricas_cols_all[: min(3, len(rubricas_cols_all))]
        rubricas_sel = st.multiselect("Rubricas", rubricas_cols_all, default=default_sel)

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

    topn = st.slider("Top N", 3, 30, 10)
    out = top_n_weights(df_ref, rubricas_cols_all, "despesa_total", n=topn)
    st.dataframe(out, use_container_width=True)

elif menu == "Elasticidades":
    st.header("Elasticidades (Top 5) — igual lógica do notebook (variação % / variação %)")
    feature_cols = [c for c in ["Segurados", "População_empregada", "Pensionista_INPS", "Inflação", "PIB_WB_USD", "log_PIB_WB_USD"] if c in df_periodo.columns]
    if not feature_cols:
        st.info("Não encontrei variáveis numéricas típicas do notebook para elasticidade (ex: Segurados, PIB_WB_USD, etc.).")
        st.stop()

    out = compute_elasticities(df_periodo, "despesa_total", feature_cols, top_n=5)
    if out.empty:
        st.info("Não foi possível calcular elasticidades (dados insuficientes / variação nula).")
    else:
        st.dataframe(out, use_container_width=True)

elif menu == "ML (igual notebook)":
    st.header("ML — igual ao notebook (baseline lag1 + Ridge/Lasso/RF com CV temporal)")

    test_years = st.slider("Quantos anos usar como teste (últimos N)?", 2, 6, 3)

    run = st.button("Rodar ML")
    if not run:
        st.info("Clique em **Rodar ML** para executar (leve).")
        st.stop()

    try:
        out = run_ml_like_notebook(df_periodo, target_col="despesa_total", year_col="Ano", test_years=test_years)
    except Exception as e:
        st.error(f"Falha no ML: {e}")
        st.stop()

    st.subheader("Resultados (RMSE menor = melhor)")
    st.dataframe(out["res_df"], use_container_width=True)
    st.success(f"Melhor modelo (RMSE): **{out['best_name']}**")

    # Real vs previsto (teste temporal)
    test = out["test"]
    fig = plt.figure(figsize=(10, 4))
    plt.plot(test["Ano"].values, out["y_test"], marker="o", label="Real")
    plt.plot(test["Ano"].values, out["pred_best"], marker="o", label="Previsto")
    plt.legend()
    plt.xlabel("Ano")
    plt.ylabel("despesa_total")
    plt.title("Real vs Previsto (teste temporal)")
    st.pyplot(fig, clear_figure=True)

    # Guardar para Agent AI
    st.session_state["ml_results_df"] = out["res_df"]

elif menu == "Agent AI":
    st.header("Agent AI — diagnóstico automático (baseado no dataset e no ML)")

    base = df_periodo[["Ano", "despesa_total"]].copy()
    res_df = st.session_state.get("ml_results_df", None)

    agent_out = run_agent(base, results_df=res_df)
    report_md = agent_out.get("report_md", "")

    st.markdown(report_md if report_md else "Sem relatório (report_md vazio).")

    if agent_out.get("best_model"):
        st.info(f"Recomendação do agente: **{agent_out.get('best_model')}**")
