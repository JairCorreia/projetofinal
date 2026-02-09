from pathlib import Path
import pandas as pd

def read_taxa_cobertura(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name="TAXA_COBERTURA")
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    if "Ano" not in df.columns:
        if "Unnamed:_0" in df.columns:
            df = df.rename(columns={"Unnamed:_0": "Ano"})
        elif "Unnamed:0" in df.columns:
            df = df.rename(columns={"Unnamed:0": "Ano"})
        else:
            df = df.rename(columns={df.columns[0]: "Ano"})
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")
    return df

def read_expenses_long(excel_path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(excel_path)
    expense_sheets = [s for s in xls.sheet_names if s.upper() != "TAXA_COBERTURA"]

    def read_sheet(sheet_name: str) -> pd.DataFrame:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]

        if "Ano" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Ano"})

        candidates = [c for c in df.columns if c.lower() in ("valor", "despesa", "total", "montante")]
        vcol = candidates[0] if candidates else df.columns[-1]
        df = df.rename(columns={vcol: "Valor"})

        df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce")
        df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")
        df["Tipo"] = sheet_name
        return df[["Ano", "Tipo", "Valor"]].dropna(subset=["Ano", "Valor"])

    return pd.concat([read_sheet(s) for s in expense_sheets], ignore_index=True)

def build_despesa_total(excel_path: Path) -> pd.DataFrame:
    despesas_long = read_expenses_long(excel_path)
    return (
        despesas_long.groupby("Ano", as_index=False)["Valor"].sum()
        .rename(columns={"Valor": "despesa_total"})
    )

def build_dataset(excel_path: Path) -> pd.DataFrame:
    taxa = read_taxa_cobertura(excel_path)
    desp = build_despesa_total(excel_path)
    return taxa.merge(desp, on="Ano", how="inner").sort_values("Ano").reset_index(drop=True)

def save_processed(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path
