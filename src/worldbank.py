import requests
import pandas as pd

WB_BASE = "https://api.worldbank.org/v2"

def fetch_indicator(country_iso3: str, indicator: str, start_year: int, end_year: int, timeout: int = 30) -> pd.DataFrame:
    url = f"{WB_BASE}/country/{country_iso3}/indicator/{indicator}"
    params = {"format": "json", "per_page": 20000, "date": f"{start_year}:{end_year}"}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("Unexpected response from World Bank API")
    rows = data[1] or []
    out = []
    for row in rows:
        year = row.get("date")
        value = row.get("value")
        if year is None:
            continue
        out.append((int(year), value))
    df = pd.DataFrame(out, columns=["Ano", indicator]).dropna(subset=["Ano"])
    df = df.sort_values("Ano").reset_index(drop=True)
    return df

def fetch_gdp_current_usd_cpv(start_year: int = 2010, end_year: int = 2025) -> pd.DataFrame:
    """GDP (current US$) for Cabo Verde (CPV) — indicator NY.GDP.MKTP.CD"""
    return fetch_indicator("CPV", "NY.GDP.MKTP.CD", start_year, end_year).rename(columns={"NY.GDP.MKTP.CD":"PIB_WB_USD"})
