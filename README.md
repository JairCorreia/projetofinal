# Análise económica das despesas com cuidados de saúde do INPS 
## 📌 Visão Geral do Projeto

Este projeto foi desenvolvido para analisar, integrar e modelar dados macroeconómicos e das despesas do ramo doença e maternidade, permitindo:

- Compreender a evolução temporal da despesa  
- Relacionar custos com variáveis macroeconómicas  
- Criar modelos preditivos interpretáveis  
- Disponibilizar resultados num dashboard interativo  
- Apoiar decisões públicas e estratégicas  

> **Mais do que prever valores, o projeto explica comportamentos e tendências.**

## 🧭 Menu de Navegação (Notion)

 Cada item abaixo como uma **sub‑página** no Notion:
 Link: https://funky-appeal-4d7.notion.site/An-lise-e-Previs-o-de-Despesa-em-Medicamentos-2f7950ae31168079930bf696ded7d202
- 🧱 Estrutura do Projeto  
- 🔄 Fluxo Lógico do Sistema  
- 🧹 Preparação e Qualidade dos Dados  
- 📈 Análise Exploratória  
- 🤖 Modelos Estatísticos e Machine Learning  
- 🧠 Agent AI – Análise Automática  
- 📊 Dashboard Interativo (Streamlit)  
- 📜 Regras de Negócio  
- 🚀 Evolução Futura do Projeto  

## 🧱 Estrutura do Projeto

```
projeto_final_G3/
├── app/                     # Interface visual (Streamlit)
│   └── streamlit_app.py
├── data/
│   ├── raw/                 # Dados brutos (sem tratamento)
│   └── processed/           # Dados tratados e integrados
├── src/                     # Lógica principal do sistema
│   ├── preprocessing.py     # Limpeza e preparação dos dados
│   ├── worldbank.py         # Integração com API do Banco Mundial
│   ├── time_series.py       # Modelação de séries temporais
│   ├── ml.py                # Modelos de Machine Learning
│   └── agent_ai.py          # Análise automática e relatórios
├── notebooks/               # Exploração e validação
│   └── Trabalho_final_PAGD_G3.ipynb
├── reports/                 # Relatórios gerados
├── figs/                    # Gráficos e imagens
├── README.md
└── requirements.txt
```

📌 **Princípio estrutural:** cada pasta tem uma única responsabilidade, garantindo organização, manutenção e escalabilidade.

## 🔄 Fluxo Lógico do Sistema (resumo)

1) **Entrada de Dados**: Excel (despesa) + API Banco Mundial (macro)  
2) **Limpeza e preparação** (`src/preprocessing.py`)  
3) **Integração** (alinhamento temporal) → `data/processed/dataset_merge_wb_gdp.csv`  
4) **EDA** (notebook)  
5) **Modelos** (`src/ml.py`, `src/time_series.py`)  
6) **Agent AI** (`src/agent_ai.py`) → `reports/agent_report.md`  
7) **Dashboard** (`app/streamlit_app.py`)  

## 📜 Regras de Negócio

- Nunca apagar dados brutos  
- Transformações sempre rastreáveis  
- Modelos explicáveis e interpretáveis  
- Separação clara entre análise e visualização  
- Código modular e reutilizável  

## 🚀 Evolução Futura

- Novos indicadores económicos  
- Previsões multi‑cenário  
- Alertas automáticos  
- Integração com bases governamentais  
- Publicação como sistema de apoio à decisão  

---

## O que este projeto entrega (em linguagem simples)

- Um dataset consolidado com despesas por categoria e **despesa_total** (por ano)
- PIB de Cabo Verde obtido via **API do Banco Mundial** e integrado ao dataset
- AED com gráficos e correlação (figuras exportadas para `figs/`)
- Modelação econométrica (OLS) como referência
- Machine Learning (obrigatório): regressão linear, Ridge, Lasso e Random Forest, com métricas e validação temporal
- Dashboard interativo em Streamlit para explorar dados e resultados
- (Bónus) Agent AI para monitorização da qualidade dos dados e recomendações


## Relatórios automáticos
- O Agent AI gera um relatório em `reports/agent_report.md`.

## Notebooks (inclui V2 completo)

- `notebooks/Trabalho_final_PAGD_G3_original.ipynb`: versão Final submetida .

## Dados (V2)

- `data/raw/DADOS_DM_2010_2025_V2.xlsx`: ficheiro de dados utilizado na versão V2.


## Dashboard (Streamlit)
Para correr o dashboard:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```


## Extensão: Séries Temporais (ARIMA/ETS)
- Implementado no notebook V2 organizado e no dashboard (tab Séries Temporais).


## Notebook de entrega
- `notebooks/Trabalho_final_PAGD_G3_ENTREGA.ipynb` (principal)
- Inclui: API Banco Mundial (PIB), ML, Agent AI, extensão ARIMA/ETS, backtest temporal.
