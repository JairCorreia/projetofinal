# Agent AI Report

## Qualidade dos dados
- Duplicados por ano: **0**
- Anos em falta: **nenhum**

### Missing values (% por coluna) — top 10
Inflação                        7.142857
PIB                             7.142857
Ano                             0.000000
População                       0.000000
Segurados                       0.000000
Pensionista_INPS                0.000000
Pensionista_FP                  0.000000
Pensionistas_Total              0.000000
Salario_Médio_Segurados         0.000000
Taxa_cobertura_Pop_Empregada    0.000000

## Outliers (IQR) — despesa_total
- Quantidade de outliers: **0**
- Limites IQR: **[324638000.0, 4539822000.0]**

## Drift (mudança estrutural recente)
- Drift detectado? **True**
- Variação média (últimos 3 anos vs anteriores): **60.6%**
- Variação da variância: **-40.6%**

## Recomendação
- Melhor modelo (por RMSE): **Ridge (CV)**
- Sugestão: revalidar anualmente e re-treinar se o drift for persistente.