# O impacto dos dados no comportamento do modelo
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
consumo_campus = pd.read_csv("consumo_campus.csv")
consumo_serra = pd.read_csv("consumo_campus_serra.csv")
colunas_de_entrada = ["temperatura_c", "ocupacao_pct", "dia_util", "evento"]
entradas_treino, entradas_teste, consumo_treino, consumo_teste = train_test_split(
 consumo_campus[colunas_de_entrada], consumo_campus["consumo_kwh"], test_size=0.25, random_state=42)

# comparando a curva de aprendizado de um modelo linear e de uma árvore de decisão para o arquivo consumo_campus.csv

print("\n--- Curva de aprendizado consumo_campus ---")

# curva de aprendizado de uma arvore de Decisão para o arquivo consumo_campus.csv
tamanhos_do_treino, erros_treino, erros_validacao = learning_curve(
 estimator=DecisionTreeRegressor(max_depth= 6, random_state= 42),
X=consumo_campus[colunas_de_entrada], y=consumo_campus[ "consumo_kwh" ],
 train_sizes=[ 0.05, 0.1, 0.2, 0.4, 0.7, 1.0], cv=5,
scoring="neg_mean_absolute_error" , shuffle=True, random_state= 42)
tabela_curva = pd.DataFrame({ "linhas_no_treino" : tamanhos_do_treino,
"erro_treino" : -erros_treino.mean(axis= 1), "erro_validacao" :
-erros_validacao.mean(axis= 1)})
print("\n--- Árvore de Decisão ---")
print(tabela_curva. round(1).to_string(index= False))


# curva de aprendizado de uma regressão linear para o arquivo consumo_campus.csv
tamanhos_do_treino, erros_treino, erros_validacao = learning_curve(
 estimator= LinearRegression(),
X=consumo_campus[colunas_de_entrada], y=consumo_campus[ "consumo_kwh" ],
 train_sizes=[ 0.05, 0.1, 0.2, 0.4, 0.7, 1.0], cv=5,
scoring="neg_mean_absolute_error" , shuffle=True, random_state= 42)
tabela_curva = pd.DataFrame({ "linhas_no_treino" : tamanhos_do_treino,
"erro_treino" : -erros_treino.mean(axis= 1), "erro_validacao" :
-erros_validacao.mean(axis= 1)})
print("\n--- Regressão Linear ---")
print(tabela_curva. round(1).to_string(index= False))


# comparando a curva de aprendizado de um modelo linear e de uma árvore de decisão para o arquivo consumo_campus_serra.csv

print("\n--- Curva de aprendizado consumo_campus_serra ---")

# curva de aprendizado de uma arvore de Decisão para o arquivo consumo_serra.csv
tamanhos_do_treino, erros_treino, erros_validacao = learning_curve(
 estimator=DecisionTreeRegressor(max_depth= 6, random_state= 42),
X=consumo_serra[colunas_de_entrada], y=consumo_serra[ "consumo_kwh" ],
 train_sizes=[ 0.05, 0.1, 0.2, 0.4, 0.7, 1.0], cv=5,
scoring="neg_mean_absolute_error" , shuffle=True, random_state= 42)
tabela_curva = pd.DataFrame({ "linhas_no_treino" : tamanhos_do_treino,
"erro_treino" : -erros_treino.mean(axis= 1), "erro_validacao" :
-erros_validacao.mean(axis= 1)})
print("\n--- Árvore de Decisão ---")
print(tabela_curva. round(1).to_string(index= False))


# curva de aprendizado de uma regressão linear para o arquivo consumo_campus_serra.csv
tamanhos_do_treino, erros_treino, erros_validacao = learning_curve(
 estimator= LinearRegression(),
X=consumo_serra[colunas_de_entrada], y=consumo_serra[ "consumo_kwh" ],
 train_sizes=[ 0.05, 0.1, 0.2, 0.4, 0.7, 1.0], cv=5,
scoring="neg_mean_absolute_error" , shuffle=True, random_state= 42)
tabela_curva = pd.DataFrame({ "linhas_no_treino" : tamanhos_do_treino,
"erro_treino" : -erros_treino.mean(axis= 1), "erro_validacao" :
-erros_validacao.mean(axis= 1)})
print("\n--- Regressão Linear ---")
print(tabela_curva. round(1).to_string(index= False))

print("\n--- Comparando a ocupação nos dois campus com KS---")

# comparação ocupacao_pct nos dois campus com o ks_ssample
print(pd.DataFrame({"campus_principal": consumo_campus[["temperatura_c", "ocupacao_pct",
"consumo_kwh"]].mean(),
 "campus_serra": consumo_serra[["temperatura_c", "ocupacao_pct",
"consumo_kwh"]].mean()}).round(1))
estatistica_ks, p_valor_ks = stats.ks_2samp(consumo_campus["ocupacao_pct"],
consumo_serra["ocupacao_pct"])
print(f"teste KS da ocupação: estatística = {estatistica_ks:.2f} | p-valor = {p_valor_ks:.2g}")
modelo_final = LinearRegression().fit(entradas_treino, consumo_treino)
previsoes_serra = modelo_final.predict(consumo_serra[colunas_de_entrada])
residuos_serra = consumo_serra["consumo_kwh"] - previsoes_serra
print(f"erro no teste do campus principal = {mean_absolute_error(consumo_teste,
modelo_final.predict(entradas_teste)):.1f} kWh")
print(f"erro no campus da serra = {mean_absolute_error(consumo_serra['consumo_kwh'],
previsoes_serra):.1f} kWh")
print(f"viés (erro médio com sinal) = {residuos_serra.mean():+.1f} kWh")