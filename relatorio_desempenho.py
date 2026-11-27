# Avaliação de erro e desempenho
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
consumo_campus = pd.read_csv("consumo_campus.csv")
colunas_de_entrada = ["temperatura_c", "ocupacao_pct", "dia_util", "evento"]
entradas_treino, entradas_teste, consumo_treino, consumo_teste = train_test_split(
 consumo_campus[colunas_de_entrada], consumo_campus["consumo_kwh"], test_size=0.25, random_state=42)
modelo_linear = LinearRegression().fit(entradas_treino, consumo_treino)
previsoes_teste = modelo_linear.predict(entradas_teste)

#Calculando as métricas de erro para o modelo linear
erro_absoluto_medio = mean_absolute_error(consumo_teste, previsoes_teste)
erro_quadratico_medio = mean_squared_error(consumo_teste, previsoes_teste)
raiz_do_erro_quadratico = np.sqrt(erro_quadratico_medio)
erro_percentual_medio = mean_absolute_percentage_error(consumo_teste, previsoes_teste)
coeficiente_de_determinacao = r2_score(consumo_teste, previsoes_teste)
print("\n--- Avaliação de erro e desempenho do modelo linear ---")
print(f"MAE = {erro_absoluto_medio:.1f} kWh")
print(f"MSE = {erro_quadratico_medio:.1f} kWh²")
print(f"RMSE = {raiz_do_erro_quadratico:.1f} kWh")
print(f"MAPE = {erro_percentual_medio:.1%}")
print(f"R² = {coeficiente_de_determinacao:.3f}")

#Comparação com o Baseline (DummyRegressor) e cálculo do R²

previsoes_baseline = DummyRegressor(strategy="mean").fit(entradas_treino,
consumo_treino).predict(entradas_teste)
soma_erros_modelo = ((consumo_teste - previsoes_teste) ** 2).sum()
soma_erros_baseline = ((consumo_teste - previsoes_baseline) ** 2).sum()
print("\n--- Comparação com o Baseline (DummyRegressor) ---")
print(f"R² do baseline (sempre a média) = {r2_score(consumo_teste, previsoes_baseline):.3f}")
print(f"R² do modelo = 1 - {soma_erros_modelo:,.0f} / {soma_erros_baseline:,.0f} = {1 -
soma_erros_modelo / soma_erros_baseline:.3f}")

#Validação cruzada com Kfold e comparação de modelos
erros_por_divisao = []
for semente in range(5):
 entradas_a, entradas_b, consumo_a, consumo_b = train_test_split(consumo_campus[colunas_de_entrada],
consumo_campus["consumo_kwh"], test_size=0.25, random_state=semente)
 erros_por_divisao.append(mean_absolute_error(consumo_b, LinearRegression().fit(entradas_a,
consumo_a).predict(entradas_b)))
print(f"5 divisões treino/teste diferentes: MAE de {min(erros_por_divisao):.1f} a {max(erros_por_divisao):.1f} kWh" )
divisor_de_validacao = KFold(n_splits=5, shuffle=True, random_state=42)
modelos_para_comparar = {"baseline (média)": DummyRegressor(strategy="mean"), "regressão linear": LinearRegression(),
"árvore (profundidade 4)": DecisionTreeRegressor(max_depth=4, random_state=42)}
for nome_do_modelo, modelo in modelos_para_comparar.items():
 erros = -cross_val_score(modelo, consumo_campus[colunas_de_entrada], consumo_campus[ "consumo_kwh"],
cv=divisor_de_validacao, scoring="neg_mean_absolute_error")
 print(f"{nome_do_modelo:24s} MAE = {erros.mean():5.1f} ± {erros.std():.1f} kWh")

#Viés e erro médio absoluto (MAE) somente nos dias úteis
residuos_teste = consumo_teste - previsoes_teste
residuos_dias_uteis = residuos_teste[entradas_teste["dia_util"] == 1]
vies_dias_uteis = residuos_dias_uteis.mean()
mae_dias_uteis = residuos_dias_uteis.abs().mean()
print("\n--- Viés e erro médio absoluto (MAE) somente nos dias úteis ---")
print(f"Viés somente nos dias úteis (erro médio com sinal) = {vies_dias_uteis:+.2f} kWh")
print(f"MAE somente nos dias úteis (erro médio absoluto)   = {mae_dias_uteis:.2f} kWh")
print(f"percentil 90 do erro absoluto = {np.percentile(np.abs(residuos_teste), 90):.1f} kWh")