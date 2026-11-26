import numpy as np
import pandas as pd
import matplotlib
matplotlib.use( "Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
consumo_campus = pd.read_csv( "consumo_campus.csv" , parse_dates=[ "data"])
gerador_aleatorio = np.random.default_rng(seed= 42)

# Conhecendo o dataset 

print(consumo_campus.shape)
print(consumo_campus.head(3))
print(consumo_campus["consumo_kwh"].describe().round(2))

# Comparando uma amostra contra mil amostras da variável Umidade relativa

umidade_relativa = consumo_campus["umidade_pct"]
media_populacao = umidade_relativa.mean()
amostra_de_30_dias = umidade_relativa.sample(n= 30 , random_state= 42 )
print(f"Media da população 730 DIAS) = {media_populacao:.2f}")
print(f"Media da amostra de 30 dias = {amostra_de_30_dias.mean():.2f}")
medias_das_amostras = [umidade_relativa.sample(n= 30 , random_state= i ).mean() for i in range(1000)]
media_das_amostras = pd.Series(medias_das_amostras)
print(f"Media das 1000 amostras de 30 dias = {media_das_amostras.mean():.2f} / Desvio padrão das 1000 amostras de 30 dias = {media_das_amostras.std():.2f}")
erro_padrao_teorico = umidade_relativa.std() / np.sqrt(30)
print(f"Erro padrão teórico = {erro_padrao_teorico:.2f}")

# Verificação da incerteza com Bootstrap com intervalo de 95% de confiança na variável ocupação
ocupacao = consumo_campus["ocupacao_pct"]
mediana_populacao = ocupacao.median()
amostra_de_40_dias = ocupacao.sample(n= 40 , random_state= 42 )
valores_da_amostra = amostra_de_40_dias.to_numpy()
medianas_bootstrap = [np.median(gerador_aleatorio.choice(valores_da_amostra, size=40, replace=True)) for i in range(2000)]
limite_inferior, limite_superior = np.percentile(medianas_bootstrap, [2.5, 97.5])
print(f"mediana da amostra de 40 dias = {np.median(valores_da_amostra):.2f}")
print(f"Intervalo de confiança de 95% = [{limite_inferior:.2f}, {limite_superior:.2f}]")
print(f"a mediana real da população = {mediana_populacao:.2f} esta dentro do intervalo de confiança? {limite_inferior <= mediana_populacao <= limite_superior}")

#Treinando regressão linear com a coluna temperatura para estimar o consumo de energia
Temperatura = consumo_campus[["temperatura_c"]]
Consumo = consumo_campus["consumo_kwh"] 
entradas_treino, entradas_teste, consumo_treino, consumo_teste = train_test_split(Temperatura, Consumo, test_size=0.25, random_state=42)
modelo_linear = LinearRegression()
modelo_linear.fit(entradas_treino, consumo_treino)
print(f"Intercepto  = {modelo_linear.intercept_:.2f}")
print(f"Coeficiente angular da temperatura = {modelo_linear.coef_[0]:.2f}")
dia_novo = pd.DataFrame({"temperatura_c": [30]})
print(f"Estimativa de consumo para 30ºC = {modelo_linear.predict(dia_novo)[0]:.2f} kWh")

# Comparando com um modelo DummyRegressor (Baseline)
modelo_baseline = DummyRegressor(strategy= "mean")
modelo_baseline.fit(entradas_treino, consumo_treino)
erro_baseline = mean_absolute_error(consumo_teste, modelo_baseline.predict(entradas_teste))
erro_modelo = mean_absolute_error(consumo_teste, modelo_linear.predict(entradas_teste))
print(f"baseline (media = {modelo_baseline.constant_[0][0]:.2f} kwh) : erro medio = {erro_baseline:.2f} kwh")
print(f"regressão linear: erro medio = {erro_modelo:.2f} kwh")
print(f"reducao do erro = {1 - erro_modelo / erro_baseline:.1%}")
