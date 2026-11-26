1- Análise da variável umidade relativa, considerando uma amostra de 30 dias e 1000 amostras de 30 dias. Dados encontrados:
Media da população 730 DIAS) = 69.77
Media da amostra de 30 dias = 68.83
Media das 1000 amostras de 30 dias = 69.79 / Desvio padrão das 1000 amostras de 30 dias = 1.64
Erro padrão teórico = 1.63

Podemos verificar que a média da amostra com 30 dias está muito próxima da média da população e o erro padrão teórico também é muito próximo do desvio padrão da média das 1000 amostras de 30 dias.Portanto poderiamos considerar que quaisquer amostra de 30 dias seria bastante significativa da população.
Outro ponto verificado é que as 1000 amostras de 30 dias comprova o TCL, pois a média da população se aproxima muito da média da população.

2- Medição da incerteza da variável Ocupação. A incerteza foi calculada com amostras de 40 dias, com bootstrap com repetição de 2000 vezes. 
Dados obtidos:
mediana da amostra de 40 dias = 64.00
Intervalo de confiança de 95% = [53.50, 70.00]
a mediana real da população = 65.00 esta dentro do intervalo de confiança? True
Podemos verificar que a mediana da ocupação provalvelmente estará entre 53.70% e 70.00%.
Podemos verificar que a mediana real da população está muito próxima da amostra.
Conclusçao a média da população está dentro do intervalo de confiança de 95%, 
Somente tomar cuidado pois a variação de 16,5% pode mostrar que dias de semanas pode ter uma influência significativa do consumo.

3- Comparação DummyRegressor e LinearRegression para prever consumo com base somente na temperatura. Dados encontrados:

Intercepto  = 352.09
Coeficiente angular da temperatura = 6.40
Estimativa de consumo para 30ºC = 543.95 kWh
baseline (media = 492.04 kwh) : erro medio = 81.96 kwh
regressão linear: erro medio = 80.45 kwh
reducao do erro = 1.8%

Apresentou uma redução de apenas 1,8% no erro da Baseline,sendo muito pouco significativa. Teremos que buscar agrupar outras variãveis, como fins de semana para tentar aumentar a significância da temperatura. 