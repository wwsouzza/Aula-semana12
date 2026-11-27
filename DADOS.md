Podemos verificar na comparação dos modelos de regressão linear e Arvore de Decisaão, que tanto para o consumo_campus e o consumo_serra, o modelo de Regressão linear precisa de menos dados, conforme a seguir:--- Curva de aprendizado consumo_campus ---

--- Árvore de Decisão ---
 linhas_no_treino  erro_treino  erro_validacao
               29          4.2            50.4
               58         11.5            45.0
              116         15.9            41.5
              233         20.5            37.4
              408         23.6            34.2
              584         25.4            32.9

--- Regressão Linear ---
 linhas_no_treino  erro_treino  erro_validacao
               29         32.7            38.0
               58         31.3            35.6
              116         32.0            34.6
              233         32.4            34.1
              408         32.5            33.6
              584         32.5            33.5

--- Curva de aprendizado consumo_campus_serra ---

--- Árvore de Decisão ---
 linhas_no_treino  erro_treino  erro_validacao
                7         -0.0            57.5
               14          0.6            41.0
               28          3.1            43.7
               57          8.6            44.0
              100         12.9            39.2
              144         14.9            39.5

--- Regressão Linear ---
 linhas_no_treino  erro_treino  erro_validacao
                7         16.3            48.6
               14         23.5            39.8
               28         27.5            40.3
               57         27.5            36.4
              100         29.4            35.0
              144         29.8            34.7

2- No teste Ks da ocuupação nos dois campus, verificou se que temos um padrão bem diferente nos dois campus: teste KS da ocupação: estatística = 0.38 | p-valor = 6.2e-19

3- Verificamos um erro significativo de 55.0 usando os dados de treino consumo_campus para o campus_serra.