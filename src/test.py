import numpy as np

N_DEZENAS = 6
N_SORTEIOS = 40

resultados = np.arange(1002).reshape((-1, 6))
N_RESULT = len(resultados)

i0 = np.arange(N_RESULT-N_SORTEIOS).reshape((N_RESULT-N_SORTEIOS,1))

print(i0)

print('-----')
Is = i0 + np.arange(0, N_SORTEIOS + 1)

print(Is)

ys = resultados[Is]

print('YS =============')
print(ys)

# X_series,y_series = ys[:, :-1].reshape(-1, N_SORTEIOS, N_DEZENAS), ys[:, 1:].reshape(-1, N_SORTEIOS, N_DEZENAS)

# print(X_series,'aaaaa',y_series)


# #separa 70% dos dados para treino e restante para validação
# N_TREINO = int(0.7 * N_RESULT)
# X_train, y_train = X_series[:N_TREINO], y_series[:N_TREINO]
# X_valid, y_valid = X_series[:-N_TREINO], y_series[:-N_TREINO]

# print(':SD:A:DAS:DA:SD:ASD:ASD:ASD:ASD:AS:DAS:DAS:D:ASD:ASD:ASD:AS:DA:SD:ASD:AS:DAS:D')
# print(X_train)
# print(':SD:A:DAS:DA:SD:ASD:ASD:ASD:ASD:AS:DAS:DAS:D:ASD:ASD:ASD:AS:DA:SD:ASD:AS:DAS:D')
# print(X_valid)
