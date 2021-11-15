
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflowjs as tfjs

# Le arquivo de dados
df = pd.read_json("../db/megasena-v2.json")

print(df.tail())
# Obtem algumas informações básicas dos dados

# Gera lista com todas dezenas
dezenas = pd.DataFrame(df['d1'].tolist() + df['d2'].tolist() + df['d3'].tolist() + df['d4'].tolist() + df['d5'].tolist() + df['d6'].tolist(), columns=['numeros'])

# Mostra frequencia das dezenas
print('Frequencia das Dezenas')
frequencia_dezenas = dezenas['numeros'].value_counts().sort_values(ascending=False)
print(frequencia_dezenas)

# --------------------------------------------------

# definição do modelo
N_DEZENAS = 6
N_NEURONIOS = 50
N_EPOCHS = 50
N_SORTEIOS = 40
NUM_DEZENAS = 60

# Note que aqui o modelo de declarado de outra forma, em forma de função
# Este é apenas outra forma de declarar o modelo.

model = tf.keras.Sequential([
    keras.layers.SimpleRNN(N_NEURONIOS, return_sequences=True, input_shape=[None, N_DEZENAS], activation="relu"),
    keras.layers.Dense(60,activation="sigmoid")
])
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model.summary()

# --------------------------------------------------
# Sequencia de resultados
resultados = pd.DataFrame(df[['d1','d2','d3',
                              'd4','d5','d6',]]).to_numpy()
N_RESULT = len(resultados)

# X são as dezena dos ultimos N_SORTEIOS de entrada, Y são os N_SORTEIOS deslocados em uma posição para frente
i0 = np.arange(N_RESULT-N_SORTEIOS).reshape((N_RESULT-N_SORTEIOS,1))
Is = i0 + np.arange(0, N_SORTEIOS + 1)
ys = resultados[Is]

X_series = ys[:, :-1].reshape(-1, N_SORTEIOS, N_DEZENAS)
y_series = ys[:, 1:].reshape(-1, N_SORTEIOS, N_DEZENAS)

#separa 70% dos dados para treino e restante para validação
N_TREINO = int(0.7 * N_RESULT)
X_train, y_train = X_series[:N_TREINO], y_series[:N_TREINO]
X_valid, y_valid = X_series[:-N_TREINO], y_series[:-N_TREINO]

# history = model.fit(X_train, y_train, epochs=N_EPOCHS,
#                    validation_data=(X_valid, y_valid))
# # --------------------------------------------------

# ultimo_resultado = y_series[-1:]
# print(ultimo_resultado)

# Faz previsão da próxima dezena utilizando o ultimo resultado
# previsao = model.predict(ultimo_resultado)
# print(previsao[0,-1,:])

# codifica dezenas para one-hot encoded
def encode(C):
  x = np.zeros(NUM_DEZENAS,dtype=int)
  for val in C:
    x[val-1] = 1
  return(x)

def encode_matrix(M):
  O = np.empty([M.shape[0],M.shape[1],60],dtype=int)
  for i in range(M.shape[0]):
    for j in range(M.shape[1]):
      O[i,j] = encode(M[i,j])
  return(O)

y_train_encoded = encode_matrix(y_train)
y_valid_encoded = encode_matrix(y_valid)
# Exemplo de como as dezenas são alteradas para classe.
#Caso nas dezenas tenha o número 1 o primeiro elemento do array será 1. Se houver a dezena 10, o décimo elemento será 1.
print(y_train[0,0])
print(y_train_encoded[0,0])

history = model.fit(X_train, y_train_encoded, epochs=N_EPOCHS,
validation_data=(X_valid, y_valid_encoded))
tfjs.converters.save_keras_model(model, './assets/model')
model.save('assets/model.h5')

# previsao = model.predict(ultimo_resultado)
# # previsao = model.predict([[[10,19,11,22,47,21]]])
# print(len(previsao[0,-1]))
# dezenas_pred = previsao[0,-1,:]
# print(dezenas_pred)

# probabilidade_dezenas = pd.DataFrame({'Dezena': range(1,61),'Probabilidade' : dezenas_pred[:]*100}).sort_values(by=['Probabilidade'], ascending=False)

# print(probabilidade_dezenas)
