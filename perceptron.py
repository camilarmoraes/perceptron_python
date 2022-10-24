import numpy as np
import pandas as pd
np.random.seed(0)

entradas = [[2.7810836, 2.550537003, 0],
            [1.465489372, 2.362125076, 0],
            [3.396561688, 4.400293529, 0],
            [1.38807019, 1.850220317, 0],
            [3.06407232, 3.005305973, 0],
            [7.627531214, 2.759262235, 1],
            [5.332441248, 2.088626775, 1],
            [6.922596716, 1.77106367, 1],
            [8.675418651, -0.242068655, 1],
            [7.673756466, 3.508563011, 1]]

dados = pd.DataFrame(entradas,columns=['X','Y','saída'])

X = np.array(dados[['X','Y']].copy())
Y = np.array(dados['saída'].copy())


def predicao(X, W, b): #O que o modelo prediz
    ativa = (np.matmul(X,W)+b)
    if ativa >= 0:
        return 1
    return 0

def perceptron(X, y, W, b, learn_rate):
  for i in range(len(X)):
    pred = predicao(X[i],W,b) # Valores 0 ou 1
    erro = y[i] - pred #Diferença entre a saída encontrada e a desejada
    bias = b
    if (pred-y[i]) < 0:
            W[0] = W[0] - (learn_rate*X[i][0]*erro)
            W[1] = W[1] - (learn_rate*X[i][1]*erro)
            bias = bias - learn_rate*erro
    elif (pred- y[i]) > 0:
            W[0] = W[0] + (learn_rate*X[i][0]*erro)
            W[1] = W[1] + (learn_rate*X[i][1]*erro)
            bias = bias + learn_rate*erro
  
  return W, bias

def treinarPerceptron(X, y, taxa_aprendizado, num_epochs):
  #W = np.array(np.random.rand(2,1)) #Pesos aleatórios
  W = np.array([[0.01],[0.03]])

  #b = 0 #bias deve começar em 0
  #W = [[0.1,0.1]]
  for i in range(num_epochs):
    W, b = perceptron(X, y, W, 0, taxa_aprendizado)
    print(f'Peso {W}, Bias {b}')
  return W,b
  
treinarPerceptron(X,Y,0.1,5) #taxa de aprendizado = 0.1 e número de epochs = 5