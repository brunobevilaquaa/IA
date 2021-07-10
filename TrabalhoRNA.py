import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

DataSet = pd.read_csv('arruela_.csv')
DataSet.head()

DataSet.describe()

DataSet.drop(['Hora', 'Tamanho','Referencia'], axis = 1, inplace = True)
DataSet.head()

X_cols = ["NumAmostra", "Delta", "Area"]
y_cols = ["Output1", "Output2"]

X = DataSet[X_cols]
y = DataSet[y_cols]

from sklearn.preprocessing import StandardScaler
sscaler = StandardScaler()
X = sscaler.fit_transform(X)
X = pd.DataFrame(X, columns = X_cols)
X.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 64)

#Tamanho do DataSet de Treinamento
n_records, n_features = X_train.shape

#Arquitetura da MPL
N_input = 3
neuronios = 8
N_output = 2
learnrate = 0.1

weights_input_hidden = np.random.normal(0, scale = learnrate, size = (N_input, neuronios))
weights_hidden_output = np.random.normal(0, scale = learnrate, size = (neuronios, N_output))

epocas = 11700
last_loss = None
ErrorEvolution = []
ErrorIndex = []

for e in range(epocas):
    delta_w_i_h = np.zeros(weights_input_hidden.shape)
    delta_w_h_o = np.zeros(weights_hidden_output.shape)

    for xi, yi in zip(X_train.values, y_train.values):
        hidden_layer_input = np.dot(xi, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
        output = sigmoid(output_layer_in)
        error = yi - output
        output_error_term = error * output * (1 - output)
        hidden_error = np.dot(weights_hidden_output,output_error_term)
        hidden_error_term = hidden_error * hidden_layer_output * (1 - hidden_layer_output)
        delta_w_h_o += output_error_term*hidden_layer_output[:, None]
        delta_w_i_h += hidden_error_term * xi[:, None]
        
    weights_input_hidden += learnrate * delta_w_i_h / n_records
    weights_hidden_output += learnrate * delta_w_h_o / n_records
    
    if  e % (epocas / 20) == 0:
        hidden_output = sigmoid(np.dot(xi, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output, weights_hidden_output))
                             
        loss = np.mean((out - yi) ** 2)
        
        if last_loss and last_loss < loss:
            print("Erro quadratico no treinamento: ", loss, " Atencao: O erro esta aumentando")
        else:
            print("Erro quadratico no treinamento: ", loss)
            
        last_loss = loss
         
        ErrorEvolution.append(loss)
        ErrorIndex.append(e)
        
plt.plot(ErrorIndex, ErrorEvolution, 'r')
plt.xlabel('Epoca')
plt.ylabel('Erro Quadratico')
plt.title('Evolucao do Erro no treinamento da MPL')
plt.show()

n_records, n_features = X_test.shape
predictions = 0

for xi, yi in zip(X_test.values, y_test.values):
        hidden_layer_input = np.dot(xi, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
        output = sigmoid(output_layer_in)

        if (output[0] > output[1]):
            if (yi[0] > yi[1]):
                predictions += 1 
                
        if (output[1] >= output[0]):
            if (yi[1] > yi[0]):
                predictions += 1

print("A Acuracia da Predicaoo e de: {:.3f}".format(predictions/n_records))