from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
import statsmodels.api as sm

### Técnicas de normalización y desnormalización
def normalizar(X):
    l_min = X.min()
    l_max = X.max()
    return (l_max-X) / (l_max - l_min)

def denormalizar(X_original, X):
    l_min = X_original.min()
    l_max = X_original.max()
    return l_max-(X * (l_max - l_min))

## Carga de datos
data_train=pd.read_csv("C:\\Users\\bihonda\\Desktop\\UTEC\\C5\\ML\\P1_Data\\train.csv")
data_test=pd.read_csv("C:\\Users\\bihonda\\Desktop\\UTEC\\C5\\ML\\P1_Data\\test.csv")
y_train= data_train['MTO_PIA'].to_numpy()
Y=data_train['MTO_PIA']
X=data_train['SEC_FUNC']
x_train = data_train['SEC_FUNC'].to_numpy()
x_test = data_test['SEC_FUNC'].to_numpy()

## Modelo de Regresión No Lineal
def h(X, W):
    vres=np.dot(X,np.transpose(W))
    return vres

## Función de pérdida
### Sin regularización
def Error(X, W, Y):
  y_pred=h(X,W)
  n=(np.linalg.norm((Y-y_pred))**2)/(2*len(Y))
  return n

### Regularización L1 - Lasso
def ErrorL1(X, W, Y,lam):
  y_pred=h(X,W)
  n=(np.linalg.norm((Y-y_pred))**2)/(2*len(Y))
  lamb=lam*(np.linalg.norm((W)))
  return n+lamb

### Regularización L2 - Ridge
def ErrorL2(X, W, Y,lam):
  y_pred=h(X,W)
  n=(np.linalg.norm((Y-y_pred))**2)/(2*len(Y))
  lamb=lam*(np.linalg.norm((W))**2)
  return n+lamb

## Derivadas
### Sin regularización
def derivada(X, W, Y):
  y_pred=h(X,W)
  dw=np.matmul(Y-y_pred,-X)/len(Y)
  return dw

### Regularización L1 - Lasso
def derivadaL1(X, W, Y, lam):
  y_pred=h(X,W)
  dw=np.matmul(Y-y_pred,-X)/len(Y)
  return dw+ lam

### Regularización L2 - Ridge
def derivadaL2(X, W, Y, lam):
  y_pred=h(X,W)
  dw=np.matmul(Y-y_pred,-X)/len(Y)
  return dw+ 2*lam*W

## Actualización de pesos
def update(W,  dW, alpha):
  W=W-alpha*dW
  return W

## Entrenamiento
### Entrenamiento sin regularización
def train(X, Y, epochs, alfa):
    np.random.seed(2001)
    W = np.array([np.random.rand() for i in range(X.shape[1])])
    L = Error(X,W,Y)
    loss = []
    for i in range(epochs):
        dW = derivada(X, W, Y)
        W = update(W, dW, alfa)
        L = Error(X, W,Y)
        loss.append(L)
    return W, loss

### Entrenamiento con regularización L1 - Lasso
def trainL1(X, Y, epochs, alfa,lam):
    np.random.seed(2001)
    W = np.array([np.random.rand() for i in range(X.shape[1])])
    L = ErrorL1(X,W,Y,lam)
    loss = []
    for i in range(epochs):
        dW = derivadaL1(X, W, Y, lam)
        W = update(W, dW, alfa)
        L = ErrorL1(X, W,Y, lam)
        loss.append(L)
    return W, loss

### Entrenamiento con regularización L2 - Ridge
def trainL2(X, Y, epochs, alfa,lam):
    np.random.seed(2001)
    W = np.array([np.random.rand() for i in range(X.shape[1])])
    L = ErrorL2(X,W,Y,lam)
    loss = []
    for i in range(epochs):
        dW = derivadaL2(X, W, Y, lam)
        W = update(W, dW, alfa)
        L = ErrorL2(X, W,Y, lam)
        loss.append(L)
    return W, loss

# Convertir a matriz
def Convertir_a_Matriz(X,grado):
  X_matrix = np.vander(X, grado, increasing=True)
  return X_matrix

## Experimentación
p=10

# Normalizamos los datos de data_train
X_norm = normalizar(x_train)
Y_norm = normalizar(y_train)
X_normL1 = normalizar(x_train)
Y_normL1 = normalizar(y_train)
X_normL2 = normalizar(x_train)
Y_normL2 = normalizar(y_train)

# Se convierte en un matriz
x_matrix = Convertir_a_Matriz(X_norm,p)
x_matrixL1 = Convertir_a_Matriz(X_normL1,p)
x_matrixL2 = Convertir_a_Matriz(X_normL2,p)
epochs = 300000
alfa = 0.9
lam = 0.00001

# Entrenamiento del modelo
### Entrenamiento sin regularización
W, loss = train(x_matrix, Y_norm, epochs, alfa)
### Entrenamiento con regularización L1 - Lasso
WL1, lossL1 = trainL1(x_matrixL1, Y_normL1, epochs, alfa, lam)
### Entrenamiento con regularización L2 - Ridge
WL2, lossL2 = trainL2(x_matrixL2, Y_normL2, epochs, alfa, lam)

# Calcular el error promedio de cada entrenamiento
error_promedio_sin_regularizacion = sum(loss) / len(loss)
error_promedio_L1 = sum(lossL1) / len(lossL1)
error_promedio_L2 = sum(lossL2) / len(lossL2)

print("Error promedio sin regularización:", error_promedio_sin_regularizacion)
print("Error promedio con regularización L1 (Lasso):", error_promedio_L1)
print("Error promedio con regularización L2 (Ridge):", error_promedio_L2)



#ahora usamos los datos de data_test para verificar que tan bien entrenado esta el modelo, pero antes de ello lo normalizamos y luego se convierte en una matriz
y_aprox = h(Convertir_a_Matriz(normalizar(x_test), p), W)
y_aproxL1 = h(Convertir_a_Matriz(normalizar(x_test), p), WL1)
y_aproxL2 = h(Convertir_a_Matriz(normalizar(x_test), p), WL2)
np.savetxt("Resultado.csv", denormalizar(y_train, y_aprox), delimiter=",")
np.savetxt("ResultadoL1.csv", denormalizar(y_train, y_aproxL1), delimiter=",")
np.savetxt("ResultadoL2.csv", denormalizar(y_train, y_aproxL2), delimiter=",")

## Y promedio
yf=denormalizar(y_train, y_aprox)
yfl1=denormalizar(y_train, y_aproxL1)
yfl2=denormalizar(y_train, y_aproxL2)
print("Y promedio sin regularización:", np.sum(yf)/len(yf))
print("Y promedio con regularización L1 (Lasso):",np.sum(yfl1)/len(yfl1))
print("Y promedio con regularización L2 (Ridge):",np.sum(yfl2)/len(yfl2))

## Ploteos
plt.plot(x_test, denormalizar(y_train, y_aprox), "*")
plt.title("Gráfico de Predicciones sin Regularización")
plt.xlabel("Valores de x_test")
plt.ylabel("Valores de y_pred")
plt.show()

plt.plot(x_test, denormalizar(y_train, y_aproxL1), "*")
plt.title("Gráfico de Predicciones con Regularización L1 (Lasso)")
plt.xlabel("Valores de x_test")
plt.ylabel("Valores de y_pred")
plt.show()

plt.plot(x_test, denormalizar(y_train, y_aproxL2), "*")
plt.title("Gráfico de Predicciones con Regularización L2 (Ridge)")
plt.xlabel("Valores de x_test")
plt.ylabel("Valores de y_pred")
plt.show()

# Visualizar las predicciones
plt.plot(x_test, denormalizar(y_train, y_aprox), "*", label="Sin Regularización")
plt.plot(x_test, denormalizar(y_train, y_aproxL1), "*", label="Regularización L1 (Lasso)")
plt.plot(x_test, denormalizar(y_train, y_aproxL2), "*", label="Regularización L2 (Ridge)")
plt.xlabel("Valores de x_test")
plt.ylabel("Valores de y_pred")
plt.legend()
plt.title("Comparación de Predicciones entre Modelos")
plt.show()


