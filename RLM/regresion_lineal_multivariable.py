from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from subprocess import check_output
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

def normalizar(X):
    l_min = X.min()
    l_max = X.max()
    return (X - l_min) / (l_max - l_min)

def denormalizar(X_original, X):
    l_min = X_original.min()
    l_max = X_original.max()
    return X * (l_max - l_min) + l_min

data_train = pd.read_csv("C:\\Users\\bihonda\\Desktop\\UTEC\\C5\\ML\\P1_Data\\train.csv")
data_test=pd.read_csv("C:\\Users\\bihonda\\Desktop\\UTEC\\C5\\ML\\P1_Data\\test.csv")

#convertimos la variable categorica en numerica para train
#Primero para la variable TIPO_ACT_OBRA_ACCINV
c1 = data_train[['TIPO_ACT_OBRA_ACCINV']]
encoder = OneHotEncoder()
columna_codificada = encoder.fit_transform(c1)
columna_codificada_df = pd.DataFrame(columna_codificada.toarray(), columns=encoder.get_feature_names_out(['TIPO_ACT_OBRA_ACCINV']))
df = pd.concat([data_train.drop(columns=['TIPO_ACT_OBRA_ACCINV']), columna_codificada_df], axis=1)
#Luego para la variable TIPO_PROD_PROY
c2 = data_train[['TIPO_PROD_PROY']]
encoder = OneHotEncoder()
columna_codificada = encoder.fit_transform(c2)
columna_codificada_df = pd.DataFrame(columna_codificada.toarray(), columns=encoder.get_feature_names_out(['TIPO_PROD_PROY']))
df = pd.concat([df.drop(columns=['TIPO_PROD_PROY']), columna_codificada_df], axis=1)
#Luego para la variable DEPARTAMENTO
c2 = data_train[['DEPARTAMENTO']]
encoder = OneHotEncoder()
columna_codificada = encoder.fit_transform(c2)
columna_codificada_df = pd.DataFrame(columna_codificada.toarray(), columns=encoder.get_feature_names_out(['DEPARTAMENTO']))
df = pd.concat([df.drop(columns=['DEPARTAMENTO']), columna_codificada_df], axis=1)
#Luego para la variable CATEGORIA_GASTO
c2 = data_train[['CATEGORIA_GASTO']]
encoder = OneHotEncoder()
columna_codificada = encoder.fit_transform(c2)
columna_codificada_df = pd.DataFrame(columna_codificada.toarray(), columns=encoder.get_feature_names_out(['CATEGORIA_GASTO']))
df = pd.concat([df.drop(columns=['CATEGORIA_GASTO']), columna_codificada_df], axis=1)


#convertimos la variable categorica en numerica para test
c3 = data_test[['TIPO_ACT_OBRA_ACCINV']]
encoder = OneHotEncoder()
columna_codificada = encoder.fit_transform(c3)
columna_codificada_df = pd.DataFrame(columna_codificada.toarray(), columns=encoder.get_feature_names_out(['TIPO_ACT_OBRA_ACCINV']))
df_test = pd.concat([data_test.drop(columns=['TIPO_ACT_OBRA_ACCINV']), columna_codificada_df], axis=1)

c4 = data_test[['TIPO_PROD_PROY']]
encoder = OneHotEncoder()
columna_codificada = encoder.fit_transform(c4)
columna_codificada_df = pd.DataFrame(columna_codificada.toarray(), columns=encoder.get_feature_names_out(['TIPO_PROD_PROY']))
df_test = pd.concat([df_test.drop(columns=['TIPO_PROD_PROY']), columna_codificada_df], axis=1)

c4 = data_test[['CATEGORIA_GASTO']]
encoder = OneHotEncoder()
columna_codificada = encoder.fit_transform(c4)
columna_codificada_df = pd.DataFrame(columna_codificada.toarray(), columns=encoder.get_feature_names_out(['CATEGORIA_GASTO']))
df_test = pd.concat([df_test.drop(columns=['CATEGORIA_GASTO']), columna_codificada_df], axis=1)

c4 = data_test[['DEPARTAMENTO']]
encoder = OneHotEncoder()
columna_codificada = encoder.fit_transform(c4)
columna_codificada_df = pd.DataFrame(columna_codificada.toarray(), columns=encoder.get_feature_names_out(['DEPARTAMENTO']))
df_test = pd.concat([df_test.drop(columns=['DEPARTAMENTO']), columna_codificada_df], axis=1)

#variable objetivo
Y=df['MTO_PIA'].to_numpy()
X= df[['TIPO_PROD_PROY_2.PROYECTO', 'TIPO_PROD_PROY_3.PRODUCTO', 'TIPO_ACT_OBRA_ACCINV_5.ACTIVIDAD', 'TIPO_ACT_OBRA_ACCINV_6.ACCION DE INVERSION','CATEGORIA_GASTO_5.GASTOS CORRIENTES','CATEGORIA_GASTO_6.GASTOS DE CAPITAL','META','DEPARTAMENTO_LIMA','DEPARTAMENTO_CAJAMARCA','DEPARTAMENTO_AYACUCHO','DEPARTAMENTO_HUANUCO','DEPARTAMENTO_JUNIN','DEPARTAMENTO_CUSCO','DEPARTAMENTO_UCAYALI','DEPARTAMENTO_SAN MARTIN']].to_numpy()
#x_train, data_test, y_train, y_test = train_test_split(X, Y , random_state=104,  test_size=0.30,    shuffle=True)
X_test=df_test[['TIPO_PROD_PROY_2.PROYECTO', 'TIPO_PROD_PROY_3.PRODUCTO', 'TIPO_ACT_OBRA_ACCINV_5.ACTIVIDAD', 'TIPO_ACT_OBRA_ACCINV_6.ACCION DE INVERSION','CATEGORIA_GASTO_5.GASTOS CORRIENTES','CATEGORIA_GASTO_6.GASTOS DE CAPITAL','META','DEPARTAMENTO_15.LIMA','DEPARTAMENTO_06.CAJAMARCA','DEPARTAMENTO_05.AYACUCHO','DEPARTAMENTO_10.HUANUCO','DEPARTAMENTO_12.JUNIN','DEPARTAMENTO_08.CUSCO','DEPARTAMENTO_25.UCAYALI','DEPARTAMENTO_22.SAN MARTIN']].to_numpy()

#normalizamos las variables
X_norm = normalizar(X)
Y_norm =normalizar(Y)
x_test=normalizar(X_test)

# Regresion Lineal Multivariable
#modelo
def h(x,w,b):
  return np.dot(x,w.transpose()) + b

#loss function
def Error(x,y,w,b):
   y_aprox = h(x,w,b)
   s= np.linalg.norm(y - y_aprox)
   return  s/(2*len(y))

#derivadas
def derivada(x,y,w,b):
  sum1 = 0
  ncols = x.shape[1] #numero de col
  sum2 = np.zeros(ncols) # matriz con el num de col

  for i in range(len(y)):
    sum1 += (y[i] -h(x[i], w, b)) * (-1)
    sum2 += (y[i] -h(x[i], w, b)) * (-x[i])

  db = sum1/len(y)
  dw = sum2/len(y)
  return db,dw

# update
def update(w,b, alpha,db,dw):
  db =   b  - alpha*db
  dw =   w  - alpha*dw
  return db,dw

#entrenamiento
def train(x,y,umbral, alfa):
  # No borrar ni cambiar
  np.random.seed(12)
  #w = np.random.rand()

  w = np.full((1, X_norm.shape[1]), np.random.rand())

  b = np.random.rand()
  L = Error(x,y,w,b)
  mse = 0
  i=0
  while(L > umbral):
    db, dw = derivada(x,y,w,b)
    b,w = update(w,b,alfa,db,dw)
    L = Error(x,y,w,b)
    mse = mse + L
    if(i%100==0):
      print("mse en iteración" + str(i) + " : " + str(L) )
    i=i+1
  return b,w,mse/(i+1)

alfa = 0.008
umbral = 0.033

#Training the model
b,w, avg_mse = train(X_norm, Y_norm, umbral, alfa)
#Show the average mean square error
print("avg_mse :" + str(avg_mse))

y_aprox = h(x_test, w,b)
np.savetxt("Resultado.csv", denormalizar(Y, y_aprox), delimiter=",")
#np.savetxt("Y_original.csv", y_test, delimiter=",")
