# %% [markdown]
# ### PROYECTO FINAL

# %% [markdown]
# En este proyecto se trabajarán con los datos de la compañía Interconnect, que es un operador de telecomunicaciones.
# 
# - Problema a resolver:
# 
# Al operador de telecomunicaciones Interconnect le gustaría poder pronosticar su tasa de cancelación de clientes.
# 
# Si se descubre que un usuario o usuaria planea irse, se le ofrecerán códigos promocionales y opciones de planes especiales.
# 
# El equipo de marketing de Interconnect ha recopilado algunos de los datos personales de sus clientes, incluyendo información sobre sus planes y contratos.

# %%
#Importación de Librerías

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# %%
#Importación de dataframes

try:
    df_contract = pd.read_csv('contract.csv')
    df_internet = pd.read_csv('internet.csv')
    df_personal = pd.read_csv('personal.csv')
    df_phone = pd.read_csv('phone.csv')
except:
    df_contract = pd.read_csv('/datasets/contract.csv')
    df_internet = pd.read_csv('/datasets/internet.csv')
    df_personal = pd.read_csv('/datasets/personal.csv')
    df_phone = pd.read_csv('/datasets/phone.csv')

# %% [markdown]
# ### Descripción de los datos
# 
# Los datos consisten en 4 archivos obtenidos de diferentes fuentes:
# 
# - `contract.csv` — información del contrato;
# - `personal.csv` — datos personales del cliente;
# - `internet.csv` — información sobre los servicios de Internet;
# - `phone.csv` — información sobre los servicios telefónicos.
# 
# En cada archivo, la columna `customerID` (ID de cliente) contiene un código único asignado a cada cliente.

# %% [markdown]
# ### Análisis exploratorio

# %% [markdown]
# ### Dataframe 'contract'

# %%
df_contract.head()

# %%
df_contract.info()

# %% [markdown]
# - En este punto se descubrirá por qué 'TotalCharges' es tipo Object y no númerico.
# - Se revisará si hay valores ausentes.

# %%
valores_unicos = df_contract['TotalCharges'].unique()
valores_unicos_ordenados = np.sort(valores_unicos)
print(valores_unicos_ordenados)

# %%
df_contract[df_contract['TotalCharges'] == ' ']

# %% [markdown]
# - Efectivamente la columna tiene valores ausentes, pero que en realidad el valor es un espacio (' '), por esto mismo, no se estaba considerando como un ausente propiamente tal.
# - Estos valores vacíos son nuevos contratos que llevan menos de un mes, por lo que tiene sentido que no tengan un acumulado de cobros.

# %% [markdown]
# ### Dataframe 'internet'

# %%
df_internet.head()

# %%
df_internet.info()

# %% [markdown]
# - Sobre este dataframe se puede mencionar que tiene menos filas que el de contrato, esto puede deberse a que no todos los usuarios han contratado el servicio de internet.

# %% [markdown]
# ### Dataframe 'personal'

# %%
df_personal.head()

# %%
df_personal.info()

# %% [markdown]
# - Este dataframe muestra características personlas de los clientes, y las filas son las mismas que el df de contract, por lo que no hay valores ausentes.

# %% [markdown]
# ### Dataframe 'phone'

# %%
df_phone.head()

# %%
df_phone.info()

# %% [markdown]
# - Sobre este dataframe se puede mencionar que tiene menos filas que el de contrato, esto puede deberse a que no todos los usuarios han contratado el servicio de teléfono.

# %% [markdown]
# ### Consolidando los dataframes

# %%
df_consolidado = pd.merge(df_contract, df_internet, on='customerID', how='outer')
df_consolidado = pd.merge(df_consolidado, df_personal, on='customerID', how='outer')
df_consolidado = pd.merge(df_consolidado, df_phone, on='customerID', how='outer')

# %%
df_consolidado.info()

# %%
df_consolidado.head()

# %% [markdown]
# ### Corrección de formatos

# %% [markdown]
# - En el siguiente paso se convertirá la columna 'SeniorCitizen' al tiipo int8, que ahora es del tipo int64, esto con el fin de que al procesar los modelos, estos no tarden tanto tiempo, ni consuman tanta memoria.

# %%
df_consolidado['SeniorCitizen'] = df_consolidado['SeniorCitizen'].astype('int8')

# %% [markdown]
# - En el siguiente paso se convertirál algunos tipos object a category, esto con el fin de mejorar la eficiencia en el procesamiento.

# %%
df_consolidado = df_consolidado.astype({'Type': 'category', 'PaperlessBilling': 'category', 'PaymentMethod': 'category','InternetService': 'category','OnlineSecurity': 'category','OnlineBackup': 'category',
                                        'DeviceProtection': 'category', 'TechSupport': 'category', 'StreamingTV': 'category', 'StreamingMovies': 'category', 'gender': 'category',
                                        'Partner': 'category', 'Dependents': 'category', 'MultipleLines': 'category'})

# %%
df_consolidado.info()

# %% [markdown]
# - En este paso se convertirán los valores de la columna TotalCharge a float64, con el fin de poder comparalo con el valor del MonthlyCharges.

# %%
df_consolidado['TotalCharges'] = pd.to_numeric(df_consolidado['TotalCharges'], errors='coerce').astype('float64')

# %% [markdown]
# - En este paso se convertirán las columnas 'BeginDate' y  'EndDate' a un formato 'datetime', con el fin de poder hacer cálculos con esos valores.

# %%
df_consolidado['BeginDate'] = pd.to_datetime(df_consolidado['BeginDate'], format='%Y-%m-%d')
df_consolidado['EndDate'] = pd.to_datetime(df_consolidado['EndDate'], format='%Y-%m-%d %H:%M:%S', errors='coerce')


# %%
df_consolidado['EndDate'].unique()

# %%
df_consolidado.head()

# %% [markdown]
# - Actualizando el formato de los encabezados a snake_case

# %%
df_consolidado = df_consolidado.rename(columns={'customerID': 'customer_id', 'BeginDate': 'begin_date', 'EndDate': 'end_date', 'Type': 'type','PaperlessBilling': 'paperless_billing', 'PaymentMethod': 'payment_method',
                                                'MonthlyCharges': 'monthly_charges', 'TotalCharges': 'total_charges', 'InternetService': 'internet_service', 'OnlineSecurity': 'online_security', 'OnlineBackup': 'online_backup',
                                                'DeviceProtection': 'device_protection', 'TechSupport': 'tech_support', 'StreamingTV': 'streaming_tv', 'StreamingMovies': 'streaming_movies', 'gender': 'gender',
                                                'SeniorCitizen': 'senior_citizen', 'Partner': 'partner', 'Dependents': 'dependents', 'MultipleLines': 'multiple_lines'})

# %%
df_consolidado.info()

# %% [markdown]
# ### Valores Ausentes.
# Los valores ausentes serán rellenados con No, porque se asumirá que los clientes no cuentan con esos servicios.

# %%
df_consolidado[df_consolidado['internet_service'].isnull()].head()

# %%
columnas = ['internet_service', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'multiple_lines']
valor = 'No'

# Agrega la categoría 'No' a las columnas categóricas solo si no existe
for columna in columnas:
    if valor not in df_consolidado[columna].cat.categories:
        df_consolidado[columna] = df_consolidado[columna].cat.add_categories(valor)

# Llena los valores nulos con 'No'
df_consolidado[columnas] = df_consolidado[columnas].fillna(valor)


# %% [markdown]
# - Acá se rellenarán los valores ausente de 'total_charges' con ceros, recordemos que estos valores ausentes corresponden a usuarios que llevan menos de un mes en la compañía.

# %%
df_consolidado['total_charges'] = df_consolidado['total_charges'].fillna(0)

# %%
df_consolidado.info()

# %% [markdown]
# - En este paso se rellenarán los ausentes de 'end_date' con 'No'.

# %%
df_consolidado['end_date'] = df_consolidado['end_date'].fillna('No')

# %%
df_consolidado['end_date'].value_counts()

# %%
df_consolidado.info()

# %% [markdown]
# ### Funciones para 'end_date'

# %% [markdown]
# - En este paso, agregaremos una columna con valores booleanos, en donde 0 representará los clientes que siguen en la compañía, y 1 a los clientes que dejaron la compañía.

# %%
def end_date_2(row):
    if row['end_date'] == 'No':
        return 0
    else:
        return 1
    
df_consolidado['end_date_2'] = df_consolidado.apply(end_date_2, axis=1)

# %%
df_consolidado['end_date_2'].unique()

# %% [markdown]
# - A continuación se calcularán los meses de uso desde el inicio del contrato. Se utilizará la fecha 31-01-2020 como la fecha de hoy.

# %%
def end_date_3(row):
    if row['end_date'] == 'No':
        return pd.to_datetime('2020-01-31',format='%Y-%m-%d')
    else:
        return row['end_date']
    
df_consolidado['end_date_3'] = df_consolidado.apply(end_date_3, axis=1)

# %%
df_consolidado['end_date_3'].value_counts()

# %%
def months(row):
    months_diff = row['end_date_3'] - row['begin_date']
    return round(months_diff / pd.Timedelta(days=30))

df_consolidado['months_using_service'] = df_consolidado.apply(months, axis=1)

# %% [markdown]
# - Ahora, se calculará los cobros adicionales al cobro mensual, se hará calculand la diferencia entre 'total_charges' y 'monthly_charges' multiplicado por 'months_using_service', que serían los meses que el usuario ha usado el servicio.

# %%
def ttl_charges(row):
    total_charges_diff = (row['monthly_charges'] * row['months_using_service']) - row['total_charges']
    return total_charges_diff

df_consolidado['total_charges_diff'] = df_consolidado.apply(ttl_charges, axis=1)

# %%
df_consolidado.head()

# %% [markdown]
# ### Análisis a través de Gráficos

# %% [markdown]
# - Se iniciará el análisis de los datos a través de los gráficos, primero se analizará el dataframe 'contract'.

# %%
df_consolidado.groupby('end_date_2')['total_charges'].sum().plot(kind='bar')
plt.xlabel('End Date')
plt.ylabel('Total Charges')
plt.title('Total Charges by End Date')
plt.show()

# %%
df_consolidado['customer_id'].nunique()

# %%
df_consolidado['total_charges'].sum()

# %%
(df_consolidado.groupby('end_date_2')['total_charges'].sum() / df_consolidado['total_charges'].sum()) * 100

# %% [markdown]
# - Sobre los ingresos totales, se puede decir que de los 7043 clientes que se manejan en el dataframe, los que aún siguen en la compañía, aportan el 82% de la venta, siendo esta de un total de $16M. Sin embargo, la empresa quiere predecir qué clientes pueden dejar de contratar sus servicios, con el fin de disminuir el 18% de clientes perdidos.

# %% [markdown]
# ### Analizando los métodos de pago

# %%
df_consolidado.groupby('type', observed=False)['total_charges'].sum().plot(kind='bar')
plt.xlabel('Type')
plt.ylabel('Total Charges')
plt.title('Total Charges by Type')
plt.show()

# %%
(df_consolidado.groupby('type', observed=False)['total_charges'].sum() / df_consolidado['total_charges'].sum()) * 100

# %%
df_consolidado.groupby('paperless_billing', observed=False)['total_charges'].sum().plot(kind='bar')
plt.xlabel('Paperless billing')
plt.ylabel('Total Charges')
plt.title('Total Charges by Paperless billing')
plt.show()

# %%
(df_consolidado.groupby('paperless_billing', observed=False)['total_charges'].sum() / df_consolidado['total_charges'].sum()) * 100

# %%
df_consolidado.groupby('payment_method', observed=False)['total_charges'].sum().plot(kind='bar')
plt.xlabel('Payment method')
plt.ylabel('Total Charges')
plt.title('Total Charges by Payment method')
plt.show()

# %%
(df_consolidado.groupby('payment_method', observed=False)['total_charges'].sum() / df_consolidado['total_charges'].sum()) * 100

# %% [markdown]
# - Con respecto a los tipos de pago no se ven grandes variaciones entre ellas, siendo la que mayormente los clientes prefieren, el tipo de pago 'Two year' con un 39% sobre los ingresos totales, seguido del tipo de pago 'Month-to-month' con un 33%, y finalmente el tipo de pago 'One year' con un 28%.
# - Con respecto a la facturación electrónica, se puede ver claramente que el 67% está bajo la modalidad de facturación electrónica, y el 33% no lo está.
# - Con respecto a los métodos de pagos, se ve una tendencia similar entre 3 de los 4 métodos, por un lado tenemos a los métodos de pago de 'Electronic check', 'Bank transfer (automatic)' y 'Credit card (automatic)' con 31%, 30% y 29% respectivamente. Por último el método de pago 'Mailed check' con un 10% de uso. Esto nos dice que los clientes de la compañía tienden a usar métodos digitales de pago.

# %% [markdown]
# ### Diferencias entre el pago total y el pago mensual

# %%
df_consolidado.groupby('end_date_2', observed=False)['total_charges_diff'].mean().plot(kind='bar')
plt.xlabel('End date')
plt.ylabel('Total Charges difference')
plt.title('Total Charges difference by End date')
plt.show()

# %%
df_consolidado.groupby('end_date_2', observed=False)['total_charges_diff'].mean()

# %% [markdown]
# - En el gráfico anterior se da a conocer los cobros adicionales al pago mensual que tiene los clientes de Interconnect.
# - Los usuarios que siguen en la compañía, pagan en promedio $37 más sobre el pago mensual desde que contraron el servicio.
# - Los usuarios que dejaron de contratar los servicios pagaron $18 más sobres el pago mensual dentro de todo el período que usaron el servicio.

# %% [markdown]
# ### Analizando los servicios de internet

# %%
df_consolidado.groupby('internet_service', observed=False)['total_charges'].sum().plot(kind='bar')
plt.xlabel('Internet service')
plt.ylabel('Total Charges')
plt.title('Total Charges by Internet service')
plt.show()

# %%
(df_consolidado.groupby('internet_service', observed=False)['total_charges'].sum() / df_consolidado['total_charges'].sum()) * 100

# %% [markdown]
# - Con respecto a tipo de internet que los usuarios prefieren, se ve una alta preferencia por la Fibra óptica, con un 62% del total de ingresos, luego el tipo de internet por DSL con un 32%, y finalmente los que no contrataron el servicio de internet representan el 6%.

# %%
print(df_consolidado['online_security'].value_counts())
print()
print(df_consolidado['online_backup'].value_counts())
print()
print(df_consolidado['device_protection'].value_counts())
print()
print(df_consolidado['tech_support'].value_counts())
print()
print(df_consolidado['streaming_tv'].value_counts())
print()
print(df_consolidado['streaming_movies'].value_counts())

# %% [markdown]
# - Por otro lado, del total de usarios del dataframe, más de 4000 usuarios, no usan los servicios extras de internet.

# %% [markdown]
# ### Analizando el dataframe Personal

# %%
df_consolidado.groupby('gender', observed=False)['total_charges'].sum().plot(kind='bar')
plt.xlabel('Gender')
plt.ylabel('Total Charges')
plt.title('Total Charges by Gender')
plt.show()

# %%
(df_consolidado.groupby('gender', observed=False)['total_charges'].sum() / df_consolidado['total_charges'].sum()) * 100

# %%
(df_consolidado['gender'].value_counts() / df_consolidado['gender'].count()) * 100

# %% [markdown]
# - En términos del género del contratane del servicio, el gráfico demuestra que entre el genero feminino y masculino hay 0.5% de diferencia sobre el total de cargos, lo que nos dice que el género de los usuarios del dataframe son practicamente iguales.

# %%
df_consolidado.groupby('senior_citizen', observed=False)['total_charges'].sum().plot(kind='bar')
plt.xlabel('Senior citizen')
plt.ylabel('Total Charges')
plt.title('Total Charges by Senior citizen')
plt.show()

# %%
(df_consolidado['senior_citizen'].value_counts() / df_consolidado['senior_citizen'].count()) * 100

# %% [markdown]
# - Sobre los 'senior_citizen' se puede decir que forman parte del 16% del total del dataframe.

# %%
df_consolidado.groupby('partner', observed=False)['total_charges'].sum().plot(kind='bar')
plt.xlabel('Partner')
plt.ylabel('Total Charges')
plt.title('Total Charges by Partner')
plt.show()

# %%
(df_consolidado['partner'].value_counts() / df_consolidado['partner'].count()) * 100

# %% [markdown]
# - Sobre los 'partner' se puede decir que los que no tienen partner forman el 52% del dataframe, contra el 48% que sí tienen partner.

# %%
df_consolidado.groupby('dependents', observed=False)['total_charges'].sum().plot(kind='bar')
plt.xlabel('Dependents')
plt.ylabel('Total Charges')
plt.title('Total Charges by Dependents')
plt.show()

# %%
(df_consolidado['dependents'].value_counts() / df_consolidado['dependents'].count()) * 100

# %% [markdown]
# - Sobre los 'dependents' se puede decir que los que no tienen dependents forman el 70% del dataframe, contra el 30% que sí.

# %% [markdown]
# ### Analizando el dataframe Phone

# %%
df_consolidado.groupby('multiple_lines', observed=False)['total_charges'].sum().plot(kind='bar')
plt.xlabel('Multiple lines')
plt.ylabel('Total Charges')
plt.title('Total Charges by Multiple lines')
plt.show()

# %%
(df_consolidado['multiple_lines'].value_counts() / df_consolidado['multiple_lines'].count()) * 100

# %% [markdown]
# - Sobre los 'multiple_lines' se puede decir que los que tienen multiple lines forman el 42% del dataframe, contra el 58% que no.

# %% [markdown]
# ### Analizando la estadía de los usuarios que dejaron la compañía

# %%
df_consolidado[df_consolidado['end_date_2'] == 1]['months_using_service'].hist(bins=25, figsize=(20, 6))
plt.xlim(0, 75)
plt.show()

# %%
print('Meses promedio de uso de usuarios que dejaron la compañía:', round(df_consolidado[df_consolidado['end_date_2'] == 1]['months_using_service'].mean()))
print('Mínimo de meses de uso de usuarios que dejaron la compañía:',df_consolidado[df_consolidado['end_date_2'] == 1]['months_using_service'].min())
print('Máximo de meses de uso de usuarios que dejaron la compañía:',df_consolidado[df_consolidado['end_date_2'] == 1]['months_using_service'].max())
print('Total de usuarios que dejaron la compañía:',df_consolidado[df_consolidado['end_date_2'] == 1]['customer_id'].count())

# %%
df_consolidado[df_consolidado['end_date_2'] == 1]['months_using_service'].value_counts() 

# %%
(df_consolidado[df_consolidado['end_date_2'] == 1]['months_using_service'].value_counts()  / df_consolidado[df_consolidado['end_date_2'] == 1]['customer_id'].count()) * 100

# %%
df_consolidado[df_consolidado['end_date_2'] == 1]['total_charges_diff'].mean() 

# %% [markdown]
# - Como se muestra en el gráfico, la mayoría de los usuarios que dejaron la compañía, permanecieron menos de 10 meses.
# - Un 38% estuvo menos de 5 meses contratando los servicios de la empresa.
# - En promedios los usuarios que dejaron la compañía, estuvieron 18 meses.
# - Los usuarios que dejaron la compañía pagaron $17 de más sobre su pago mensual.

# %% [markdown]
# ### Meses de uso del servicio

# %%
df_consolidado['months_using_service'].hist(bins=25, figsize=(20, 6))
plt.xlim(0, 75)
plt.show()

# %%
df_consolidado['months_using_service'].mean()

# %% [markdown]
# - En el gráfico se puede apreciar que tenemos una gran concentración en usuarios nuevos, los que llevan menos de 12 meses en la compañía, y por otro lado otra gran concentración en usuarios que llevan más de 70 meses en la empresa, ¡esos son más de 5 años!

# %% [markdown]
# ### Analizando usuarios activos y desactivos

# %%
_= df_consolidado.pivot_table(index='type', columns='end_date_2', values='total_charges', aggfunc='mean').plot(kind='bar')

# %% [markdown]
# - Los usuarios que dejaron la compañía tenían preferencia por el tipo de pago a 2 años.

# %%
_= df_consolidado.pivot_table(index='payment_method', columns='end_date_2', values='total_charges', aggfunc='mean').plot(kind='bar')

# %% [markdown]
# - La tendencia del método de pago es similiar para los usuarios activos y desactivos, ambos prefieron los métodos de pago digitales.

# %%
_= df_consolidado.boxplot(column='months_using_service', by='end_date_2', figsize=(15,6))

# %% [markdown]
# - Sobre los usuarios que dejaron la compañía tenemos dos grupos marcaddos:
# - Usuarios con menos de 10 meses desde que contrataron los servicios de la compañía.
# - Usuarios antigüos con más de 70 meses en la compañía, siendo estos además valores atípicos.

# %% [markdown]
# ### Selección de características para los modelos

# %% [markdown]
# - A continuación se definirá que características se usarán para los modelos:

# %% [markdown]
# | Variable         | Decisión |
# |--------------|--------------|
# | customer_id    | No se usará en los modelos porque no influye en el objetivo, dado que es el identificador el usuario.  |
# |  begin_date    | No se usará en los modelos porque no influye en el objetivo, dado que es una fecha. |
# | end_date  | No se usará en los modelos porque no influye en el objetivo, dado que es una fecha.   |
# |  type| Se usará en los modelos porque influye en el objetivo, dado que dependiendo del tipo, el usuario puede dar de baja el servicio desde un mes hasta dos años.  |
# | paperless_billing | No se usará en los modelos porque no influye en el objetivo, dado que es una preferencia de cada usuario. |
# | payment_method| No se usará en los modelos porque no influye en el objetivo, dado que es una preferencia de cada usuario. |
# | monthly_charges | Se usará en los modelos porque influye en el objetivo, dado que es el monto del pago mensual del servicio. |
# |  total_charges | Se usará en los modelos porque influye en el objetivo, dado que es el monto del pago total del servicio. |
# | internet_service| Se usará en los modelos porque influye en el objetivo, dado que identifica el tipo de internet que el usuario elige. |
# | online_security | No se usará en los modelos porque no influye en el objetivo, dado que es un servicio adicional opcional. |
# | online_backup| No se usará en los modelos porque no influye en el objetivo, dado que es un servicio adicional opcional. |
# | device_protection |No se usará en los modelos porque no influye en el objetivo, dado que es un servicio adicional opcional. |
# | tech_support | No se usará en los modelos porque no influye en el objetivo, dado que es un servicio adicional opcional. |
# | streaming_tv | No se usará en los modelos porque no influye en el objetivo, dado que es un servicio adicional opcional. |
# | streaming_movies | No se usará en los modelos porque no influye en el objetivo, dado que es un servicio adicional opcional. |
# | gender | Se usará en los modelos porque influye en el objetivo, dado que identifica el género de cada usuario. |
# | senior_citizen |Se usará en los modelos porque influye en el objetivo, dado que identifica si el usuario es senior. |
# | partner | No se usará en los modelos porque no influye en el objetivo, dado que es un dato personal de cada usuario. |
# | dependents | No se usará en los modelos porque no influye en el objetivo, dado que es un dato personal de cada usuario. |
# | multiple_lines | No se usará en los modelos porque no influye en el objetivo, dado que es un servicio adicional opcional. |
# | end_date_2 | Objetivo. |
# | end_date_3 | No se usará en los modelos porque no influye en el objetivo, dado que es una fecha. |
# | months_using_service | Se usará en los modelos porque influye en el objetivo, dado que identifica los meses que el usuario ha usado el servicio. |
# | total_charges_diff | Se usará en los modelos porque influye en el objetivo, dado que identifica los cobros extras por sobre el monto mensual. |
# 

# %% [markdown]
# - Ahora, se dejarán fuera del dataframe las variables que no se usarán en los modelos.

# %%
df_consolidado = df_consolidado.drop(['begin_date', 'end_date', 'paperless_billing', 'payment_method', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
                                      'streaming_movies', 'partner', 'dependents', 'multiple_lines', 'end_date_3'], axis=1)

# %%
df_consolidado.head()

# %% [markdown]
# - En el siguiente paso se utilizará One Hot Encoder para los valores de clasificación.

# %%
df_consolidado_ohe = pd.get_dummies(df_consolidado[['type', 'internet_service', 'gender']], drop_first=True)

# %%
df_consolidado_ohe.head()

# %% [markdown]
# - A continuación se une el dataframe al cual se aplicó el enconder con el resto del dataframe.

# %%
df_consolidado = pd.merge(df_consolidado, df_consolidado_ohe, left_index=True, right_index=True, how='outer')

# %%
df_consolidado.head()

# %% [markdown]
# - A continuación se quitan del dataframe los valores que fueron pasados por el enconder, además del customer id, dado que ya no vamos a unir dataframes.

# %%
df_consolidado = df_consolidado.drop(['customer_id', 'type', 'internet_service', 'gender'], axis=1)

# %%
df_consolidado.head()

# %% [markdown]
# - Ahora se separará el dataframe en features, target y df de entrenamiento y validación.

# %%
features = df_consolidado.drop(['end_date_2'], axis=1)
target  = df_consolidado['end_date_2']

features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)

# %% [markdown]
# - En el siguiente paso se hará un escalado a las variables no booleanas.

# %%
numeric = ['monthly_charges', 'total_charges', 'months_using_service', 'total_charges_diff']

scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])

features_train_scaled = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])

# %% [markdown]
# # Entrenado los modelos

# %% [markdown]
# ### Árbol de decisiones

# %% [markdown]
# -  Con el siguiente bucle for se pretende encontrar los mejores hiperparámetros para el modelo.

# %%
for d in range(1, 20):
    model = DecisionTreeClassifier(random_state=1234, max_depth = d)
    model.fit(features_train, target_train)
    predictions_dtc = model.predict(features_valid)
    print('roc_auc_score:', roc_auc_score(target_valid, predictions_dtc), 'Max depth:', d)

# %%
model_dtc = DecisionTreeClassifier(random_state=1234, max_depth=6)
model_dtc.fit(features_train, target_train)

predictions_dtc = model_dtc.predict(features_valid)

auc_roc_dtc = roc_auc_score(target_valid, predictions_dtc)
accuracy = accuracy_score(target_valid, predictions_dtc)

print("AUC-ROC Score:", auc_roc_dtc)

# %% [markdown]
# - Con el mejor max_depth (6), el auc-roc es de 0.69, valor que se aleja del exigido por la compañía. El actual score nos dice que este modelo tiene una capacidad de discriminación pobre y es dificilmente útil para la clasifiación.

# %% [markdown]
# ### Aplicando Sobre y Submuestreo

# %%
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=1234)

    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(features_train, target_train, 3)

for d in range(1, 20):
    model = DecisionTreeClassifier(random_state=1234, max_depth = d)
    model.fit(features_upsampled, target_upsampled)
    predictions_dtc_upsampled = model.predict(features_valid)
    print('roc_auc_score:', roc_auc_score(target_valid, predictions_dtc_upsampled), 'Max depth:', d)

# %%
def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat([features_zeros.sample(frac=fraction, random_state=1234)] + [features_ones])
    target_downsampled = pd.concat([target_zeros.sample(frac=fraction, random_state=1234)] + [target_ones])

    features_downsampled, target_downsampled = shuffle(features_downsampled, target_downsampled, random_state=1234)

    return features_downsampled, target_downsampled

features_downsampled, target_downsampled = downsample(features_train, target_train, 0.3)

for d in range(1, 20):
    model = DecisionTreeClassifier(random_state=1234, max_depth = d)
    model.fit(features_downsampled, target_downsampled)
    predictions_dtc_downsampled = model.predict(features_valid)
    print('roc_auc_score:', roc_auc_score(target_valid, predictions_dtc_downsampled), 'Max depth:', d)

# %%
model_dtc = DecisionTreeClassifier(random_state=1234, max_depth=4)
model_dtc.fit(features_downsampled, target_downsampled)

predictions_dtc_downsampled = model_dtc.predict(features_valid)

auc_roc_dtc_downsampled = roc_auc_score(target_valid, predictions_dtc_downsampled)

print("AUC-ROC Score:", auc_roc_dtc_downsampled)

# %% [markdown]
# - El Submuestreo tiene un mayor valor auc-roc.
# - El score del auc-roc con el submuestreo es de 0.76 el cual es mayor al score original de 0.69.
# - Aún así, el score sigue siendo  bajo para lo requerido por la compañía.

# %% [markdown]
# ### Random Forest Regressor

# %% [markdown]
# -  Con el siguiente Grid Search se pretende encontrar los mejores hiperparámetros para el modelo.

# %%
model_rfr = RandomForestRegressor(random_state=1234)

param_grid = {
    'n_estimators': [5, 10, 15],
    'max_depth': [2, 4, 8]
    #'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model_rfr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(features_train, target_train)
best_params = grid_search.best_params_
print("Mejores Hiperparámetros:", best_params)
best_model = grid_search.best_estimator_

# %%
%%time
model_rfr = RandomForestRegressor(random_state=1234, max_depth=4, n_estimators=15)
model_rfr.fit(features_train, target_train)
predictions_rfr = model_rfr.predict(features_valid)

auc_roc_rfr = roc_auc_score(target_valid, predictions_rfr)

print("AUC-ROC Score:", auc_roc_rfr)

# %% [markdown]
# - Con este modelo obtenemos un score de 0.83, siendo este un valor superior a los modelos anteriores.
# - El valor del score nos dice que el modelo tiene una capacidad de discriminación aceptable y puede ser útil para la clasificación en algunos casos.

# %% [markdown]
# ### Aplicando Sobre y Submuestreo

# %%
model_rfr = RandomForestRegressor(random_state=1234)

param_grid = {
    'n_estimators': [5, 10, 55],
    'max_depth': [2, 4, 40]
    #'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model_rfr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(features_upsampled, target_upsampled)
best_params = grid_search.best_params_
print("Mejores Hiperparámetros:", best_params)
best_model = grid_search.best_estimator_

# %%
model_rfr = RandomForestRegressor(random_state=1234, max_depth=40, n_estimators=55)
model_rfr.fit(features_upsampled, target_upsampled)
predictions_rfr = model_rfr.predict(features_valid)

auc_roc = roc_auc_score(target_valid, predictions_rfr)

print("AUC-ROC Score:", auc_roc)

# %%
model_rfr = RandomForestRegressor(random_state=1234)

param_grid = {
    'n_estimators': [5, 10, 55],
    'max_depth': [2, 4, 40]
    #'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model_rfr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(features_downsampled, target_downsampled)
best_params = grid_search.best_params_
print("Mejores Hiperparámetros:", best_params)
best_model = grid_search.best_estimator_

# %%
model_rfr = RandomForestRegressor(random_state=1234, max_depth=4, n_estimators=55)
model_rfr.fit(features_downsampled, target_downsampled)
predictions_rfr = model_rfr.predict(features_valid)

auc_roc = roc_auc_score(target_valid, predictions_rfr)

print("AUC-ROC Score:", auc_roc)

# %% [markdown]
# - Los sobre y submuestreos en este modelo no generan mejores resultados, para este modelo se recomienda usar la versión original.

# %% [markdown]
# ### Regresión Lineal

# %% [markdown]
# -  Con el siguiente Grid Search se pretende encontrar los mejores hiperparámetros para el modelo.

# %%
model_lr = LinearRegression()

param_grid = {
    'fit_intercept': [True, False]
}

grid_search = GridSearchCV(estimator=model_lr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(features_train, target_train)
best_params = grid_search.best_params_
print("Mejores Hiperparámetros:", best_params)
best_model = grid_search.best_estimator_

# %%
%%time
model_lr = LinearRegression(fit_intercept=True)
model_lr.fit(features_train, target_train)
predictions_lr = model_lr.predict(features_valid)

auc_roc_lr = roc_auc_score(target_valid, predictions_lr)

print("AUC-ROC Score:", auc_roc_lr)

# %% [markdown]
# - Con este modelo obtenemos un score de 0.83, siendo este un valor superior a los modelos anteriores y similar al RandomForestRegressor.
# - El valor del score nos dice que el modelo tiene una capacidad de discriminación aceptable y puede ser útil para la clasificación en algunos casos.
# - Si bien es un score similar al de RandomForestRegressor, la velocidad de procesamiento de este modelo, es superior.

# %% [markdown]
# ### Aplicando sobre y submuestreo

# %%
model_lr = LinearRegression(fit_intercept=True)
model_lr.fit(features_upsampled, target_upsampled)
predictions_lr = model_lr.predict(features_valid)

auc_roc = roc_auc_score(target_valid, predictions_lr)

print("AUC-ROC Score:", auc_roc)

# %%
model_lr = LinearRegression(fit_intercept=True)
model_lr.fit(features_downsampled, target_downsampled)
predictions_lr = model_lr.predict(features_valid)

auc_roc = roc_auc_score(target_valid, predictions_lr)

print("AUC-ROC Score:", auc_roc)

# %% [markdown]
# - Con el sobre y submuestreo la diferencia en de décimas, por lo que se recomiendo usar la versión original del modelo.

# %% [markdown]
# ### Light GBM

# %%
%%time
# Configuración básica del modelo
params = {
    'objective': 'regression',  # Puedes cambiar a 'binary' para clasificación binaria, etc.
    'metric': 'mse',  # MSE (Error Cuadrático Medio) como métrica de evaluación
    'boosting_type': 'gbdt',  # Puedes probar 'dart' o 'goss' también
    'early_stopping_rounds': 10,  # Número de rondas para esperar antes de detener el entrenamiento si no mejora la métrica
}

# Crear conjunto de datos de LightGBM
train_data = lgb.Dataset(features_train, label=target_train)
valid_data = lgb.Dataset(features_valid, label=target_valid, reference=train_data)

# Entrenar el modelo
num_round = 100  # Número de rondas de entrenamiento
model = lgb.train(params, train_data, num_round, valid_sets=[valid_data], valid_names=['test'])

# %%
predictions_lgb = model.predict(features_valid, num_iteration=model.best_iteration)
auc_roc_lgbm = roc_auc_score(target_valid, predictions_lgb)
print("AUC-ROC Score:", auc_roc_lgbm)

# %% [markdown]
# - Con este modelo obtenemos un score de 0.83, siendo este un valor similar a los últimos modelos.
# - El valor del score nos dice que el modelo tiene una capacidad de discriminación aceptable y puede ser útil para la clasificación en algunos casos.
# - Si bien es un score similar a los últimos modelos, la velocidad de procesamiento de este modelo, es mayor.

# %% [markdown]
# ### Aplicando sobre y submuestreo

# %%
# Configuración básica del modelo
params = {
    'objective': 'regression',  # Puedes cambiar a 'binary' para clasificación binaria, etc.
    'metric': 'mse',  # MSE (Error Cuadrático Medio) como métrica de evaluación
    'boosting_type': 'gbdt',  # Puedes probar 'dart' o 'goss' también
    'early_stopping_rounds': 10,  # Número de rondas para esperar antes de detener el entrenamiento si no mejora la métrica
}

# Crear conjunto de datos de LightGBM
train_data = lgb.Dataset(features_downsampled, label=target_downsampled)
valid_data = lgb.Dataset(features_valid, label=target_valid, reference=train_data)

# Entrenar el modelo
num_round = 100  # Número de rondas de entrenamiento
model = lgb.train(params, train_data, num_round, valid_sets=[valid_data], valid_names=['test'])

# %%
predictions_lgb = model.predict(features_valid, num_iteration=model.best_iteration)
auc_roc = roc_auc_score(target_valid, predictions_lgb)
print("AUC-ROC Score:", auc_roc)

# %%
# Configuración básica del modelo
params = {
    'objective': 'regression',  # Puedes cambiar a 'binary' para clasificación binaria, etc.
    'metric': 'mse',  # MSE (Error Cuadrático Medio) como métrica de evaluación
    'boosting_type': 'gbdt',  # Puedes probar 'dart' o 'goss' también
    'early_stopping_rounds': 10,  # Número de rondas para esperar antes de detener el entrenamiento si no mejora la métrica
}

# Crear conjunto de datos de LightGBM
train_data = lgb.Dataset(features_upsampled, label=target_upsampled)
valid_data = lgb.Dataset(features_valid, label=target_valid, reference=train_data)

# Entrenar el modelo
num_round = 100  # Número de rondas de entrenamiento
model = lgb.train(params, train_data, num_round, valid_sets=[valid_data], valid_names=['test'])

# %%
predictions_lgb = model.predict(features_valid, num_iteration=model.best_iteration)
auc_roc = roc_auc_score(target_valid, predictions_lgb)
print("AUC-ROC Score:", auc_roc)

# %% [markdown]
# - Los sobre y submuestreos en este modelo no generan mejores resultados, para este modelo se recomienda usar la versión original.

# %% [markdown]
# ### Regresión Logística

# %% [markdown]
# -  Con el siguiente Grid Search se pretende encontrar los mejores hiperparámetros para el modelo.

# %%
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularización
    'penalty': ['l1', 'l2'],  # Tipo de regularización
    'solver': ['liblinear', 'saga']  # Algoritmo de optimización
}

# Crear el objeto GridSearchCV
grid_search = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=param_grid,
    cv=5,  # Número de validaciones cruzadas
    scoring='accuracy',  # Métrica de evaluación
    verbose=1,  # Muestra información detallada
    n_jobs=-1  # Utiliza todos los núcleos de CPU disponibles
)

# Entrenar el objeto GridSearchCV
grid_search.fit(features_train, target_train)

# Obtener los mejores parámetros y el mejor modelo
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Mejores parámetros:", best_params)
print("Mejor modelo:", best_model)

# %%
#model_log_r = LogisticRegression(random_state=1234, C=10, penalty='l2', solver='liblinear')
model_log_r = LogisticRegression(random_state=1234, C=10, penalty='l2', class_weight='balanced', solver='liblinear')
#model_log_r = LogisticRegression(random_state=1234, class_weight='balanced', solver='liblinear'
model_log_r.fit(features_train, target_train)
predictions_log_r = model_log_r.predict(features_valid)
auc_roc = roc_auc_score(target_valid, predictions_log_r)
print("AUC-ROC Score:", auc_roc)

# %% [markdown]
# - Se usan los mejores hiperparámetros para este modelo: C=10, penalty='l2', class_weight='balanced', solver='liblinear'.
# - El auc-roc para el modelo es de 0.69, valor que se aleja del exigido por la compañía.
# - El actual score nos dice que este modelo tiene una capacidad de discriminación pobre y es dificilmente útil para la clasifiación.

# %% [markdown]
# ### Aplicando sobre y submuestreo

# %%
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularización
    'penalty': ['l1', 'l2'],  # Tipo de regularización
    'solver': ['liblinear', 'saga']  # Algoritmo de optimización
}

# Crear el objeto GridSearchCV
grid_search = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=param_grid,
    cv=5,  # Número de validaciones cruzadas
    scoring='accuracy',  # Métrica de evaluación
    verbose=1,  # Muestra información detallada
    n_jobs=-1  # Utiliza todos los núcleos de CPU disponibles
)

# Entrenar el objeto GridSearchCV
grid_search.fit(features_upsampled, target_upsampled)

# Obtener los mejores parámetros y el mejor modelo
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Mejores parámetros:", best_params)
print("Mejor modelo:", best_model)

# %%
#model_log_r = LogisticRegression(random_state=1234, C=10, penalty='l2', solver='liblinear')
model_log_r = LogisticRegression(random_state=1234, C=1, penalty='l1', class_weight='balanced', solver='saga')
#model_log_r = LogisticRegression(random_state=1234, class_weight='balanced', solver='liblinear'
model_log_r.fit(features_upsampled, target_upsampled)
predictions_log_r = model_log_r.predict(features_valid)
auc_roc = roc_auc_score(target_valid, predictions_log_r)
print("AUC-ROC Score:", auc_roc)

# %%
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularización
    'penalty': ['l1', 'l2'],  # Tipo de regularización
    'solver': ['liblinear', 'saga']  # Algoritmo de optimización
}

# Crear el objeto GridSearchCV
grid_search = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=param_grid,
    cv=5,  # Número de validaciones cruzadas
    scoring='accuracy',  # Métrica de evaluación
    verbose=1,  # Muestra información detallada
    n_jobs=-1  # Utiliza todos los núcleos de CPU disponibles
)

# Entrenar el objeto GridSearchCV
grid_search.fit(features_downsampled, target_downsampled)

# Obtener los mejores parámetros y el mejor modelo
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Mejores parámetros:", best_params)
print("Mejor modelo:", best_model)

# %%
#model_log_r = LogisticRegression(random_state=1234, C=10, penalty='l2', solver='liblinear')
model_log_r = LogisticRegression(random_state=1234, C=0.1, penalty='l2', class_weight='balanced', solver='liblinear')
#model_log_r = LogisticRegression(random_state=1234, class_weight='balanced', solver='liblinear'
model_log_r.fit(features_upsampled, target_upsampled)
predictions_log_r_upsampled = model_log_r.predict(features_valid)
auc_roc_log_r_upsampled = roc_auc_score(target_valid, predictions_log_r_upsampled)
print("AUC-ROC Score:", auc_roc_log_r_upsampled)

# %% [markdown]
# - Los sobre y submuestreos en este modelo no generan mejores resultados, para este modelo se recomienda usar la versión original.

# %% [markdown]
# ### XGBClassifier

# %%
%%time
# Configurar los parámetros del modelo
params = {
    'objective': 'binary:logistic',  # Problema de clasificación binaria
    'eval_metric': 'logloss',  # Métrica de evaluación para la clasificación
    'early_stopping_rounds': 10,  # Detener el entrenamiento si no mejora después de 10 iteraciones
    # Puedes agregar más parámetros según sea necesario
}

model = XGBClassifier(**params)
model.fit(features_train, target_train, eval_set=[(features_valid, target_valid)])
predictions_xgb = model.predict_proba(features_valid)[:, 1]
auc_roc_xgb = roc_auc_score(target_valid, predictions_xgb)
print("AUC-ROC Score:", auc_roc_xgb)

# %% [markdown]
# - Con este modelo obtenemos un score de 0.83.

# %% [markdown]
# ### Aplicando sobre y submuestreo

# %%
%%time
# Configurar los parámetros del modelo
params = {
    'objective': 'binary:logistic',  # Problema de clasificación binaria
    'eval_metric': 'logloss',  # Métrica de evaluación para la clasificación
    'early_stopping_rounds': 10,  # Detener el entrenamiento si no mejora después de 10 iteraciones
    # Puedes agregar más parámetros según sea necesario
}

model = XGBClassifier(**params)
model.fit(features_upsampled, target_upsampled, eval_set=[(features_valid, target_valid)])
predictions_xgb = model.predict_proba(features_valid)[:, 1]
auc_roc_xgb = roc_auc_score(target_valid, predictions_xgb)
print("AUC-ROC Score:", auc_roc_xgb)

# %%
%%time
# Configurar los parámetros del modelo
params = {
    'objective': 'binary:logistic',  # Problema de clasificación binaria
    'eval_metric': 'logloss',  # Métrica de evaluación para la clasificación
    'early_stopping_rounds': 10,  # Detener el entrenamiento si no mejora después de 10 iteraciones
    # Puedes agregar más parámetros según sea necesario
}

model = XGBClassifier(**params)
model.fit(features_downsampled, target_downsampled, eval_set=[(features_valid, target_valid)])
predictions_xgb = model.predict_proba(features_valid)[:, 1]
auc_roc_xgb = roc_auc_score(target_valid, predictions_xgb)
print("AUC-ROC Score:", auc_roc_xgb)

# %% [markdown]
# - Los sobre y submuestreos en este modelo no generan mejores resultados, para este modelo se recomienda usar la versión original.

# %% [markdown]
# ### KNeighborsClassifier

# %%
model_knn = KNeighborsClassifier(n_neighbors=100)
model_knn.fit(features_train, target_train)
predictions_knn = model_knn.predict_proba(features_valid)[:, 1]
auc_roc_kn = roc_auc_score(target_valid, predictions_knn)
print("AUC-ROC Score:", auc_roc_kn)

# %% [markdown]
# - Con este modelo obtenemos un score de 0.826.

# %% [markdown]
# ### Aplicando sobre y submuestreo

# %%
model_knn = KNeighborsClassifier(n_neighbors=100)
model_knn.fit(features_upsampled, target_upsampled)
predictions_knn = model_knn.predict_proba(features_valid)[:, 1]
auc_roc_kn = roc_auc_score(target_valid, predictions_knn)
print("AUC-ROC Score:", auc_roc_kn)

# %%
model_knn = KNeighborsClassifier(n_neighbors=100)
model_knn.fit(features_downsampled, target_downsampled)
predictions_knn = model_knn.predict_proba(features_valid)[:, 1]
auc_roc_kn = roc_auc_score(target_valid, predictions_knn)
print("AUC-ROC Score:", auc_roc_kn)

# %% [markdown]
# - Los sobre y submuestreos en este modelo no generan mejores resultados, para este modelo se recomienda usar la versión original.

# %% [markdown]
# ### MLPClassifier

# %%
model_mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=1234)
model_mlp.fit(features_train, target_train)
predictions_mlp = model_mlp.predict_proba(features_valid)[:, 1]
auc_roc_mlp = roc_auc_score(target_valid, predictions_mlp)
print("AUC-ROC Score:", auc_roc_mlp)

# %% [markdown]
# - Con este modelo obtenemos un score de 0.81.

# %% [markdown]
# ### Aplicando sobre y submuestreo

# %%
model_mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=1234)
model_mlp.fit(features_upsampled, target_upsampled)
predictions_mlp = model_mlp.predict_proba(features_valid)[:, 1]
auc_roc_mlp = roc_auc_score(target_valid, predictions_mlp)
print("AUC-ROC Score:", auc_roc_mlp)

# %%
model_mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=1234)
model_mlp.fit(features_downsampled, target_downsampled)
predictions_mlp = model_mlp.predict_proba(features_valid)[:, 1]
auc_roc_mlp = roc_auc_score(target_valid, predictions_mlp)
print("AUC-ROC Score:", auc_roc_mlp)

# %% [markdown]
# - Los sobre y submuestreos en este modelo no generan mejores resultados, para este modelo se recomienda usar la versión original.

# %% [markdown]
# ### Resultados

# %%
print('AUC-ROC Árbol de decisiones:', round(auc_roc_dtc_downsampled,2))
print('AUC-ROC Regresión Bosque Aleatorio:', round(auc_roc_rfr,2))
print('AUC-ROC Regresión Lineal:', round(auc_roc_lr,2))
print('AUC-ROC Light GBM:', round(auc_roc_lgbm,2))
print('AUC-ROC Regresión Logística:', round(auc_roc_log_r_upsampled,2))
print('AUC-ROC XGB Classifier:', round(auc_roc_xgb,2))
print('AUC-ROC KNeighbors Classifier:', round(auc_roc_kn,2))
print('AUC-ROC MLP Classifier:', round(auc_roc_mlp,2))

# %% [markdown]
# ### Modelo Final

# %% [markdown]
# - Se decide utilizar como modelo final, la Regresión Lineal, porque es de las más utilizadas y además por el tiempo de procesamiento que tiene, siendo uno de los más rápidos en procesar el modelo.

# %%
model_lr = LinearRegression(fit_intercept=True)
model_lr.fit(features_train, target_train)
predictions_lr = model_lr.predict(features_valid)

auc_roc_lr = roc_auc_score(target_valid, predictions_lr)

print("AUC-ROC Score:", auc_roc_lr)

# %% [markdown]
# ### Conclusiones Generales

# %% [markdown]
# - Este proyecto se inicia con una introducción del problema que se quiere resolver.
# - Seguido, se importan las librerías y los 4 dataframes que se utilizarán en el proyecto.
# - Luego se encuentra el análisis exploratorio de cada dataframe, en donde se abordan los tipos de variables y valores ausentes.
# - Finalizado el paso anterior se unen todos los dataframes, y se proceden a corregir formatos, con el fin de obtener un mejor rendimiento de procesamiento.
# - El rellenar los valores ausentes fue el siguiente paso.
# - Seguido, se crea una función para 'end_date' con el fin de crear un valor booleano para identifcar a los activos de los que abandonaron la compañía. Esta columna será usado como nuevo 'target'.
# - Avanzando a través del proyecto se encuentraa luego el análisis a través de gráficos, se destaca que lo usuarios que dejaron de usar los servicios aportaron un 18% de la venta total, también que el método de pago de preferencia es el de 'Two year', y que el 67% prefiere facturación electrónica.
# - Otro dato relevante es que la concentración de usuarios nuevos y antiguos es muy notoria, eso quiere decir usuarios con menos de 12 meses en la compañía o con más de 5 años.
# - Para el entrenamiento de los modelos se utilizarán 8 características.
# - Se entrenaron 8 modelos.
# - Para cada modelo se utilizó el análisis a través del sub y sobremuestreo.
# - 5 de los modelos arrojaron un auc-roc de 0.83.
# - Dado la versatilidad y rapidez de procesamiento del modelo, se elige como mejor modelo la Regresión Lineal.


