import os, sys
sys.path.append(os.getcwd()) 

# import pandas as pd
from imports import *

def load_data():
    df_contract = pd.read_csv('data/contract.csv')
    df_internet = pd.read_csv('data/internet.csv')
    df_personal = pd.read_csv('data/personal.csv')
    df_phone = pd.read_csv('data/phone.csv')
    return df_contract, df_internet, df_personal, df_phone

# df_contract, df_internet, df_personal, df_phone = load_data()
# print(df_contract.head())

def merge_dataframes(df_contract, df_internet, df_personal, df_phone):
    df_consolidado = pd.merge(df_contract, df_internet, on='customerID', how='outer')
    df_consolidado = pd.merge(df_consolidado, df_personal, on='customerID', how='outer')
    df_consolidado = pd.merge(df_consolidado, df_phone, on='customerID', how='outer')
    return df_consolidado

def preprocess_data(df_consolidado):
    df_consolidado['SeniorCitizen'] = df_consolidado['SeniorCitizen'].astype('int8')

    # Convertir columnas a categorías
    categorical_columns = ['Type', 'PaperlessBilling', 'PaymentMethod', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'gender', 'Partner', 'Dependents', 'MultipleLines']
    for column in categorical_columns:
        df_consolidado[column] = df_consolidado[column].astype('category')

    # Convertir columna 'TotalCharges' a numérica
    df_consolidado['TotalCharges'] = pd.to_numeric(df_consolidado['TotalCharges'], errors='coerce').astype('float64')

    # Convertir columnas de fecha
    df_consolidado['BeginDate'] = pd.to_datetime(df_consolidado['BeginDate'], format='%Y-%m-%d')
    df_consolidado['EndDate'] = pd.to_datetime(df_consolidado['EndDate'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Renombrar columnas
    df_consolidado = df_consolidado.rename(columns={'customerID': 'customer_id', 'BeginDate': 'begin_date', 'EndDate': 'end_date', 'Type': 'type',
                                                    'PaperlessBilling': 'paperless_billing', 'PaymentMethod': 'payment_method', 'MonthlyCharges': 'monthly_charges',
                                                    'TotalCharges': 'total_charges', 'InternetService': 'internet_service', 'OnlineSecurity': 'online_security',
                                                    'OnlineBackup': 'online_backup', 'DeviceProtection': 'device_protection', 'TechSupport': 'tech_support',
                                                    'StreamingTV': 'streaming_tv', 'StreamingMovies': 'streaming_movies', 'gender': 'gender', 'SeniorCitizen': 'senior_citizen',
                                                    'Partner': 'partner', 'Dependents': 'dependents', 'MultipleLines': 'multiple_lines'})

    # Llenar valores faltantes
    df_consolidado['total_charges'] = df_consolidado['total_charges'].fillna(0)
    df_consolidado['end_date'] = df_consolidado['end_date'].fillna('No')

    columnas = ['internet_service', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'multiple_lines']
    valor = 'No'
    # Agrega la categoría 'No' a las columnas categóricas solo si no existe
    for columna in columnas:
        if valor not in df_consolidado[columna].cat.categories:
            df_consolidado[columna] = df_consolidado[columna].cat.add_categories(valor)
    # Llena los valores nulos con 'No'
    df_consolidado[columnas] = df_consolidado[columnas].fillna(valor)

    def end_date_2(row):
        if row['end_date'] == 'No':
            return 0
        else:
            return 1
    df_consolidado['end_date_2'] = df_consolidado.apply(end_date_2, axis=1)

    def end_date_3(row):
        if row['end_date'] == 'No':
            return pd.to_datetime('2020-01-31',format='%Y-%m-%d')
        else:
            return row['end_date']
    df_consolidado['end_date_3'] = df_consolidado.apply(end_date_3, axis=1)

    def months(row):
        months_diff = row['end_date_3'] - row['begin_date']
        return round(months_diff / pd.Timedelta(days=30))
    df_consolidado['months_using_service'] = df_consolidado.apply(months, axis=1)

    def ttl_charges(row):
        total_charges_diff = (row['monthly_charges'] * row['months_using_service']) - row['total_charges']
        return total_charges_diff
    df_consolidado['total_charges_diff'] = df_consolidado.apply(ttl_charges, axis=1)

    df_consolidado = df_consolidado.drop(['begin_date', 'end_date', 'paperless_billing', 'payment_method', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
                                      'streaming_movies', 'partner', 'dependents', 'multiple_lines', 'end_date_3'], axis=1)

    df_consolidado_ohe = pd.get_dummies(df_consolidado[['type', 'internet_service', 'gender']], drop_first=True)

    df_consolidado = pd.merge(df_consolidado, df_consolidado_ohe, left_index=True, right_index=True, how='outer')

    df_consolidado = df_consolidado.drop(['customer_id', 'type', 'internet_service', 'gender'], axis=1)

    return df_consolidado

def features_target(df_consolidado):
    features = df_consolidado.drop(['end_date_2'], axis=1)
    target  = df_consolidado['end_date_2']
    features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)

    numeric = ['monthly_charges', 'total_charges', 'months_using_service', 'total_charges_diff']
    scaler = StandardScaler()
    scaler.fit(features_train[numeric])
    features_train[numeric] = scaler.transform(features_train[numeric])
    features_valid[numeric] = scaler.transform(features_valid[numeric])
    
    return features_train, features_valid, target_train, target_valid

def consolidado():
    # Carga de datos
    df_contract, df_internet, df_personal, df_phone = load_data()
    # print("-------------------------------")
    # Unir dataframes
    df_consolidado = merge_dataframes(df_contract, df_internet, df_personal, df_phone)

    # Preprocesamiento de datos
    df_consolidado = preprocess_data(df_consolidado)

    # Realizar otras tareas como cálculos, visualizaciones, entrenamiento de modelos, etc.
    # print(df_consolidado.head())

    features_train, features_valid, target_train, target_valid = features_target(df_consolidado)
    # print(target_valid)

# if __name__ == "__main__":
consolidado()