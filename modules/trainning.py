from imports import *
from load_data import load_data, merge_dataframes, preprocess_data, features_target

df_contract, df_internet, df_personal, df_phone = load_data()
df_consolidado = merge_dataframes(df_contract, df_internet, df_personal, df_phone)
df_consolidado = preprocess_data(df_consolidado)
features_train, features_valid, target_train, target_valid = features_target(df_consolidado)

def trainning(features_train, features_valid, target_train, target_valid):

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

    # DecisionTreeClassifier

    max_auc_roc_dtc = float('-inf')
    best_d_dtc = None
    
    for d in range(1, 20):
        model = DecisionTreeClassifier(random_state=1234, max_depth = d)
        model.fit(features_train, target_train)
        predictions_dtc = model.predict(features_valid)
        auc_roc = roc_auc_score(target_valid, predictions_dtc)
        if auc_roc > max_auc_roc_dtc:
            max_auc_roc_dtc = auc_roc
            best_d_dtc = d

    model_dtc = DecisionTreeClassifier(random_state=1234, max_depth=best_d_dtc)
    model_dtc.fit(features_train, target_train)
    predictions_dtc = model_dtc.predict(features_valid)
    auc_roc_dtc = roc_auc_score(target_valid, predictions_dtc)
    accuracy = accuracy_score(target_valid, predictions_dtc)

    max_auc_roc_dtc_up = float('-inf')
    best_d_dtc_up = None
    
    for d in range(1, 20):
        model = DecisionTreeClassifier(random_state=1234, max_depth = d)
        model.fit(features_upsampled, target_upsampled)
        predictions_dtc_upsampled = model.predict(features_valid)
        auc_roc_dtc_up = roc_auc_score(target_valid, predictions_dtc_upsampled)
        if auc_roc_dtc_up > max_auc_roc_dtc_up:
            max_auc_roc_dtc_up = auc_roc_dtc_up
            best_d_dtc_up = d

    model_dtc = DecisionTreeClassifier(random_state=1234, max_depth=best_d_dtc_up)
    model_dtc.fit(features_upsampled, target_upsampled)
    predictions_dtc_upsampled = model_dtc.predict(features_valid)
    auc_roc_dtc_up = roc_auc_score(target_valid, predictions_dtc_upsampled)

    max_auc_roc_dtc_down = float('-inf')
    best_d_dtc_down = None

    for d in range(1, 20):
        model = DecisionTreeClassifier(random_state=1234, max_depth = d)
        model.fit(features_downsampled, target_downsampled)
        predictions_dtc_downsampled = model.predict(features_valid)
        auc_roc_dtc_down = roc_auc_score(target_valid, predictions_dtc_downsampled)
        if auc_roc_dtc_down > max_auc_roc_dtc_down:
            max_auc_roc_dtc_down = auc_roc_dtc_down
            best_d_dtc_down = d
        
    model_dtc = DecisionTreeClassifier(random_state=1234, max_depth=best_d_dtc_down)
    model_dtc.fit(features_downsampled, target_downsampled)
    predictions_dtc_downsampled = model_dtc.predict(features_valid)
    auc_roc_dtc_down = roc_auc_score(target_valid, predictions_dtc_downsampled)

    # Random Forest Regressor

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


    model_rfr = RandomForestRegressor(random_state=1234, **best_params)
    model_rfr.fit(features_train, target_train)
    predictions_rfr = model_rfr.predict(features_valid)
    auc_roc_rfr = roc_auc_score(target_valid, predictions_rfr)

    param_grid = {
        'n_estimators': [5, 10, 55],
        'max_depth': [2, 4, 40]
        #'min_samples_split': [2, 5, 10],
        #'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=model_rfr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(features_upsampled, target_upsampled)
    best_params = grid_search.best_params_

    model_rfr = RandomForestRegressor(random_state=1234, **best_params)
    model_rfr.fit(features_upsampled, target_upsampled)
    predictions_rfr = model_rfr.predict(features_valid)
    auc_roc_rfr_up = roc_auc_score(target_valid, predictions_rfr)

    param_grid = {
        'n_estimators': [5, 10, 55],
        'max_depth': [2, 4, 40]
        #'min_samples_split': [2, 5, 10],
        #'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=model_rfr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(features_downsampled, target_downsampled)
    best_params = grid_search.best_params_

    model_rfr = RandomForestRegressor(random_state=1234, **best_params)
    model_rfr.fit(features_downsampled, target_downsampled)
    predictions_rfr = model_rfr.predict(features_valid)
    auc_roc_rfr_down = roc_auc_score(target_valid, predictions_rfr)

    # LinearRegression

    model_lr = LinearRegression()

    param_grid = {
        'fit_intercept': [True, False]
    }

    grid_search = GridSearchCV(estimator=model_lr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(features_train, target_train)
    best_params = grid_search.best_params_

    model_lr = LinearRegression(**best_params)
    model_lr.fit(features_train, target_train)
    predictions_lr = model_lr.predict(features_valid)
    auc_roc_lr = roc_auc_score(target_valid, predictions_lr)

    model_lr.fit(features_upsampled, target_upsampled)
    predictions_lr = model_lr.predict(features_valid)
    auc_roc_lr_up = roc_auc_score(target_valid, predictions_lr)

    model_lr.fit(features_downsampled, target_downsampled)
    predictions_lr = model_lr.predict(features_valid)
    auc_roc_lr_up_down = roc_auc_score(target_valid, predictions_lr)

    # Light GBM

    params = {
        'objective': 'regression',  # Puedes cambiar a 'binary' para clasificación binaria, etc.
        'metric': 'mse',  # MSE (Error Cuadrático Medio) como métrica de evaluación
        'boosting_type': 'gbdt',  # Puedes probar 'dart' o 'goss' también
        'early_stopping_rounds': 10,  # Número de rondas para esperar antes de detener el entrenamiento si no mejora la métrica
    }

    train_data = lgb.Dataset(features_train, label=target_train)
    valid_data = lgb.Dataset(features_valid, label=target_valid, reference=train_data)
    num_round = 100
    model = lgb.train(params, train_data, num_round, valid_sets=[valid_data], valid_names=['test'])
    predictions_lgb = model.predict(features_valid, num_iteration=model.best_iteration)
    auc_roc_lgbm = roc_auc_score(target_valid, predictions_lgb)

    train_data = lgb.Dataset(features_upsampled, label=target_upsampled)
    valid_data = lgb.Dataset(features_valid, label=target_valid, reference=train_data)
    num_round = 100
    model = lgb.train(params, train_data, num_round, valid_sets=[valid_data], valid_names=['test'])
    predictions_lgb = model.predict(features_valid, num_iteration=model.best_iteration)
    auc_roc_lgbm_up = roc_auc_score(target_valid, predictions_lgb)

    train_data = lgb.Dataset(features_downsampled, label=target_downsampled)
    valid_data = lgb.Dataset(features_valid, label=target_valid, reference=train_data)
    num_round = 100
    model = lgb.train(params, train_data, num_round, valid_sets=[valid_data], valid_names=['test'])
    predictions_lgb = model.predict(features_valid, num_iteration=model.best_iteration)
    auc_roc_lgbm_down = roc_auc_score(target_valid, predictions_lgb)

    # LogisticRegression

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularización
        'penalty': ['l1', 'l2'],  # Tipo de regularización
        'solver': ['liblinear', 'saga']  # Algoritmo de optimización
    }

    grid_search = GridSearchCV(
        estimator=LogisticRegression(),
        param_grid=param_grid,
        cv=5,  # Número de validaciones cruzadas
        scoring='accuracy',  # Métrica de evaluación
        verbose=1,  # Muestra información detallada
        n_jobs=-1  # Utiliza todos los núcleos de CPU disponibles
    )

    grid_search.fit(features_train, target_train)
    best_params = grid_search.best_params_
    model_log_r = LogisticRegression(random_state=1234, class_weight='balanced', **best_params)
    model_log_r.fit(features_train, target_train)
    predictions_log_r = model_log_r.predict(features_valid)
    auc_roc_logr = roc_auc_score(target_valid, predictions_log_r)

    grid_search.fit(features_upsampled, target_upsampled)
    best_params = grid_search.best_params_
    model_log_r = LogisticRegression(random_state=1234, class_weight='balanced', **best_params)
    model_log_r.fit(features_upsampled, target_upsampled)
    predictions_log_r = model_log_r.predict(features_valid)
    auc_roc_logr_up = roc_auc_score(target_valid, predictions_log_r)

    grid_search.fit(features_downsampled, target_downsampled)
    best_params = grid_search.best_params_
    model_log_r = LogisticRegression(random_state=1234, class_weight='balanced', **best_params)
    model_log_r.fit(features_downsampled, target_downsampled)
    predictions_log_r = model_log_r.predict(features_valid)
    auc_roc_logr_down = roc_auc_score(target_valid, predictions_log_r)

    # XGBClassifier

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

    model.fit(features_upsampled, target_upsampled, eval_set=[(features_valid, target_valid)])
    predictions_xgb = model.predict_proba(features_valid)[:, 1]
    auc_roc_xgb_up = roc_auc_score(target_valid, predictions_xgb)

    model.fit(features_downsampled, target_downsampled, eval_set=[(features_valid, target_valid)])
    predictions_xgb = model.predict_proba(features_valid)[:, 1]
    auc_roc_xgb_down = roc_auc_score(target_valid, predictions_xgb)

    # KNeighborsClassifier

    model_knn = KNeighborsClassifier(n_neighbors=100)
    model_knn.fit(features_train, target_train)
    predictions_knn = model_knn.predict_proba(features_valid)[:, 1]
    auc_roc_kn = roc_auc_score(target_valid, predictions_knn)

    model_knn.fit(features_upsampled, target_upsampled)
    predictions_knn = model_knn.predict_proba(features_valid)[:, 1]
    auc_roc_kn_up = roc_auc_score(target_valid, predictions_knn)

    model_knn.fit(features_downsampled, target_downsampled)
    predictions_knn = model_knn.predict_proba(features_valid)[:, 1]
    auc_roc_kn_down = roc_auc_score(target_valid, predictions_knn)

    # MLPClassifier

    model_mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=1234)
    model_mlp.fit(features_train, target_train)
    predictions_mlp = model_mlp.predict_proba(features_valid)[:, 1]
    auc_roc_mlp = roc_auc_score(target_valid, predictions_mlp)

    model_mlp.fit(features_upsampled, target_upsampled)
    predictions_mlp = model_mlp.predict_proba(features_valid)[:, 1]
    auc_roc_mlp_up = roc_auc_score(target_valid, predictions_mlp)

    model_mlp.fit(features_downsampled, target_downsampled)
    predictions_mlp = model_mlp.predict_proba(features_valid)[:, 1]
    auc_roc_mlp_down = roc_auc_score(target_valid, predictions_mlp)




    return auc_roc_dtc, auc_roc_dtc_up, auc_roc_dtc_down, auc_roc_rfr, auc_roc_rfr_up, auc_roc_rfr_down, auc_roc_lr, auc_roc_lr_up, auc_roc_lr_up_down, auc_roc_lgbm, auc_roc_lgbm_down, auc_roc_lgbm_up, auc_roc_logr, auc_roc_logr_up, auc_roc_logr_down, auc_roc_xgb, auc_roc_xgb_up, auc_roc_xgb_down, auc_roc_kn, auc_roc_kn_up, auc_roc_kn_down, auc_roc_mlp,auc_roc_mlp_up, auc_roc_mlp_down



auc_roc_dtc, auc_roc_dtc_up, auc_roc_dtc_down, auc_roc_rfr, auc_roc_rfr_up, auc_roc_rfr_down, auc_roc_lr, auc_roc_lr_up, auc_roc_lr_up_down, auc_roc_lgbm, auc_roc_lgbm_down, auc_roc_lgbm_up, auc_roc_logr, auc_roc_logr_up, auc_roc_logr_down, auc_roc_xgb, auc_roc_xgb_up, auc_roc_xgb_down, auc_roc_kn, auc_roc_kn_up, auc_roc_kn_down, auc_roc_mlp, auc_roc_mlp_up, auc_roc_mlp_down = trainning(features_train, features_valid, target_train, target_valid)
print('Mejores Resultados por Modelo')
print("-----------------------")
# print('AUC-ROC Score, Ábol de decisiones:', auc_roc_dtc)
# print("-----------------------")
# print('roc_auc_score:', auc_roc_dtc_up)
# print("-----------------------")
print('AUC-ROC Score Ábol de decisiones - Submuestreo:', round(auc_roc_dtc_down, 2))
print("-----------------------")
print('AUC-ROC Score Regresión Bosque Aleatorio:', round(auc_roc_rfr,2))
print("-----------------------")
# print('roc_auc_score:', auc_roc_rfr_up)
# print("-----------------------")
# print('roc_auc_score:', auc_roc_rfr_down)
# print("-----------------------")
print('AUC-ROC Score Regresión Lineal:', round(auc_roc_lr,2))
print("-----------------------")
# print('roc_auc_score:', auc_roc_lr_up)
# print("-----------------------")
# print('roc_auc_score:', auc_roc_lr_up_down)
# print("-----------------------")
print('AUC-ROC Score Light GBM:', round(auc_roc_lgbm,2))
print("-----------------------")
# print('roc_auc_score:', auc_roc_lgbm_down)
# print("-----------------------")
# print('roc_auc_score:', auc_roc_lgbm_up)
# print("-----------------------")
# print('roc_auc_score:', auc_roc_logr)
# print("-----------------------")
print('AUC-ROC Score Regresión Logística - Sobremuestreo:', round(auc_roc_logr_up,2))
print("-----------------------")
# print('roc_auc_score:', auc_roc_logr_down)
# print("-----------------------")
print('AUC-ROC Score XGB Classifier:', round(auc_roc_xgb,2))
print("-----------------------")
# print('roc_auc_score:', auc_roc_xgb_up)
# print("-----------------------")
# print('roc_auc_score:', auc_roc_xgb_down)
# print("-----------------------")
print('AUC-ROC Score KNeighbors Classifier:', round(auc_roc_kn,2))
print("-----------------------")
# print('roc_auc_score:', auc_roc_kn_up)
# print("-----------------------")
# print('roc_auc_score:', auc_roc_kn_down)
# print("-----------------------")
print('AUC-ROC Score MLP Classifier:', round(auc_roc_mlp,2))
print("-----------------------")
# print('roc_auc_score:', auc_roc_mlp_up)
# print("-----------------------")
# print('roc_auc_score:', auc_roc_mlp_down)
print('Modelo Final:')
print('AUC-ROC Score Regresión Lineal:', round(auc_roc_lr,2))