from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression(features, target, feature_list, model, test_size = 0.3, scaler = None, cross_validation = False, cv = 10,
               random_state = 42):

    if scaler != None: # The case is to use scaling
        if cross_validation == False:
            scaler = scaler
            feature_scaled = scaler.fit_transform(features)
            feature_scaled = pd.DataFrame(feature_scaled, columns = features.columns)

            X_train, X_test, y_train, y_test = train_test_split(feature_scaled, target, test_size = test_size, random_state = random_state)
            print(f'X_train shape : {X_train.shape}')
            print(f'X_test shape : {X_test.shape}')
            print(f'y_train shape : {y_train.shape}')
            print(f'y_test shape : {y_test.shape}')

            model = model.fit(X_train[feature_list], y_train)
            pred = model.predict(X_test[feature_list])
            mse = mean_squared_error(pred, y_test)
            mae = mean_absolute_error(pred, y_test)
            r2 = r2_score(pred, y_test)
            print(f'{model} {feature_list} MSE : {mse:.4f}')
            print(f'{model} {feature_list} MAE : {mae:.4f}')
            print(f'{model} {feature_list} R2 : {r2:.4f}')
            return pred
        else:
            scaler = scaler
            feature_scaled = scaler.fit_transform(features)
            feature_scaled = pd.DataFrame(feature_scaled, columns = features.columns)

            X_train, X_test, y_train, y_test = train_test_split(feature_scaled, target, test_size = test_size, random_state = random_state)
            print(f'X_train shape : {X_train.shape}')
            print(f'X_test shape : {X_test.shape}')
            print(f'y_train shape : {y_train.shape}')
            print(f'y_test shape : {y_test.shape}')

            model = model.fit(X_train[feature_list], y_train)
            pred = model.predict(X_test[feature_list])
            mse = mean_squared_error(pred, y_test)
            mae = mean_absolute_error(pred, y_test)
            r2 = r2_score(pred, y_test)
            cv_score = np.round(cross_val_score(model, X_test, y_test, cv = cv), 4)
            print(f'{model} {feature_list} MSE : {mse:.4f}')
            print(f'{model} {feature_list} MAE : {mae:.4f}')
            print(f'{model} {feature_list} R2 : {r2:.4f}')
            print(f'{model} {feature_list} Cross Validation Score : {cv_score}')
            return pred
    
    else: # The case isn't to use scaling
        if cross_validation == False:
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = test_size, random_state = random_state)
            print(f'X_train shape : {X_train.shape}')
            print(f'X_test shape : {X_test.shape}')
            print(f'y_train shape : {y_train.shape}')
            print(f'y_test shape : {y_test.shape}')

            model = model.fit(X_train[feature_list], y_train)
            pred = model.predict(X_test[feature_list])
            mse = mean_squared_error(pred, y_test)
            mae = mean_absolute_error(pred, y_test)
            r2 = r2_score(pred, y_test)
            print(f'{model} {feature_list} MSE : {mse:.4f}')
            print(f'{model} {feature_list} MAE : {mae:.4f}')
            print(f'{model} {feature_list} R2 : {r2:.4f}')
            return pred
        else:
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = test_size, random_state = random_state)
            print(f'X_train shape : {X_train.shape}')
            print(f'X_test shape : {X_test.shape}')
            print(f'y_train shape : {y_train.shape}')
            print(f'y_test shape : {y_test.shape}')

            model = model.fit(X_train[feature_list], y_train)
            pred = model.predict(X_test[feature_list])
            mse = mean_squared_error(pred, y_test)
            mae = mean_absolute_error(pred, y_test)
            r2 = r2_score(pred, y_test)
            cv_score = np.round(cross_val_score(model, X_test, y_test, cv = cv), 4)
            print(f'{model} {feature_list} MSE : {mse:.4f}')
            print(f'{model} {feature_list} MAE : {mae:.4f}')
            print(f'{model} {feature_list} R2 : {r2:.4f}')
            print(f'{model} {feature_list} Cross Validation Score : {cv_score}')
            return pred