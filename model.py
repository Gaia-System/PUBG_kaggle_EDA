'''
X_train, y_train, X_test, y_test와 모델에 넣을 feature를 입력하면
그에 대한 Linear regression 모델이 생성됨과 동시에
MSE, MAE, R2 지표의 값이 소수점 4번째까지 반환됩니다.
'''
import numpy as np

def linearEvaluate(X_train, y_train, X_test, y_test, feature_list):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(X_train[feature_list], y_train)
    pred = model.predict(X_test[feature_list])
    mse = np.round(mean_squared_error(pred, y_test), 4)
    mae = np.round(mean_absolute_error(pred, y_test), 4)
    r2 = np.round(r2_score(pred, y_test), 4)
    print(f'{feature_list} MSE : {mse}')
    print(f'{feature_list} MAE : {mae}')
    print(f'{feature_list} R2 : {r2}')