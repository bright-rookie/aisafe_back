import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import numpy as np
from scipy.optimize import minimize
from aisafe_xgboost.utils import MockData


def prepare_with_mock(dataframe, mockdata, y_label):
    column_idx = 1
    for df in mockdata:
        start, column_idx = column_idx, column_idx + len(df.columns)
        yield xgb.DMatrix(dataframe.iloc[:, start:column_idx], label=y_label)

def rmse_loss(weights, predictions, y_test):
    weighted_predictions = sum([w * p for w, p in zip(weights, predictions)])
    rmse = root_mean_squared_error(y_test, weighted_predictions)
    return rmse

def main():
    mockdata = MockData()
    mockdatalist = [
        mockdata.info_vector, 
        mockdata.bruise_vector, 
        mockdata.response_vector, 
        mockdata.lab_vector, 
        mockdata.xray_vector, 
        mockdata.video_vector
    ]
    datanames = ['info', 'bruise', 'response', 'lab', 'xray', 'video']
    X = pd.concat(mockdatalist, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, mockdata.true, test_size=0.3, random_state=None)

    dtrain = prepare_with_mock(X_train, mockdatalist, y_train)
    dtest = prepare_with_mock(X_test, mockdatalist, y_test)
    params = {'objective': 'binary:logistic', 'max_depth': 5, 'eta': 0.1}
    num_round = 500

    models, predictions = [], []
    for train in dtrain:
        model = xgb.train(params, train, num_round)
        models.append(model)
    for model, test in zip(models, dtest):
        prediction = model.predict(test)
        predictions.append(prediction)

    initial_weights = np.ones(6) / 6
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * 6
    rmse = lambda w: rmse_loss(w, predictions, y_test)
    result = minimize(rmse, initial_weights, bounds=bounds, constraints=constraints)
    optimal_weights = result.x

    np.save("models/optimal_weights.npy", optimal_weights)
    for model, name in zip(models, datanames):
        model.save_model(f"models/model_{name}.ubj")
    np.set_printoptions(precision=2, suppress=True)
    print(f"Optimal weights: {optimal_weights}")
    return models, optimal_weights

if __name__ == "__main__":
    main()
