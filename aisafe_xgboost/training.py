# MIT License
# 
# Copyright (c) 2024 bright-rookie
# 
# This repository ("aisafe_back") and the website aisafe.qbio.page are 
# educational resources maintained by bright-rookie. All content, incl
# -uding code, data, and models, are provided strictly for educational 
# and demonstration purposes.

# All data and examples in this repository are mock/synthetic demonst-
# rations. The repository contains no functional AI models or real med
# -ical data. Any outputs generated are entirely fictional and have no
# basis in real medical analysis or diagnostics.

# DO NOT USE FOR MEDICAL PURPOSES UNDER ANY CIRCUMSTANCES
# This repository and website are not intended for clinical use under 
# any circumstances. The content must not be used for medical decision
# -making or as a substitute for professional medical advice. Any medi
# -cal concerns should be directed to qualified healthcare professionals.

# bright-rookie and contributors assume no liability for any damages ari
# -sing from the use or misuse of this repository or website. Use of an
# y repository contents and website data is entirely at your own risk. 
# No warranties are provided regarding the accuracy or completeness of 
# any information contained herein.

# Links to third-party content within this repository or website are pr
# -ovided solely for convenience. bright-rookie and contributors neith
# er endorse nor verify the content of external resources. Access and 
# use of any external resources referenced herein is entirely at your
# own risk.

# This repository and website do not provide medical or health advice
# in any form. The contents are intended exclusively for machine lear
# -ning education and demonstration. The repository cannot and does no
# -t provide treatment recommendations. All materials are unsuitable f
# -or diagnostic purposes.

# Permission is hereby granted, free of charge, to any person obtaining 
# a copy of this software and associated documentation files (the "Soft
# -ware"), to deal in the Software without restriction, including witho
# ut limitation the rights to use, copy, modify, merge, publish, distri
# -bute, sublicense, and/or sell copies of the Software, and to permit 
# persons to whom the Software is furnished to do so, subject to the f
# -ollowing conditions:

# The above copyright notice, this permission notice, and all disclaime
# -rs shall be included in all copies or substantial portions of the So
# -ftware.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRES
# -S OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANT
# -ABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO 
# EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
# MEDICAL DISCLAIMER: THIS SOFTWARE AND ANY ASSOCIATED MATERIALS ARE FOR
# EDUCATIONAL PURPOSES ONLY AND SHALL NOT BE USED FOR MEDICAL DECISIONS 
# OR CLINICAL PURPOSES.



import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import numpy as np
from scipy.optimize import minimize
from aisafe_xgboost.utils import MockData


def prepare_with_mock(dataframe, mockdata, y_label, column_idx = 0):
    for df in mockdata:
        start, column_idx = column_idx, column_idx + len(df.columns)
        df = dataframe.iloc[:, start:column_idx]
        print(df.columns.tolist())
        yield xgb.DMatrix(df, label=y_label)

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
    rmse = lambda w: rmse_loss(w, predictions, y_test) #noqa
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
