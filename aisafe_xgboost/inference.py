import pandas as pd
import xgboost as xgb
import numpy as np
from pathlib import Path
from aisafe_xgboost.utils import MockData

PACKAGEDIR = Path(__file__).parent.absolute()
MODELDIR = PACKAGEDIR / 'models'

def model(**kwargs):
    # load data
    mockdata = MockData()
    expected_cols = mockdata.expected_columns
    dfs = [pd.DataFrame([arg], columns = expected_cols[type]) for type, arg in kwargs.items()]
    dmtx = [xgb.DMatrix(df) for df in dfs]
    models = [xgb.Booster(model_file= MODELDIR / f"model_{name}.ubj") for name in kwargs.keys()]
    weights = np.load(MODELDIR / "optimal_weights.npy")
    predictions = [model.predict(dmatrix) for model, dmatrix in zip(models, dmtx)]
    predicted = [predictions[0] for _ in range(len(predictions[0]))]
    final_pred = sum(w * pred for w, pred in zip(weights, predicted))
    final_pred_sub = [w * pred for w, pred in zip(weights, predicted)]
    final_pred_sub_rounded = []
    for element in final_pred_sub :
        final_pred_sub_rounded.append(np.round(element, 3))

    prediction_names = np.array(['신체 계측치', '멍 정보', '문진 정보', 'Lab 수치', 'X-ray 영상', '진료 영상'])
    sorted_indices = np.argsort(final_pred_sub)[::-1]
    sorted_weights = np.array(final_pred_sub_rounded)[sorted_indices]
    sorted_names = prediction_names[sorted_indices]

    explainability = list(zip(sorted_names, sorted_weights))

    return final_pred, explainability
    
