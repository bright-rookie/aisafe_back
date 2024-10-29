import pandas as pd
import xgboost as xgb
import numpy as np
from pathlib import Path

PACKAGEDIR = Path(__file__).parent.absolute()
MODELDIR = PACKAGEDIR / 'models'

def model(**kwargs):
    # load data
    dfs = [pd.read_csv(arg) for arg in kwargs.values()]
    dmtx = [xgb.DMatrix(df) for df in dfs]
    models = [xgb.Booster(model_file=f"{str(MODELDIR)}/model_{name}.ubj") for name in kwargs.keys()]
    weights = np.load(f"{str(MODELDIR)}/optimal_weights.npy")
    predictions = [model.predict(dmatrix) for model, dmatrix in zip(models, dmtx)]
    predicted = [predictions[0] for _ in range(len(predictions[0]))]

    final_pred = sum(w * pred for w, pred in zip(weights, predictions))
    final_pred_sub = [w * pred for w, pred in zip(weights, predictions)]
    final_pred_sub_rounded = []
    for element in final_pred_sub :
      final_pred_sub_rounded.append(round(element, 3))

    prediction_names = np.array(['신체 계측치', '멍 정보', '문진 정보', 'Lab 수치', 'X-ray 영상', '진료 영상'])

    sorted_indices = np.argsort(final_pred_sub)[::-1]
    sorted_weights = np.array(final_pred_sub_rounded)[sorted_indices]
    sorted_names = prediction_names[sorted_indices]

    explainability = list(zip(sorted_names, sorted_weights))

    return final_pred, explainability
    