import pandas as pd
import xgboost as xgb
import numpy as np
from pathlib import Path
from .utils import MockData

PACKAGEDIR = Path(__file__).parent.absolute()
MODELDIR = PACKAGEDIR / 'models'

def model(**kwargs):
    # load data
    mockdata = MockData()
    expected_cols = mockdata.expected_columns
    dfs = [pd.DataFrame([list(arg)], columns = expected_cols[type]) for type, arg in kwargs.items()]
    dmtx = [xgb.DMatrix(df) for df in dfs]
    models = [xgb.Booster(model_file= MODELDIR / f"model_{name}.ubj") for name in kwargs.keys()]
    weights = np.load(MODELDIR / "optimal_weights.npy")
    predicted = np.array([model.predict(dmatrix) for model, dmatrix in zip(models, dmtx)]).squeeze()
    final_pred = np.sum(weights * predicted)
    final_pred_sub = weights * predicted
    final_pred_sub_rounded = np.round(final_pred_sub, 3)

    prediction_names = ['신체 계측치', '멍 정보', '문진 정보', 'Lab 수치', 'X-ray 영상', '진료 영상']
    sorted_indices = np.argsort(final_pred_sub)[::-1]
    sorted_weights = np.array(final_pred_sub_rounded)[sorted_indices]
    sorted_names = np.array(prediction_names)[sorted_indices]

    explainability = list(zip(sorted_names, sorted_weights))

    return final_pred, explainability

if __name__ == "__main__":
    output=model(
    info=(120,0,11.67,13.84),
    bruise=[0 for _ in range(11)],
    response=[0 for _ in range(9)],
    lab=[0 for _ in range(19)],
    xray=[0 for _ in range(9)],
    video=[0 for _ in range(30)],
    )
    print(f"Final prediction: {output}")
