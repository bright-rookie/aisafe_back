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
import numpy as np
from pathlib import Path
from aisafe_xgboost.utils import MockData

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
