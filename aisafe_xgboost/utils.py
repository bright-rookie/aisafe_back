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
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

PACKAGEDIR = Path(__file__).parent.absolute()
DATADIR = PACKAGEDIR / 'mock_data'
GROWTHDIR = PACKAGEDIR / 'growth'

@dataclass
class MockData:
    info_data = pd.read_csv(DATADIR / "basic_info.csv")
    bruise_data = pd.read_csv(DATADIR / "bruise.csv" )
    response_data = pd.read_csv(DATADIR / "exam.csv" )
    lab_data = pd.read_csv(DATADIR / "lab.csv" )
    xray_data = pd.read_csv(DATADIR / "fracture.csv" )
    video_data = pd.read_csv(DATADIR / "emotion.csv" )
    true_data = pd.read_csv(DATADIR / "true_label.csv" )

    @cached_property
    def info_vector_pre(self):
        return self.info_data.iloc[:, 1:5]

    @cached_property
    def bruise_vector(self):
        return self.bruise_data.iloc[:, 1:12]

    @cached_property
    def response_vector(self):
        return self.response_data.iloc[:, 1:10]

    @cached_property
    def lab_vector(self):
        return self.lab_data.iloc[:, 1:20]

    @cached_property
    def xray_vector(self):
        return self.xray_data.iloc[:, 1:10]

    @cached_property
    def video_vector(self):
        return self.video_data.iloc[:, 1:31]

    @cached_property
    def true(self):
        return self.true_data.iloc[:, 1]

    @cached_property
    def info_vector(self):
        info_columns = ['patient_age', 'patient_sex', 'height_percentile', 'weight_percentile']
        ages = self.info_vector_pre.iloc[:, 0]
        sexes = self.info_vector_pre.iloc[:, 1].astype(int)
        heights = self.info_vector_pre.iloc[:, 2]
        weights = self.info_vector_pre.iloc[:, 3]
        info_vector_list = [
            ParseGrowth(age, sex, height, weight).get_percentiles()
            for age, sex, height, weight in zip(ages, sexes, heights, weights)
        ]
        return pd.DataFrame(info_vector_list, columns=info_columns)

    @cached_property
    def expected_columns(self):
        types = ['info', 'bruise', 'response', 'lab', 'xray', 'video']
        dfs = [self.__getattribute__(f"{type}_vector") for type in types]
        col_names = [df.columns.tolist() for df in dfs]
        return {type: names for type, names in zip(types, col_names)}


class ParseGrowth:
    def __init__(self, patient_age: int, patient_sex: int, patient_height: float, patient_weight: float):
        self.patient_age = int(patient_age) # Patient age must be given in months
        self.patient_sex = int(patient_sex) # 0 for male, 1 for female
        self.patient_height = patient_height # given in cm
        self.patient_weight = patient_weight # given in kg

    @staticmethod
    def load_growth_data(patient_sex: int, data_type: str) -> pd.DataFrame:
        sex_list = ["male", "female"]
        assert data_type in ["height", "weight"], "Data type must be either 'height' or 'weight'"

        try:
            file_path = f"{str(GROWTHDIR)}/{data_type}_{sex_list[patient_sex]}.csv"
            growth_data = pd.read_csv(file_path)
            growth_data["Age(Months)"] = growth_data["Age(Months)"].astype(int)
            return growth_data

        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None

    def calculate_percentile(self, value: float, age_data: pd.DataFrame):
        percentiles = age_data.columns[1:].astype(
            float
        )
        values = age_data.iloc[self.patient_age, 1:].values.astype(float).tolist()

        if value <= values[0]:
            return 0.01*100
        elif value >= values[-1]:
            return 0.99*100

        percentile = 0

        for i, val in enumerate(values):
            if val > value:
                continue
            lower_bound = val
            upper_bound = values[i + 1]
            lower_percentile = percentiles[i]
            upper_percentile = percentiles[i + 1]
            percentile = lower_percentile + (
                (value - lower_bound) / (upper_bound - lower_bound)
            ) * (upper_percentile - lower_percentile)
            break

        return round(percentile, 2)

    def get_percentiles(self):
        height_data = self.load_growth_data(self.patient_sex, "height")
        weight_data = self.load_growth_data(self.patient_sex, "weight")

        if height_data is None or weight_data is None:
            return None, None

        height_percentile = self.calculate_percentile(self.patient_height, height_data)
        weight_percentile = self.calculate_percentile(self.patient_weight, weight_data)

        return self.patient_age, self.patient_sex, height_percentile, weight_percentile
