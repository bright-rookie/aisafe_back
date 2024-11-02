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
    def load_growth_data(sex: int, data_type: str) -> pd.DataFrame:
        sex_list = ["male", "female"]
        assert data_type in ["height", "weight"], "Data type must be either 'height' or 'weight'"
        assert isinstance(sex, int), "Sex must be an integer (0 for male, 1 for female)"

        try:
            file_path = f"{str(GROWTHDIR)}/{data_type}_{sex_list[sex]}.csv"
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
