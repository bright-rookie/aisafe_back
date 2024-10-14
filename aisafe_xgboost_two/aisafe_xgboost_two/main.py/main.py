import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.optimize import minimize

# CSV 파일을 읽고 첫 번째 행을 변수명으로 사용
info_data = pd.read_csv("./final_files/Basic_information.CSV" )
bruise_data = pd.read_csv("./final_files/Bruise.CSV" )
response_data = pd.read_csv("./final_files/Medical_examination.CSV" )
lab_data = pd.read_csv("./final_files/EMR_lab.CSV" )
xray_data = pd.read_csv("./final_files/Fractures.CSV" )
video_data = pd.read_csv("./final_files/emotion.CSV" )
true_data = pd.read_csv("./final_files/true_label.CSV" )

# 벡터별로 컬럼
info_vector_pre = info_data.iloc[:, 1:5]  # column 2-5
bruise_vector = bruise_data.iloc[:, 1:12]  # column 2-12
response_vector = response_data.iloc[:, 1:10]  # column 2-10
lab_vector = lab_data.iloc[:, 1:20]  # column 2-20
xray_vector = xray_data.iloc[:, 1:10] # column 2-10
video_vector = video_data.iloc[:, 1:31] # column 2-31

y = true_data.iloc[:, 1]

# 표준 성장 데이터 로드 함수
def load_growth_data(sex, data_type):
    # 성별과 데이터 타입에 따라 파일 선택
    if sex == 0 and data_type == "height":
        file_path = "./csv/height_male.csv"
    elif sex == 1 and data_type == "height":
        file_path = "./csv/height_female.csv"
    elif sex == 0 and data_type == "weight":
        file_path = "./csv/weight_male.csv"
    elif sex == 1 and data_type == "weight":
        file_path = "./csv/weight_female.csv"
    else:
        return None

    # 해당 CSV 파일 로드
    return pd.read_csv(file_path)

# 환자의 값을 퍼센타일 구간에 맞추어 선형 보간법으로 계산하는 함수
def calculate_percentile(value, age_data):
    percentiles = age_data.columns[1:].astype(
        float
    )  # 퍼센타일 구간 (1%, 3%, 5%, ...)

    # 퍼센타일 값에 해당하는 데이터 (height 또는 weight)
    values = age_data.iloc[0, 1:].values.astype(float)

    # 만약 주어진 값이 값의 최소값보다 작으면 1% 미만
    if value <= values[0]:
        return 1

    # 만약 주어진 값이 값의 최대값보다 크면 최대 퍼센타일 이상
    if value >= values[-1]:
        return 99

    # 두 값 사이에서 선형 보간법 적용
    for i in range(len(values) - 1):
        if values[i] <= value <= values[i + 1]:
            # 선형 보간법 공식 적용
            lower_bound = values[i]
            upper_bound = values[i + 1]
            lower_percentile = percentiles[i]
            upper_percentile = percentiles[i + 1]

            # 선형 보간 계산
            percentile = lower_percentile + (
                (value - lower_bound) / (upper_bound - lower_bound)
            ) * (upper_percentile - lower_percentile)
            return round(percentile, 2)

    return None

# 환자의 키/체중 퍼센타일을 계산하는 함수
def get_percentiles(patient_age, patient_sex, patient_height, patient_weight):
    # 표준 데이터 로드
    height_data = load_growth_data(patient_sex, "height")
    weight_data = load_growth_data(patient_sex, "weight")

    if height_data is None or weight_data is None:
        st.error("성장 데이터 파일을 찾을 수 없습니다.")
        return None, None

    # 연령에 따른 데이터 필터링 (데이터 타입을 명시적으로 정수로 변환하여 비교)
    height_data["Age(Months)"] = height_data["Age(Months)"].astype(int)
    weight_data["Age(Months)"] = weight_data["Age(Months)"].astype(int)
    patient_age = int(patient_age)  # 데이터 타입 일치

    # 필터링 후 데이터 확인
    filtered_height = height_data[height_data["Age(Months)"] == patient_age]
    filtered_weight = weight_data[weight_data["Age(Months)"] == patient_age]

    # 환자의 키와 체중 퍼센타일 계산
    height_percentile = calculate_percentile(patient_height, filtered_height)
    weight_percentile = calculate_percentile(patient_weight, filtered_weight)

    return height_percentile, weight_percentile


# 변환된 결과를 저장할 리스트 초기화
info_vector_list = []

# 각 row마다 퍼센타일을 계산하여 변환
for _, row in info_vector_pre.iterrows():
    # row의 각 column을 순서대로 patient 정보로 할당
    patient_age = row[0]
    patient_sex = int(row[1])
    patient_height = row[2]
    patient_weight = row[3]

    # 퍼센타일 계산 함수 호출
    height_percentile, weight_percentile = get_percentiles(patient_age, patient_sex, patient_height, patient_weight)

    # 변환된 데이터 형태로 리스트에 추가
    info_vector_list.append([patient_age, patient_sex, height_percentile, weight_percentile])

# 변환된 결과를 DataFrame으로 변환
info_vector = pd.DataFrame(info_vector_list, columns=['patient_age', 'patient_sex', 'height_percentile', 'weight_percentile'])


# 패딩된 벡터들을 데이터프레임으로 결합
X = pd.concat([pd.DataFrame(info_vector), pd.DataFrame(bruise_vector),
               pd.DataFrame(response_vector), pd.DataFrame(lab_vector), pd.DataFrame(xray_vector), pd.DataFrame(video_vector)], axis=1)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# 데이터 형식 확인 및 변환
X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)

# numpy 배열로 변환
X_train_np = X_train.values  # 또는 X_train.to_numpy()
y_train_np = y_train.values

# DMatrix로 변환
dtrain = xgb.DMatrix(X_train_np, label=y_train_np)
dtest = xgb.DMatrix(X_test.values, label=y_test.values)


# 모델 파라미터 설정
params = {
    'objective': 'binary:logistic',
    'max_depth': 5,
    'eta': 0.1,
    'verbosity': 0
}
num_round = 500

# 모델 학습
trained_model = xgb.train(params, dtrain, num_round)


def model(info, bruise, response, lab, xray, video) :
    new_info_data = pd.DataFrame(info)
    new_bruise_data = pd.DataFrame(bruise)
    new_response_data = pd.DataFrame(response)
    new_lab_data = pd.DataFrame(lab)
    new_xray_data = pd.DataFrame(xray)
    new_video_data = pd.DataFrame(video)

    concat = np.concatenate((new_info_data, new_bruise_data, new_response_data, new_lab_data, new_xray_data, new_video_data))

    dnew_concat = xgb.DMatrix(concat)

    final_pred = trained_model.predict(dnew_concat)[0]

    # Extract the top 5 causes from the feature importances
    importance = trained_model.get_score(importance_type='weight')

    # Normalize the importance values to sum to 1 for involvement rate calculation


    all_features = list(importance.keys())
    importance_weights = np.array([importance.get(f, 0) for f in all_features])

    # 입력 벡터와 element-wise 곱하기
    input_vector = np.array(concat)
    contributions = {f: input_vector[i] * importance_weights[i] for i, f in enumerate(all_features)}
    total_contributions = sum(contributions.values())
    normalized_contributions = {k: v / total_contributions for k, v in contributions.items()}

    # 기여도에 따라 피처 정렬
    sorted_contributions = sorted(normalized_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_5_causes = sorted_contributions[:5]

    # Map feature indices to actual input vector names for interpretation
    feature_names = ['age', 'sex', 'height', 'weight', 'head_count', 'head_length', 'arms_count', 'arms_length', 'legs_count', 'legs_length',
                    'torso_count', 'torso_length', 'buttocks_count', 'buttocks_length', 'shape_abnormality', 'consciousness', 'guardian_status',
                    'abuse_likely', 'match_explanation', 'developemental_stage', 'treatment_delayed', 'consistent_history', 'poor_condition', 'inappropriate_relationship',
                    'CBC_RBC', 'CBC_WBC', 'CBC_Platelet', 'Hb', 'PT_INR', 'aPTT', 'AST', 'ALT', 'ALP', 'Na', 'K', 'C',
                    'Calcium', 'Phosphorus', '25hydroxyvitaminD', 'Serum_albumin', 'Pre_albumin', 'Transferrin', 'Glucose'
                    'Skull', 'Rib', 'Humerus', 'Radius_Ulna', 'Femur', 'Tibia_Fibula', 'Spiral_fx', 'Metaphyseal_fx',
                    'Happiness1', 'Sadness1', 'Anger1', 'Surprise1', 'Fear1', 'Happiness2', 'Sadness2', 'Anger2', 'Surprise2', 'Fear2',
                    'Happiness3', 'Sadness3', 'Anger3', 'Surprise3', 'Fear3', 'Happiness3', 'Sadness3', 'Anger3', 'Surprise3', 'Fear3'
                    'Happiness5', 'Sadness5', 'Anger5', 'Surprise5', 'Fear5', 'Happiness6', 'Sadness6', 'Anger6', 'Surprise6', 'Fear6'
                    ]

    # Create abuse_cause vector with top 5 features and normalize involvement rates
    explainability = []
    for feature, importance in top_5_causes:
        # Extract the index from feature (e.g., 'f0' -> 0)
        feature_index = int(feature[1:])  # Remove 'f' and convert to int
        feature_name = feature_names[feature_index]  # Map to human-readable feature name
        involvement_rate = round(importance.item(), 3)  # Normalize to 0-1 range and round for better readability
        explainability.append((feature_name, involvement_rate))

    return [final_pred, explainability]
