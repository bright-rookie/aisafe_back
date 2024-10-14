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



# 전체 데이터에서 벡터 결합 (벡터별로 분리된 상태 유지)
X = pd.concat([info_vector, bruise_vector, response_vector, lab_vector, xray_vector, video_vector], axis=1)

# 전체 데이터에서 train_test_split 수행
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# 분할된 X_train과 X_test에서 벡터별로 데이터 추출
X_train_info = X_train.iloc[:, 1:5].values
X_train_bruise = X_train.iloc[:, 5:16].values
X_train_response = X_train.iloc[:, 16:25].values
X_train_lab = X_train.iloc[:, 25:44].values
X_train_xray = X_train.iloc[:, 44:53].values
X_train_video = X_train.iloc[:, 53:83].values

X_test_info = X_test.iloc[:, 1:5].values
X_test_bruise = X_test.iloc[:, 5:16].values
X_test_response = X_test.iloc[:, 16:25].values
X_test_lab = X_test.iloc[:, 25:44].values
X_test_xray = X_test.iloc[:, 44:53].values
X_test_video = X_test.iloc[:, 53:83].values

# 각 벡터별 DMatrix 생성
dtrain_info = xgb.DMatrix(X_train_info, label=y_train)
dtest_info = xgb.DMatrix(X_test_info, label=y_test)
dtrain_bruise = xgb.DMatrix(X_train_bruise, label=y_train)
dtest_bruise = xgb.DMatrix(X_test_bruise, label=y_test)
dtrain_response = xgb.DMatrix(X_train_response, label=y_train)
dtest_response = xgb.DMatrix(X_test_response, label=y_test)
dtrain_lab = xgb.DMatrix(X_train_lab, label=y_train)
dtest_lab = xgb.DMatrix(X_test_lab, label=y_test)
dtrain_xray = xgb.DMatrix(X_train_xray, label=y_train)
dtest_xray = xgb.DMatrix(X_test_xray, label=y_test)
dtrain_video = xgb.DMatrix(X_train_video, label=y_train)
dtest_video = xgb.DMatrix(X_test_video, label=y_test)

# 모델 파라미터 설정
params = {'objective': 'binary:logistic', 'max_depth': 5, 'eta': 0.1}
num_round = 500

# 개별 모델 학습
model_info = xgb.train(params, dtrain_info, num_round)
model_bruise = xgb.train(params, dtrain_bruise, num_round)
model_response = xgb.train(params, dtrain_response, num_round)
model_lab = xgb.train(params, dtrain_lab, num_round)
model_xray = xgb.train(params, dtrain_xray, num_round)
model_video = xgb.train(params, dtrain_video, num_round)

# 개별 모델 예측
y_pred_info = model_info.predict(dtest_info)
y_pred_bruise = model_bruise.predict(dtest_bruise)
y_pred_response = model_response.predict(dtest_response)
y_pred_lab = model_lab.predict(dtest_lab)
y_pred_xray = model_xray.predict(dtest_xray)
y_pred_video = model_video.predict(dtest_video)

# 예측 값들을 리스트로 저장
predictions = [y_pred_info, y_pred_bruise, y_pred_response, y_pred_lab, y_pred_xray, y_pred_video]

# 초기 가중치 (각 가중치의 합이 1이 되도록 초기값 설정)
initial_weights = np.array([1/6] * 6)

# RMSE를 최소화하는 최적화 함수 정의
def rmse_loss(weights):
    weighted_pred = sum(w * pred for w, pred in zip(weights, predictions))
    rmse = mean_squared_error(y_test, weighted_pred, squared=False)
    return rmse

# 제약 조건: 가중치의 합이 1이 되도록 설정
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1)] * 6

# 최적화 수행
result = minimize(rmse_loss, initial_weights, bounds=bounds, constraints=constraints)

# 최적 가중치 확인
optimal_weights = result.x



def model(info, bruise, response, lab, xray, video) :
    new_info_data = pd.DataFrame(info)
    new_bruise_data = pd.DataFrame(bruise)
    new_response_data = pd.DataFrame(response)
    new_lab_data = pd.DataFrame(lab)
    new_xray_data = pd.DataFrame(xray)
    new_video_data = pd.DataFrame(video)

    # DMatrix로 변환
    dnew_info = xgb.DMatrix(new_info_data)
    dnew_bruise = xgb.DMatrix(new_bruise_data)
    dnew_response = xgb.DMatrix(new_response_data)
    dnew_lab = xgb.DMatrix(new_lab_data)
    dnew_xray = xgb.DMatrix(new_xray_data)
    dnew_video = xgb.DMatrix(new_video_data)

    # 개별 모델로 예측 수행
    pred_info = model_info.predict(dnew_info)
    pred_bruise = model_bruise.predict(dnew_bruise)
    pred_response = model_response.predict(dnew_response)
    pred_lab = model_lab.predict(dnew_lab)
    pred_xray = model_xray.predict(dnew_xray)
    pred_video = model_video.predict(dnew_video)

        # 예측 값 리스트 생성
    predictions = [pred_info[0], pred_bruise[0], pred_response[0], pred_lab[0], pred_xray[0], pred_video[0]]

    # 최적 가중치를 적용하여 최종 예측 계산
    final_pred = sum(w * pred for w, pred in zip(optimal_weights, predictions))
    final_pred_sub = [w * pred for w, pred in zip(optimal_weights, predictions)]
    final_pred_sub_rounded = []
    for element in final_pred_sub :
      final_pred_sub_rounded.append(round(element, 3))

    prediction_names = np.array(['신체 계측치', '멍 정보', '문진 정보', 'Lab 수치', 'X-ray 영상', '진료 영상'])

    sorted_indices = np.argsort(final_pred_sub)[::-1]
    sorted_weights = np.array(final_pred_sub_rounded)[sorted_indices]
    sorted_names = prediction_names[sorted_indices]

    explainability = list(zip(sorted_names, sorted_weights))

    return [final_pred, explainability]
    