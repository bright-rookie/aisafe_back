<Input 벡터>

0. patient_number(int)
1. Basic information: csv input (col 2-5)
=> info_vector = [patient_age, patient_sex, height, weight] *앞에서부터 각각 int, 0(남자), 1(여자), Float, Float
2. Bruise: streamlit을 이용한 manual input (col 2-12)
=> bruise_vector = [head_count, head_length, arms_count, arms_length, legs_count, legs_length, torso_count, torso_length, buttocks_count, buttocks_length, specific_shape]*_count(int), _length(float), specific_shape(0 or 1)
3. History taking: streamlit을 이용한 manual input (col 2-10)
=> response_vector = [consciousness_state, guardian_status, abuse_likely, match_explanation, developemental_stage, treatment_delayed, consistent_history, poor_condition, inappropriate_relationship] *0(예) or 1(아니오) or None(유보)
4. Lab: csv input (col 2-20)
=> lab_vector = [CBC_RBC, CBC_WBC, CBC_Platelet, Hb, PT_INR, aPTT, AST, ALT, ALP, Na, K, Cl, Calcium, Phosphorus, 25hydroxyvitaminD, Serum_albumin, Pre_albumin, Transferrin, Glucose] *모두 float
5. X-ray assessment: txt input(여러 부위의 .txt형식 판독문을 합쳐서 하나의 .txt 파일로 input) (col 2-10)
=> xray_vector = [skull, ribs, humerus, radius_ulna, femur, tibia_fibula, pelvis, spiral_fx, metaphyseal_fx] *0(없음) or 1(1개/마지막 두 항목 - 있음) or 2 (여러 개)
6. Video/Audio: Video(.mp4) input (col 2-31)
=> emotion_vector = [Happiness1, Sadness1, Anger1, Surprise1, Fear1, ..., Surprise6, Fear6] *모두 float(0-1), 각 숫자당 5개 값 합 1
7. true_label : 0 (아동학대 아님), 1 (아동학대)

<Output 벡터> 

=> final_pred, explainability = [(원인1(str), 관여율1(float)), (원인2(str), 관여율2(float)), (원인3(str), 관여율3(float)), ...]
*현재 임의의 데이터 x_train, y_train을 이용해 training한 XGBoost 사용