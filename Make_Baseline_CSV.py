import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 1. CSV 데이터 불러오기
file_path = 'data_Mono12_Row_20250124_120851.csv'  # CSV 파일 경로
data = pd.read_csv(file_path)

# X축 값과 Intensity 값 가져오기
x_values = data['cm^-1'].values  # 'X' 컬럼
intensity = data['intensity'].values  # 'Y' 컬럼

# X축 감소 방향으로 정렬
sorted_indices = np.argsort(x_values)[::-1]  # 내림차순 정렬
x_values = x_values[sorted_indices]
intensity = intensity[sorted_indices]

# 2. 사용자 정의 Baseline 설정
plt.figure(figsize=(10, 6))
plt.xlim([max(x_values), min(x_values)])  # X축 감소 방향
plt.plot(x_values, intensity, label='Intensity', color='blue')
plt.xlabel('X-axis')
plt.ylabel('Intensity (a.u.)')
plt.title('Select Baseline Points')
plt.legend()
# plt.show()

print("그래프에서 Baseline으로 사용할 포인트를 선택하세요 (엔터로 종료).")
baseline_points = plt.ginput(n=-1, timeout=0)  # 사용자로부터 포인트 입력 받기
baseline_points = np.array(baseline_points)

# 중복된 X 좌표 제거
baseline_x = baseline_points[:, 0]
baseline_y = baseline_points[:, 1]
unique_indices = np.unique(baseline_x, return_index=True)[1]  # 중복 제거
baseline_x = baseline_x[unique_indices]
baseline_y = baseline_y[unique_indices]

# 3. Baseline 계산 (스플라인 보간)
spline = make_interp_spline(baseline_x, baseline_y, k=3)  # 3차 스플라인 보간
baseline_fit = spline(x_values)  # X축 감소 방향에 맞게 보간

# 4. 데이터 보정
corrected_intensity = intensity - baseline_fit

# 5. 보정된 데이터와 Baseline 시각화
plt.figure(figsize=(10, 6))
plt.xlim([max(x_values), min(x_values)])  # X축 감소 방향
plt.plot(x_values, intensity, label='Original Intensity', color='blue', alpha=0.5)
plt.plot(x_values, baseline_fit, label='Baseline', color='green', linestyle='--')
plt.plot(x_values, corrected_intensity, label='Corrected Intensity', color='red')
plt.xlabel('X-axis')
plt.ylabel('Intensity (a.u.)')
plt.title('Baseline Correction with Spline (CSV Data)')
plt.legend()
plt.show()

# # 6. 결과 저장 (CSV 파일)
# output_file = 'corrected_data.csv'
# output_data = pd.DataFrame({'X': x_values, 'Original Intensity': intensity, 
#                             'Baseline': baseline_fit, 'Corrected Intensity': corrected_intensity})
# output_data.to_csv(output_file, index=False)
# print(f"Corrected data saved to {output_file}")
