import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# CSV 파일 읽기
file_path = "./20250312/data_Mono12_Row_20250312_100304.csv"
data = pd.read_csv(file_path)

# x와 y 값 추출
x = data['cm^-1'].values
y = data['intensity'].values

# 보간 함수 생성 (선형 보간)
interp_function = interp1d(x, y, kind='linear', fill_value="extrapolate")

x_new = np.arange(x.min(), x.max(), 0.001)
x_combined = np.unique(np.concatenate((x, x_new)))  # 기존 x값과 새로운 x값 병합

# 보간된 y 값 계산
y_combined = interp_function(x_combined)  # 보간된 y 값 계산

# y의 최대값에서 50% 계산
half_max = max(y_combined) * 0.5

# y = half_max와 교차점 계산 (선형 보간 방식)
intersection_xs = []
for i in range(len(y_combined) - 1):
    if (y_combined[i] - half_max) * (y_combined[i + 1] - half_max) < 0:
        # 선형 보간 계산
        x_interp = x_combined[i] + (x_combined[i + 1] - x_combined[i]) * ((half_max - y_combined[i]) / (y_combined[i + 1] - y_combined[i]))
        intersection_xs.append(x_interp)

# 시작점과 끝점 계산
if len(intersection_xs) >= 2:
    intersection_start = intersection_xs[0]  # 첫 교차점
    intersection_end = intersection_xs[-1]  # 마지막 교차점
    distance = intersection_end - intersection_start
else:
    intersection_start = intersection_end = distance = None

# 보간 데이터 시각화
plt.figure(figsize=(8, 6))
plt.plot(x_combined, y_combined, 'o', label="Interpolated Points (0.001 step)", alpha=0.8, color='green', markersize=0.1)
# plt.plot(x, y,'o' ,alpha=0.8, color='red', markersize=3)
plt.axhline(half_max, color='red', linestyle='--', label=f"Half Maximum ({half_max:.4f})")
if intersection_start is not None and intersection_end is not None:
    plt.axvline(intersection_start, color='blue', linestyle='--', label=f"Intersection Start ({intersection_start:.4f})")
    plt.axvline(intersection_end, color='purple', linestyle='--', label=f"Intersection End ({intersection_end:.4f})")
    # 두 교차점 사이를 선으로 연결하여 표시
    plt.plot([intersection_start, intersection_end], [half_max, half_max], color='black', linestyle='-', label=f"Distance ({distance:.4f})")
plt.title("Original + Interpolated Data with Half Maximum")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# 결과 출력
print(f"Intersection Start: {intersection_start:.4f}")
print(f"Intersection End: {intersection_end:.4f}")
print(f"Distance between intersections: {distance:.4f}")

# 원본 데이터의 y 최대값
original_y_max = y.max()

# 보간된 데이터의 y 최대값
interpolated_y_max = y_combined.max()

# 두 값의 차이 계산
difference = abs(original_y_max - interpolated_y_max)

print(f"Original y max: {original_y_max}")
print(f"Interpolated y max: {interpolated_y_max}")
print(f"Difference: {difference}")
