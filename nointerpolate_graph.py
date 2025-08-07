import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일 읽기
file_path = "data_Mono12_Row_20250121_145322.csv"
data = pd.read_csv(file_path)

# x와 y 값 추출
x = data['x_original'].values
y = data['Intensity'].values

# y의 최대값에서 50% 계산
half_max = max(y) * 0.5

# y = half_max와의 교점 계산
intersection_xs = []
for i in range(len(x) - 1):
    if (y[i] - half_max) * (y[i + 1] - half_max) < 0:
        # 점과 점을 잇는 직선의 방정식으로 교점 계산
        x_interp = x[i] + (x[i + 1] - x[i]) * (half_max - y[i]) / (y[i + 1] - y[i])
        intersection_xs.append(x_interp)

# 시작점과 끝점 계산
if len(intersection_xs) >= 2:
    intersection_start = intersection_xs[0]  # 첫 교차점
    intersection_end = intersection_xs[-1]  # 마지막 교차점
    distance = intersection_end - intersection_start
else:
    intersection_start = intersection_end = distance = None

# 시각화
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Original Data", color='blue', markersize=3)  # 점을 연결한 선
plt.axhline(half_max, color='red', linestyle='--', label=f"Half Maximum ({half_max:.4f})")
if intersection_start is not None and intersection_end is not None:
    plt.axvline(intersection_start, color='green', linestyle='--', label=f"Intersection Start ({intersection_start:.4f})")
    plt.axvline(intersection_end, color='purple', linestyle='--', label=f"Intersection End ({intersection_end:.4f})")
    plt.plot([intersection_start, intersection_end], [half_max, half_max], color='black', linestyle='-', label=f"Distance ({distance:.4f})")
plt.title("Original Data with Half Maximum Intersection")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# 결과 출력
print(f"Intersection Start: {intersection_start:.4f}")
print(f"Intersection End: {intersection_end:.4f}")
print(f"Distance between intersections: {distance:.4f}")
