import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 1. 이미지 데이터 불러오기
image = cv2.imread('capture_Mono8_Row_20250116_162222.bmp', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Image not found. Check the file path.")
    exit()

# 2. 데이터 추출 (100개의 row 합산)
start_row = max(0, image.shape[0] // 2 - 50)
end_row = min(image.shape[0], image.shape[0] // 2 + 50)
intensity = np.sum(image[start_row:end_row, :], axis=0)
x_values = np.arange(image.shape[1])
x_index = np.linspace(950, 750, len(x_values))  # X축 감소 방향 설정

# X축 감소 방향에 맞게 데이터 정렬
sorted_indices = np.argsort(x_index)[::-1]  # 내림차순 정렬
x_index = x_index[sorted_indices]
intensity = intensity[sorted_indices]

# 3. 사용자 정의 Baseline 설정
plt.figure(figsize=(10, 6))
plt.xlim([950, 750])  # X축 감소 방향
plt.plot(x_index, intensity, label='Intensity', color='blue')
plt.xlabel('Pixel Position (x-axis)')
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

# 4. Baseline 계산 (스플라인 보간)
spline = make_interp_spline(baseline_x, baseline_y, k=3)  # 3차 스플라인 보간
baseline_fit = spline(x_index)  # X축 감소 방향에 맞게 보간

# 5. 데이터 보정
corrected_intensity = intensity - baseline_fit

# 6. 보정된 데이터 시각화
plt.figure(figsize=(10, 6))
plt.xlim([950, 750])  # X축 감소 방향
plt.plot(x_index, intensity, label='Original Intensity', color='blue', alpha=0.1)
plt.plot(x_index, baseline_fit, label='Baseline', color='green', alpha=0.1)
plt.plot(x_index, corrected_intensity, label='Corrected Intensity', color='red')
plt.xlabel('Pixel Position (x-axis)')
plt.ylabel('Intensity (a.u.)')
plt.title('Baseline Correction with Spline')
plt.savefig('baseline_corrected_graph.png')
plt.legend()
plt.show()