import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1. 이미지 데이터 불러오기
image = cv2.imread('graph_Mono8_Row_20250106_162355.bmp', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Image not found. Check the file path.")
    exit()

# 2. 데이터 추출 (100개의 row 합산)
start_row = max(0, image.shape[0] // 2 - 50)
end_row = min(image.shape[0], image.shape[0] // 2 + 50)
intensity = np.sum(image[start_row:end_row, :], axis=0)
x_values = np.arange(image.shape[1])

# 3. 낙폭 기준 추가 함수
# def filter_peaks_by_falloff(peaks, intensity, falloff_threshold=0.5, check_range=80):
#     valid_peaks = []
#     for peak in peaks:
#         # 피크 이후 검사 범위 내 최소값 계산
#         end_idx = min(len(intensity), peak + check_range)
#         post_peak_min = np.min(intensity[peak + 1:end_idx])  # 피크 이후 최소값

#         # 낙폭 기준 조건 확인
#         if post_peak_min > falloff_threshold * intensity[peak]:
#             valid_peaks.append(peak)

#     return np.array(valid_peaks)

# 4. 원본 데이터에서 피크 탐지
# 히스토그램으로 동적 threshold 계산
hist, bin_edges = np.histogram(intensity, bins=50)
threshold_index = int(0.2 * len(intensity))  # 상위 20%
dynamic_height = np.sort(intensity)[threshold_index]

# 원본 데이터에서 피크 탐지
peaks_original, properties_original = find_peaks(
    intensity,
    prominence=0.5 * dynamic_height,  # 두드러짐 기준
    height=dynamic_height,           # 동적 height
    distance=50                      # 최소 거리
)

# 낙폭 기준으로 피크 필터링
# filtered_peaks = filter_peaks_by_falloff(peaks_original, intensity)

# 5. 하이라이트 영역만 남기기
highlighted_intensity = np.zeros_like(intensity)  # 초기화
for peak in peaks_original:
    highlighted_intensity[peak] = intensity[peak]  # 피크 위치에만 값 유지

x_index = np.linspace(950, 750, len(x_values))

# 6. 그래프 시각화
plt.figure(figsize=(10, 6))
plt.plot(x_index, intensity, label='Original Intensity', color='blue')
plt.plot(x_index, highlighted_intensity, label='Highlighted Intensity', color='red')

# 피크 위치 텍스트로 표시
for peak in peaks_original:
    plt.text(x_index[peak], intensity[peak] + 10, f"{x_index[peak]:.2f}", color='green', fontsize=8, ha='center')

plt.xlim([950, 750])  # X축 범위 강제 설정
plt.xlabel('Pixel Position (x-axis)')
plt.ylabel('Intensity (a.u.)')
plt.title('Original Intensity with Highlighted Peaks')
plt.legend()
plt.savefig("XRD_XPS_Origin_Filtered.png", dpi=300)

# 7. 그래프 시각화 (파란색 그래프 제외)
plt.figure(figsize=(10, 6))
plt.plot(x_index, highlighted_intensity, label='Highlighted Intensity', color='red')

# 피크 위치 텍스트로 표시
for peak in peaks_original:
    plt.text(x_index[peak], highlighted_intensity[peak] + 10, f"{x_index[peak]:.2f}", color='green', fontsize=8, ha='center')

plt.xlim([950, 750])  # X축 범위 강제 설정
plt.xlabel('Pixel Position (x-axis)')
plt.ylabel('Intensity (a.u.)')
plt.title('Highlighted Peaks Only (Filtered)')
plt.legend()
plt.savefig("XRD_XPS_HightLight_Filtered.png", dpi=300)

plt.show()
