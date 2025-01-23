import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

# CSV 파일 경로
file_path = 'data_Mono12_Row_20250123_093123.csv'

# 데이터 로드
data = pd.read_csv(file_path)

# X와 Y 값 추출
x_values = data['X'].values  # Wavelength
intensity = data['Y'].values  # Intensity

# 스무딩 적용 (Savitzky-Golay 필터)
window_length = 51  # 윈도우 크기 (데이터 길이보다 작아야 함)
polyorder = 1       # 다항식 차수
smoothed_intensity = savgol_filter(intensity, window_length=window_length, polyorder=polyorder)

distance = 30
prominence_ratio = 0.2
threshold_index = int(0.3 * len(smoothed_intensity))
dynamic_height = np.sort(smoothed_intensity)[threshold_index]
peaks, _ = find_peaks(
    smoothed_intensity,
    prominence=prominence_ratio * dynamic_height,
    height=dynamic_height,
    distance=distance
)

# peaks = np.delete(peaks, [1])


# 스무딩된 데이터와 피크 시각화
plt.figure(figsize=(12, 6))
plt.plot(x_values, smoothed_intensity, label='Smoothed Intensity', color='red', linewidth=2)
plt.scatter(x_values[peaks], smoothed_intensity[peaks], color='green', s=50, label='Detected Peaks', zorder=5)
plt.xlabel('Wavelength (X)')
plt.ylabel('Intensity (Y)')
plt.title('Peak Detection on Smoothed Intensity')
plt.legend()
plt.grid()
plt.show()