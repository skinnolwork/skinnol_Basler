import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# 데이터 로드
file_path = 'normalized_raman_data.csv'
data = pd.read_csv(file_path)

# 데이터 크기에 맞게 설정 (예: 1D -> 2D)
rows, cols = 3840, 1  # 데이터 크기 = 3840
intensity_data = data['Intensity'].values.reshape((rows, cols))

# X축과 Y축 전치 (3840 x 1 -> 1 x 3840)
transposed_data = intensity_data.T

# Y축 범위 확장: 데이터 반복
y_scale_factor = 2160  # Y축 확대 배율
expanded_data = np.repeat(transposed_data, y_scale_factor, axis=0)  # 데이터 복제

# 데이터를 0-255 범위로 정규화
normalized_data = 255 * (expanded_data - np.min(expanded_data)) / (np.max(expanded_data) - np.min(expanded_data))
normalized_data = normalized_data.astype(np.uint8)

# 그라데이션 마스크 생성 (Y축 방향으로 그라데이션)
gradient_mask = np.linspace(0, 1, normalized_data.shape[0]).reshape(-1, 1) ** 2
gradient_data = normalized_data * gradient_mask

# 0-255 범위로 다시 정규화
gradient_data = 255 * (gradient_data - np.min(gradient_data)) / (np.max(gradient_data) - np.min(gradient_data))
gradient_data = gradient_data.astype(np.uint8)

# Bitmap 표시
plt.imshow(gradient_data, cmap='gray', aspect='auto')
plt.title("Raman Bitmap with Gradient")
plt.colorbar(label='Intensity (Normalized with Gradient)')
plt.show()

# BMP 파일로 저장
image = Image.fromarray(gradient_data, mode='L')  # 'L'은 그레이스케일 모드
image.save("raman_bitmap_gradient.bmp")

print("BMP 파일이 저장되었습니다: raman_bitmap_with_gradient.bmp")