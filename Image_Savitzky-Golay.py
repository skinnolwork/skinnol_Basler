import numpy as np
import imageio.v3 as iio
import scipy.signal
from PIL import Image
import matplotlib.pyplot as plt

# 파일 경로
input_tiff_path = "original_Mono12_20250226_151813.tiff"
output_tiff_path = "Denoise_Image.tiff"

# TIFF 이미지 로드
image = iio.imread(input_tiff_path).astype(np.float32)

# 사비츠키-골레이 필터 적용 (윈도우 크기=11, 차수=3)
window_size = 41  # 홀수여야 함
poly_order = 12   # 다항식 차수

filtered_image = np.zeros_like(image)

for row in range(image.shape[0]):
    filtered_image[row, :] = scipy.signal.savgol_filter(image[row, :], window_size, poly_order)

# # # TIFF 파일로 저장
filtered_image = np.clip(filtered_image, 0, 4095).astype(np.uint16)  # 12-bit TIFF 대응
Image.fromarray(filtered_image).save(output_tiff_path)

# 이미지 출력
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 원본 이미지 표시
axes[0].imshow(image, cmap='gray', aspect='auto')
axes[0].set_title("Original Image")
axes[0].axis("off")

# 필터링된 이미지 표시
axes[1].imshow(filtered_image, cmap='gray', aspect='auto')
axes[1].set_title("Filtered Image (Savitzky-Golay)")
axes[1].axis("off")

# 출력
plt.show()

# 필터링 정보 출력
print(f"Savitzky-Golay Filter Applied: Window Size={window_size}, Polynomial Order={poly_order}")
