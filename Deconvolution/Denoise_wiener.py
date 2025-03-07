import numpy as np
import tifffile  # TIFF 파일을 읽기/저장하기 위한 라이브러리
import matplotlib.pyplot as plt
from skimage import restoration
import datetime

# 📌 1D 가우시안 PSF (세로 방향) 생성
def vertical_gaussian_psf(height, sigma):
    y = np.linspace(-height // 2, height // 2, height)
    psf = np.exp(-y**2 / (2 * sigma**2))
    psf /= psf.sum()
    psf = psf[:, np.newaxis]  # 세로 방향 PSF 생성
    return psf

# 📌 12-bit TIFF 이미지 불러오기
image_path = "original_Mono12_20250304_112653.tiff"
blurred_image = tifffile.imread(image_path)

# 📌 12-bit 정규화 (0~1 범위)
if blurred_image.dtype == np.uint16:
    blurred_image = blurred_image.astype(np.float32) / 4095.0  

# 📌 PSF 크기 및 블러 강도 조정
psf_height = 200  # 기존 200 → 100으로 변경
psf_sigma = 5  # 기존 3 → 10으로 증가
psf = vertical_gaussian_psf(psf_height, psf_sigma)

# 📌 Wiener 필터 적용 (balance 값 증가)
balance = 0.02  # 기존 0.1 → 0.02로 조정 (노이즈 억제)
deconvolved_image = restoration.wiener(blurred_image, psf, balance)

# 📌 밝기 조정 및 TIFF 저장
deconvolved_image = np.clip(deconvolved_image, 0, 1)  # 값 범위 조정
output_filename = f"Deconvolved_wiener_Mono12_20250304_112653.tiff"
tifffile.imwrite(output_filename, (deconvolved_image * 4095).astype(np.uint16))  # 12-bit 저장

# 📌 결과 시각화
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(blurred_image, cmap='gray')
ax[0].set_title("Blurred Image")
ax[1].imshow(deconvolved_image, cmap='gray')
ax[1].set_title("Deconvolved Image (Wiener Filter)")
plt.show()

print(f"Deconvolved image saved at: {output_filename}")
