import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import restoration
from skimage.restoration import denoise_tv_chambolle
import datetime
import tifffile

# 12-bit TIFF 이미지 불러오기
image_path = "original_Mono12_20250226_151813.tiff"
blurred_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 12-bit 유지

# 12-bit 범위를 고려하여 정규화 (0~1 사이로 변환)
blurred_image = blurred_image.astype(np.float32) / 4096.0

# 세로 방향 1D 가우시안 PSF 생성
def vertical_gaussian_psf(height, sigma):
    y = np.linspace(-(height // 2), height // 2, height)
    psf = np.exp(- (y ** 2) / (2 * sigma ** 2))  # 세로 방향 가우시안
    psf /= np.sum(psf)  # 정규화
    psf = psf[:, np.newaxis]  # (height, 1) 형태로 변환 (세로 방향만 적용)
    return psf

# PSF 크기 및 시그마 설정
height = 10 # 세로 방향 PSF 크기
sigma = 5  # 가우시안 블러 표준편차
psf = vertical_gaussian_psf(height, sigma)

# Richardson-Lucy 디컨볼루션 적용
num_iterations = 20  # 반복 횟수 증가
deconvolved_image = restoration.richardson_lucy(blurred_image, psf, num_iterations)

# 디컨볼루션 후 12-bit 범위로 복원
deconvolved_image = cv2.normalize(deconvolved_image, None, 0, 4096, cv2.NORM_MINMAX)

# 결과 시각화 (밝기 조정 추가)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(blurred_image, cmap='gray', vmin=0, vmax=1)  # 원본 정규화된 이미지
ax[0].set_title("Blurred Image (12-bit scaled to 0-1)")

ax[1].imshow(deconvolved_image, cmap='gray', vmin=0, vmax=4096)  # 밝기 복원된 이미지
ax[1].set_title("Deconvolved Image (12-bit restored)")

# 저장 코드 수정 (변수명 수정)
filename = f"Deconvolved_RL_Mono12_20250304_112653.tiff"
tifffile.imwrite(filename, deconvolved_image.astype(np.uint16))  # 16-bit TIFF로 저장
print(f"Deconvolved image saved as {filename}")

plt.show()
