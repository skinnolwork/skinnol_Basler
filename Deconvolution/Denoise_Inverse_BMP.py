import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# 📌 1D 가우시안 PSF (세로 방향) 생성
def vertical_gaussian_psf(height, sigma):
    y = np.linspace(-height // 2, height // 2, height)
    psf = np.exp(-y**2 / (2 * sigma**2))
    return psf[:, np.newaxis]  # 정규화 X

# 📌 역 필터 적용 (PSF 조정이 반영되도록 변경)
def inverse_filter(image, psf):
    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(psf, s=image.shape)

    # 너무 작은 값 방지 (0 나누기 방지) - 기존 0.2 → 1e-3로 변경
    psf_fft[np.abs(psf_fft) < 0.4] = 0.4

    # 단순 역 필터 적용
    deconv_fft = image_fft / psf_fft
    deconvolved = np.fft.ifft2(deconv_fft).real  # 실수부만 취함
    return deconvolved

# 📌 8-bit BMP 이미지 불러오기
image_path = "original_Mono8_20250305_162122.bmp"
blurred_image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)

# 📌 PSF 크기 조정 (더 큰 값으로 테스트)
psf_height = 10  # 기존 20 → 200 (PSF가 영향 미치도록 변경)
psf_sigma = 1   # 기존 10 → 50 (블러 효과가 반영되도록 변경)

psf = vertical_gaussian_psf(psf_height, psf_sigma)

# 📌 PSF를 이미지 크기로 확장
psf_2d = np.zeros_like(blurred_image)
psf_2d[:psf_height, :blurred_image.shape[1]] = np.tile(psf, (1, blurred_image.shape[1]))
psf_2d = np.fft.ifftshift(psf_2d)  # PSF 위치 조정

# 📌 디컨볼루션 적용 (푸리에 변환 + 역 컨볼루션만 수행)
deconvolved_image = inverse_filter(blurred_image, psf_2d)

# 📌 8-bit 값 변환
deconvolved_image_uint8 = np.clip(deconvolved_image, 0, 255).astype(np.uint8)

# 📌 결과 저장 (BMP 포맷)
output_path = "Deconvolved_Mono8_8bit.bmp"
Image.fromarray(deconvolved_image_uint8).save(output_path)

# 📌 결과 시각화
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
ax[0].imshow(blurred_image, cmap='gray', vmin=0, vmax=255)
ax[0].set_title("Original 8-bit BMP Image")
ax[0].axis("off")

ax[1].imshow(deconvolved_image_uint8, cmap='gray', vmin=0, vmax=255)
ax[1].set_title("Deconvolved 8-bit BMP Image (Fixed)")
ax[1].axis("off")

plt.show()

print(f"Deconvolved BMP image saved at: {output_path}")
