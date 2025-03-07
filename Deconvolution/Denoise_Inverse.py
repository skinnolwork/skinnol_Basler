import numpy as np
import tifffile
import matplotlib.pyplot as plt

# 📌 1D 가우시안 PSF (세로 방향) 생성 (정규화 추가)
def vertical_gaussian_psf(height, sigma):
    y = np.linspace(-height // 2, height // 2, height)
    psf = np.exp(-y**2 / (2 * sigma**2))
    psf /= np.sum(psf)  # ✅ PSF 정규화 추가
    return psf[:, np.newaxis]

# 📌 역 필터 적용 (PSF 정규화 및 FFT 안정화 추가)
def inverse_filter(image, psf):
    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(psf, s=image.shape)

    # ✅ PSF FFT 값 정규화 (값이 너무 커지는 문제 방지)
    psf_fft /= np.max(np.abs(psf_fft))

    # ✅ Divide by Zero 방지
    psf_fft[np.abs(psf_fft) < 1e-8] = 1e-8  

    # ✅ 역 필터 적용
    deconv_fft = image_fft / psf_fft
    deconvolved = np.fft.ifft2(deconv_fft).real  # 실수부만 취함

    # ✅ 디컨볼루션 후 값 확인
    print(f"Deconvolved min: {np.min(deconvolved)}, max: {np.max(deconvolved)}")

    # ✅ 정규화 적용 (12-bit 스케일 조정)
    deconvolved -= np.min(deconvolved)  # 음수 제거
    deconvolved = (deconvolved / np.max(deconvolved)) * 4095  # 12-bit 범위로 정규화

    return deconvolved

# 📌 12-bit TIFF 이미지 불러오기
image_path = "original_Mono12_20250226_151813.tiff"
blurred_image = tifffile.imread(image_path).astype(np.float32)

# 📌 PSF 크기 조정 (실제 이미지 크기에 맞게 자동 설정)
psf_height = blurred_image.shape[0] // 10  # 이미지 크기의 1/10 정도로 설정
psf_sigma = max(5, psf_height // 5)

psf = vertical_gaussian_psf(psf_height, psf_sigma)

# 📌 PSF를 이미지 크기로 확장
psf_2d = np.zeros_like(blurred_image)
psf_2d[:psf_height, :blurred_image.shape[1]] = np.tile(psf, (1, blurred_image.shape[1]))
psf_2d = np.fft.fftshift(psf_2d)  # ✅ `ifftshift()` 대신 `fftshift()` 사용

# 📌 디컨볼루션 적용 (순수한 FFT 역 컨볼루션)
deconvolved_image = inverse_filter(blurred_image, psf_2d)

# 📌 12-bit 값 변환 (4095 범위 유지)
deconvolved_image_uint16 = np.clip(deconvolved_image, 0, 4095).astype(np.uint16)

# 📌 결과 저장 (TIFF 포맷)
output_path = "Deconvolved_Mono12_12bit_Fixed.tiff"
tifffile.imwrite(output_path, deconvolved_image_uint16)

# 📌 결과 시각화 (12-bit 이미지를 8-bit로 변환해서 보기)
visual_image = (deconvolved_image / 4095 * 255).astype(np.uint8)

fig, ax = plt.subplots(1, 2, figsize=(15, 7))
ax[0].imshow(blurred_image / 4095, cmap='gray', vmin=0, vmax=1)
ax[0].set_title("Original 12-bit TIFF Image")
ax[0].axis("off")

ax[1].imshow(visual_image, cmap='gray', vmin=0, vmax=255)
ax[1].set_title("Deconvolved 12-bit TIFF Image (Fixed)")
ax[1].axis("off")

plt.show()

print(f"Deconvolved TIFF image saved at: {output_path}")
