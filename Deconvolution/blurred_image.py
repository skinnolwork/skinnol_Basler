import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 📌 1D 가우시안 PSF (세로 방향) 생성
def vertical_gaussian_psf(height, sigma):
    y = np.linspace(-height // 2, height // 2, height)
    psf = np.exp(-y**2 / (2 * sigma**2))
    psf /= np.sum(psf)  # ✅ 합이 1이 되도록 정규화 (max가 아닌 sum 사용!)
    return psf[:, np.newaxis]

# 📌 역 필터 적용 (푸리에 역 컨볼루션)
def inverse_filter(image, psf, epsilon=1e-6):
    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(psf, s=image.shape)

    psf_fft[np.abs(psf_fft) < epsilon] = epsilon  # 0 나누기 방지
    deconv_fft = image_fft / psf_fft
    deconvolved = np.fft.ifft2(deconv_fft).real

    print("Deconvolved Image Min/Max before normalization:", np.min(deconvolved), np.max(deconvolved))
    deconvolved = (deconvolved - np.min(deconvolved)) / (np.max(deconvolved) - np.min(deconvolved)) * 255
    return np.clip(deconvolved, 0, 255)

# 📌 메인 실행 함수
def main():
    # 📌 8-bit BMP 이미지 불러오기
    image_path = "original_Mono8_20250305_161550.bmp"
    original_image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)

    # 📌 PSF 설정
    psf_sigma = 200
    psf = vertical_gaussian_psf(original_image.shape[0], psf_sigma)

    # ✅ PSF를 이미지 중앙 한 줄에만 적용
    psf_2d = np.zeros_like(original_image)
    psf_2d[:, original_image.shape[1] // 2] = psf[:, 0]  # 중앙 한 줄만 PSF 적용
    psf_2d = np.fft.ifftshift(psf_2d)  # 중심 이동

    # 🔍 **PSF 시각화**
    print(f"PSF Sum: {np.sum(psf_2d)} (should be 1.0)")
    plt.figure(figsize=(6, 5))
    plt.imshow(psf_2d, cmap='gray')
    plt.title(f"PSF (2D) - Sum: {np.sum(psf_2d)}")
    plt.colorbar()
    plt.show()

    # ✅ Blurred Image 생성
    blurred_image = np.fft.ifft2(np.fft.fft2(original_image) * np.fft.fft2(psf_2d, s=original_image.shape)).real

    # 🔥 블러링이 정상적으로 적용됐는지 확인
    print(f"Blurred Image Min/Max before normalization: {np.min(blurred_image)}, {np.max(blurred_image)}")

    # 🔥 0~255로 값 정규화
    blurred_image = (blurred_image - np.min(blurred_image)) / (np.max(blurred_image) - np.min(blurred_image)) * 255
    blurred_image = np.clip(blurred_image, 0, 255)

    # 🔍 **Blurred Image 확인**
    plt.figure(figsize=(6, 5))
    plt.imshow(blurred_image, cmap='gray')
    plt.title("Blurred Image (PSF 적용 후)")
    plt.colorbar()
    plt.show()

    # 📌 디컨볼루션 적용
    deconvolved_image = inverse_filter(blurred_image, psf_2d)

    # 📌 8-bit 값 변환
    deconvolved_image_uint8 = np.clip(deconvolved_image, 0, 255).astype(np.uint8)

    # 📌 결과 저장
    output_path = "Deconvolved_Mono8_8bit.bmp"
    Image.fromarray(deconvolved_image_uint8).save(output_path)

    # 📌 결과 시각화
    fig, ax = plt.subplots(1, 3, figsize=(15, 7))

    ax[0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
    ax[0].set_title("Original 8-bit BMP Image")
    ax[0].axis("off")

    ax[1].imshow(blurred_image, cmap='gray', vmin=0, vmax=255)
    ax[1].set_title("Blurred 8-bit BMP Image")
    ax[1].axis("off")

    ax[2].imshow(deconvolved_image_uint8, cmap='gray', vmin=0, vmax=255)
    ax[2].set_title("Deconvolved 8-bit BMP Image (Pure FFT)")
    ax[2].axis("off")

    plt.show()

    print(f"Deconvolved BMP image saved at: {output_path}")

# 📌 코드 실행
if __name__ == "__main__":
    main()
