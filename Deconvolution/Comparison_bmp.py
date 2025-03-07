import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from PIL import Image


def vertical_gaussian_psf(height, sigma):
    y = np.linspace(-height // 2, height // 2, height)
    psf = np.exp(-y**2 / (2 * sigma**2))
    return psf[:, np.newaxis]

def inverse_filter(image, psf):
    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(psf, s=image.shape)

    psf_fft[np.abs(psf_fft) < 0.4] = 0.4

    deconv_fft = image_fft / psf_fft
    deconvolved = np.fft.ifft2(deconv_fft).real
    return deconvolved

# 📌 Column 선택 후 그래프 비교
def plot_graph(original_img, deconvolved_img, selected_peaks):
    """ 선택한 Column 범위에서 Original vs. Deconvolved 그래프 비교 """
    if len(selected_peaks) != 2:
        print("두 개의 Column 좌표를 선택해야 합니다.")
        return

    start_col, end_col = sorted(selected_peaks)
    y_axis = np.arange(original_img.shape[0])  # 이미지 높이 (픽셀 인덱스)

    # 선택한 범위 내 모든 컬럼을 합산
    original_column_sum = np.sum(original_img[:, start_col:end_col + 1], axis=1)
    deconvolved_column_sum = np.sum(deconvolved_img[:, start_col:end_col + 1], axis=1)

    original_column_sum = savgol_filter(original_column_sum, window_length=51, polyorder=1)
    deconvolved_column_sum = savgol_filter(deconvolved_column_sum, window_length=51, polyorder=1)


    plt.figure(figsize=(12, 6))

    # 📌 원본 Intensity 그래프
    plt.plot(original_column_sum, y_axis, label="Original Intensity", linestyle="solid", color="blue", alpha=1.0)
    plt.plot(deconvolved_column_sum, y_axis, label="Deconvolved Intensity", linestyle="solid", color="red", alpha=0.5)

    plt.ylabel("Column Index")
    plt.xlabel("Intensity")
    plt.title(f"Comparison of Intensity (Original vs Deconvolved)")

    plt.gca().set_ylim(300, 1500)

    plt.legend()
    plt.grid()
    plt.gca().invert_yaxis()  # Y축 반전 (위가 0부터 시작)

    plt.show()

# 📌 마우스 클릭 이벤트 처리 (Column 선택)
def mouse_callback(event, x, y, flags, param):
    global selected_peaks

    original_img, deconvolved_img, img_display, scale_factor = param

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_peaks) == 2:
            selected_peaks = []  # 두 개를 초과하면 초기화
        selected_peaks.append(int(x / scale_factor))  # 🔥 클릭 위치를 원래 크기로 변환

        print(f"Selected Column: {selected_peaks[-1]}")

        # 선택된 Column 표시 (축소된 이미지에서 좌표 변환 필요)
        for peak in selected_peaks:
            cv2.line(img_display, (int(peak * scale_factor), 0),
                     (int(peak * scale_factor), img_display.shape[0]), (0, 255, 255), 2)

        cv2.imshow("Original Image", img_display)

        # 두 번째 Column 선택 시 그래프 그리기
        if len(selected_peaks) == 2:
            plot_graph(original_img, deconvolved_img, selected_peaks)

# 📌 메인 실행 함수
def main():
    global selected_peaks
    selected_peaks = []  # Column 선택 초기화

    # 📌 8-bit BMP 이미지 불러오기
    image_path = "original_Mono8_20250305_162122.bmp"
    blurred_image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)

    # 📌 PSF 설정
    psf_height = 2000  # 기존 20 → 200 (PSF가 영향 미치도록 변경)
    psf_sigma = 1   # 기존 10 → 50 (블러 효과가 반영되도록 변경)

    psf = vertical_gaussian_psf(psf_height, psf_sigma)

    # 📌 PSF를 이미지 크기로 확장
    psf_2d = np.zeros_like(blurred_image)
    psf_2d[:psf_height, :blurred_image.shape[1]] = np.tile(psf, (1, blurred_image.shape[1]))
    psf_2d = np.fft.ifftshift(psf_2d)  # PSF 위치 조정

    # 📌 디컨볼루션 적용 (푸리에 변환 + 역 컨볼루션만 수행)
    deconvolved_image = inverse_filter(blurred_image, psf_2d)

    # 📌 8-bit 값 변환 (외부 보정 없이 순수한 결과 저장)
    deconvolved_image_uint8 = np.clip(deconvolved_image, 0, 255).astype(np.uint8)

    # 📌 결과 저장 (BMP 포맷)
    output_path = "Deconvolved_Mono8_8bit.bmp"
    Image.fromarray(deconvolved_image_uint8).save(output_path)

    # 📌 결과 시각화
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(blurred_image, cmap='gray', vmin=0, vmax=255)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(deconvolved_image_uint8, cmap='gray', vmin=0, vmax=255)
    ax[1].set_title("Deconvolved Image")
    ax[1].axis("off")

    plt.show()

    print(f"Deconvolved BMP image saved at: {output_path}")

    # 📌 Original 이미지 출력 (Column 선택 가능)
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original Image", 1024, 768)  # 창 크기 강제 지정

    # 🔥 원본 이미지를 화면에 맞게 리사이즈 (축소 비율 조정 가능)
    scale_factor = 0.5  # 50% 크기로 축소
    img_display = cv2.resize(blurred_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # 8-bit로 변환 후 컬러 적용
    img_display = cv2.normalize(img_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

    while True:
        temp_display = img_display.copy()  # 선택된 Column 반영을 위해 복사본 사용

        for peak in selected_peaks:
            cv2.line(temp_display, (int(peak * scale_factor), 0),
                     (int(peak * scale_factor), temp_display.shape[0]), (0, 255, 255), 2)

        cv2.imshow("Original Image", temp_display)

        # 🔥 마우스 콜백을 위한 데이터 전달
        cv2.setMouseCallback("Original Image", mouse_callback,
                             param=(blurred_image, deconvolved_image_uint8, temp_display, scale_factor))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키로 종료
            break

    cv2.destroyAllWindows()

# 📌 코드 실행
if __name__ == "__main__":
    main()
