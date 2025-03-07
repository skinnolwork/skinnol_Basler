import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile
import datetime
import pandas as pd
from scipy.signal import savgol_filter
from skimage import restoration

# 📌 이미지 경로 설정
ORIGINAL_IMAGE_PATH = "original_Mono12_20250226_151813.tiff"
DECONVOLVED_IMAGE_PATH = f"deconvolved_Mono12_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tiff"

# 📌 RL 파라미터 설정
height = 200  # PSF 세로 크기 (너무 크면 정보 손실 발생)
sigma = 10  # 가우시안 블러 표준편차
iterations = 10  # RL 디컨볼루션 반복 횟수
selected_peaks = []  # 선택한 두 개의 Column 좌표


def load_image(image_path):
    """ TIFF 이미지를 불러오는 함수 (12-bit Mono 유지) """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

    img = img.astype(np.float32) / 4096.0  # 12-bit Mono 데이터를 0~1 사이로 정규화
    return img


def vertical_gaussian_psf(height, sigma):
    """ 세로 방향 1D 가우시안 PSF 생성 """
    y = np.linspace(-(height // 2), height // 2, height)
    psf = np.exp(- (y ** 2) / (2 * sigma ** 2))  # 세로 방향 가우시안
    psf /= np.sum(psf)  # 정규화
    psf = psf[:, np.newaxis]  # (height, 1) 형태로 변환 (세로 방향만 적용)
    return psf


def apply_rl_deconvolution(image, psf, iterations):
    """ RL 디컨볼루션 수행 """
    deconvolved = restoration.richardson_lucy(image, psf, iterations)
    deconvolved = cv2.normalize(deconvolved, None, 0, 4096, cv2.NORM_MINMAX)  # 원본과 동일한 12-bit 스케일로 변환
    return deconvolved


def save_deconvolved_image(deconvolved_img):
    """ 디컨볼루션된 이미지를 저장 """
    global DECONVOLVED_IMAGE_PATH
    tifffile.imwrite(DECONVOLVED_IMAGE_PATH, deconvolved_img.astype(np.uint16))
    print(f"Deconvolved image saved as {DECONVOLVED_IMAGE_PATH}")


def plot_graph(original_img, deconvolved_img, selected_peaks):
    """ 선택한 Column 범위에서 Original vs. Deconvolved 그래프 비교 """
    if len(selected_peaks) != 2:
        print("두 개의 Column 좌표를 선택해야 합니다.")
        return

    start_col, end_col = sorted(selected_peaks)
    y_axis = np.arange(original_img.shape[0])  # 이미지 높이 (픽셀 인덱스)

    # 선택한 범위 내 모든 컬럼을 합산 (12-bit 값 그대로 사용)
    original_column_sum = np.sum(original_img[:, start_col:end_col + 1], axis=1)
    deconvolved_column_sum = np.sum(deconvolved_img[:, start_col:end_col + 1], axis=1)

    # 🔥 원본 데이터를 12-bit 스케일로 변환하여 디컨볼루션 데이터와 비교
    original_column_sum = original_column_sum * 4096.0  

    # 사비츠키-골레이 필터 적용 (부드러움 조정)
    original_column_sum = savgol_filter(original_column_sum, window_length=21, polyorder=2)
    deconvolved_column_sum = savgol_filter(deconvolved_column_sum, window_length=21, polyorder=2)

    plt.figure(figsize=(12, 6))

    # 📌 원본 Intensity 그래프 (12-bit 값 그대로)
    plt.plot(original_column_sum, y_axis, label="Original Intensity", linestyle="solid", color="blue", alpha=1.0)
    plt.plot(deconvolved_column_sum, y_axis, label="Deconvolved Intensity", linestyle="solid", color="red", alpha=0.5)

    plt.ylabel("Column Index")
    plt.xlabel("Intensity (Raw 12-bit Values)")  # 라벨 수정
    plt.title(f"Comparison of Raw Intensity (Original vs Deconvolved)\n"
              f"Height: {height}, Sigma: {sigma}, Iteration: {iterations}")

    plt.legend()
    plt.grid()
    plt.gca().invert_yaxis()  # Y축 반전 (위가 0부터 시작)

    plt.show()



def mouse_callback(event, x, y, flags, param):
    """ 사용자가 Original 이미지에서 Column 좌표 선택 """
    global selected_peaks

    # 전달된 데이터 언팩 (scale_factor 추가)
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

def main():
    """ RL 디컨볼루션을 수행하고, Original + Deconvolved 이미지 비교 후 그래프 분석 """
    global height, sigma, iterations, selected_peaks

    # 1️⃣ 원본 이미지 로드
    original_img = load_image(ORIGINAL_IMAGE_PATH)

    # 2️⃣ RL 디컨볼루션 수행
    psf = vertical_gaussian_psf(height, sigma)
    deconvolved_img = apply_rl_deconvolution(original_img, psf, iterations)

    # 3️⃣ 디컨볼루션된 이미지 저장
    save_deconvolved_image(deconvolved_img)

    # 4️⃣ Original 이미지 출력 (선택한 Column 표시)
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original Image", 1024, 768)  # 창 크기 강제 지정

    # 🔥 원본 이미지를 화면에 맞게 리사이즈 (축소 비율 조정 가능)
    scale_factor = 0.5  # 50% 크기로 축소
    img_display = cv2.resize(original_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # 8-bit로 변환 후 컬러 적용
    img_display = cv2.normalize(img_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

    while True:
        temp_display = img_display.copy()  # 선택된 Column 반영을 위해 복사본 사용

        for peak in selected_peaks:
            cv2.line(temp_display, (int(peak * scale_factor), 0), 
                     (int(peak * scale_factor), temp_display.shape[0]), (0, 255, 255), 2)

        cv2.imshow("Original Image", temp_display)

        # 🔥 마우스 콜백을 위한 데이터 전달 (이미지 크기 스케일 고려)
        cv2.setMouseCallback("Original Image", mouse_callback, param=(original_img, deconvolved_img, temp_display, scale_factor))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키로 종료
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
