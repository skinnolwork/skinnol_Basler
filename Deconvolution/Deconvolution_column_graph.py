import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile
import datetime
import pandas as pd
from scipy.signal import savgol_filter

# 변수 설정
IMAGE_PATH = "original_Mono12_20250226_151813.tiff"  # 불러올 이미지 파일 경로
selected_peaks = []  # 사용자가 선택한 두 개의 피크 좌표


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
    
    img = img.astype(np.float32)  # 12-bit Mono12 이미지를 정규화 없이 사용
    return img


def save_image_and_graph(img, selected_peaks, y_axis, column_sum):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 📌 그래프 이미지 저장
    filename_graph = f"graph_{timestamp}.png"
    plt.savefig(filename_graph)
    print(f"Graph saved as {filename_graph}")

    # 📌 CSV 데이터 저장
    filename_csv = f"original_graph_data_{timestamp}.csv"
    data = {
        "Column": y_axis,
        "Intensity": column_sum,
    }
    df = pd.DataFrame(data)
    df.to_csv(filename_csv, index=False)
    print(f"Graph data saved as {filename_csv}")


def plot_graph(img, selected_peaks):
    if len(selected_peaks) != 2:
        print("두 개의 피크를 선택해야 합니다.")
        return
    
    start_col, end_col = sorted(selected_peaks)
    column_sum = np.sum(img[:, start_col:end_col+1], axis=1)  # 선택한 범위 내 모든 컬럼을 합산
    y_axis = np.arange(img.shape[0])  # 이미지 높이 (픽셀 인덱스)
    
    # 사비츠키-골레이 필터 적용 (부드러움 조정)
    column_sum = savgol_filter(column_sum, window_length=21, polyorder=2)
    
    plt.figure(figsize=(8, 6))
    plt.plot(column_sum, y_axis, label=f'Smoothed Column Sum ({start_col} - {end_col})', color='green')
    plt.gca().invert_yaxis()  # Y축 반전 (이미지 좌표와 맞추기)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Index (Row)")
    plt.title("Intensity of Selected Columns")
    plt.legend()
    plt.grid()
    
    # 📌 그래프, CSV 저장
    save_image_and_graph(img, selected_peaks, y_axis, column_sum)
    
    plt.show()


def mouse_callback(event, x, y, flags, param):
    global selected_peaks
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_peaks) == 2:
            selected_peaks = []  # 두 개를 초과하면 초기화
        selected_peaks.append(x)
        print(f"Selected Peak: {x}")
        
        # 먼저 세로 라인 업데이트 (8-bit 변환 후 출력)
        img_display = cv2.normalize(param, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # 정규화 변환
        img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)  # 흑백 이미지를 컬러로 변환
        for peak in selected_peaks:
            cv2.line(img_display, (peak, 0), (peak, img_display.shape[0]), (0, 255, 255), 3)  # 노란색으로 변경
        cv2.imshow("Select Peaks", img_display)
        cv2.waitKey(1)  # 화면 갱신을 위해 잠깐 대기
        
        # 두 번째 피크가 선택되면 그래프 그리기
        if len(selected_peaks) == 2:
            plot_graph(param, selected_peaks)


def main():
    global selected_peaks
    img = load_image(IMAGE_PATH)

    cv2.namedWindow("Select Peaks", cv2.WINDOW_NORMAL)  # 창 크기 조절 가능하게 설정
    cv2.resizeWindow("Select Peaks", 1024, 768)  # 원하는 창 크기로 제한
    cv2.setMouseCallback("Select Peaks", mouse_callback, param=img)  # 마우스 클릭 이벤트 추가

    while True:
        # 12-bit 이미지 → 8-bit 변환하여 OpenCV로 표시
        img_display = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  
        img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)  # 흑백 이미지를 컬러로 변환
        for peak in selected_peaks:
            cv2.line(img_display, (peak, 0), (peak, img.shape[0]), (0, 255, 255), 3)  # 노란색으로 변경
        
        cv2.imshow("Select Peaks", img_display)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
            break

if __name__ == "__main__":
    main()
