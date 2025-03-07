import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile
import datetime
import pandas as pd
from scipy.signal import savgol_filter

# 변수 설정
IMAGE_PATH = "Deconvolved_Mono12_20250304_112645.tiff"  # 불러올 이미지 파일 경로
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
    filename_csv = f"deconvolution_graph_data_{timestamp}.csv"
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
    
    # 사비츠키-골레이 필터 적용 (윈도우 크기 11, 다항식 차수 3)
    column_sum = savgol_filter(column_sum, window_length=21, polyorder=2)
    
    plt.figure(figsize=(8, 6))
    plt.plot(column_sum, y_axis, label=f'Smoothed Column Sum ({start_col} - {end_col})', color='green')
    plt.gca().invert_yaxis()  # Y축 반전 (이미지 좌표와 맞추기)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Index (Row)")
    plt.title("Intensity of Selected Columns")
    plt.legend()
    plt.grid()
    
    # 📌 그래프 및 CSV 저장
    save_image_and_graph(img, selected_peaks, y_axis, column_sum)
    
    plt.show()


def main():
    global selected_peaks
    img = load_image(IMAGE_PATH)
    
    print("두 개의 column index를 입력하세요 (예: 100 200):")
    start_col, end_col = map(int, input().split())
    selected_peaks = [start_col, end_col]
    
    plot_graph(img, selected_peaks)


if __name__ == "__main__":
    main()
