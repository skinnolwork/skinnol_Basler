import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv

# 이미지 로드 함수
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 이미지를 흑백으로 로드
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

# 특정 행 선택 및 그래프 그리기 함수
def process_image_row(image, row_index):
    if row_index < 0 or row_index >= image.shape[0]:
        raise ValueError("Row index is out of bounds.")

    row_data = image[row_index, :]  # 선택한 행 데이터

    # 그래프 그리기
    plt.plot(row_data)
    plt.title(f"Row {row_index} Intensity Profile")
    plt.xlabel("Column Index")
    plt.ylabel("Intensity")
    plt.show()

    return row_data

# CSV 파일로 저장 함수
def save_to_csv(row_data, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Column Index", "Intensity"])
        for i, value in enumerate(row_data):
            writer.writerow([i, value])

    print(f"CSV file saved at {output_csv_path}")

# 메인 실행 코드
def main():
    image_path = "capture_Mono8_Column_20250114_161906.bmp"  # 처리할 이미지 경로
    output_csv_path = "row_data.csv"  # 저장할 CSV 경로
    row_index = 100  # 선택할 행 인덱스

    try:
        # 이미지 로드
        image = load_image(image_path)

        # 특정 행 처리 및 그래프 그리기
        row_data = process_image_row(image, row_index)

        # 데이터를 CSV로 저장
        save_to_csv(row_data, output_csv_path)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
