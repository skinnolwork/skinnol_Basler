import matplotlib.pyplot as plt
import numpy as np

def plot_csv_data(csv_file):
    """
    저장된 CSV 파일에서 데이터를 읽어와 그래프를 그리는 함수.
    """
    # CSV 파일 읽기
    try:
        data = np.loadtxt(csv_file, delimiter=",", skiprows=1)  # 첫 줄은 헤더이므로 건너뜀
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # X값과 Y값 분리
    x_values = data[:, 0]  # 첫 번째 열: X 값
    y_values = data[:, 1]  # 두 번째 열: Y 값

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, color='blue', linewidth=1.5, label='Intensity')
    plt.title("Saved Intensity Data", fontsize=14)
    plt.xlabel("X (Wavelength)", fontsize=12)
    plt.ylabel("Y (Intensity)", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 그래프 표시
    plt.show()

if __name__ == "__main__":
    # CSV 파일 경로
    csv_file_path = "./data_Mono8_Row_20250116_160403.csv"  # 파일 경로를 저장된 CSV 파일 이름으로 수정
    plot_csv_data(csv_file_path)
