import pandas as pd
import matplotlib.pyplot as plt

file1 = "original_graph_data_20250304_111912.csv"
file2 = "deconvolution_graph_data_20250304_112139.csv"

def load_csv(file_path):
    """ CSV 파일을 로드하고 데이터 반환 """
    df = pd.read_csv(file_path)
    return df["Column"], df["Intensity"]

def plot_two_graphs(file1, file2):
    """ 두 개의 CSV 데이터를 불러와 그래프를 그린다 """
    x1, y1 = load_csv(file1)
    x2, y2 = load_csv(file2)

    plt.figure(figsize=(8, 6))
    plt.plot(y1, x1, label=f"{file1}", linestyle='-', color='blue')
    plt.plot(y2, x2, label=f"{file2}", linestyle='-', color='red', alpha=0.4)

    plt.gca().invert_yaxis()  # Y축 반전
    plt.xlabel("Intensity")
    plt.ylabel("Column")
    plt.title("Comparison of Two Graphs")
    plt.legend()
    plt.grid()
    plt.show()

# ✅ 실행 시 자동으로 두 개의 파일을 그래프로 표시
plot_two_graphs(file1, file2)
