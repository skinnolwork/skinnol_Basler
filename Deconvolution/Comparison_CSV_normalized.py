import pandas as pd
import matplotlib.pyplot as plt

# 📌 CSV 파일 경로 (본인 파일명으로 변경)
original_csv_path = "original_graph_data_20250304_114409.csv"
deconvolved_csv_path = "deconvolution_graph_data_20250304_122358.csv"

# 📌 CSV 데이터 불러오기
original_df = pd.read_csv(original_csv_path)
deconvolved_df = pd.read_csv(deconvolved_csv_path)

# 📌 데이터 컬럼 확인 (컬럼명이 다를 경우 직접 확인 후 수정)
print("Original CSV Columns:", original_df.columns)
print("Deconvolved CSV Columns:", deconvolved_df.columns)

# 📌 Intensity 정규화 함수 (0~1 스케일 조정)
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

# 📌 Intensity 표준화 함수 (평균 0, 표준편차 1로 조정)
def standardize(data):
    return (data - data.mean()) / data.std()

# 📌 원본 및 디컨볼루션 데이터 정규화 및 표준화 적용
original_df["Intensity_norm"] = normalize(original_df["Intensity"])
deconvolved_df["Intensity_norm"] = normalize(deconvolved_df["Intensity"])

original_df["Intensity_std"] = standardize(original_df["Intensity"])
deconvolved_df["Intensity_std"] = standardize(deconvolved_df["Intensity"])

# 📌 데이터 크기가 너무 크면 샘플링 (메모리 절약)
sample_rate = 10  # 10개마다 하나씩 샘플링
original_sampled = original_df.iloc[::sample_rate, :]
deconvolved_sampled = deconvolved_df.iloc[::sample_rate, :]

# 📌 시각화 - 정규화된 데이터 비교 (X, Y축 바꾸고, 위에서 아래로 증가하도록 설정)
plt.figure(figsize=(10, 6))
plt.plot(original_sampled["Intensity_norm"], original_sampled["Column"], label="Original (Normalized)", linestyle="dashed", color="blue")
plt.plot(deconvolved_sampled["Intensity_norm"], deconvolved_sampled["Column"], label="Deconvolved (Normalized)", linestyle="solid", color="red")
plt.xlabel("Normalized Intensity (0-1)")
plt.ylabel("Column Index")
plt.title("Comparison of Normalized Intensity (Original vs Deconvolved)")
plt.legend()
plt.grid()
plt.gca().invert_yaxis()  # Y축 반전 (위가 0부터 시작)
plt.show()

# 📌 시각화 - 표준화된 데이터 비교 (X, Y축 바꾸고, 위에서 아래로 증가하도록 설정)
plt.figure(figsize=(10, 6))
plt.plot(original_sampled["Intensity_std"], original_sampled["Column"], label="Original (Standardized)", linestyle="dashed", color="blue")
plt.plot(deconvolved_sampled["Intensity_std"], deconvolved_sampled["Column"], label="Deconvolved (Standardized)", linestyle="solid", color="red")
plt.xlabel("Standardized Intensity (Mean=0, Std=1)")
plt.ylabel("Column Index")
plt.title("Comparison of Standardized Intensity (Original vs Deconvolved)")
plt.legend()
plt.grid()
plt.gca().invert_yaxis()  # Y축 반전 (위가 0부터 시작)
plt.show()
