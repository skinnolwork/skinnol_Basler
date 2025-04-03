import pandas as pd
import matplotlib.pyplot as plt

# CSV 경로
file_path = "data_Mono12_Row_20250205_161744.csv"

# CSV 불러오기
df = pd.read_csv(file_path)

# x축은 보정된 'cm^-1', y축은 'intensity'
x = df["nm"]
y = df["intensity"]

# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(x, y, label="Raman Spectrum", color="blue")
plt.xlabel("Raman Shift (1/cm)")
plt.ylabel("Intensity")
plt.title("Raman Spectrum Analysis")
plt.legend()
plt.grid(True)
plt.show()
