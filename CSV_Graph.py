import pandas as pd
import matplotlib.pyplot as plt

file_path = "skin_3s_back_1.csv"
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

data_start_index = lines.index("XYDATA=\n") + 1
data_lines = lines[data_start_index:]

x_values = []
y_values = []

for line in data_lines:
    try:
        x, y = map(float, line.strip().split(","))
        x_values.append(x)
        y_values.append(y)
    except ValueError:
        continue

plt.figure(figsize=(10, 5))
plt.plot(x_values, y_values, label="Raman Spectrum", color="blue")
plt.xlabel("Raman Shift (1/cm)")
plt.ylabel("Intensity")
plt.title("Raman Spectrum Analysis")
plt.legend()
plt.grid(True)
plt.show()
