import pandas as pd
import numpy as np

# CSV 파일 로드 (라만 데이터 예시)
file_path = 'a_3840.csv'
data = pd.read_csv(file_path)

# x축과 y축 데이터 추출
x = data.iloc[:, 0]  # Raman Shift
y = data.iloc[:, 1]  # Intensity

# y축 정규화 (2160px 해상도에 맞게 스케일링)
y_min = y.min()
y_max = y.max()
y_scaled = (y - y_min) / (y_max - y_min) * 2160  # 0 ~ 2160 범위로 스케일링

# 정규화된 데이터 저장
normalized_data = pd.DataFrame({'Raman Shift': x, 'Intensity': y_scaled})
output_file = 'normalized_raman_data.csv'
normalized_data.to_csv(output_file, index=False)

print(f"정규화된 데이터가 저장되었습니다: {output_file}")