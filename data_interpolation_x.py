## x축 보간 ##

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# 입력 파일 경로
input_file = 'a.csv'  # 기존 CSV 파일 이름을 입력하세요.
output_file = 'a_3840.csv'  # 변환된 파일 저장 이름

# CSV 파일 로드
data = pd.read_csv(input_file)
data.columns = ['Raman Shift', 'Intensity']  # 컬럼 이름 정리

# 새 x축 생성 (0 ~ 3840)
x_mapped = np.linspace(0, 3840, 3840)

# 기존 x축을 0 ~ 3840으로 선형 변환 (보간 함수 생성)
interp_func_mapped = interp1d(
    data['Raman Shift'], data['Intensity'], kind='cubic'
)

# 새 x축에 맞는 y값 계산
y_mapped = interp_func_mapped(np.linspace(data['Raman Shift'].min(), data['Raman Shift'].max(), 3840))

# 새로운 데이터프레임 생성
mapped_data = pd.DataFrame({'Raman Shift': x_mapped, 'Intensity': y_mapped})

# 변환된 파일 저장
mapped_data.to_csv(output_file, index=False)