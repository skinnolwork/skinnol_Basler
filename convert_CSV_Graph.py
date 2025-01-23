import pandas as pd
import matplotlib.pyplot as plt

# 1. CSV 파일 로드
file_path = 'processed_data.csv'  # 이전 스크립트에서 저장된 CSV 파일 경로
data = pd.read_csv(file_path)

# 2. x_original vs Intensity 그래프
plt.figure(figsize=(10, 6))
plt.plot(data['x_original'], data['Intensity'], label='x_original vs Intensity', color='blue')
plt.xlabel('x_original')
plt.ylabel('Intensity (a.u.)')
plt.title('x_original vs Intensity')
plt.gca().invert_xaxis()  # X축 반전 (nm 감소 방향)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 3. nm vs Intensity 그래프
plt.figure(figsize=(10, 6))
plt.plot(data['nm'], data['Intensity'], label='nm vs Intensity', color='green')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.title('Wavelength (nm) vs Intensity')
plt.gca().invert_xaxis()  # X축 반전 (nm 감소 방향)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 4. cm^-1 vs Intensity 그래프
plt.figure(figsize=(10, 6))
plt.plot(data['cm^-1'], data['Intensity'], label='cm^-1 vs Intensity', color='red')
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('Intensity (a.u.)')
plt.title('Wavenumber (cm^-1) vs Intensity')
plt.gca().invert_xaxis()  # X축 반전 (cm^-1 감소 방향)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
