import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드 (그레이스케일로 변환)
image_path = "original_Mono8_20250305_162122.bmp"  # BMP 파일 경로 지정
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 이미지 출력
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')

plt.show()
