import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from pybaselines import Baseline

# 🔹 전처리 함수 (내장)
def normalize(y):
    return (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)

def baseline_correction(y, lam=1e5, p=0.01):
    baseline, _ = Baseline().asls(y, lam=lam, p=p)
    return y - baseline

# 이미지 불러오기
image_path = "sma_Mono12_20250806_150017.tiff"
original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
if original_image is None:
    raise FileNotFoundError("이미지를 불러올 수 없습니다.")

# 프론트와 동일한 이미지 처리 (min-max 대비 stretch)
norm_img = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX)
display_base = norm_img.astype(np.uint8)

# 이미지 리사이징
def resize_to_fit_screen(img, max_width=1280, max_height=720):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return resized, scale
    return img.copy(), 1.0

display_image, resize_scale = resize_to_fit_screen(display_base)
selected_rows = []

# x축은 nm (원본 기준)
x_original = np.linspace(950, 750, original_image.shape[1])  # reverse

# 선택된 행 표시
def draw_selected_rows():
    temp = cv2.cvtColor(display_image.copy(), cv2.COLOR_GRAY2BGR)
    for y in selected_rows:
        y_scaled = int(y * resize_scale)
        cv2.line(temp, (0, y_scaled), (temp.shape[1], y_scaled), (0, 255, 0), 2)
    cv2.imshow("Image - Select Rows", temp)

# 마우스 클릭 이벤트
def mouse_callback(event, x, y, flags, param):
    global selected_rows
    if event == cv2.EVENT_LBUTTONDOWN:
        original_y = int(y / resize_scale)
        if len(selected_rows) == 2:
            selected_rows = []
        selected_rows.append(original_y)
        draw_selected_rows()

        if len(selected_rows) == 2:
            start, end = sorted(selected_rows)
            selected_data = original_image[start:end, :]
            intensity = np.sum(selected_data, axis=0).astype(np.float64)

            # 전처리 순서: baseline → normalize → smooth
            corrected = baseline_correction(intensity)
            normalized = normalize(corrected)
            smoothed = savgol_filter(normalized, window_length=11, polyorder=3)

            # 피크 검출 (더 민감하게)
            peaks, _ = find_peaks(smoothed, distance=1, prominence=0.06)

            # 시각화
            plt.figure(figsize=(12, 7))
            plt.plot(x_original, smoothed, color='blue', label='Processed Spectrum')
            plt.plot(x_original[peaks], smoothed[peaks], 'ro', label='Detected Peaks', alpha=0.6)
            plt.xlim(x_original[0], x_original[-1])
            plt.xlabel("Wavelength (nm)")
            plt.title(f"Rows {start}~{end} — Detected Peaks: {len(peaks)}")
            plt.grid()
            plt.legend()
            plt.gca().invert_xaxis()
            plt.tight_layout()
            plt.show()

            selected_rows = []
            draw_selected_rows()

# OpenCV 실행
cv2.namedWindow("Image - Select Rows")
cv2.setMouseCallback("Image - Select Rows", mouse_callback)
draw_selected_rows()

print("행 두 개 클릭, ESC 종료")
while True:
    if cv2.waitKey(20) == 27:
        break
cv2.destroyAllWindows()
