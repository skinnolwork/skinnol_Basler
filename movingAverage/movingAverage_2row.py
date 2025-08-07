import tifffile
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_tiff(filepath):
    img = tifffile.imread(filepath).astype(np.float64)
    return img

def compute_intensity(img, row1, row2):
    start, end = sorted([row1, row2])
    roi = img[start:end+1, :]
    intensity = np.sum(roi, axis=0)
    return intensity

def simple_moving_average(data, window):
    if len(data) < window:
        raise ValueError("데이터 길이가 window보다 짧습니다.")
    sma = np.convolve(data, np.ones(window)/window, mode='same')
    return sma

def plot_intensity(original, smoothed):
    plt.figure(figsize=(12, 6))
    plt.plot(original, label=f"Original ({len(original)})", alpha=0.3, linewidth=1)
    plt.plot(np.arange(len(smoothed)), smoothed, color='red', label=f"SMA ({len(smoothed)})", linewidth=1)
    plt.xlabel("Pixel index")
    plt.ylabel("Intensity")
    plt.title("TIFF Intensity & SMA")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def resize_to_fit_screen(img, max_width=1280, max_height=720):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)  # 더 클 때만 축소
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale

if __name__ == "__main__":
    filepath = "original_Mono12_20250411_110738.tiff"
    img = load_tiff(filepath)

    # OpenCV로 띄울 이미지 준비
    display_img = img.copy()
    display_img = (display_img / display_img.max() * 255).astype(np.uint8)
    display_img_color = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    display_img_color, scale = resize_to_fit_screen(display_img_color)

    rows_clicked = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            original_y = int(y / scale)
            rows_clicked.append(original_y)
            cv2.line(display_img_color, (0, y), (display_img_color.shape[1], y), (0, 255, 0), 1)
            cv2.imshow("Select 2 Rows", display_img_color)

    cv2.imshow("Select 2 Rows", display_img_color)
    cv2.setMouseCallback("Select 2 Rows", click_event)

    print("이미지에서 row를 2번 클릭 (ESC로 종료)")

    while True:
        key = cv2.waitKey(1)
        if key == 27 or len(rows_clicked) >= 2:
            break

    cv2.destroyAllWindows()

    if len(rows_clicked) < 2:
        raise RuntimeError("2개의 row를 선택")

    intensity = compute_intensity(img, rows_clicked[0], rows_clicked[1])
    print(f"선택한 row: {rows_clicked}")
    print(f"Intensity 길이: {len(intensity)}")

    sma = simple_moving_average(intensity, window=4)
    plot_intensity(intensity, sma)
