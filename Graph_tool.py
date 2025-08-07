import numpy as np
import matplotlib.pyplot as plt
import cv2
from tifffile import imread

def load_cm_axis(length):
    return np.linspace(1850, 362, length)

def resize_to_fit_screen(img, max_width=1280, max_height=720):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale

def select_row(img):
    display_img = img.copy()
    if img.dtype == np.uint16:
        display_img = ((img / 4095.0) * 255).astype(np.uint8)

    display_img_color = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    display_img_resized, scale = resize_to_fit_screen(display_img_color)

    selected = []

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) == 0:
            original_y = int(y / scale)
            selected.append(original_y)
            cv2.line(display_img_resized, (0, y), (display_img_resized.shape[1], y), (0, 255, 0), 1)
            cv2.imshow("Select Row", display_img_resized)

    cv2.imshow("Select Row", display_img_resized)
    cv2.setMouseCallback("Select Row", mouse_cb)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or len(selected) == 1:
            break
    cv2.destroyAllWindows()

    return selected[0]

def main(filename):
    img = imread(filename)
    h, w = img.shape
    cm_axis = load_cm_axis(w)

    row = select_row(img)
    start = max(row - 150, 0)
    end = min(row + 150, h - 1)
    roi = img[start:end+1, :]
    intensity = np.sum(roi, axis=0).astype(np.float64)

    plt.figure(figsize=(12, 6))
    plt.plot(cm_axis, intensity, color='blue')
    plt.title(f"Row {row} ± 150")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity (Arb. )")
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main("sma_Mono12_20250806_150017.tiff")
