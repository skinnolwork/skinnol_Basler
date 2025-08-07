import numpy as np
import matplotlib.pyplot as plt
import cv2
from tifffile import imread

def load_cm_axis(length):
    return np.linspace(194, 1800, length)

def pure_peak_area(x_sel, y_sel):
    if x_sel[0] > x_sel[-1]:
        x_sel = x_sel[::-1]
        y_sel = y_sel[::-1]

    if len(x_sel) < 2:
        return 0.0

    m = (y_sel[-1] - y_sel[0]) / (x_sel[-1] - x_sel[0] + 1e-9)
    b = y_sel[0] - m * x_sel[0]
    y_line = m * x_sel + b

    area_curve = area_base = 0.0
    for i in range(len(x_sel)-1):
        dx = abs(x_sel[i+1] - x_sel[i])
        area_curve += 0.5*(y_sel[i]+y_sel[i+1])*dx
        area_base  += 0.5*(y_line[i]+y_line[i+1])*dx

    return area_curve - area_base

from matplotlib import cm

def plot_area_and_intensity(areas, intensity_list, cm_axis, x1, x2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    reds = plt.colormaps['Reds']  # 최신 방식
    color_levels = np.linspace(1, 0.1, len(intensity_list))  # 위→아래 진함

    offset = max([np.max(intensity) for intensity in intensity_list]) * 0.2

    # 왼쪽: intensity
    for idx, (intensity, cval) in enumerate(zip(reversed(intensity_list), color_levels)):
        shifted = intensity + idx * offset
        ax1.plot(cm_axis, shifted, color=reds(cval), alpha=0.9, linewidth=1)

    ax1.axvspan(x1, x2, color='green', alpha=0.2)
    ax1.set_xlabel("cm⁻¹")
    ax1.set_ylabel("Intensity (offset)")
    ax1.set_title("Intensity Profiles (offset)")
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    

    # 오른쪽: 적분
    y_vals = np.arange(1, len(areas)+1)
    ax2.plot(areas, y_vals, '-', color='lightgray', linewidth=1, zorder=0)

    for area, y, cval in zip(areas, y_vals, reversed(color_levels)):
        ax2.plot(area, y, marker='o', color=reds(cval))

    ax2.set_xlabel("Area")
    ax2.set_ylabel("Section")
    ax2.set_title("Integration Areas")
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.show()

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
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(selected) == 0:
                original_y = int(y / scale)
                selected.append(original_y)
                cv2.line(display_img_resized, (0, y), (display_img_resized.shape[1], y), (0, 255, 0), 1)
                cv2.imshow("Select 1 row", display_img_resized)

    cv2.imshow("Select 1 row", display_img_resized)
    cv2.setMouseCallback("Select 1 row", mouse_cb)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or len(selected) == 1:
            break
    cv2.destroyAllWindows()

    return selected[0]

def select_x1x2(x, y):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y)
    plt.title("Click two points (x1, x2)")
    plt.xlabel("cm⁻¹")
    plt.ylabel("Intensity")
    plt.grid()
    plt.gca().invert_xaxis()
    pts = plt.ginput(2)
    plt.close()
    return sorted([pts[0][0], pts[1][0]])

def main(filename):
    img = imread(filename)
    h, w = img.shape
    cm_axis = load_cm_axis(w)

    row = select_row(img)
    start = max(row - 150, 0)
    end = min(row + 150, h-1)
    roi = img[start:end+1, :]
    intensity = np.sum(roi, axis=0).astype(np.float64)
    x1, x2 = select_x1x2(cm_axis, intensity)
    print(f"선택된 x1, x2: {x1:.2f}, {x2:.2f}")

    NUM_SECTIONS = 30
    step = h / NUM_SECTIONS
    areas = []
    intensity_list = []

    for i in range(NUM_SECTIONS):
        center_row = int(i * step + step/2)
        s = max(center_row - 150, 0)
        e = min(center_row + 150, h-1)

        roi = img[s:e+1, :].astype(np.float32)
        intensity = np.sum(roi, axis=0)
        intensity_list.append(intensity)

        mask = (cm_axis >= x1) & (cm_axis <= x2)
        x_sel = cm_axis[mask]
        y_sel = intensity[mask]

        area = pure_peak_area(x_sel, y_sel)
        areas.append(area)

    plot_area_and_intensity(areas, intensity_list, cm_axis, x1, x2)

if __name__ == "__main__":
    main("20250806-160127_sma.tiff")
