# 라만 데이터 적분
# 1. 이미지 화면
#   Row 두곳 클릭
# 2. 첫번째 그래프 화면
#   피크 두곳 클릭 : x_original -> nm -> cm^-1 변환
# 3. 두번째 그래프 화면
#   원하는 두곳 클릭해서 라만 데이터 적분하여 넓이 계산

import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "original_Mono12_20250205_161744.tiff"
original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
if original_image is None:
    raise FileNotFoundError("이미지를 불러올 수 없습니다.")

if original_image.dtype == np.uint16:
    display_base = ((original_image / 4095.0) * 255).astype(np.uint8)
else:
    display_base = original_image.copy()

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

x_original = np.linspace(950, 750, original_image.shape[1])
x_cm = 1e7 / x_original
use_cm = False

def adjust_axis_cm(peaks_idx, x_original, target_nm, target_cm):
    i2, i1 = peaks_idx
    current_nm = x_original[[i1, i2]]
    m_nm = (target_nm[1] - target_nm[0]) / (current_nm[1] - current_nm[0])
    b_nm = target_nm[0] - m_nm * current_nm[0]
    x_nm_adj = m_nm * x_original + b_nm
    x_cm_temp = 1e7 / x_nm_adj
    current_cm = x_cm_temp[[i1, i2]]
    m_cm = (target_cm[1] - target_cm[0]) / (current_cm[1] - current_cm[0])
    b_cm = target_cm[0] - m_cm * current_cm[0]
    return m_cm * x_cm_temp + b_cm

def draw_selected_rows():
    temp = cv2.cvtColor(display_image.copy(), cv2.COLOR_GRAY2BGR)
    for y in selected_rows:
        y_scaled = int(y * resize_scale)
        cv2.line(temp, (0, y_scaled), (temp.shape[1], y_scaled), (0, 255, 0), 2)
    cv2.imshow("Image - Select Rows", temp)

def interactive_integration_plot(x, y):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x, y, color='blue')
    ax.set_title("Click two points to integrate")
    ax.set_xlabel("Wavenumber (cm⁻¹)" if use_cm else "Wavelength (nm)")
    ax.grid()
    ax.invert_xaxis()

    clicked = []
    click_start_pos = [None]

    def get_nearest_index(x_array, value):
        return np.abs(np.asarray(x_array) - value).argmin()

    def on_press(event):
        if event.inaxes != ax:
            return
        click_start_pos[0] = (event.x, event.y)

    def on_release(event):
        if event.inaxes != ax or event.button != 1:
            return
        x0, y0 = click_start_pos[0]
        if abs(event.x - x0) < 5 and abs(event.y - y0) < 5:
            handle_click(event)

    def handle_click(event):
        clicked.append(event.xdata)
        idx = get_nearest_index(x, event.xdata)
        x_val, y_val = x[idx], y[idx]
        ax.plot(x_val, y_val, 'ro', markersize=5)
        fig.canvas.draw()


        if len(clicked) == 2:
            x1, x2 = clicked
            idx1 = get_nearest_index(x, x1)
            idx2 = get_nearest_index(x, x2)
            start_idx = min(idx1, idx2)
            end_idx = max(idx1, idx2)

            x_sel = x[start_idx:end_idx+1]
            y_sel = y[start_idx:end_idx+1]

            m = (y[end_idx] - y[start_idx]) / (x[end_idx] - x[start_idx])
            b = y[start_idx] - m * x[start_idx]
            y_line = m * x_sel + b

            # 전체 넓이: 그래프 아래 면적
            area_curve = 0
            area_base = 0
            for i in range(len(x_sel) - 1):
                dx = abs(x_sel[i+1] - x_sel[i])
                
                avg_height_a = (y_sel[i] + y_sel[i+1]) / 2
                avg_height_b = (y_line[i] + y_line[i+1]) / 2

                area_curve += avg_height_a * dx                
                area_base += avg_height_b * dx

            # 순수 피크 영역 = 초록색
            pure_area = area_curve - area_base

            # 시각화: 실제 넓이 부분만 색칠
            ax.fill_between(x_sel, y_line, y_sel, interpolate=True, color='green', alpha=0.5)
            ax.fill_between(x_sel, y_sel, y_line, interpolate=True, color='yellow', alpha=0.5)

            # 피크 표시
            peak_idx = np.argmax(y_sel)
            peak_x = x_sel[peak_idx]
            peak_y = y_sel[peak_idx]
            ax.plot(peak_x, peak_y, 'ko')
            ax.text(peak_x, peak_y + 2000, f'Peak : {peak_x:,.2f}', fontsize=10, color='black')


            # 넓이 표시
            center_x = (x1 + x2) / 2
            center_y = max(y_sel) * 0.75
            ax.text(center_x, center_y, f'{int(pure_area):,}', color='red', ha='center')


            fig.canvas.draw()

        elif len(clicked) == 3:
            # 확대된 범위 저장
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            clicked.clear()
            ax.cla()
            ax.plot(x, y, color='blue')
            ax.set_title("Click two points to integrate")
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.grid()
            if use_cm:
                ax.invert_xaxis()

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            fig.canvas.draw()




    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    plt.show(block=False)
    plt.pause(0.1)

def mouse_callback(event, x, y, flags, param):
    global selected_rows, x_cm, use_cm
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

            plt.figure(figsize=(12, 7))
            plt.plot(x_original, intensity, color='red')
            plt.xlim(x_original[0], x_original[-1])
            plt.xlabel("Wavelength (nm)")
            plt.title(f"Rows {start}~{end}")
            plt.grid()
            plt.show(block=False)

            clicked = plt.ginput(2, timeout=0)
            if len(clicked) == 2:
                plt.close()
                idx1 = np.abs(x_original - clicked[0][0]).argmin()
                idx2 = np.abs(x_original - clicked[1][0]).argmin()
                peaks_idx = sorted([idx1, idx2])
                target_peaks_nm = (852.1, 898.2)
                target_peaks_cm = (1001.4, 1602.3)
                x_cm = adjust_axis_cm(peaks_idx, x_original, target_peaks_nm, target_peaks_cm)
                use_cm = True

                plt.figure(figsize=(12, 7))
                plt.plot(x_cm, intensity, color='blue')
                plt.xlabel("Wavenumber (cm⁻¹)")
                plt.title("Calibrated Spectrum")
                plt.grid()
                plt.gca().invert_xaxis()
                plt.close()

                interactive_integration_plot(x_cm, intensity)

            selected_rows = []
            draw_selected_rows()

cv2.namedWindow("Image - Select Rows")
cv2.setMouseCallback("Image - Select Rows", mouse_callback)
draw_selected_rows()

print("행 두 개 클릭, ESC 종료")
while True:
    if cv2.waitKey(20) == 27:
        break
cv2.destroyAllWindows()
