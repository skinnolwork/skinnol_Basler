import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일 읽기
file_path = "data_Mono12_Row_20250409_110125.csv"
data = pd.read_csv(file_path)

# x와 y 값 추출
x = data['cm^-1'].values
y = data['Intensity'].values

clicked_points = []
scatter_points = []
color_cycle = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
color_index = 0  # 색 인덱스

def on_click(event):
    global clicked_points, scatter_points, color_index
    if event.xdata is not None and event.ydata is not None:
        clicked_points.append((event.xdata, event.ydata))
        current_color = color_cycle[color_index % len(color_cycle)]
        scatter = ax.scatter(event.xdata, event.ydata, color=current_color, s=15, zorder=5)
        scatter_points.append(scatter)
        plt.draw()

        if len(clicked_points) == 3:
            run_fgwm_analysis(clicked_points, current_color)
            clicked_points.clear()
            color_index += 1  # 다음 색상으로 변경

def run_fgwm_analysis(points, color):
    x1, _ = points[0]
    x2, _ = points[1]
    y3 = points[2][1]

    x_min, x_max = sorted([x1, x2])
    mask = (x >= x_min) & (x <= x_max)
    x_filtered = x[mask]
    y_filtered = y[mask]

    peak_y = max(y_filtered)
    custom_half_max = (y3 + peak_y) / 2

    intersection_xs = []
    for i in range(len(x_filtered) - 1):
        y1, y2 = y_filtered[i], y_filtered[i + 1]
        if (y1 - custom_half_max) * (y2 - custom_half_max) < 0:
            x_interp = x_filtered[i] + (x_filtered[i + 1] - x_filtered[i]) * (custom_half_max - y1) / (y2 - y1)
            intersection_xs.append(x_interp)

    if len(intersection_xs) >= 2:
        intersection_xs_sorted = sorted(intersection_xs)
        intersection_start = intersection_xs_sorted[0]
        intersection_end = intersection_xs_sorted[-1]
        distance = abs(intersection_end - intersection_start)
    else:
        intersection_start = intersection_end = distance = None

    # FGWM 그래프
    fig_fgwm, ax_fgwm = plt.subplots(figsize=(8, 6))
    ax_fgwm.plot(x_filtered, y_filtered, label="Filtered Data", color=color)
    ax_fgwm.axhline(custom_half_max, color=color, linestyle='--', label=f"Half Max ({custom_half_max:.4f})")

    if intersection_start is not None and intersection_end is not None:
        ax_fgwm.axvline(intersection_start, color='black', linestyle='--', label=f"Start ({intersection_start:.2f})")
        ax_fgwm.axvline(intersection_end, color='black', linestyle='--', label=f"End ({intersection_end:.2f})")
        ax_fgwm.plot([intersection_start, intersection_end], [custom_half_max]*2,
                     color='black', linestyle='-', label=f"Distance ({distance:.2f})")

    ax_fgwm.set_title("FGWM Result")
    ax_fgwm.set_xlabel("x")
    ax_fgwm.set_ylabel("y")
    ax_fgwm.legend()
    ax_fgwm.grid()
    plt.show(block=False)

# 메인 그래프
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, label="Original Data", color='blue', linewidth=1)
ax.set_title("Click twice to select x-range, third click for custom y")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid()

fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
