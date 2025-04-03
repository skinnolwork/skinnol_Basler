# FGWM 구하는 스크립트
# 첫번째 클릭 : 구하고자 하는 FGWM x범위 시작점
# 두번쨰 클릭 : 구하고자 하는 FGWM x범위 끝점
# 세번쨰 클릭 : FGWM의 50% 구하기위한 밑 설정

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일 읽기
file_path = "data_Mono12_Row_20250205_161744.csv"
data = pd.read_csv(file_path)

# x와 y 값 추출
x = data['cm^-1'].values
y = data['intensity'].values

# 전역 변수로 클릭 위치 저장
clicked_points = []

# 클릭 이벤트 처리 함수
def on_click(event):
    global clicked_points, ax
    if event.xdata is not None and event.ydata is not None:
        clicked_points.append((event.xdata, event.ydata))
        ax.scatter(event.xdata, event.ydata, color='red', s=10, label="Selected Point" if len(clicked_points) == 1 else "")
        plt.draw()

        if len(clicked_points) == 3:
            plt.close()

# 그래프 표시 및 클릭 이벤트 연결
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, label="Original Data", color='blue', markersize=3)
ax.set_title("Click twice to select x-range, third click to select custom y")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid()

fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

# 클릭 지점이 3개일 때 계산
if len(clicked_points) == 3:
    # x 범위는 앞의 두 클릭으로 결정
    x1, _ = clicked_points[0]
    x2, _ = clicked_points[1]
    y3 = clicked_points[2][1]

    x_min, x_max = sorted([x1, x2])
    mask = (x >= x_min) & (x <= x_max)
    x_filtered = x[mask]
    y_filtered = y[mask]

    # 해당 구간 내 최대 y값
    peak_y = max(y_filtered)

    # 새로운 기준선: (y3 + peak_y) / 2
    custom_half_max = (y3 + peak_y) / 2

    # 교차점 계산
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
        distance = abs(intersection_end - intersection_start)
    else:
        intersection_start = intersection_end = distance = None

    # 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(x_filtered, y_filtered, label="Filtered Data", color='blue', markersize=3)
    plt.axhline(custom_half_max, color='red', linestyle='--', label=f"Half Maximum ({custom_half_max:.4f})")

    if intersection_start is not None and intersection_end is not None:
        plt.axvline(intersection_start, color='green', linestyle='--', label=f"Intersection Start ({intersection_start:.4f})")
        plt.axvline(intersection_end, color='purple', linestyle='--', label=f"Intersection End ({intersection_end:.4f})")
        plt.plot([intersection_start, intersection_end], [custom_half_max, custom_half_max], color='black', linestyle='-', label=f"Distance ({distance:.4f})")

    plt.title("Filtered Data with Half Maximum Intersection")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

else:
    print("You need to click 3 times: 2 for x-range and 1 for custom y level.")
