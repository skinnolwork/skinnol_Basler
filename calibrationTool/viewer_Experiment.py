# 같은 폴더 내에 calibration_data.csv 파일 필요
# b 키를 눌러 저장
# 저장 항목
#     1. 선택 row 표시된 opencv 이미지 (png)
#     2. 밝기 보정된 썸네일 이미지 (png)
#     3. cm⁻¹, Intensity 데이터 (csv)
#     4. pyqtgraph 그래프 이미지 (png)
#     5. 16bit 원본 시각화 이미지 (tiff)
#     6. 12bit 원본 이미지 (tiff)
#     7. 썸네일 이미지 (png)
#     8. Moving Average 적용한 이미지 (tiff)

# 실행
# 카메라 영상에서 row 2곳 선택
# 'b' 저장, 'esc' 종료

from pypylon import pylon
import numpy as np
import cv2
import datetime
import tifffile
import os
import keyboard
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence

CAMERA_SIZE = [3840, 2160] # 카메라 해상도
EXPOSURE_TIME = 1000000.0 # 노출 시간 (µs)
GAIN = 40.0 # 신호 증폭
MONO_MODE = "Mono12" # 픽셀 포맷 (고정)
CALIBRATION_FILE = "calibration_data.csv" # 보정 파일


class BaslerCamera:
    def __init__(self, camera_size, exposure_time, gain, mono_mode):
        self.camera_size = camera_size
        self.exposure_time = exposure_time
        self.gain = gain
        self.mono_mode = mono_mode
        self.camera = None
        self.x_cm = None # cm⁻¹ x축 정보
        self.clicked_rows = [] # 이미지 클릭 row 좌표
        self.pg_app = None # pyqtgraph GUI 앱
        self.pg_win = None # pyqtgraph 윈도우

    # pyqtgraph 앱
    def setup_plot(self):
        self.pg_app = QtWidgets.QApplication([])
        self.pg_win = pg.GraphicsLayoutWidget(title="실시간 스펙트럼")
        self.pg_win.show()

        # 그래프를 3개의 subplot으로 구성 ( Top / middle / Bottom )
        # Top
        self.top_plot = self.pg_win.addPlot()
        self.top_plot.showGrid(x=True, y=True)
        self.top_plot.setLabel("bottom", "Wavenumber", units="cm⁻¹")
        self.top_plot.setLabel("left", "Intensity (Arb.)")
        self.top_curve = self.top_plot.plot(pen='y')

        # Middle
        self.pg_win.nextRow()
        self.middle_plot = self.pg_win.addPlot()
        self.middle_plot.showGrid(x=True, y=True)
        self.middle_plot.setLabel("bottom", "Wavenumber", units="cm⁻¹")
        self.middle_plot.setLabel("left", "Intensity (Arb.)")
        self.middle_curve = self.middle_plot.plot(pen='y')

        # Bottom
        self.pg_win.nextRow()
        self.bottom_plot = self.pg_win.addPlot()
        self.bottom_plot.showGrid(x=True, y=True)
        self.bottom_plot.setLabel("bottom", "Wavenumber", units="cm⁻¹")
        self.bottom_plot.setLabel("left", "Intensity (Arb.)")
        self.bottom_curve = self.bottom_plot.plot(pen='y')

        # cm⁻¹ 눈금 설정 
        if self.x_cm is not None:
            min_cm = int(np.floor(np.min(self.x_cm)))
            max_cm = int(np.ceil(np.max(self.x_cm)))

            major_ticks = [(v, str(v)) for v in range(min_cm, max_cm + 1) if v % 100 == 0]
            minor_ticks = [(v, "") for v in range(min_cm, max_cm + 1) if v % 50 == 0 and v % 100 != 0]

            for plot in [self.top_plot, self.middle_plot, self.bottom_plot]:
                axis = plot.getAxis("bottom")
                axis.setTicks([major_ticks, minor_ticks])

    # 실시간 그래프 업데이트
    def update_plot(self, intensity):
        if self.x_cm is None or intensity is None:
            return

        n = len(self.x_cm)
        n1 = n // 3
        n2 = 2 * n // 3

        max_y = np.max(intensity)

        # 세 plot의 y축 범위 통일 (0 ~ 가장 큰 Intensity)
        self.top_plot.setYRange(0, max_y)
        self.middle_plot.setYRange(0, max_y)
        self.bottom_plot.setYRange(0, max_y)

        # 세 구간으로 나누어 그리기
        self.top_curve.setData(x=self.x_cm[:n1], y=intensity[:n1])
        self.middle_curve.setData(x=self.x_cm[n1:n2], y=intensity[n1:n2])
        self.bottom_curve.setData(x=self.x_cm[n2:], y=intensity[n2:])

        QtWidgets.QApplication.processEvents()

    # 카메라 연결 설정
    def initialize_camera(self):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        print(f"[✓] Device: {self.camera.GetDeviceInfo().GetModelName()}")

        self.camera.Width.SetValue(self.camera_size[0])
        self.camera.Height.SetValue(self.camera_size[1])
        self.camera.PixelFormat.SetValue(self.mono_mode)
        self.camera.ExposureTime.SetValue(self.exposure_time)
        self.camera.Gain.SetValue(self.gain)

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # 카메라 및 창 종료 처리
    def release_resources(self):
        self.camera.StopGrabbing()
        self.camera.Close()
        cv2.destroyAllWindows()
        print("[✓] Camera closed.")

    # 보정 파일 호출
    def load_latest_calibration(self):
        if not os.path.exists(CALIBRATION_FILE):
            print("[ERROR] calibration_data.csv not found.")
            return

        with open(CALIBRATION_FILE, "r") as f:
            lines = f.readlines()[1:]
            if not lines:
                print("[ERROR] calibration_data.csv is empty.")
                return

            latest = lines[-1].strip().split(",")
            _, cm_start_str, cm_end_str = latest
            cm_start = float(cm_start_str)
            cm_end = float(cm_end_str)

            # cm⁻¹ 값 보정
            self.x_cm = np.linspace(cm_end, cm_start, self.camera_size[0])
            print(f"[✓] Loaded cm⁻¹ range: {cm_start:.2f} ~ {cm_end:.2f}")

    # Moving Average
    def compute_sma(self, img_array, window=9): # 9개의 평균으로 설정
        img_float = img_array.astype(np.float64)
        h, _ = img_array.shape
        sma_img = np.zeros_like(img_float)

        pad = window // 2
        kernel = np.ones(window) / window

        for i in range(h):
            padded_row = np.pad(img_float[i, :], (pad, pad), mode='edge')
            sma_row = np.convolve(padded_row, kernel, mode='valid')
            sma_img[i, :] = sma_row

        print("SMA min/max:", np.min(sma_img), np.max(sma_img))

        return np.clip(sma_img, 0, 4095).astype(np.uint16)
    
    # 저장
    def save_data(self, img_array, intensity, color_img, display_img):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        date_folder = datetime.datetime.now().strftime("%Y%m%d")
        save_directory = f"./{date_folder}_experiment"
        os.makedirs(save_directory, exist_ok=True)

        # 원본 이미지 (tiff)
        tifffile.imwrite(f"{save_directory}/original_{self.mono_mode}_{timestamp}.tiff",
                         img_array, dtype="uint16", compression=None)
        
        # MovingAverage 이미지 저장 (tiff) - 노이즈 감소
        sma_img = self.compute_sma(img_array, window=9)
        tifffile.imwrite(f"{save_directory}/sma_{self.mono_mode}_{timestamp}.tiff",
                        sma_img, dtype="uint16", compression=None)

        # 12bit 원본 -> 16bit로 저장 - 원본 이미지 시각화 (tiff)
        normalized = (img_array / 4095.0 * 65535).astype(np.uint16)
        tifffile.imwrite(f"{save_directory}/normalized_{self.mono_mode}_{timestamp}.tiff", normalized)

        # 썸네일용 8bit 이미지 (png)
        resized_png = cv2.resize((img_array / 4095.0 * 255).astype(np.uint8), (384, 216))
        cv2.imwrite(f"{save_directory}/resized_{self.mono_mode}_{timestamp}.png", resized_png)
        cv2.imwrite(f"{save_directory}/annotated_{self.mono_mode}_{timestamp}.png", display_img)

        # 보정을 통해 밝기 증가한 이미지 (png)
        bright_resized_png = cv2.resize(color_img, (384, 216))
        cv2.imwrite(f"{save_directory}/bright_resized_{self.mono_mode}_{timestamp}.png", bright_resized_png)

        # cm^1, intesity를 헤더로 가지는 csv파일 (csv)
        data_to_save = np.column_stack((self.x_cm, intensity))
        header = "cm^-1,intensity"
        np.savetxt(f"{save_directory}/data_{self.mono_mode}_{timestamp}.csv",
                   data_to_save, delimiter=",", fmt="%.2f", header=header, comments="")
        
        # 그래프 캡쳐
        self.pg_win.grab().save(f"{save_directory}/graph_cm_{self.mono_mode}_{timestamp}.png")
        print("[✓] All data saved.")

    # 이미지 클릭 시 row 좌표 기록
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            scale = self.camera_size[1] / 540
            row = int(y * scale)
            if len(self.clicked_rows) >= 2:
                self.clicked_rows = []
            self.clicked_rows.append(row)

    # 전체 루프 동작
    def run(self):
        self.initialize_camera() # 카메라 설정
        self.load_latest_calibration() # x 보정값 설정
        self.setup_plot() # pyqtgraph 설정

        cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera View", 960, 540)
        cv2.setMouseCallback("Camera View", self.mouse_callback)

        try:
            while self.camera.IsGrabbing():
                grab_result = self.camera.RetrieveResult(int(self.exposure_time/1000 + 500), pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    # 이미지 처리
                    img_array = grab_result.Array & 0xFFF
                    disp_img = img_array.astype(np.float32)
                    min_val = np.percentile(disp_img, 1)
                    max_val = np.percentile(disp_img, 99)

                    range_val = max_val - min_val
                    if range_val < 1e-5:
                        range_val = 1.0

                    disp_img = np.clip((disp_img - min_val) / range_val * 255.0, 0, 255).astype(np.uint8)

                    # 디스플레이 용 이미지 생성
                    color_img = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2BGR)
                    display_img = cv2.resize(color_img, (960, 540))

                    # 클릭된 row 시각화
                    scale = self.camera_size[1] / 540
                    for row in self.clicked_rows:
                        y_rescaled = int(row / scale)
                        cv2.line(display_img, (0, y_rescaled), (959, y_rescaled), (0, 255, 0), 1)

                    # cm⁻¹ 눈금 표시
                    if self.x_cm is not None:
                        step = len(self.x_cm) // 30
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        for i in range(30):
                            x_index = i * step
                            cm_value = self.x_cm[x_index]
                            x_pos = int(x_index / len(self.x_cm) * display_img.shape[1])

                            cv2.line(display_img, (x_pos, 530), (x_pos, 540), (255, 255, 255), 1)

                            cv2.putText(display_img,
                                        f"{int(cm_value)}",
                                        (x_pos - 5, 525),
                                        font, 0.2, (0, 255, 0), 1)

                    # 디스플레이 출력
                    cv2.imshow("Camera View", display_img)

                    # row 2개 클릭 시 pyqtgraph 출력
                    if len(self.clicked_rows) == 2:
                        r1, r2 = sorted(self.clicked_rows)
                        spectrum = np.sum(img_array[r1:r2, :], axis=0).astype(float)
                        self.update_plot(spectrum)

                    # b 키를 눌러 저장
                    if keyboard.is_pressed("b"):
                        self.save_data(img_array, spectrum, color_img, display_img)

                    # ESC 키를 눌러 종료, 현재 opencv창에서 밖에 안먹음
                    key = cv2.waitKey(10)
                    if key == 27:
                        if self.pg_app is not None:
                            self.pg_app.quit()
                        break
                grab_result.Release()
        finally:
            self.release_resources()


if __name__ == '__main__':
    camera = BaslerCamera(CAMERA_SIZE, EXPOSURE_TIME, GAIN, MONO_MODE)
    camera.run()
