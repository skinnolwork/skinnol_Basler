from pypylon import pylon
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keyboard
import datetime
import tifffile
import os

# 변수 설정
CAMERA_SIZE = [3840, 2160]
EXPOSURE_TIME = 1000000.0
GAIN = 40.0
MONO_MODE = "Mono12"

class BaslerCamera:
    def __init__(self, camera_size, exposure_time, gain, mono_mode):
        self.camera_size = camera_size
        self.exposure_time = exposure_time
        self.gain = gain
        self.mono_mode = mono_mode
        self.camera = None

        self.x_original_nm = np.linspace(950, 750, self.camera_size[0])  # 원래 nm
        self.x_axis = self.x_original_nm.copy()  # 보정된 x축 (초기엔 동일)
        self.awaiting_peaks = True
        self.clicked_rows = []

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

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self.mouse_click_callback)

    def mouse_click_callback(self, event):
        if not self.awaiting_peaks or not event.inaxes:
            return

        if not hasattr(self, 'clicked_temp'):
            self.clicked_temp = []

        self.clicked_temp.append(event.xdata)
        print(f"[CLICK] x={event.xdata:.2f}")
        self.ax.plot(event.xdata, event.ydata, 'o', color='blue', markersize=5)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if len(self.clicked_temp) == 2:
            self.awaiting_peaks = False
            peak_indices = [np.abs(self.x_original_nm - x).argmin() for x in self.clicked_temp]
            print("[INFO] 두 피크 선택 완료 → x축 재정의 중...")
            self.adjust_axis_based_on_peaks(
                peak_indices[0], peak_indices[1],
                1001.4, 1602.3  # 원하는 보정값 (임의의 선형 단위)
            )
            del self.clicked_temp

    def adjust_axis_based_on_peaks(self, idx1, idx2, new_x1, new_x2):
        if idx1 == idx2:
            print("[ERROR] 동일한 두 피크를 선택했습니다.")
            return

        old_x1 = self.x_original_nm[idx1]
        old_x2 = self.x_original_nm[idx2]
        slope = (new_x2 - new_x1) / (old_x2 - old_x1)
        intercept = new_x1 - slope * old_x1
        self.x_axis = slope * self.x_original_nm + intercept
        print(f"[✓] X축 재정의 완료: {self.x_axis[0]:.1f} ~ {self.x_axis[-1]:.1f}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            scale = self.camera_size[1] / 540
            row = int(y * scale)
            if len(self.clicked_rows) >= 2:
                self.clicked_rows = []
            self.clicked_rows.append(row)

    def release_resources(self):
        self.camera.StopGrabbing()
        self.camera.Close()
        cv2.destroyAllWindows()
        print("[✓] Camera closed.")

    def save_calibration_csv(self):
        x_start, x_end = sorted([self.x_axis[0], self.x_axis[-1]])
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = f"{date_str},{x_start:.2f},{x_end:.2f}\n"
        csv_path = "calibration_data.csv"
        write_header = not os.path.exists(csv_path)

        with open(csv_path, "a") as f:
            if write_header:
                f.write("date,x_start,x_end\n")
            f.write(row)

        print(f"[✓] Calibration saved with x range: {x_start:.2f} ~ {x_end:.2f}")

    def process_image(self, img_array):
        if len(self.clicked_rows) == 2:
            r1, r2 = sorted(self.clicked_rows)
            sub_img = img_array[r1:r2+1, :]
            intensity = np.sum(sub_img, axis=0).astype(float)
        else:
            intensity = np.sum(img_array, axis=0).astype(float)
        return intensity, img_array

    def display_camera_feed(self, img_array, intensity):
        y_max = np.max(intensity)
        self.ax.clear()
        self.ax.plot(self.x_axis, intensity, color='red')
        self.ax.set_xlabel("Corrected X-Axis")
        self.ax.set_ylabel("Intensity")
        self.ax.set_ylim([0, y_max])
        self.ax.set_title("Real-Time Spectrum")
        self.ax.grid(True)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self):
        self.initialize_camera()

        cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera View", 960, 540)
        cv2.setMouseCallback("Camera View", self.mouse_callback)

        try:
            while self.camera.IsGrabbing():
                grab_result = self.camera.RetrieveResult(int(self.exposure_time), pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    img_array = grab_result.Array & 0xFFF if self.mono_mode == "Mono12" else grab_result.Array
                    disp_img = img_array.astype(np.float32)
                    min_val = np.percentile(disp_img, 1)
                    max_val = np.percentile(disp_img, 99)

                    if max_val - min_val < 1e-5:
                        disp_img = np.zeros_like(disp_img, dtype=np.uint8)
                    else:
                        disp_img = np.clip((disp_img - min_val) / (max_val - min_val) * 255.0, 0, 255).astype(np.uint8)

                    color_img = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2BGR)
                    display_img = cv2.resize(color_img, (960, 540))

                    scale = self.camera_size[1] / 540
                    for row in self.clicked_rows:
                        y_rescaled = int(row / scale)
                        cv2.line(display_img, (0, y_rescaled), (959, y_rescaled), (0, 255, 0), 1)

                    cv2.imshow("Camera View", display_img)

                    intensity, _ = self.process_image(img_array)
                    if len(self.clicked_rows) == 2:
                        self.display_camera_feed(img_array, intensity)

                    if keyboard.is_pressed("b"):
                        self.save_calibration_csv()

                    if keyboard.is_pressed("esc"):
                        plt.close('all')
                        break

                grab_result.Release()
                plt.pause(0.01)
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            self.release_resources()

if __name__ == '__main__':
    camera = BaslerCamera(CAMERA_SIZE, EXPOSURE_TIME, GAIN, MONO_MODE)
    camera.run()
