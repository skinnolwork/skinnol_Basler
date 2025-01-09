from pypylon import pylon
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keyboard
import datetime
import tifffile

# 변수 설정
CAMERA_SIZE = [3840, 2160]  # Width, Height
EXPOSURE_TIME = 3000.0
GAIN = 10.0
SINGLE_OR_MULTI = False  # True: 단일 열 모드, False: 범위 모드
MONO_MODE = "Mono8"  # Mono8 또는 Mono12 설정


class BaslerCamera:
    def __init__(self, camera_size, exposure_time, gain, single_or_multi, mono_mode):
        self.camera_size = camera_size
        self.exposure_time = exposure_time
        self.gain = gain
        self.single_or_multi = single_or_multi
        self.mono_mode = mono_mode
        self.camera = None
        self.selected_columns = []  # 선택한 열 범위 저장

    def initialize_camera(self):
        """카메라 초기화 및 설정"""
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        print(f"Using device: {self.camera.GetDeviceInfo().GetModelName()}")

        # 카메라 크기 및 설정
        self.camera.Width.SetValue(self.camera_size[0])
        self.camera.Height.SetValue(self.camera_size[1])
        self.camera.PixelFormat.SetValue(self.mono_mode)
        self.camera.ExposureTime.SetValue(self.exposure_time)
        self.camera.Gain.SetValue(self.gain)

        # 캡처 시작
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def release_resources(self):
        """카메라 및 리소스 해제"""
        self.camera.StopGrabbing()
        self.camera.Close()
        cv2.destroyAllWindows()
        print("Camera and resources released successfully.")

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭 이벤트 핸들러"""
        if self.single_or_multi:  # 단일 열 모드
            if event == cv2.EVENT_LBUTTONDOWN:
                self.selected_columns = [x]  # 단일 열 설정
                print(f"Selected Column: {x}")
        else:  # 범위 모드
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.selected_columns) == 2:
                    self.selected_columns = []
                self.selected_columns.append(x)
                if len(self.selected_columns) == 2:
                    self.selected_columns.sort()
                    print(f"Selected Column Range: {self.selected_columns}")

    def save_image(self, img_array):
        """이미지 저장"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.mono_mode == "Mono8":
            # 이미지 저장
            filename = f"./capture_Mono8_Column_{timestamp}.bmp"
            cv2.imwrite(filename, img_array)
            print(f"Image saved as {filename} (BMP format)")
            # 그래프 저장
            filename_graph = f"./graph_Mono8_Column_{timestamp}.png"
            plt.savefig(filename_graph)
            print(f"Graph saved as {filename_graph}")
        elif self.mono_mode == "Mono12":
            # 이미지 저장
            filename = f"./capture_Mono12_Column_{timestamp}.tiff"
            img_array_16bit = (img_array << 4).astype(np.uint16)  # Mono12 데이터를 16비트로 확장
            tifffile.imwrite(filename, img_array_16bit, compression=None)
            print(f"Image saved as {filename} (16-bit TIFF format)")
            # 그래프 저장
            filename_graph = f"./graph_Mono12_Column_{timestamp}.png"
            plt.savefig(filename_graph)
            print(f"Graph saved as {filename_graph}")

    def process_image(self, img_array):
        """이미지 데이터 처리 및 그래프 업데이트"""
        plt.clf()

        # Mono12 데이터를 8비트로 스케일링
        if self.mono_mode == "Mono12":
            img_array = ((img_array / 4095) * 255).astype(np.uint8)

        if self.single_or_multi:
            # 단일 열 데이터 처리
            if self.selected_columns:
                col = self.selected_columns[0]
                column_intensity = img_array[:, col]
            else:
                column_intensity = img_array[:, 2000]  # 기본 열
            plt.xlim([0, 255])  # 기본 Intensity 범위
        else:
            # 범위 모드 데이터 처리
            if len(self.selected_columns) == 2:
                start_col, end_col = self.selected_columns
                column_data = img_array[:, start_col:end_col]
                column_intensity = np.sum(column_data, axis=1)
                plt.xlim([0, (end_col - start_col) * 255])
            else:
                column_intensity = img_array[:, 2000]  # 기본 열
                plt.xlim([0, 255])

        # 그래프 업데이트
        y_axis = np.arange(self.camera_size[1])  # 이미지의 높이 (row index)
        plt.plot(column_intensity, y_axis, color='blue', linewidth=1)
        title = f"{self.mono_mode} - Single Column Intensity" if self.single_or_multi else f"{self.mono_mode} - Range Column Intensity"
        plt.title(title)
        plt.xlabel("Intensity")
        plt.ylabel("Pixel Index (Row)")
        plt.gca().invert_yaxis()  # Y축 반전
        plt.grid()
        plt.pause(0.1)

    def display_camera_feed(self, img_array):
        """카메라 화면 표시"""
        if self.mono_mode == "Mono12":
            img_array = ((img_array / 4095) * 255).astype(np.uint8)

        img_colored = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        text = ""
        if self.single_or_multi and self.selected_columns:
            col = self.selected_columns[0]
            text = f"Column: {col}"
            cv2.line(img_colored, (col, 0), (col, self.camera_size[1]), (0, 255, 0), 3)
        elif not self.single_or_multi:
            if len(self.selected_columns) >= 1:
                start_col = self.selected_columns[0]
                text = f"Start: {start_col}"
                cv2.line(img_colored, (start_col, 0), (start_col, self.camera_size[1]), (0, 255, 0), 3)
            if len(self.selected_columns) == 2:
                start_col, end_col = self.selected_columns
                text = f"Range: {start_col} ~ {end_col}"
                cv2.line(img_colored, (start_col, 0), (start_col, self.camera_size[1]), (0, 255, 0), 3)
                cv2.line(img_colored, (end_col, 0), (end_col, self.camera_size[1]), (0, 255, 0), 3)

        # X축 표시 (linspace 사용)
        x_length, y_length = self.camera_size  # X축과 Y축 크기 가져오기
        axis_wavelength = np.linspace(950, 750, x_length)  # X축 linspace 생성
        for i, wavelength in enumerate(axis_wavelength):
            if i % 100 == 0:  # 100픽셀 간격으로 표시
                start_point = (i, y_length - 40)
                end_point = (i, y_length - 20)
                cv2.line(img_colored, start_point, end_point, (255, 0, 0), 3)
                cv2.putText(img_colored, f"{int(wavelength)}", (i , y_length - 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(img_colored, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow(f'Basler Camera - {self.mono_mode}', img_colored)

    def run(self):
        """메인 실행 함수"""
        self.initialize_camera()
        cv2.namedWindow(f'Basler Camera - {self.mono_mode}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'Basler Camera - {self.mono_mode}', 960, 540)
        cv2.setMouseCallback(f'Basler Camera - {self.mono_mode}', self.mouse_callback)

        try:
            while self.camera.IsGrabbing():
                grab_result = self.camera.RetrieveResult(int(self.exposure_time), pylon.TimeoutHandling_ThrowException)

                if grab_result.GrabSucceeded():
                    img_array = grab_result.Array
                    if self.mono_mode == "Mono12":
                        img_array &= 0xFFF  # Mono12의 하위 12비트 유지

                    self.process_image(img_array)
                    self.display_camera_feed(img_array)

                    if keyboard.is_pressed("b"):
                        self.save_image(img_array)

                    if keyboard.is_pressed("esc"):
                        break

                grab_result.Release()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.release_resources()


if __name__ == '__main__':
    camera = BaslerCamera(CAMERA_SIZE, EXPOSURE_TIME, GAIN, SINGLE_OR_MULTI, MONO_MODE)
    camera.run()