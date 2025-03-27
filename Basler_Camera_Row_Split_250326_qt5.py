from pypylon import pylon
import numpy as np
import cv2
import keyboard
import datetime
import tifffile
from scipy.signal import find_peaks
import os
from graph_window import GraphWindow  # 위에서 만든 클래스

# 변수 설정
CAMERA_SIZE = [3840, 2160]  # Width, Height
EXPOSURE_TIME = 4000.0
GAIN = 10.0

MONO_MODE = "Mono12"  # Mono8 또는 Mono12 설정

class BaslerCamera:
    def __init__(self, camera_size, exposure_time, gain, mono_mode):
        self.camera_size = camera_size
        self.exposure_time = exposure_time
        self.gain = gain
        self.mono_mode = mono_mode
        self.camera = None
        self.selected_rows = []  # 선택한 행 범위 저장
        self.x_original_nm = np.linspace(950, 750, self.camera_size[0])
        self.x_nm = self.x_nm = self.x_original_nm.copy()
        self.x_cm = 1e7 / self.x_nm  # 초기 cm 변환
        self.peaks = []  # 선택한 피크 인덱스를 저장
        self.use_cm = False
        self.zoom_range = None

        self.graph_1st = GraphWindow("1st")
        self.graph_2nd = GraphWindow("2nd")
        self.graph_3rd = GraphWindow("3rd")

        # 피크 선택용 창
        self.graph_peak = GraphWindow("Select Peaks")
        self.graph_peak.peak_callback = self.handle_peak_selection

        self.graph_1st.show()
        self.graph_2nd.show()
        self.graph_3rd.show()
        self.graph_peak.show()
        self.real_time_peak_mode = True  # Select Peaks 창 실시간 갱신 여부

    def handle_peak_selection(self, x_clicks):
        print("Peaks selected:", x_clicks)
        self.peaks = [np.abs(self.x_original_nm - x).argmin() for x in x_clicks]

        self.adjust_axis_based_on_peaks(
            target_x_peak1_nm=852.1, target_x_peak2_nm=898.2,
            target_x_peak1_cm=1001.4, target_x_peak2_cm=1602.3
        )

    def initialize_camera(self):
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
        self.camera.StopGrabbing()
        self.camera.Close()
        cv2.destroyAllWindows()
        print("Camera and resources released successfully.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.selected_rows) == 2:
                # 클릭 상태 초기화
                self.selected_rows = []
            self.selected_rows.append(y)
            if len(self.selected_rows) == 2:
                self.selected_rows.sort()  # 선택한 행 정렬
                print(f"Selected Rows Range: {self.selected_rows}")

    def detect_peaks(self, intensity, distance=50, prominence_ratio=0.5):
        threshold_index = int(0.2 * len(intensity))
        dynamic_height = np.sort(intensity)[threshold_index]
        peaks, _ = find_peaks(
            intensity,
            prominence=prominence_ratio * dynamic_height,
            height=dynamic_height,
            distance=distance
        )

        # filtered_peaks = self.filter_peaks_by_falloff(peaks, intensity)

        highlighted_intensity = np.zeros_like(intensity)
        for peak in peaks:
            highlighted_intensity[peak] = intensity[peak]

        return peaks, highlighted_intensity

    def save_image(self, grab_result, row_intensity):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        date_folder = datetime.datetime.now().strftime("%Y%m%d")
        save_directory = f"./{date_folder}"
        os.makedirs(save_directory, exist_ok=True)

        # 파일 이름 설정
        original_filename = f"{save_directory}/original_{self.mono_mode}_{timestamp}.{'bmp' if self.mono_mode == 'Mono8' else 'tiff'}"
        resized_png_filename = f"{save_directory}/resized_{self.mono_mode}_{timestamp}.png"
        annotated_png_filename = f"{save_directory}/annotated_{self.mono_mode}_{timestamp}.png"
        normalized_tiff_filename = f"{save_directory}/normalized_{self.mono_mode}_{timestamp}.tiff"

        # 이미지 데이터 가져오기
        img_array = grab_result.Array

        if self.mono_mode == "Mono12":
            img_array &= 0xFFF  # 12비트 데이터 유지

            # TIFF 저장 (원본)
            tifffile.imwrite(original_filename, img_array, dtype="uint16", compression=None)
            print(f"Original TIFF saved as {original_filename} (16-bit TIFF)")

            # 정규화 (0-65535로 확장)
            img_array_normalized = (img_array / 4095.0 * 65535).astype(np.uint16)
            tifffile.imwrite(normalized_tiff_filename, img_array_normalized, dtype="uint16", compression=None)
            print(f"Normalized TIFF saved as {normalized_tiff_filename} (16-bit TIFF)")

            # 384x216 PNG 저장 (정규화 후)
            img_resized = cv2.resize((img_array / 4095.0 * 255).astype(np.uint8), (384, 216), interpolation=cv2.INTER_AREA)
            cv2.imwrite(resized_png_filename, img_resized)
            print(f"Resized PNG saved as {resized_png_filename} (384x216)")

            # 주석이 포함된 PNG 저장
            annotated_image = self.add_annotations(img_array)
            cv2.imwrite(annotated_png_filename, annotated_image)
            print(f"Annotated PNG saved as {annotated_png_filename} (full-size with annotations)")

        elif self.mono_mode == "Mono8":
            # BMP 저장 (원본)
            cv2.imwrite(original_filename, img_array)
            print(f"Original BMP saved as {original_filename} (8-bit BMP)")

            # 384x216 PNG 저장
            img_resized = cv2.resize(img_array, (384, 216), interpolation=cv2.INTER_AREA)
            cv2.imwrite(resized_png_filename, img_resized)
            print(f"Resized PNG saved as {resized_png_filename} (384x216)")

            # 주석이 포함된 PNG 저장
            annotated_image = self.add_annotations(img_array)
            cv2.imwrite(annotated_png_filename, annotated_image)
            print(f"Annotated PNG saved as {annotated_png_filename} (full-size with annotations)")


        if self.use_cm:
            filename_graph = f"{save_directory}/graph_cm_{self.mono_mode}_Row_{timestamp}.png"
            print(f"Graph PNG saved as {filename_graph} (384x216)")

        # CSV 저장
        x_length = len(row_intensity)
        x_original = self.x_original_nm  # x_original 값 (픽셀 인덱스)
        x_nm = self.x_nm  # 변환된 nm 값
        x_cm = self.x_cm if self.use_cm else 1e7 / x_nm  # cm^-1 값 (use_cm 여부에 따라 결정)
        intensity = row_intensity  # Intensity 값

        # 데이터를 열 단위로 결합
        data_to_save = np.column_stack((x_original, x_nm, x_cm, intensity))
        
        filename_csv = f"{save_directory}/data_{self.mono_mode}_Row_{timestamp}.csv"
        header = "x_original,nm,cm^-1,intensity"  # CSV 헤더
        np.savetxt(filename_csv, data_to_save, delimiter=",", fmt="%.2f", header=header, comments="")
        print(f"CSV file saved as {filename_csv}")

    def process_image(self, img_array):

        # 범위 모드 데이터 처리
        if len(self.selected_rows) == 2:
            start_row, end_row = self.selected_rows
            rows_to_sum = img_array[start_row:end_row, :]
            row_intensity = np.sum(rows_to_sum, axis=0).astype(float)

        else:
            row_intensity = img_array[2000, :].astype(float)  # 기본 행



         # 반환: 계산된 강도 데이터와 전체 이미지 데이터
        return row_intensity, img_array

        
    # display에 표시되는 정보 출력 함수
    def add_annotations(self, img_array):

        # 선택된 행/범위가 표시된 이미지를 생성
        img_colored = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        if self.mono_mode == "Mono8":
            # 선택한 행/범위 표시 추가
            if len(self.selected_rows) == 1:
                start_row = self.selected_rows[0]
                cv2.line(img_colored, (0, start_row), (self.camera_size[0], start_row), (0, 255, 0), 3)
            elif len(self.selected_rows) == 2:
                start_row, end_row = self.selected_rows
                cv2.line(img_colored, (0, start_row), (self.camera_size[0], start_row), (0, 255, 0), 3)
                cv2.line(img_colored, (0, end_row), (self.camera_size[0], end_row), (0, 255, 0), 3)

        elif self.mono_mode == "Mono12":
            # Mono12 데이터를 16비트로 확장
            img_array_16bit = (img_array << 4).astype(np.uint16)  

            # 단일 채널을 3채널 컬러 이미지로 변환
            img_colored = cv2.merge((img_array_16bit, img_array_16bit, img_array_16bit))

            # 이미지를 컬러로 변환하지 않고 16비트 이미지 그대로 선 그리기
            if len(self.selected_rows) == 1:
                start_row = self.selected_rows[0]
                cv2.line(img_colored, (0, start_row), (self.camera_size[0], start_row), (0, 65535, 0), 3)
            elif len(self.selected_rows) == 2:
                start_row, end_row = self.selected_rows
                cv2.line(img_colored, (0, start_row), (self.camera_size[0], start_row), (0, 65535, 0), 3)
                cv2.line(img_colored, (0, end_row), (self.camera_size[0], end_row), (0, 65535, 0), 3)


        # Row 위치 텍스트 생성
        text = ""
        if len(self.selected_rows) >= 1:
            start_row = self.selected_rows[0]
            text = f"Start: {start_row}"
        if len(self.selected_rows) == 2:
            start_row, end_row = self.selected_rows
            text = f"Range: {start_row} ~ {end_row}"

        if self.mono_mode == "Mono8":
            cv2.putText(img_colored, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        elif self.mono_mode == "Mono12":
            cv2.putText(img_colored, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 65535, 0), 3)

        # X축 표시 추가
        x_length, y_length = self.camera_size  # X축과 Y축 크기 가져오기
        axis_wavelength = np.linspace(950, 750, x_length)  # X축 linspace 생성
        for i, wavelength in enumerate(axis_wavelength):
            if i % 100 == 0:  # 100픽셀 간격으로 표시
                start_point = (i, y_length - 40)
                end_point = (i, y_length - 20)
                if (self.mono_mode == "Mono8"):
                    cv2.line(img_colored, start_point, end_point, (255, 0, 0), 3)  # 파란색 선
                    cv2.putText(img_colored, f"{int(wavelength)}", (i, y_length - 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.line(img_colored, start_point, end_point, (0, 0, 65535), 3)  # 파란색 선
                    cv2.putText(img_colored, f"{int(wavelength)}", (i, y_length - 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 65535, 0), 2)

        return img_colored

    def adjust_axis_based_on_peaks(self, target_x_peak1_nm, target_x_peak2_nm, target_x_peak1_cm, target_x_peak2_cm):
        # 선택한 두 피크 위치
        current_x_peak1_nm = self.x_original_nm[self.peaks[0]]
        current_x_peak2_nm = self.x_original_nm[self.peaks[1]]

        # nm 변환 계산
        m_nm = (target_x_peak2_nm - target_x_peak1_nm) / (current_x_peak2_nm - current_x_peak1_nm)
        b_nm = target_x_peak1_nm - m_nm * current_x_peak1_nm
        
        self.x_nm = m_nm * self.x_original_nm + b_nm
        self.x_cm = 1e7 / self.x_nm

        current_x_peak1_cm = self.x_cm[self.peaks[0]]
        current_x_peak2_cm = self.x_cm[self.peaks[1]]

        m_cm = (target_x_peak2_cm - target_x_peak1_cm) / (current_x_peak2_cm - current_x_peak1_cm)
        b_cm = target_x_peak1_cm - m_cm * current_x_peak1_cm
        self.x_cm = m_cm * self.x_cm + b_cm

        self.use_cm = True
        print("X-axis adjusted based on selected peaks.")


    def run(self):
        self.initialize_camera()
        cv2.namedWindow(f'Basler Camera - {self.mono_mode}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'Basler Camera - {self.mono_mode}', 960, 540)
        cv2.setMouseCallback(f'Basler Camera - {self.mono_mode}', self.mouse_callback)

        p_pressed_count = 0  # "P" 키 눌린 횟수 추적
        try:
            while self.camera.IsGrabbing():
                grab_result = self.camera.RetrieveResult(int(self.exposure_time), pylon.TimeoutHandling_ThrowException)

                if grab_result.GrabSucceeded():
                    img_array = grab_result.Array
                    if self.mono_mode == "Mono12":
                        img_array &= 0xFFF  # Mono12의 하위 12비트 유지

                    row_intensity, processed_img = self.process_image(img_array)

                    # 실시간 그래프 표시 (확대 모드 반영)
                    self.display_camera_feed(img_array, row_intensity)

                    if keyboard.is_pressed("b"):
                        self.save_image(grab_result, row_intensity)

                    if keyboard.is_pressed("p"):
                        self.real_time_peak_mode = False
                        self.graph_peak.update_plot(self.x_original_nm, row_intensity)
                        print("Select 2 peaks on the 'Select Peaks' window.")

                    if keyboard.is_pressed("esc"):
                        break

                grab_result.Release()

        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.release_resources()

    def display_camera_feed(self, img_array, row_intensity):
        if self.mono_mode == "Mono12":
            img_array = ((img_array / 4095) * 255).astype(np.uint8)

        img_colored = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Row 위치 텍스트 생성
        text = ""
        if len(self.selected_rows) >= 1:
            start_row = self.selected_rows[0]
            text = f"Start: {start_row}"
            cv2.line(img_colored, (0, start_row), (self.camera_size[0], start_row), (0, 255, 0), 3)
        if len(self.selected_rows) == 2:
            start_row, end_row = self.selected_rows
            text = f"Range: {start_row} ~ {end_row}"
            cv2.line(img_colored, (0, start_row), (self.camera_size[0], start_row), (0, 255, 0), 3)
            cv2.line(img_colored, (0, end_row), (self.camera_size[0], end_row), (0, 255, 0), 3)

        # X축 표시 (linspace 사용)
        x_length, y_length = self.camera_size  # X축과 Y축 크기 가져오기
        axis_wavelength = np.linspace(950, 750, x_length)  # X축 linspace 생성
        for i, wavelength in enumerate(axis_wavelength):
            if i % 100 == 0:  # 100픽셀 간격으로 표시
                start_point = (i, y_length - 40)
                end_point = (i, y_length - 20)
                cv2.line(img_colored, start_point, end_point, (255, 0, 0), 3)
                cv2.putText(img_colored, f"{int(wavelength)}", (i , y_length - 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
        # Row 텍스트 표시
        cv2.putText(img_colored, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)        
        cv2.imshow(f'Basler Camera - {self.mono_mode}', img_colored)
        cv2.waitKey(1)

        x_axis = self.x_cm if self.use_cm else self.x_original_nm
        label = "Wavenumber (cm⁻¹)" if self.use_cm else "Wavelength (nm)"

        if not self.use_cm:
            if self.real_time_peak_mode:
                self.graph_peak.update_plot(self.x_original_nm, row_intensity)
        else:
            total = len(x_axis)
            third = total // 3

            self.graph_1st.update_plot(x_axis[:third], row_intensity[:third], xlabel=label)
            self.graph_2nd.update_plot(x_axis[third:2*third], row_intensity[third:2*third], xlabel=label)
            self.graph_3rd.update_plot(x_axis[2*third:], row_intensity[2*third:], xlabel=label)

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    camera = BaslerCamera(CAMERA_SIZE, EXPOSURE_TIME, GAIN, MONO_MODE)
    camera.run()

    sys.exit(app.exec_())  # PyQt 이벤트 루프 실행
