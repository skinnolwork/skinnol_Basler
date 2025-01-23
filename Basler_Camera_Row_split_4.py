from pypylon import pylon
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keyboard
import datetime
import tifffile
from scipy.signal import find_peaks
import os

# 변수 설정
CAMERA_SIZE = [3840, 2160]  # Width, Height
EXPOSURE_TIME = 3000.0
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
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.selected_rows) == 2:
                # 클릭 상태 초기화
                self.selected_rows = []
            self.selected_rows.append(y)
            if len(self.selected_rows) == 2:
                self.selected_rows.sort()  # 선택한 행 정렬
                print(f"Selected Rows Range: {self.selected_rows}")

    def detect_peaks(self, intensity, distance=50, prominence_ratio=0.5):
        """피크를 감지하고 하이라이트 영역을 반환"""
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

    def save_image(self, row_intensity, img_array):
        """이미지 저장"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        date_folder = datetime.datetime.now().strftime("%Y%m%d")  # 오늘 날짜 폴더 이름
        # 저장할 디렉토리 경로 생성
        save_directory = f"./{date_folder}"
        os.makedirs(save_directory, exist_ok=True)  # 디렉토리 생성 (이미 존재하면 무시)

        # 선택된 행/범위가 표시된 이미지를 생성
        img_colored = self.add_annotations(img_array)
        
        filename_image = f"{save_directory}/capture_{self.mono_mode}_Row_{timestamp}.{'bmp' if self.mono_mode == 'Mono8' else 'tiff'}"
        # 카메라 저장
        if self.mono_mode == "Mono8":
            cv2.imwrite(filename_image, img_colored)
            print(f"Image saved as {filename_image} (8-bit BMP format)")

        elif self.mono_mode == "Mono12":
            tifffile.imwrite(filename_image, img_colored, compression=None)
            print(f"Image saved as {filename_image} (16-bit TIFF format)")

        # 그래프 저장
        filename_graph = f"{save_directory}/graph_{self.mono_mode}_Row_{timestamp}.png"
        plt.savefig(filename_graph)
        print(f"Graph saved as {filename_graph}")

        # CSV 저장
        x_length = len(row_intensity)
        x_values = np.linspace(950, 750, x_length)  # X축 값 생성
        data_to_save = np.column_stack((x_values, row_intensity))  # X와 Y 값을 2D 배열로 결합

        filename_csv = f"{save_directory}/data_{self.mono_mode}_Row_{timestamp}.csv"
        np.savetxt(filename_csv, data_to_save, delimiter=",", fmt="%.2f", header="X,Y", comments="")
        print(f"CSV file saved as {filename_csv}")

    def process_image(self, img_array):
        """이미지 데이터 처리 및 그래프 업데이트"""
        plt.clf()

        # # Mono12 데이터를 8비트로 스케일링
        # if self.mono_mode == "Mono12":
        #     img_array = ((img_array / 4095) * 255).astype(np.uint8)


        # 범위 모드 데이터 처리
        if len(self.selected_rows) == 2:
            start_row, end_row = self.selected_rows
            rows_to_sum = img_array[start_row:end_row, :]
            row_intensity = np.sum(rows_to_sum, axis=0).astype(float)

            if (self.mono_mode == "Mono8"):
                plt.ylim([0, (end_row - start_row) * 255])
            else:
                plt.ylim([0, (end_row - start_row) * 4095])

        else:
            row_intensity = img_array[2000, :].astype(float)  # 기본 행
            if (self.mono_mode == "Mono8"):
                plt.ylim([0, 255])  # 기본 Intensity 범위
            else:
                plt.ylim([0, 4095])  # 기본 Intensity 범위


         # 반환: 계산된 강도 데이터와 전체 이미지 데이터
        return row_intensity, img_array

    def display_camera_feed(self, img_array, row_intensity):
        """카메라 화면 표시"""
        if self.mono_mode == "Mono12":
            img_array = ((img_array / 4095) * 255).astype(np.uint8)

        img_colored = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Row 위치 텍스트 생성
        text = ""
        if len(self.selected_rows) >= 1:
            start_row = self.selected_rows[0]
            cv2.line(img_colored, (0, start_row), (self.camera_size[0], start_row), (0, 255, 0), 3)
            text = f"Start: {start_row}"
        if len(self.selected_rows) == 2:
            start_row, end_row = self.selected_rows
            cv2.line(img_colored, (0, start_row), (self.camera_size[0], start_row), (0, 255, 0), 3)
            cv2.line(img_colored, (0, end_row), (self.camera_size[0], end_row), (0, 255, 0), 3)
            text = f"Range: {start_row} ~ {end_row}"

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

        # 피크 검출
        # peaks, highlighted_intensity = self.detect_peaks(row_intensity)

        # X축 데이터 설정
        # x_length = self.camera_size[0]  # 이미지의 가로 크기
        # axis_wavelength = np.linspace(950, 750, x_length)  # X축 범위와 동일한 Wavelength 생성

        # 그래프 업데이트
        # x_axis = np.linspace(0, self.camera_size[0], self.camera_size[0])
        # plt.plot(axis_wavelength, row_intensity, color='red', linewidth=1)
        # # 피크 검출
        # plt.plot(axis_wavelength, highlighted_intensity, label='Highlighted Intensity', color='blue')
        # for peak in peaks:
        #     plt.text(axis_wavelength[peak], row_intensity[peak] + 10, f"{axis_wavelength[peak]:.2f}", color='green', fontsize=8, ha='center')
    
        # X축 데이터 4등분
        x_length = len(row_intensity)
        axis_wavelength = np.linspace(950, 750, x_length)  # 전체 X축 데이터
        num_splits = 4  # 4등분
        split_size = x_length // num_splits  # 각 구간의 크기
        split_ranges = [slice(i * split_size, (i + 1) * split_size) for i in range(num_splits)]

        # 기존 창을 초기화하고 한 번에 4개 그래프 그리기
        plt.clf()  # 이전 그래프 초기화
        for i, split_range in enumerate(split_ranges):
            plt.subplot(4, 1, i + 1)  # 4행 1열의 서브플롯 중 i+1번째
            plt.plot(
                axis_wavelength[split_range],
                row_intensity[split_range],
                color="blue"
            )
            
            y_min = np.min(row_intensity[split_range]) 
            y_max = np.max(row_intensity[split_range])

            plt.ylim(y_min, y_max)
            plt.ylabel("Intensity")
            plt.grid()

            # X축 레이블과 간격 표시
            x_ticks = np.linspace(axis_wavelength[split_range.start], axis_wavelength[split_range.stop - 1], 5)  # 5개 간격으로 표시
            plt.xticks(x_ticks)  # X축에 간격 적용

            # 모든 서브플롯에서 X축 레이블 표시
            plt.tick_params(labelbottom=True)  # X축 레이블 표시

        plt.tight_layout()  # 서브플롯 간격 조정
        plt.pause(0.1)  # 그래프 갱신

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

                    row_intensity, processed_img = self.process_image(img_array)
                    self.display_camera_feed(img_array, row_intensity)

                    if keyboard.is_pressed("b"):
                        self.save_image(row_intensity, processed_img)

                    if keyboard.is_pressed("esc"):
                        break

                grab_result.Release()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.release_resources()


if __name__ == '__main__':
    camera = BaslerCamera(CAMERA_SIZE, EXPOSURE_TIME, GAIN, MONO_MODE)
    camera.run()