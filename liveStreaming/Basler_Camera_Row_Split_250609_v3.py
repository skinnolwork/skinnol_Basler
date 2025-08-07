from pypylon import pylon
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keyboard
import datetime
import tifffile
from scipy.signal import find_peaks
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 변수 설정
CAMERA_SIZE = [3840, 2160]  # Width, Height
EXPOSURE_TIME = 1000000.0
GAIN = 40.0

MONO_MODE = "Mono12"  # Mono8 또는 Mono12 설정

class BaslerCamera:
    def __init__(self, camera_size, exposure_time, gain, mono_mode):
        self.camera_size = camera_size
        self.exposure_time = exposure_time
        self.gain = gain
        self.mono_mode = mono_mode
        self.camera = None
        self.x_original_nm = np.linspace(950, 750, self.camera_size[0])
        self.x_nm = self.x_nm = self.x_original_nm.copy()
        self.x_cm = 1e7 / self.x_nm  # 초기 cm 변환
        self.peaks = []  # 선택한 피크 인덱스를 저장
        self.use_cm = False
        self.zoom_range = None

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


    def save_image(self, grab_result, intensity1, intensity2, intensity3):
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
            plt.savefig(filename_graph)

        # CSV 저장        
        x_original = self.x_original_nm  # x_original 값 (픽셀 인덱스)
        x_nm = self.x_nm  # 변환된 nm 값
        x_cm = self.x_cm if self.use_cm else 1e7 / x_nm  # cm^-1 값 (use_cm 여부에 따라 결정)

        # 데이터를 열 단위로 결합
        data_to_save = np.column_stack((x_original, x_nm, x_cm, intensity1, intensity2, intensity3))
        
        filename_csv = f"{save_directory}/data_{self.mono_mode}_Row_{timestamp}.csv"
        header = "x_original,nm,cm^-1,intensity1,intensity2,intensity3"
        np.savetxt(filename_csv, data_to_save, delimiter=",", fmt="%.2f", header=header, comments="")
        print(f"CSV file saved as {filename_csv}")

    def process_image(self, img_array):
        plt.clf()

        # 🔧 고정된 3분할 영역 설정 (30px 제거 후 700px씩)
        section_height = 700
        base_start = 30

        section1 = img_array[base_start:base_start + section_height, :]
        section2 = img_array[base_start + section_height:base_start + 2 * section_height, :]
        section3 = img_array[base_start + 2 * section_height:base_start + 3 * section_height, :]

        intensity1 = np.sum(section1, axis=0).astype(float)
        intensity2 = np.sum(section2, axis=0).astype(float)
        intensity3 = np.sum(section3, axis=0).astype(float)

        return (intensity1, intensity2, intensity3), img_array



    def add_annotations(self, img_array):
        img_colored = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        if self.mono_mode == "Mono12":
            img_array_16bit = (img_array << 4).astype(np.uint16)
            img_colored = cv2.merge((img_array_16bit, img_array_16bit, img_array_16bit))

        # 선택된 행/범위 없음 → 주석 표시 생략
        text = ""  # 표시할 텍스트 없음

        if hasattr(self, "column_cutoff"):
            col_x = self.column_cutoff
            if self.mono_mode == "Mono8":
                cv2.line(img_colored, (col_x, 0), (col_x, self.camera_size[1]), (0, 255, 0), 3)
            elif self.mono_mode == "Mono12":
                cv2.line(img_colored, (col_x, 0), (col_x, self.camera_size[1]), (0, 65535, 0), 3)

        if text:
            if self.mono_mode == "Mono8":
                cv2.putText(img_colored, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            elif self.mono_mode == "Mono12":
                cv2.putText(img_colored, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 65535, 0), 3)

        # X축 표시 그대로 유지
        x_length, y_length = self.camera_size
        axis_wavelength = np.linspace(950, 750, x_length)
        for i, wavelength in enumerate(axis_wavelength):
            if i % 100 == 0:
                start_point = (i, y_length - 40)
                end_point = (i, y_length - 20)
                if self.mono_mode == "Mono8":
                    cv2.line(img_colored, start_point, end_point, (255, 0, 0), 3)
                    cv2.putText(img_colored, f"{int(wavelength)}", (i, y_length - 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.line(img_colored, start_point, end_point, (0, 0, 65535), 3)
                    cv2.putText(img_colored, f"{int(wavelength)}", (i, y_length - 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 65535, 0), 2)

        return img_colored


    def select_two_peaks(self):
        print("Select two peaks on the graph (Press Enter after selection).")
        peak_points = plt.ginput(n=2, timeout=0)
        if len(peak_points) != 2:
            print("Error: Two peaks must be selected.")
            return None
        # 두 피크의 인덱스를 반환
        return [np.abs(self.x_original_nm - peak[0]).argmin() for peak in peak_points]

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

        p_pressed_count = 0
        try:
            while self.camera.IsGrabbing():
                grab_result = self.camera.RetrieveResult(int(self.exposure_time), pylon.TimeoutHandling_ThrowException)

                if grab_result.GrabSucceeded():
                    img_array = grab_result.Array
                    if self.mono_mode == "Mono12":
                        img_array &= 0xFFF

                    (intensity1, intensity2, intensity3), processed_img = self.process_image(img_array)

                    self.display_camera_feed(img_array, (intensity1, intensity2, intensity3))

                    if keyboard.is_pressed("b"):
                        self.save_image(grab_result, intensity1, intensity2, intensity3)

                    if keyboard.is_pressed("'"):
                        p_pressed_count += 1
                        if p_pressed_count == 1:
                            self.peaks = self.select_two_peaks()
                            if self.peaks:
                                self.adjust_axis_based_on_peaks(
                                    852.1, 898.2, 1001.4, 1602.3
                                )
                        elif p_pressed_count == 2:
                            self.select_zoom_region()
                            p_pressed_count = 0

                    if keyboard.is_pressed("esc"):
                        break

                grab_result.Release()
                plt.pause(0.01)

        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.release_resources()

    def select_zoom_region(self):
        """ P를 한 번 더 눌렀을 때, 확대할 영역을 설정하는 함수 """
        zoom_points = plt.ginput(n=2, timeout=0)  # 두 점 선택

        if len(zoom_points) == 2:
            (x1, y1), (x2, y2) = zoom_points
            self.zoom_x_range = [max(x1, x2), min(x1, x2)]
            self.zoom_y_range = [min(y1, y2), max(y1, y2)]
            print(f"확대 범위 설정: X [{self.zoom_x_range[0]:.2f}, {self.zoom_x_range[1]:.2f}], Y [{self.zoom_y_range[0]:.2f}, {self.zoom_y_range[1]:.2f}]")

    def display_camera_feed(self, img_array, intensity_tuple):
        intensity1, intensity2, intensity3 = intensity_tuple
        if self.mono_mode == "Mono12":
            img_array = ((img_array / 4095) * 255).astype(np.uint8)

        img_colored = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        cv2.imshow(f'Basler Camera - {self.mono_mode}', img_colored)

        plt.clf()
        x_axis = self.x_cm if self.use_cm else self.x_original_nm
        y_max = max(np.max(intensity1), np.max(intensity2), np.max(intensity3))

        if self.use_cm:
            total = len(x_axis)
            fifth = total // 5

            ax = plt.subplot(3, 2, 1)
            ax.plot(x_axis[:fifth], intensity1[:fifth], color='red', linewidth=1)
            ax.plot(x_axis[:fifth], intensity2[:fifth], color='green', linewidth=1)
            ax.plot(x_axis[:fifth], intensity3[:fifth], color='blue', linewidth=1)
            ax.set_ylabel("Intensity")
            ax.set_ylim([0, y_max])
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.tick_params(axis='x', labelsize=8)
            ax.grid(which='major', color='gray', linestyle='--', linewidth=0.6)

            ax = plt.subplot(3, 2, 2)
            ax.plot(x_axis, intensity1, color='red', linewidth=1, label='Top')
            ax.plot(x_axis, intensity2, color='green', linewidth=1, label='Middle')
            ax.plot(x_axis, intensity3, color='blue', linewidth=1, label='Bottom')
            ax.set_ylabel("Intensity")
            ax.set_ylim([0, y_max])
            ax.legend(loc='upper right', fontsize=8)

            ax = plt.subplot(3, 1, 2)
            ax.plot(x_axis[fifth:3*fifth], intensity1[fifth:3*fifth], color='red', linewidth=1)
            ax.plot(x_axis[fifth:3*fifth], intensity2[fifth:3*fifth], color='green', linewidth=1)
            ax.plot(x_axis[fifth:3*fifth], intensity3[fifth:3*fifth], color='blue', linewidth=1)
            ax.set_ylabel("Intensity")
            ax.set_ylim([0, y_max])
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.tick_params(axis='x', labelsize=8)
            ax.grid(which='major', color='gray', linestyle='--', linewidth=0.6)

            ax = plt.subplot(3, 1, 3)
            ax.plot(x_axis[3*fifth:], intensity1[3*fifth:], color='red', linewidth=1)
            ax.plot(x_axis[3*fifth:], intensity2[3*fifth:], color='green', linewidth=1)
            ax.plot(x_axis[3*fifth:], intensity3[3*fifth:], color='blue', linewidth=1)
            ax.set_ylabel("Intensity")
            ax.set_ylim([0, y_max])
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.tick_params(axis='x', labelsize=8)
            ax.grid(which='major', color='gray', linestyle='--', linewidth=0.6)

            plt.tight_layout()
            plt.pause(0.001)
        else:
            plt.plot(x_axis, intensity1, color='red', label='Top')
            plt.plot(x_axis, intensity2, color='green', label='Middle')
            plt.plot(x_axis, intensity3, color='blue', label='Bottom')

            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Intensity")
            plt.title("Real-Time Spectrum (3 Regions)")
            plt.ylim([0, max(np.max(intensity1), np.max(intensity2), np.max(intensity3))])
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.pause(0.001)

if __name__ == '__main__':
    camera = BaslerCamera(CAMERA_SIZE, EXPOSURE_TIME, GAIN, MONO_MODE)
    camera.run()