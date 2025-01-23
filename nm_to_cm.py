import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import make_interp_spline
import datetime
import os


class PeakDetection:
    def __init__(self, file_path):
        self.file_path = file_path
        self.x_values = None
        self.intensity = None
        self.smoothed_intensity = None
        self.peaks = None
        self.corrected_intensity = None

    def load_data(self):
        data = pd.read_csv(self.file_path)
        self.x_values = data['X'].values
        self.intensity = data['Y'].values

    def apply_smoothing(self, window_length=31, polyorder=1):
        self.smoothed_intensity = savgol_filter(self.intensity, window_length=window_length, polyorder=polyorder)

    @staticmethod
    def adjust_peaks_to_original(intensity, peaks, window=10):
        adjusted_peaks = []
        for peak in peaks:
            start = max(0, peak - window)
            end = min(len(intensity), peak + window + 1)
            local_peak = start + np.argmax(intensity[start:end])  # Find local maximum in the range
            adjusted_peaks.append(local_peak)
        return np.array(adjusted_peaks)

    def detect_peaks(self, distance=30, prominence_ratio=0.2):
        threshold_index = int(0.3 * len(self.smoothed_intensity))
        dynamic_height = np.sort(self.smoothed_intensity)[threshold_index]
        peaks, _ = find_peaks(
            self.smoothed_intensity,
            prominence=prominence_ratio * dynamic_height,
            height=dynamic_height,
            distance=distance
        )

        peaks = np.delete(peaks, [0,1])
        self.peaks = self.adjust_peaks_to_original(self.intensity, peaks, window=10)

    def adjust_graph(self, target_x_3rd, target_x_last):
        current_x_3rd = self.x_values[self.peaks[2]]
        current_x_last = self.x_values[self.peaks[-1]]

        m = (target_x_last - target_x_3rd) / (current_x_last - current_x_3rd)
        b = target_x_3rd - m * current_x_3rd

        self.x_values = m * self.x_values + b

    def convert_to_cm(self):
        self.x_values = 1e7 / self.x_values

    def baseline_correction(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_values, self.intensity, label='Intensity', color='blue')
        plt.xlabel('Pixel Position (x-axis)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Select Baseline Points')
        plt.legend()

        plt.gca().invert_xaxis()

        print("그래프에서 Baseline으로 사용할 포인트를 선택하세요 (엔터로 종료).")
        baseline_points = plt.ginput(n=-1, timeout=0)
        if len(baseline_points) == 0:
            print("Error: No points selected.")
            return

        baseline_points = np.array(baseline_points)

        # 중복된 X 좌표 제거
        baseline_x = baseline_points[:, 0]
        baseline_y = baseline_points[:, 1]
        unique_indices = np.unique(baseline_x, return_index=True)[1]
        baseline_x = baseline_x[unique_indices]
        baseline_y = baseline_y[unique_indices]

        # Baseline 계산
        spline = make_interp_spline(baseline_x, baseline_y, k=1)
        baseline_fit = spline(self.x_values)

        # 데이터 보정
        self.corrected_intensity = self.intensity - baseline_fit


    def visualize_cm(self, title):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_values, self.corrected_intensity, label='Corrected Intensity', color='red')
        plt.scatter(self.x_values[self.peaks], self.corrected_intensity[self.peaks], color='orange', label='Peaks')
        for i, (x, y) in enumerate(zip(self.x_values[self.peaks], self.corrected_intensity[self.peaks])):
            plt.text(x, y, f"{i + 1}", fontsize=9, ha='center', va='bottom')
        plt.xlabel('Wavenumber (cm^-1)')
        plt.ylabel('Intensity (a.u.)')
        plt.title(title)
        plt.legend()

        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))
        plt.grid(visible=True, which='major', axis='x', linewidth=0.5, linestyle='--', alpha=0.7)

        for label in plt.gca().get_xticklabels():
            text = label.get_text()
            try:
                if text and int(float(text)) % 100 != 0:
                    label.set_visible(False)
            except ValueError:
                label.set_visible(False)

        plt.gca().invert_xaxis()

        plt.show()


    def visualize_nm(self, title):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_values, self.corrected_intensity, label='Corrected Intensity', color='red')
        plt.scatter(self.x_values[self.peaks], self.corrected_intensity[self.peaks], color='orange', label='Peaks')
        for i, (x, y) in enumerate(zip(self.x_values[self.peaks], self.corrected_intensity[self.peaks])):
            plt.text(x, y, f"{i + 1}", fontsize=9, ha='center', va='bottom')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.title(title)
        plt.legend()

        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))  # 5 단위로 Grid 설정
        plt.grid(visible=True, which='major', axis='x', linewidth=0.5, linestyle='--', alpha=0.7)

        for label in plt.gca().get_xticklabels():
            text = label.get_text()
            try:
                if text and int(float(text)) % 5 != 0:
                    label.set_visible(False)
            except ValueError:
                label.set_visible(False)


        # X축 반전
        plt.gca().invert_xaxis()

        plt.show()

    def process_all(self, target_x_3rd_nm, target_x_last_nm, target_x_3rd_cm, target_x_last_cm):
        # CSV 데이터 처리
        self.load_data()
        self.apply_smoothing()
        self.detect_peaks()
        self.adjust_graph(target_x_3rd_nm, target_x_last_nm)
        self.baseline_correction()
        self.visualize_nm('Adjusted Graph (nm)')

        # nm → cm^-1 변환 및 피크 조정
        self.convert_to_cm()
        self.adjust_graph(target_x_3rd_cm, target_x_last_cm)
        self.visualize_cm('Adjusted Graph (cm^-1)')


date_folder = datetime.datetime.now().strftime("%Y%m%d")
save_directory = os.path.join(os.getcwd(), date_folder)

file_name = "data_Mono12_Row_20250123_094237.csv"
file_path = os.path.join(save_directory, file_name)

target_x_3rd_nm = 852.1
target_x_last_nm = 898.2
target_x_3rd_cm = 1001.4
target_x_last_cm = 1602.3

peak_detector = PeakDetection(file_path)
peak_detector.process_all(target_x_3rd_nm, target_x_last_nm, target_x_3rd_cm, target_x_last_cm)