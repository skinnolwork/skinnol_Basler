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
        self.x_original = None  # 기존 X 값
        self.x_nm = None        # nm로 변환된 X 값
        self.x_cm = None        # cm^-1로 변환된 X 값
        self.intensity = None
        self.smoothed_intensity = None
        self.peaks = None
        self.corrected_intensity = None

    def load_data(self):
        data = pd.read_csv(self.file_path)
        self.x_original = data['X'].values
        self.intensity = data['Y'].values

    def apply_smoothing(self, window_length=31, polyorder=1):
        self.smoothed_intensity = savgol_filter(self.intensity, window_length=window_length, polyorder=polyorder)

    def adjust_graph_manual(self, target_x_peak1_nm, target_x_peak2_nm, target_x_peak1_cm, target_x_peak2_cm):
        current_x_peak1 = self.x_nm[self.peaks[0]]
        current_x_peak2 = self.x_nm[self.peaks[1]]

        m_nm = (target_x_peak2_nm - target_x_peak1_nm) / (current_x_peak2 - current_x_peak1)
        b_nm = target_x_peak1_nm - m_nm * current_x_peak1
        self.x_nm = m_nm * self.x_nm + b_nm

        self.convert_to_cm()

        current_x_peak1_cm = self.x_cm[self.peaks[0]]
        current_x_peak2_cm = self.x_cm[self.peaks[1]]

        m_cm = (target_x_peak2_cm - target_x_peak1_cm) / (current_x_peak2_cm - current_x_peak1_cm)
        b_cm = target_x_peak1_cm - m_cm * current_x_peak1_cm
        self.x_cm = m_cm * self.x_cm + b_cm


    def convert_to_cm(self):
        self.x_cm  = 1e7 / self.x_nm

    def baseline_correction(self):
        sorted_indices = np.argsort(self.x_original)[::-1]  # 내림차순 정렬
        x_decreasing = self.x_original[sorted_indices]
        intensity_sorted = self.intensity[sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.plot(x_decreasing, intensity_sorted, label='Original Intensity', color='blue', alpha=0.5)  # Original graph with transparency
        plt.xlabel('Pixel Position (x-axis)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Select Baseline Points')
        plt.legend()
        plt.xlim([max(x_decreasing), min(x_decreasing)])  # X축 감소 방향 설정


        print("Select Baseline Points on the Graph (Press Enter to Finish).")
        baseline_points = plt.ginput(n=-1, timeout=0)
        if len(baseline_points) == 0:
            print("Error: No points selected.")
            return

        baseline_points = np.array(baseline_points)

        baseline_x = baseline_points[:, 0]
        baseline_y = baseline_points[:, 1]
        unique_indices = np.unique(baseline_x, return_index=True)[1]
        baseline_x = baseline_x[unique_indices]
        baseline_y = baseline_y[unique_indices]

        sorted_indices = np.argsort(baseline_x)
        baseline_x = baseline_x[sorted_indices]
        baseline_y = baseline_y[sorted_indices]

        spline = make_interp_spline(baseline_x, baseline_y, k=3)
        baseline_fit = spline(x_decreasing)

        corrected_intensity = intensity_sorted - baseline_fit

        plt.figure(figsize=(10, 6))
        plt.plot(x_decreasing, intensity_sorted, label='Original Intensity', color='blue', alpha=0.5)  # Original graph with transparency
        plt.plot(x_decreasing, baseline_fit, label='Baseline', color='green', linestyle='--')  # Baseline visualization
        plt.plot(x_decreasing, corrected_intensity, label='Corrected Intensity', color='red')  # Corrected intensity
        plt.xlabel('Pixel Position (x-axis)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Baseline Correction with Original and Corrected Graphs')
        plt.legend()
        plt.gca().invert_xaxis()
        plt.xlim([max(x_decreasing), min(x_decreasing)])  # X축 감소 방향 설정
        
        plt.show()
        plt.close()

        self.corrected_intensity = corrected_intensity
    
    def select_peaks(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_original, self.corrected_intensity, label='Corrected Intensity', color='red')
        plt.xlabel('Pixel Position (x-axis)')
        plt.ylabel('Corrected Intensity (a.u.)')
        plt.title('Select Two Peaks')
        plt.legend()
        plt.gca().invert_xaxis()

        print("Select Two Peaks on the Graph (Press Enter to Finish).")
        peak_points = plt.ginput(n=2, timeout=0)
        if len(peak_points) != 2:
            print("Error: Two peaks must be selected.")
            return

        self.peaks = [np.abs(self.x_original - peak[0]).argmin() for peak in peak_points]
        self.x_nm = self.x_original.copy()  # nm 값 초기화
        plt.close()


    def visualize(self, title):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_cm, self.corrected_intensity, label='Corrected Intensity', color='red')
        plt.scatter(
            [self.x_cm[self.peaks[0]], self.x_cm[self.peaks[1]]],
            [self.corrected_intensity[self.peaks[0]], self.corrected_intensity[self.peaks[1]]],
            color='orange',
            label='Selected Peaks'
        )
        plt.xlabel('Wavenumber (cm^-1)')
        plt.ylabel('Corrected Intensity (a.u.)')
        plt.title(title)
        plt.legend()

        if self.x_cm[0] > 1e3:  # cm^-1 단위
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
        plt.close()

    def save_to_csv(self):
        # 데이터 저장
        data = pd.DataFrame({
            'x_original': self.x_original,
            'nm': self.x_nm,
            'cm^-1': self.x_cm,
            'Intensity': self.corrected_intensity
        })

        # 파일 경로 수정
        base_name, ext = os.path.splitext(self.file_path)
        output_file = f"{base_name}_convert{ext}"
                            
        data.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")

    def process_all(self, target_x_peak1_nm, target_x_peak2_nm, target_x_peak1_cm, target_x_peak2_cm):
        self.load_data()
        self.apply_smoothing()
        self.baseline_correction()
        self.select_peaks()
        self.adjust_graph_manual(target_x_peak1_nm, target_x_peak2_nm, target_x_peak1_cm, target_x_peak2_cm)
        self.visualize("Adjusted Graph (cm^-1)")
        self.save_to_csv()

file_path = "data_Mono12_Row_20250121_145322.csv"
target_x_peak1_nm = 852.1
target_x_peak2_nm = 898.2
target_x_peak1_cm = 1001.4
target_x_peak2_cm = 1602.3

peak_detector = PeakDetection(file_path)
peak_detector.process_all(target_x_peak1_nm, target_x_peak2_nm, target_x_peak1_cm, target_x_peak2_cm)