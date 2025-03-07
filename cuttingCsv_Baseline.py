import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pywt
import scipy.signal
from scipy.signal import find_peaks



class RamanSpectrumAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.x_values = None
        self.intensity = None
        self.corrected_intensity = None
        self.baseline_fit = None
        self.clicked_points = []
        self.peaks = None

    def load_data(self):
        data = pd.read_csv(self.file_path)
        self.x_values = data['cm^-1'].values
        self.intensity = data['intensity'].values
        sorted_indices = np.argsort(self.x_values)[::-1]
        self.x_values = self.x_values[sorted_indices]
        self.intensity = self.intensity[sorted_indices]
    
    def select_xy_range(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_values, self.intensity, label="Full Data", color="blue")
        plt.xlabel("X-axis")
        plt.ylabel("Intensity (a.u.)")
        plt.title("Click Two Points to Set X & Y Limits")
        plt.legend()
        plt.xlim([min(self.x_values), max(self.x_values)])
        plt.ylim([min(self.intensity), max(self.intensity)])
        plt.gca().invert_xaxis()
        # plt.show()

        print("Select X and Y range by clicking two points.")
        points = np.array(plt.ginput(2, timeout=0))
        x_start, x_end = max(points[:, 0]), min(points[:, 0])
        y_min, y_max = min(points[:, 1]), max(points[:, 1])
        mask = (self.x_values >= x_end) & (self.x_values <= x_start)
        self.x_values = self.x_values[mask]
        self.intensity = self.intensity[mask]
    
    def select_baseline_points(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_values, self.intensity, label="Intensity", color="blue")
        plt.xlabel("X-axis")
        plt.ylabel("Intensity (a.u.)")
        plt.title("Select Baseline Points")
        plt.legend()
        plt.xlim([min(self.x_values), max(self.x_values)])
        plt.ylim([min(self.intensity), max(self.intensity)])
        plt.gca().invert_xaxis()
        # plt.show()

        print("Select baseline points by clicking (press Enter to finish).")
        baseline_points = np.array(plt.ginput(n=-1, timeout=0))
        baseline_x, baseline_y = baseline_points[:, 0], baseline_points[:, 1]
        unique_indices = np.unique(baseline_x, return_index=True)[1]
        linear_interp = interp1d(baseline_x[unique_indices], baseline_y[unique_indices], kind="linear", fill_value="extrapolate")
        self.baseline_fit = linear_interp(self.x_values)
        self.corrected_intensity = self.intensity - self.baseline_fit
    
    def wavelet_denoising(self, wavelet='db8', level=2, threshold_method='soft'):
        coeffs = pywt.wavedec(self.corrected_intensity, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(self.corrected_intensity)))
        coeffs_thresholded = [pywt.threshold(c, threshold, mode=threshold_method) if i > 0 else c for i, c in enumerate(coeffs)]
        self.corrected_intensity = pywt.waverec(coeffs_thresholded, wavelet)[:len(self.corrected_intensity)]
    
    def savgol_filter(self):
        self.corrected_intensity = scipy.signal.savgol_filter(self.corrected_intensity, window_length=21, polyorder=3)
    
    def visualize_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.x_values, self.corrected_intensity, label="Denoised Intensity", color='red', linewidth=1)

        if self.peaks is not None:
            # plt.scatter(self.x_values[self.peaks], self.corrected_intensity[self.peaks], color="blue", s=20, label="Detected Peaks")
            for px, py in zip(self.x_values[self.peaks], self.corrected_intensity[self.peaks]):
                plt.text(px, py + 100, f"{px:.1f}", fontsize=9, verticalalignment='bottom', horizontalalignment='center', color='blue')

        plt.xlabel("Raman Shift (cm^-1)")
        plt.ylabel("Intensity")
        plt.title("Wavelet Denoising on Raman Spectrum")
        plt.legend()
        plt.xlim([min(self.x_values), max(self.x_values)])
        plt.ylim([min(self.corrected_intensity), max(self.corrected_intensity) + 20000])
        plt.gca().invert_xaxis()
        plt.show()

    def detect_peaks(self, distance=10, prominence_ratio=0.3):
        if self.corrected_intensity is None:
            raise ValueError("Corrected intensity data is not available. Run baseline correction first.")
        
        threshold_index = int(0.8 * len(self.corrected_intensity))
        dynamic_height = np.sort(self.corrected_intensity)[threshold_index]
        
        self.peaks, _ = find_peaks(
            self.corrected_intensity,
            prominence=prominence_ratio * dynamic_height,
            height=dynamic_height,
            distance=distance
        )

    
    def run_analysis(self):
        self.load_data()
        # self.select_xy_range()
        self.select_baseline_points()
        self.wavelet_denoising()
        # self.savgol_filter()
        self.detect_peaks()
        self.visualize_results()


# 실행
file_path = "data_Mono12_Row_20250214_110944.csv"
analyzer = RamanSpectrumAnalyzer(file_path)
analyzer.run_analysis()