import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks


class PeakDetection:
    def __init__(self, file_path):
        self.file_path = file_path
        self.x_values = None
        self.intensity = None
        self.smoothed_intensity = None
        self.peaks = None

    def load_data(self):
        data = pd.read_csv(self.file_path)
        self.x_values = data['X'].values  # Wavelength
        self.intensity = data['Y'].values  # Intensity

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

        peaks = np.delete(peaks, [0, 1])

        self.peaks = self.adjust_peaks_to_original(self.intensity, peaks, window=10)

    def adjust_graph(self, target_x_3rd, target_x_last):
        # Get current X values for 3rd and last peaks
        current_x_3rd = self.x_values[self.peaks[2]]
        current_x_last = self.x_values[self.peaks[-1]]

        # Calculate linear transformation coefficients
        m = (target_x_last - target_x_3rd) / (current_x_last - current_x_3rd)
        b = target_x_3rd - m * current_x_3rd

        # Adjust entire X values
        self.x_values = m * self.x_values + b

    def visualize_data(self):
        original_peaks_x = self.x_values[self.peaks]
        original_peaks_y = self.intensity[self.peaks]

        plt.figure(figsize=(12, 6))

        # Plot original and smoothed data
        plt.plot(self.x_values, self.intensity, label='Adjusted Intensity', color='blue', alpha=0.7, linewidth=1)

        # Plot peaks
        plt.scatter(original_peaks_x, original_peaks_y, color='orange', s=20, label='Adjusted Peaks', zorder=6)

        x_min, x_max = int(self.x_values.min()), int(self.x_values.max())  # X축 범위 계산
        plt.xticks(np.arange(x_min, x_max + 1, 5))  # X축 눈금을 5 단위로 설정
        # Customize plot
        plt.xlabel('Wavelength (X)')
        plt.ylabel('Intensity (Y)')
        plt.title('Adjusted Graph: Peaks Aligned with Target Positions')
        plt.legend()
        plt.grid()
        plt.show()

    def process_and_save(self, target_x_3rd, target_x_last, output_file):
        self.load_data()
        self.apply_smoothing()
        self.detect_peaks()
        self.adjust_graph(target_x_3rd, target_x_last)
        
        # Save the adjusted data to CSV
        adjusted_data = pd.DataFrame({
            'Adjusted_X': self.x_values,
            'Intensity': self.intensity
        })
        adjusted_data.to_csv(output_file, index=False)
        print(f"Adjusted data saved to: {output_file}")

    def process_and_visualize(self, target_x_3rd, target_x_last):
        self.load_data()
        self.apply_smoothing()
        self.detect_peaks()
        self.adjust_graph(target_x_3rd, target_x_last)
        self.visualize_data()


# Example usage
file_path = 'data_Mono12_Row_20250121_145322.csv'
target_x_3rd = 852.1  # Target X value for the 3rd peak
target_x_last = 898.2  # Target X value for the last peak
output_file = 'test_data.csv'  # Output CSV file name

peak_detector = PeakDetection(file_path)
peak_detector.process_and_visualize(target_x_3rd, target_x_last)
peak_detector.process_and_save(target_x_3rd, target_x_last, output_file)
