import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PeakSelector:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.x_original = self.data.iloc[:, 0].values  # 원래 x 값 가져오기
        self.intensity = self.data.iloc[:, 3].values  # Intensity 값 가져오기
        self.x_nm = None
        self.x_cm = None
        self.peaks = []
        self.use_cm = False
    
    def find_nearest_peak(self, click_x, search_range=100):
        click_index = np.abs(self.x_original - click_x).argmin()
        start = max(0, click_index - search_range)
        end = min(len(self.intensity) - 1, click_index + search_range)
        local_region = self.intensity[start:end + 1]
        max_index = np.argmax(local_region)  # 가장 높은 Intensity 값의 인덱스
        return start + max_index
    
    def select_two_peaks(self):
        print("Click on two points in the graph to select peaks.")
        peak_points = plt.ginput(n=2, timeout=0)
        if len(peak_points) != 2:
            print("Error: Two peaks must be selected.")
            return None
        
        self.peaks = [self.find_nearest_peak(peak[0]) for peak in peak_points]
        print(f"Selected peaks (indices): {self.peaks}")
        self.adjust_axis_based_on_peaks()
        self.plot_adjusted_graph()

    def adjust_axis_based_on_peaks(self):
        if len(self.peaks) < 2:
            print("Error: Two peaks must be selected first.")
            return

        target_x_peak1_nm, target_x_peak2_nm = 852.1, 898.2
        target_x_peak1_cm, target_x_peak2_cm = 1001.4, 1602.3

        # 선택한 두 피크 위치
        current_x_peak1_original = self.x_original[self.peaks[0]]
        current_x_peak2_original = self.x_original[self.peaks[1]]

        # nm 변환 계산
        m_nm = (target_x_peak2_nm - target_x_peak1_nm) / (current_x_peak2_original - current_x_peak1_original)
        b_nm = target_x_peak1_nm - m_nm * current_x_peak1_original
        
        self.x_nm = m_nm * self.x_original + b_nm
        self.x_cm = 1e7 / self.x_nm

        current_x_peak1_cm = self.x_cm[self.peaks[0]]
        current_x_peak2_cm = self.x_cm[self.peaks[1]]

        m_cm = (target_x_peak2_cm - target_x_peak1_cm) / (current_x_peak2_cm - current_x_peak1_cm)
        b_cm = target_x_peak1_cm - m_cm * current_x_peak1_cm
        self.x_cm = m_cm * self.x_cm + b_cm

        self.use_cm = True
        print("X-axis adjusted based on selected peaks.")
    
    def plot_adjusted_graph(self):
        plt.figure()
        plt.plot(self.x_cm, self.intensity, color='red', linewidth=1)
        plt.xlabel("Wavenumber (cm^-1)")
        plt.ylabel("Intensity")
        plt.title("Adjusted Spectrum Data")
        plt.gca().invert_xaxis()
        plt.grid()
        
        # 선택된 피크 표시
        for peak in self.peaks:
            plt.axvline(self.x_cm[peak], color='blue', linestyle='--')
            plt.text(self.x_cm[peak], self.intensity[peak], f"{self.x_cm[peak]:.2f}", color='green', fontsize=8, ha='center')
        
        plt.show()
    
    def plot_data(self):
        plt.figure()
        plt.plot(self.x_original, self.intensity, color='red', linewidth=1)
        plt.xlabel("Original X Value")
        plt.ylabel("Intensity")
        plt.title("Spectrum Data")
        plt.gca().invert_xaxis()
        plt.grid()
        
        # 클릭 이벤트 실행
        self.select_two_peaks()

if __name__ == '__main__':
    csv_file = "data_Mono12_Row_20250214_110944.csv"
    selector = PeakSelector(csv_file)
    selector.plot_data()
