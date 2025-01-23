import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline

def process_and_plot(csv_file_path):
    # Load the CSV file
    data = pd.read_csv(csv_file_path)

    # Extract X and Y values
    x_values = data['X'].values
    intensity = data['Y'].values

# 스무딩 적용 (Savitzky-Golay 필터)
    window_length = 51  # 윈도우 크기 (데이터 길이보다 작아야 함)
    polyorder = 1       # 다항식 차수
    smoothed_intensity = savgol_filter(intensity, window_length=window_length, polyorder=polyorder)

    distance = 30
    prominence_ratio = 0.2
    threshold_index = int(0.3 * len(smoothed_intensity))
    dynamic_height = np.sort(smoothed_intensity)[threshold_index]
    peaks, _ = find_peaks(
        smoothed_intensity,
        prominence=prominence_ratio * dynamic_height,
        height=dynamic_height,
        distance=distance
    )
    peaks = np.delete(peaks, [-2])  # 0번과 1번 인덱스 삭제

    # Step 2: Mark excluded regions with ±interval around peaks
    excluded_regions_interval = []
    for peak in peaks:
        start = max(0, peak - 80)
        end = min(len(intensity), peak + 80)
        excluded_regions_interval.append((start, end))

    # Step 3: Create a baseline with intervals
    baseline_points_interval = np.arange(0, len(intensity), 5)

    # Filter out points that fall within the excluded regions
    valid_baseline_points = []
    for point in baseline_points_interval:
        in_excluded_region = any(start <= point < end for start, end in excluded_regions_interval)
        if not in_excluded_region:
            valid_baseline_points.append(point)

    valid_baseline_points = np.array(valid_baseline_points)  # Convert to numpy array

    # Calculate the average of ±1 around each valid baseline point
    adjusted_baseline_y_points = []
    for point in valid_baseline_points:
        # Get indices for ±1
        prev_index = max(0, point - 2)
        next_index = min(len(intensity) - 1, point + 2)
        # Calculate the average of ±1 intensity values
        avg_intensity = np.mean([intensity[prev_index], intensity[point], intensity[next_index]])
        adjusted_baseline_y_points.append(avg_intensity)

    baseline_x_points_interval = x_values[valid_baseline_points]
    baseline_y_points_interval = np.array(adjusted_baseline_y_points)  # Adjusted Y values

    # Sort the baseline points by X-axis for interpolation
    sorted_indices_interval = np.argsort(baseline_x_points_interval)
    baseline_x_sorted_interval = baseline_x_points_interval[sorted_indices_interval]
    baseline_y_sorted_interval = baseline_y_points_interval[sorted_indices_interval]

    # Interpolate the baseline using sorted points
    spline_baseline_interval = make_interp_spline(baseline_x_sorted_interval, baseline_y_sorted_interval, k=1)  # Linear interpolation
    baseline_fit = spline_baseline_interval(x_values)

    # Step 4: Connect excluded regions with straight lines
    for start, end in excluded_regions_interval:
        x_start = x_values[start]
        x_end = x_values[end - 1]
        y_start = baseline_fit[start]
        y_end = baseline_fit[end - 1]

        # Generate a straight line connecting the ±interval points
        x_range = np.linspace(x_start, x_end, end - start)
        y_range = np.linspace(y_start, y_end, end - start)
        baseline_fit[start:end] = y_range

    # Step 5: Correct the intensity using the final Baseline
    corrected_intensity = intensity - baseline_fit

    # Step 6: Plot the final Baseline and corrected graph
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, intensity, label='Original Intensity', color='blue', alpha=0.3)
    plt.plot(x_values, corrected_intensity, label='Corrected Intensity', color='red', linewidth=1.5)
    plt.plot(x_values, baseline_fit, label='Baseline', color='green', linestyle='--', linewidth=2)
    plt.scatter(baseline_x_sorted_interval, baseline_y_sorted_interval, color='purple', label='Baseline Points (interval)', s=5, zorder=5)  # Adjusted scatter size
    plt.xlabel('Wavelength (X)')

    # Mark the excluded regions with connecting lines
    for start, end in excluded_regions_interval:
        x_start = x_values[start]
        x_end = x_values[end - 1]
        y_start = baseline_fit[start]
        y_end = baseline_fit[end - 1]
        plt.plot([x_start, x_end], [y_start, y_end], color='orange', linewidth=2, label='Excluded Connection' if start == excluded_regions_interval[0][0] else "")

    plt.xlabel('Wavelength (X)')
    plt.ylabel('Intensity (Y)')
    plt.title('Corrected Intensity with Baseline')
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
csv_file_path = 'data_Mono12_Row_20250121_145322.csv'
process_and_plot(csv_file_path)
