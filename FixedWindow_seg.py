import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def segment_signal(signal, height=0.5, window_size_before=20, window_size_after=30):
    """
    Segment a PPG signal into windows around detected peaks.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        height (float, optional): The minimum height of peaks to be detected. Defaults to 0.5.
        window_size_before (int, optional): The number of samples to include before each peak. Defaults to 20.
        window_size_after (int, optional): The number of samples to include after each peak. Defaults to 30.

    Returns:
        list: A list of numpy arrays, where each array represents a window around a detected peak.
    """
    # Find peaks in the signal
    peaks, _ = find_peaks(signal, height=height)

    # Create windows around each peak based on x-axis values
    windows = []
    for peak in peaks:
        start_index = max(0, peak - window_size_before)
        end_index = min(len(signal), peak + window_size_after)
        window = signal[start_index:end_index]
        windows.append(window)

    return windows

def plot_signal_with_windows(signal, peaks, windows, window_size_before=20, window_size_after=30):
    """
    Plot the PPG signal with detected peaks, windows around peaks, and segment lines.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        peaks (numpy.ndarray): The indices of the detected peaks.
        windows (list): A list of numpy arrays representing the windows around each peak.
        window_size_before (int, optional): The number of samples included before each peak. Defaults to 20.
        window_size_after (int, optional): The number of samples included after each peak. Defaults to 30.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(signal)), signal, label='Signal')
    plt.plot(peaks, signal[peaks], 'ro', label='Peaks')

    for peak, window in zip(peaks, windows):
        window_x = np.arange(peak - window_size_before, peak + window_size_after)
        plt.plot(window_x, window, label='Window')
        # Plot vertical lines at the beginning and end of each segment
        plt.axvline(peak - window_size_before, color='g', linestyle='--', alpha=0.5)
        plt.axvline(peak + window_size_after - 1, color='b', linestyle='--', alpha=0.5)

    # Plot lines indicating the beginning and end of the signal
    plt.axvline(0, color='k', linestyle='--', label='Signal Start')
    plt.axvline(len(signal) - 1, color='k', linestyle='--', label='Signal End')

    plt.legend()
    plt.title('Signal with Peaks, Windows, and Segment Lines')
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.show()

# Example usage
def main():
    file_path = 'path/to/input/file'
    signal = pd.read_csv(file_path)

    height = 0.5
    window_size_before = 20
    window_size_after = 30

    peaks, _ = find_peaks(signal, height=height)
    windows = segment_signal(signal, height, window_size_before, window_size_after)

    print("Number of windows:", len(windows))
    plot_signal_with_windows(signal, peaks, windows, window_size_before, window_size_after)

if __name__ == "__main__":
    main()