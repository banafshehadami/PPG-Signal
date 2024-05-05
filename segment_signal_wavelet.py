import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

def find_peaks(signal, height=0.5, distance=None, prominence=None):
    """
    Find peaks in a PPG signal.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        height (float, optional): The minimum height of peaks to be detected. Defaults to 0.5.
        distance (int, optional): The minimum distance between peaks. Defaults to None.
        prominence (float, optional): The minimum prominence of peaks to be detected. Defaults to None.

    Returns:
        tuple: A tuple containing two numpy arrays: (peak_indices, peak_properties)
            peak_indices: The indices of the detected peaks in the signal.
            peak_properties: A dictionary containing properties of the detected peaks.
    """
    peak_indices, peak_properties = signal_find_peaks(signal, height=height, distance=distance, prominence=prominence)
    return peak_indices, peak_properties

def signal_find_peaks(x, height=0.5, distance=None, prominence=None):
    """
    Find peaks in a 1D signal using the `find_peaks` function from SciPy.

    Args:
        x (numpy.ndarray): The 1D signal array.
        height (float, optional): The minimum height of peaks to be detected. Defaults to 0.5.
        distance (int, optional): The minimum distance between peaks. Defaults to None.
        prominence (float, optional): The minimum prominence of peaks to be detected. Defaults to None.

    Returns:
        tuple: A tuple containing two elements:
            peak_indices: A numpy array containing the indices of the detected peaks.
            properties: A dictionary containing properties of the detected peaks.
    """

    peak_indices, properties = find_peaks(x, height=height, distance=distance, prominence=prominence)
    return peak_indices, properties
def segment_signal_wavelet(signal, wavelet='db4', max_level=5):
    """
    Segment a PPG signal using wavelet-based decomposition.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        wavelet (str, optional): The wavelet family to be used for decomposition. Defaults to 'db4' (Daubechies 4).
        max_level (int, optional): The maximum level of wavelet decomposition. Defaults to 5.

    Returns:
        list: A list of numpy arrays, where each array represents a window around a detected pulse or wave.
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, mode='symmetric', level=max_level)

    # Analyze the wavelet coefficients to identify pulse/wave locations
    pulse_locations = []
    for level in range(max_level + 1):
        cD = coeffs[level]
        peaks, _ = find_peaks(cD, height=0.5)
        pulse_locations.extend(peaks)

    # Create windows around each pulse/wave location
    windows = []
    window_size = 50  # Adjust this value as needed
    for loc in pulse_locations:
        start_index = max(0, loc - window_size // 2)
        end_index = min(len(signal), loc + window_size // 2)
        window = signal[start_index:end_index]
        windows.append(window)

    return windows

def plot_signal_with_windows(signal, windows):
    """
    Plot the PPG signal with detected windows around pulses or waves.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        windows (list): A list of numpy arrays representing the windows around each pulse or wave.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(signal)), signal, label='Signal')

    for i, window in enumerate(windows):
        start_index = np.argmax(window == signal)
        window_x = np.arange(start_index, start_index + len(window))
        plt.plot(window_x, window, label=f'Window {i+1}')

    plt.legend()
    plt.title('Signal with Detected Pulses/Waves')
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.show()

# Example usage
def main():
    file_path = 'path.to/input/file'
    signal = pd.read_csv(file_path)

    windows = segment_signal_wavelet(signal)

    print("Number of windows:", len(windows))
    plot_signal_with_windows(signal, windows)

if __name__ == "__main__":
    main()