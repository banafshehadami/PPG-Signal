from sklearn.cluster import KMeans
import pywt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

###############Fixed window segmentation##############
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
# def main():
#     file_path = 'path/to/input/file'
#     signal = pd.read_csv(file_path)
#
#     height = 0.5
#     window_size_before = 20
#     window_size_after = 30
#
#     peaks, _ = find_peaks(signal, height=height)
#     windows = segment_signal(signal, height, window_size_before, window_size_after)
#
#     print("Number of windows:", len(windows))
#     plot_signal_with_windows(signal, peaks, windows, window_size_before, window_size_after)
#
# if __name__ == "__main__":
#     main()



###############Wavelet##############
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
# def main():
#     file_path = 'path.to/input/file'
#     signal = pd.read_csv(file_path)
#
#     windows = segment_signal_wavelet(signal)
#
#     print("Number of windows:", len(windows))
#     plot_signal_with_windows(signal, windows)
#
# if __name__ == "__main__":
#     main()



###############derivative_segmentation##############

def derivative_segmentation(signal, threshold=0.1):
    """
    Segment a PPG signal based on its first derivative.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        threshold (float, optional): The threshold value for detecting peaks and valleys. Defaults to 0.1.

    Returns:
        list: A list of numpy arrays representing the segmented pulses or waves.
    """
    derivative = np.gradient(signal)
    peaks, _ = find_peaks(derivative, height=threshold)
    valleys, _ = find_peaks(-derivative, height=threshold)

    segments = []
    start = 0
    for peak, valley in zip(peaks, valleys[1:]):
        end = valley
        segment = signal[start:end]
        segments.append(segment)
        start = end

    return segments


###############template_matching_segmentation##############

def template_matching_segmentation(signal, template, threshold=0.5):
    """
    Segment a PPG signal based on template matching.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        template (numpy.ndarray): The template waveform to match.
        threshold (float, optional): The threshold for the correlation coefficient. Defaults to 0.5.

    Returns:
        list: A list of numpy arrays representing the segmented pulses or waves.
    """
    corr = np.correlate(signal, template, mode='same')
    peak_indices, _ = find_peaks(corr, height=threshold)

    segments = []
    for peak in peak_indices:
        start = max(0, peak - len(template) // 2)
        end = min(len(signal), peak + len(template) // 2)
        segment = signal[start:end]
        segments.append(segment)

    return segments



###############cluster_based_segmentation##############

def cluster_based_segmentation(signal, n_clusters=2):
    """
    Segment a PPG signal based on clustering.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        n_clusters (int, optional): The number of clusters to use. Defaults to 2.

    Returns:
        list: A list of numpy arrays representing the segmented pulses or waves.
    """
    X = signal.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    labels = kmeans.labels_

    segments = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            end = i
            segment = signal[start:end]
            segments.append(segment)
            start = end

    return segments


###############sliding_window_segmentation##############
def sliding_window_segmentation(signal, window_size, overlap=0.5):
    """
    Segment a PPG signal using a sliding window approach.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        window_size (int): The size of the sliding window.
        overlap (float, optional): The overlap fraction between consecutive windows. Defaults to 0.5.

    Returns:
        list: A list of numpy arrays representing the segmented windows.
    """
    step = int(window_size * (1 - overlap))
    segments = [signal[i:i + window_size] for i in range(0, len(signal) - window_size + 1, step)]
    return segments
