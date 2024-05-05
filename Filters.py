from scipy import signal
import numpy as np
from scipy import signal
from scipy import signal
import pywt
from scipy import signal
import numpy as np
import PyEMD
from scipy.ndimage import morphological_filter


def butterworth_filter(signal, cutoff_freq, order=4, filter_type='lowpass'):
    """
    Apply a Butterworth filter to the PPG signal.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        cutoff_freq (float): The cutoff frequency for the filter.
        order (int, optional): The order of the Butterworth filter. Defaults to 4.
        filter_type (str, optional): The type of filter ('lowpass', 'highpass', 'bandpass', or 'bandstop'). Defaults to 'lowpass'.

    Returns:
        numpy.ndarray: The filtered PPG signal.
    """
    nyquist_freq = 0.5 * signal.sampling_rate
    normalized_cutoff = cutoff_freq / nyquist_freq

    b, a = signal.butter(order, normalized_cutoff, btype=filter_type, analog=False)
    filtered_signal = signal.filtfilt(b, a, signal.values)

    return filtered_signal


def moving_average_filter(signal, window_size):
    """
    Apply a moving average filter to the PPG signal.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        window_size (int): The size of the window for the moving average.

    Returns:
        numpy.ndarray: The filtered PPG signal.
    """
    filtered_signal = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    return filtered_signal



def savitzky_golay_filter(signal, window_size, poly_order):
    """
    Apply a Savitzky-Golay filter to the PPG signal.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        window_size (int): The size of the window for the filter.
        poly_order (int): The order of the polynomial used for fitting.

    Returns:
        numpy.ndarray: The filtered PPG signal.
    """
    filtered_signal = signal.savgol_filter(signal, window_size, poly_order)
    return filtered_signal



def notch_filter(signal, notch_freq, quality_factor=10):
    """
    Apply a notch filter to the PPG signal to remove a specific frequency component.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        notch_freq (float): The frequency component to be removed.
        quality_factor (float, optional): The quality factor of the notch filter. Defaults to 10.

    Returns:
        numpy.ndarray: The filtered PPG signal.
    """
    nyquist_freq = 0.5 * signal.sampling_rate
    normalized_notch_freq = notch_freq / nyquist_freq

    b, a = signal.iirnotch(normalized_notch_freq, quality_factor)
    filtered_signal = signal.filtfilt(b, a, signal.values)

    return filtered_signal




def wavelet_denoising(signal, wavelet='db4', level=3, mode='soft'):
    """
    Apply wavelet denoising to the PPG signal.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        wavelet (str, optional): The wavelet family to be used for denoising. Defaults to 'db4' (Daubechies 4).
        level (int, optional): The level of wavelet decomposition. Defaults to 3.
        mode (str, optional): The thresholding mode ('soft' or 'hard'). Defaults to 'soft'.

    Returns:
        numpy.ndarray: The denoised PPG signal.
    """
    coeffs = pywt.wavedec(signal, wavelet, mode='sym', level=level)
    denoised_coeffs = pywt.threshold(coeffs, mode=mode)
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet, mode='sym')

    return denoised_signal




def band_pass_filter(signal, lowcut, highcut, order=4):
    """
    Apply a band-pass filter to the PPG signal.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        lowcut (float): The lower cutoff frequency for the band-pass filter.
        highcut (float): The upper cutoff frequency for the band-pass filter.
        order (int, optional): The order of the filter. Defaults to 4.

    Returns:
        numpy.ndarray: The filtered PPG signal.
    """
    nyquist_freq = 0.5 * signal.sampling_rate
    normalized_lowcut = lowcut / nyquist_freq
    normalized_highcut = highcut / nyquist_freq

    b, a = signal.butter(order, [normalized_lowcut, normalized_highcut], btype='band', analog=False)
    filtered_signal = signal.filtfilt(b, a, signal.values)

    return filtered_signal



def median_filter(signal, window_size):
    """
    Apply a median filter to the PPG signal.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        window_size (int): The size of the window for the median filter.

    Returns:
        numpy.ndarray: The filtered PPG signal.
    """
    filtered_signal = np.convolve(signal, np.ones(window_size), mode='same') / window_size
    return filtered_signal




def emd_denoising(signal, max_imf=10):
    """
    Apply Empirical Mode Decomposition (EMD) denoising to the PPG signal.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        max_imf (int, optional): The maximum number of Intrinsic Mode Functions (IMFs) to extract. Defaults to 10.

    Returns:
        numpy.ndarray: The denoised PPG signal.
    """
    emd = PyEMD.EMD()
    imfs = emd.emd(signal, max_imf=max_imf)
    denoised_signal = np.sum(imfs[:max_imf // 2], axis=0)

    return denoised_signal


def adaptive_filter(signal, reference, order=4):
    """
    Apply an adaptive filter to the PPG signal using a reference signal.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        reference (numpy.ndarray): The reference signal for the adaptive filter.
        order (int, optional): The order of the adaptive filter. Defaults to 4.

    Returns:
        numpy.ndarray: The filtered PPG signal.
    """
    filtered_signal, _ = signal.lfilter(signal.butter(order, 0.5)[0], reference=reference, zi=None)
    return filtered_signal




def morphological_filter_ppg(signal, mode='open'):
    """
    Apply a morphological filter to the PPG signal.

    Args:
        signal (numpy.ndarray): The PPG signal values.
        mode (str, optional): The mode of the morphological filter ('open', 'close', 'erode', or 'dilate'). Defaults to 'open'.

    Returns:
        numpy.ndarray: The filtered PPG signal.
    """
    if mode == 'open':
        filtered_signal = morphological_filter(signal, operation='opening', mode='nearest')
    elif mode == 'close':
        filtered_signal = morphological_filter(signal, operation='closing', mode='nearest')
    elif mode == 'erode':
        filtered_signal = morphological_filter(signal, operation='erosion', mode='nearest')
    elif mode == 'dilate':
        filtered_signal = morphological_filter(signal, operation='dilation', mode='nearest')
    else:
        raise ValueError("Invalid mode. Choose 'open', 'close', 'erode', or 'dilate'.")

    return filtered_signal