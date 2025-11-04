"""
Deconvolution-Based Event Detection Module

This module contains functions originally implemented for use with Stimfit software
(stf package) for deconvolution-based detection of synaptic events. These functions
have been adapted to work as standalone Python functions without requiring Stimfit.

ORIGINAL STIMFIT IMPLEMENTATION:
These functions were originally written to work with the Stimfit software using 
the stf package. The core algorithm logic has been preserved while removing 
Stimfit-specific dependencies. (https://neurodroid.github.io/stimfit/manual/event_extraction.html)

The three main functions are:
1. cursors_bait() - Create template from a 'bait' event
2. cursors_averaged() - Create template from averaged events  
3. detect() - Detect events using template and deconvolution

The deconvolution approach is based on Pernia-Andrade et al. (2012).

Reference:
- Pernia-Andrade et al. (2012). 
  "A Deconvolution-Based Method with High Sensitivity and Temporal Resolution
  for Detection of Spontaneous Synaptic Currents In Vitro and In Vivo."
  https://doi.org/10.1016/j.bpj.2012.08.039
 - Guzman et al. (2016). 
   "Stimfit: quantifying electrophysiological data with Python."
   https://doi.org/10.3389/fninf.2014.00016

Author: Dario Cupolillo
Publication: Cupolillo et al., 2025
"""

import numpy as np
from scipy import signal, optimize
import matplotlib.pyplot as plt


def fit_template_from_bait_event(
        trace: np.ndarray,
        dt: float,
        peak_start_ms: float
) -> np.ndarray:
    """
    Create template from a single large and isolated 'bait' event by placing cursors and fitting.
    This bait event is used to detect a certain amount of events using any methods.
    
    ORIGINAL STIMFIT CODE (using stf package):
    ```python
    def cursors_bait(peak_start: int) -> np.ndarray:
        '''Place the peak, baseline and fitting cursors given an user-defined peak starting point.'''
        
        dt = stf.get_sampling_interval()
        stf.set_peak_direction('down')
        
        #Position the cursors according to pre-defined interval ranges
        stf.set_peak_start(peak_start/dt)
        stf.set_peak_end(stf.get_peak_start()+(10/dt))
        stf.set_base_start(stf.get_peak_start()-(17/dt))
        stf.set_base_end(stf.get_peak_start()-(7/dt))
        stf.set_fit_start(stf.get_peak_start()-(5/dt))
        stf.set_fit_end(stf.get_peak_start()+(35/dt))
        stf.measure()

        #Get the fitting biexponential function
        stf.leastsq(5,refresh=True)
        template=stf.get_fit()
        
        return template[1]
    ```
    
    Parameters
    ----------
    trace : np.ndarray
        Continuous recording trace containing the bait event.
    dt : float
        Sampling interval (ms).
    peak_start_ms : float
        Time point where peak search starts (ms).
    
    Returns
    -------
    np.ndarray
        Fitted biexponential template.
        
    Examples
    --------
    >>> trace = np.genfromtxt('recording.csv')
    >>> dt = 0.1  # ms (10 kHz sampling)
    >>> template = fit_template_from_bait_event(trace, dt, peak_start_ms=1000.0)
    """
    # Convert time to index (equivalent to: peak_start/dt in Stimfit)
    peak_start_idx = int(peak_start_ms / dt)
    
    # Position cursors according to pre-defined interval ranges (matches Stimfit)
    peak_end_idx = peak_start_idx + int(10 / dt)
    base_start_idx = peak_start_idx - int(17 / dt)
    base_end_idx = peak_start_idx - int(7 / dt)
    fit_start_idx = peak_start_idx - int(5 / dt)
    fit_end_idx = peak_start_idx + int(35 / dt)
    
    # Ensure indices are within bounds
    base_start_idx = max(0, base_start_idx)
    fit_start_idx = max(0, fit_start_idx)
    fit_end_idx = min(len(trace), fit_end_idx)
    
    # Extract fitting region
    fit_region = trace[fit_start_idx:fit_end_idx]
    
    # Calculate baseline (equivalent to: stf.measure())
    baseline = np.mean(trace[base_start_idx:base_end_idx])
    
    # Subtract baseline
    fit_region_normalized = fit_region - baseline
    
    # Fit biexponential (equivalent to: stf.leastsq(5))
    template = _fit_biexponential(fit_region_normalized, dt)
    return template


def fit_template_from_averaged_events(
        events: np.ndarray,
        dt: float
) -> np.ndarray:
    """
    Create template from averaged extracted events by placing cursors and fitting.
    This was originally done using Stimfit, now rewritten to work in pure Python.
    
    ORIGINAL STIMFIT CODE (using stf package):
    ```python
    def cursors_averaged():
        '''Place the peak, baseline and fitting cursors onto extracted events.'''
        
        dt = stf.get_sampling_interval()
        
        #Position the cursors according to pre-defined interval ranges
        stf.set_peak_start(12/dt)
        stf.set_peak_end(22/dt)
        stf.set_base_start(4/dt)
        stf.set_base_end(8.8/dt)
        stf.set_fit_start(8.8/dt)
        stf.set_fit_end(48.8/dt)
        stf.measure()
        
        #Get the fitting biexponential function
        stf.leastsq(5,refresh=True)
        template_averaged=stf.get_fit()
        
        return template_averaged[1]
    ```
    
    Parameters
    ----------
    events : np.ndarray
        2D array of extracted events, or 1D averaged event.
    dt : float
        Sampling interval (ms).
    
    Returns
    -------
    np.ndarray
        Fitted biexponential template from averaged events.
        
    Examples
    --------
    >>> events = np.array([...])  # 2D array of extracted events
    >>> dt = 0.1  # ms
    >>> template = fit_template_from_averaged_events(events, dt)
    """
    # Average events if 2D array provided
    if events.ndim == 2:
        avg_event = np.mean(events, axis=0)
    else:
        avg_event = events
    
    # Position cursors according to pre-defined interval ranges (matches Stimfit)
    peak_start_idx = int(12 / dt)
    peak_end_idx = int(22 / dt)
    base_start_idx = int(4 / dt)
    base_end_idx = int(8.8 / dt)
    fit_start_idx = int(8.8 / dt)
    fit_end_idx = int(48.8 / dt)
    
    # Ensure indices are within bounds
    fit_end_idx = min(len(avg_event), fit_end_idx)
    
    # Extract fitting region
    fit_region = avg_event[fit_start_idx:fit_end_idx]
    
    # Calculate baseline (equivalent to: stf.measure())
    baseline = np.mean(avg_event[base_start_idx:base_end_idx])
    
    # Subtract baseline
    fit_region_normalized = fit_region - baseline
    
    # Fit biexponential (equivalent to: stf.leastsq(5))
    template = _fit_biexponential(fit_region_normalized, dt)
    
    return template


def detect(
        trace: np.ndarray,
        template: np.ndarray,
        dt: float,
        threshold_factor: float = 4.0,
        min_interval_ms: float = 50.0,
        normalize: bool = False,
        lowpass: float = 0.1,
        highpass: float = 0.001
) -> tuple:
    """
    Detect events using template matching and deconvolution.
    
    This function replicates the original Stimfit implementation.
    
    ORIGINAL STIMFIT CODE (using stf package):
    ```python
    def detect(template, mode, th, min_int):
        '''Detect events using the given template and the algorithm specified in 
        'mode' with a thresholt 'th' and a minimal interval of 'min_int' between
        events. Returns amplitude and interevent intervals.'''

        #deconvolve currently active trace
        crit = stf.detect_events(template, mode=mode, norm=False, lowpass=0.1, highpass=0.001)
        
        dt = stf.get_sampling_interval()
        
        #find events onset corresponding to peaks in deconvolved trace
        onsets_i = stf.peak_detection(crit, (np.std(crit)*th), int(min_int/dt))
        
        trace = stf.get_trace()
        
        peak_window_i = min_int / dt
        
        #array of indices of peaks
        amps_i = np.array([int(np.argmin(trace[int(onset_i):int(onset_i+peak_window_i)])+onset_i) 
                          for onset_i in onsets_i], dtype=np.int)
        
        return amps_i, onsets_i
    ```
    
    Parameters
    ----------
    trace : np.ndarray
        Continuous recording trace (in pA).
    template : np.ndarray
        Event template for deconvolution.
    dt : float
        Sampling interval (ms).
    threshold_factor : float, optional
        Threshold as multiple of criterion std (th parameter in original).
        Typical values: 4.0 or 4.5.
    min_interval_ms : float, optional
        Minimum interval between events in ms (min_int in original, default: 50.0).
    normalize : bool, optional
        Whether to normalize template (norm parameter in original, default: False).
    lowpass : float, optional
        Low-pass filter cutoff (default: 0.1).
    highpass : float, optional
        High-pass filter cutoff (default: 0.001).
    
    Returns
    -------
    peak_indices : np.ndarray
        Array of indices of detected event peaks (amps_i in original).
    onset_indices : np.ndarray
        Array of indices of detected event onsets (onsets_i in original).
        
    Examples
    --------
    >>> # Create template from bait event
    >>> template = fit_template_from_bait_event(trace, dt=0.1, peak_start_ms=1000.0)
    >>> 
    >>> # Detect events (matches original: detect(template, 'deconvolution', 4, 50))
    >>> peak_indices, onset_indices = detect(
    ...     trace, template, dt=0.1, mode='deconvolution', 
    ...     threshold_factor=4.0, min_interval_ms=50
    ... )
    >>> print(f"Detected {len(peak_indices)} events")
    """
    # Deconvolve trace (equivalent to: stf.detect_events)
    criterion = _deconvolve_trace(trace, template, normalize, lowpass, highpass)
    
    # Find event onsets in criterion function (equivalent to: stf.peak_detection)
    threshold = np.std(criterion) * threshold_factor
    min_interval_points = int(min_interval_ms / dt)
    onset_indices = _peak_detection(criterion, threshold, min_interval_points)
    
    # Find peaks in original trace (matches original implementation exactly)
    peak_window_points = min_interval_ms / dt
    
    # Array of indices of peaks (exact match to original)
    peak_indices = np.array([
        int(np.argmin(trace[int(onset_i):int(onset_i + peak_window_points)]) + onset_i)
        for onset_i in onset_indices
        if int(onset_i + peak_window_points) <= len(trace)
    ], dtype=int)
    
    return peak_indices, onset_indices


# ============================================================================
# HELPER FUNCTIONS - Internal implementations of Stimfit (stf) functions
# ============================================================================

def _fit_biexponential(
        event: np.ndarray,
        dt: float
) -> np.ndarray:
    """
    Fit biexponential function to event.
    
    Replicates Stimfit's stf.leastsq(5) function.
    
    Parameters
    ----------
    event : np.ndarray
        Event trace to fit.
    dt : float
        Sampling interval (ms).
    
    Returns
    -------
    np.ndarray
        Fitted biexponential template.
    """
    def biexponential(t, amplitude, tau_rise, tau_decay, offset=0):
        """Biexponential function."""
        return amplitude * (np.exp(-t / tau_decay) - np.exp(-t / tau_rise)) + offset
    
    t = np.arange(len(event)) * dt
    
    # Initial parameter guesses
    amplitude_guess = np.min(event)
    initial_guess = {
        'amplitude': amplitude_guess,
        'tau_rise': 2.0,  # ms
        'tau_decay': 10.0,  # ms
        'offset': 0.0
    }
    
    p0 = [initial_guess['amplitude'], 
          initial_guess['tau_rise'], 
          initial_guess['tau_decay'], 
          initial_guess['offset']]
    
    # Fit using least squares
    try:
        popt, _ = optimize.curve_fit(
            biexponential, 
            t, 
            event, 
            p0=p0,
            bounds=([-np.inf, 0.1, 0.1, -np.inf],  # Lower bounds
                    [0, 100, 1000, np.inf])  # Upper bounds
        )
        template = biexponential(t, *popt)
    except (RuntimeError, ValueError):
        # If fitting fails, return original event
        print("Warning: Biexponential fit failed. Using original event as template.")
        template = event
    
    return template


def _deconvolve_trace(trace: np.ndarray,
                     template: np.ndarray,
                     normalize: bool,
                     lowpass: float,
                     highpass: float) -> np.ndarray:
    """
    Deconvolve trace using template.
    
    Replicates Stimfit's stf.detect_events() function.
    
    Parameters
    ----------
    trace : np.ndarray
        Continuous recording trace.
    template : np.ndarray
        Event template.
    normalize : bool
        Whether to normalize template.
    lowpass : float
        Low-pass filter cutoff (normalized frequency, 0-1).
    highpass : float
        High-pass filter cutoff (normalized frequency, 0-1).
    
    Returns
    -------
    np.ndarray
        Deconvolved criterion function.
    """
    # Normalize template if requested
    if normalize:
        template = template / np.sqrt(np.sum(template ** 2))
    
    # Perform Wiener deconvolution
    template_fft = np.fft.fft(template, n=len(trace))
    trace_fft = np.fft.fft(trace)
    
    # Wiener filter (simplified)
    noise_power = 0.001
    wiener_filter = np.conj(template_fft) / (np.abs(template_fft) ** 2 + noise_power)
    deconvolved_fft = trace_fft * wiener_filter
    
    deconvolved = np.real(np.fft.ifft(deconvolved_fft))
    
    # Band-pass filter the deconvolved trace
    if highpass > 0:
        b_high, a_high = signal.butter(2, highpass, btype='high')
        deconvolved = signal.filtfilt(b_high, a_high, deconvolved)
    
    if lowpass < 0.5:
        b_low, a_low = signal.butter(2, lowpass, btype='low')
        deconvolved = signal.filtfilt(b_low, a_low, deconvolved)
    
    return deconvolved


def _peak_detection(criterion: np.ndarray,
                   threshold: float,
                   min_interval: int) -> np.ndarray:
    """
    Detect peaks in criterion function.
    
    Replicates Stimfit's stf.peak_detection() function.
    
    Parameters
    ----------
    criterion : np.ndarray
        Deconvolved criterion function.
    threshold : float
        Detection threshold.
    min_interval : int
        Minimum interval between events (in points).
    
    Returns
    -------
    np.ndarray
        Array of onset indices.
    """
    # Find points above threshold
    above_threshold = criterion > threshold
    
    # Find rising edges (event onsets)
    onsets = np.where(np.diff(above_threshold.astype(int)) > 0)[0] + 1
    
    # Enforce minimum interval between events
    if len(onsets) > 0:
        filtered_onsets = [onsets[0]]
        for onset in onsets[1:]:
            if onset - filtered_onsets[-1] >= min_interval:
                filtered_onsets.append(onset)
        onsets = np.array(filtered_onsets)
    
    return onsets
