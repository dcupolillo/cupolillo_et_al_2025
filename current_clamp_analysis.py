"""
Current Clamp Analysis Module

This module contains functions for analyzing neuronal electrophysiological properties
recorded during current clamp experiments. It provides tools for:
- Action potential detection and kinetic analysis using IPFX
- Passive membrane properties (input resistance)
- Spike feature extraction (peak, threshold, amplitude, width, velocities)
- Trace segmentation and batch processing

The analysis workflow includes:
1. Trace segmentation from concatenated recordings
2. Comprehensive spike feature extraction using IPFX
3. Passive membrane property calculations
4. Feature extraction for onset, peak, trough, FWHM, and other properties

These functions are designed for current clamp recordings with repeated
current injections to analyze both active (spikes) and passive (Rin)
membrane properties.

IPFX is the spike detection and feature extraction package developed by the
Allen Institute for Brain Science. It provides robust and validated algorithms
for electrophysiology analysis.

Reference:
- IPFX: https://github.com/AllenInstitute/ipfx

Author: Dario Cupolillo
Publication: Cupolillo et al., 2025
"""

import numpy as np
from ipfx.feature_extractor import SpikeFeatureExtractor


def split_trace(
        trace: np.ndarray,
        start: int,
        length: int,
        sweeps: int,
        sweep_duration: int
) -> np.ndarray:
    """
    Segment a concatenated trace into individual sweeps.
    
    Takes a continuous recording trace and extracts individual sweeps based on
    the specified timing parameters. This is useful for analyzing repeated
    current injection trials recorded in a single continuous trace.
    
    Parameters
    ----------
    trace : np.ndarray
        Concatenated recording trace (1D array).
    start : int
        Starting index of the first sweep (in points).
    length : int
        Length of each sweep to extract (in points).
    sweeps : int
        Number of sweeps to extract.
    sweep_duration : int
        Duration between consecutive sweep starts (in points).
    
    Returns
    -------
    np.ndarray
        2D array of segmented sweeps (sweeps × length).
    """
    sweep_starts = np.arange(start, start + (sweeps * sweep_duration), sweep_duration)
    
    segmented_sweeps = []
    for sweep_start in sweep_starts:
        segmented_sweeps.append(trace[sweep_start:sweep_start + length])
    
    return np.array(segmented_sweeps)


def create_current_trace(
        n_points: int,
        baseline_points: int = 1000,
        pulse_amplitude_pa: float = 500.0,
        pulse_duration_points: int = 10000,
        post_pulse_points: int = 1000
) -> np.ndarray:
    """
    Create a square current pulse trace for IPFX analysis.
    
    Generates a current injection waveform with baseline, pulse, and post-pulse periods.
    This is required by IPFX SpikeFeatureExtractor to identify the stimulus period.
    Recordings obtained using HEKA Patchmaster software do not include the current command trace.
    
    Parameters
    ----------
    n_points : int
        Total number of points in the trace.
    baseline_points : int, optional
        Number of baseline points before pulse (default: 1000).
    pulse_amplitude_pa : float, optional
        Amplitude of current pulse in pA (default: 500.0).
    pulse_duration_points : int, optional
        Duration of current pulse in points (default: 10000).
    post_pulse_points : int, optional
        Number of points after pulse (default: 1000).
    
    Returns
    -------
    np.ndarray
        Current injection trace (1D array, in pA).
        
    Notes
    -----
    The total number of points should equal:
    baseline_points + pulse_duration_points + post_pulse_points
    
    If n_points is larger, remaining points are filled with zeros.
    If n_points is smaller, the trace is truncated.
    """
    current = np.zeros(n_points)
    pulse_start = baseline_points
    pulse_end = baseline_points + pulse_duration_points
    
    if pulse_end <= n_points:
        current[pulse_start:pulse_end] = pulse_amplitude_pa
    else:
        current[pulse_start:] = pulse_amplitude_pa
    
    return current


def extract_spike_features(
        voltage_trace: np.ndarray,
        current_trace: np.ndarray,
        sampling_rate: float = 10000.0,
        start_time: float = 0.0,
        end_time: float = 1.1,
        filter_frequency: float = None,
        dv_cutoff: float = 20.0,
        min_peak_mv: float = 0.0
) -> dict:
    """
    Extract comprehensive spike features using IPFX SpikeFeatureExtractor.
    
    Uses the Allen Institute IPFX package to detect and analyze action potentials,
    extracting a comprehensive set of spike features including onset, peak, trough,
    width, and other kinetic properties.
    
    Parameters
    ----------
    voltage_trace : np.ndarray
        Voltage recording trace (1D array, in mV).
    current_trace : np.ndarray
        Current injection trace (1D array, in pA).
    sampling_rate : float, optional
        Sampling rate in Hz (default: 10000.0 Hz).
    start_time : float, optional
        Start time for analysis window in seconds (default: 0.0).
    end_time : float, optional
        End time for analysis window in seconds (default: 1.1).
    filter_frequency : float, optional
        Low-pass filter frequency in Hz (default: None, no filtering).
    dv_cutoff : float, optional
        Cutoff for dV/dt detection in V/s (default: 20.0).
    min_peak_mv : float, optional
        Minimum peak voltage in mV (default: 0.0).
    
    Returns
    -------
    dict
        Dictionary containing comprehensive spike features:
        - 'peak_v': Peak voltages (mV)
        - 'peak_t': Peak times (s)
        - 'peak_i': Peak indices
        - 'threshold_v': Threshold voltages (mV)
        - 'threshold_t': Threshold times (s)
        - 'threshold_i': Threshold indices
        - 'trough_v': Trough voltages (mV)
        - 'trough_t': Trough times (s)
        - 'trough_i': Trough indices
        - 'width': Spike widths (FWHM) (s)
        - 'upstroke_downstroke_ratio': Ratio of upstroke to downstroke velocities
        - 'upstroke': Maximum upstroke velocity (V/s)
        - 'downstroke': Maximum downstroke velocity (V/s)
        - 'fast_trough_v': Fast trough voltages (mV)
        - 'fast_trough_t': Fast trough times (s)
        - 'fast_trough_i': Fast trough indices
        - 'slow_trough_v': Slow trough voltages (mV)
        - 'slow_trough_t': Slow trough times (s)
        - 'slow_trough_i': Slow trough indices
        - 'clipped': Array of booleans indicating if spikes are clipped
    """
    # Create time array in seconds
    n_points = len(voltage_trace)
    dt = 1.0 / sampling_rate
    time = np.linspace(0, n_points * dt, n_points, dtype=float)
    
    # Initialize IPFX SpikeFeatureExtractor
    sfx = SpikeFeatureExtractor(
        start=start_time,
        end=end_time,
        filter=filter_frequency,
        dv_cutoff=dv_cutoff,
        min_peak=min_peak_mv
    )
    
    # Process trace and extract features
    results = sfx.process(time, voltage_trace, current_trace)
    
    return results


def extract_spike_waveforms(
        voltage_trace: np.ndarray,
        peak_times: np.ndarray,
        sampling_rate: float = 10000.0,
        pre_peak_ms: float = 5.0,
        post_peak_ms: float = 10.0
) -> np.ndarray:
    """
    Extract spike waveforms aligned to peaks.
    
    Extracts a fixed-length window around each spike peak for waveform analysis.
    Aligns all spikes to have the peak at the same relative position.
    
    Parameters
    ----------
    voltage_trace : np.ndarray
        Voltage recording trace (1D array, in mV).
    peak_times : np.ndarray
        Array of spike peak times in seconds.
    sampling_rate : float, optional
        Sampling rate in Hz (default: 10000.0 Hz).
    pre_peak_ms : float, optional
        Duration before peak to include in ms (default: 5.0 ms).
    post_peak_ms : float, optional
        Duration after peak to include in ms (default: 10.0 ms).
    
    Returns
    -------
    np.ndarray
        2D array of spike waveforms (n_spikes × n_points).
        Each row is a spike waveform with peak at relative index
        (pre_peak_ms * sampling_rate / 1000).
    """
    pre_peak_points = int(pre_peak_ms * sampling_rate / 1000)
    post_peak_points = int(post_peak_ms * sampling_rate / 1000)
    
    waveforms = []
    
    for peak_time in peak_times:
        peak_idx = int(peak_time * sampling_rate)
        start_idx = peak_idx - pre_peak_points
        end_idx = peak_idx + post_peak_points
        
        # Check boundaries
        if start_idx >= 0 and end_idx <= len(voltage_trace):
            waveform = voltage_trace[start_idx:end_idx]
            waveforms.append(waveform)
    
    return np.array(waveforms)


def calculate_relative_times(
        peak_times: np.ndarray,
        threshold_times: np.ndarray,
        trough_times: np.ndarray
) -> tuple:
    """
    Calculate relative timing of spike features.
    
    Computes time differences between spike onset (threshold), peak, and trough
    for each action potential.
    
    Parameters
    ----------
    peak_times : np.ndarray
        Array of spike peak times in seconds.
    threshold_times : np.ndarray
        Array of spike threshold (onset) times in seconds.
    trough_times : np.ndarray
        Array of spike trough (end) times in seconds.
    
    Returns
    -------
    onset_to_peak : np.ndarray
        Time from onset to peak for each spike (seconds).
    peak_to_trough : np.ndarray
        Time from peak to trough for each spike (seconds).
    """
    onset_to_peak = np.array(peak_times) - np.array(threshold_times)
    peak_to_trough = np.array(trough_times) - np.array(peak_times)
    
    return onset_to_peak, peak_to_trough


def calculate_instantaneous_frequency(
        peak_times: np.ndarray
) -> np.ndarray:
    """
    Calculate instantaneous firing frequency from consecutive spike peaks.
    
    Computes the frequency based on inter-spike intervals (ISIs) between
    consecutive action potentials.
    
    Parameters
    ----------
    peak_times : np.ndarray
        Array of spike peak times in seconds.
    
    Returns
    -------
    np.ndarray
        Array of instantaneous frequencies in Hz.
        Length is (n_spikes - 1) for n_spikes.
        Returns empty array if fewer than 2 spikes.
    """
    if len(peak_times) < 2:
        return np.array([])
    
    # Calculate inter-spike intervals in seconds
    isis = np.diff(peak_times)
    
    # Convert to frequency (Hz)
    frequencies = 1.0 / isis
    
    return frequencies


def analyze_action_potentials(
        voltage_trace: np.ndarray,
        current_trace: np.ndarray = None,
        sampling_rate: float = 10000.0,
        start_time: float = 0.0,
        end_time: float = 1.1,
        filter_frequency: float = None,
        dv_cutoff: float = 20.0,
        min_peak_mv: float = 0.0,
        current_amplitude_pa: float = 500.0
) -> dict:
    """
    Complete action potential analysis pipeline for a single sweep.
    
    Performs comprehensive spike detection and feature extraction on a voltage
    recording trace using IPFX, including waveform extraction and derived features.
    
    Parameters
    ----------
    voltage_trace : np.ndarray
        Voltage recording trace (1D array, in mV).
    current_trace : np.ndarray, optional
        Current injection trace (1D array, in pA).
        If None, a standard square pulse is created.
    sampling_rate : float, optional
        Sampling rate in Hz (default: 10000.0 Hz).
    start_time : float, optional
        Start time for analysis window in seconds (default: 0.0).
    end_time : float, optional
        End time for analysis window in seconds (default: 1.1).
    filter_frequency : float, optional
        Low-pass filter frequency in Hz (default: None).
    dv_cutoff : float, optional
        Cutoff for dV/dt detection in V/s (default: 20.0).
    min_peak_mv : float, optional
        Minimum peak voltage in mV (default: 0.0).
    current_amplitude_pa : float, optional
        Current pulse amplitude in pA (default: 500.0).
        Only used if current_trace is None.
    
    Returns
    -------
    dict
        Dictionary containing:
        - All IPFX spike features (see extract_spike_features documentation)
        - 'n_spikes': Number of detected spikes
        - 'waveforms': Extracted spike waveforms
        - 'onset_to_peak': Time from onset to peak (seconds)
        - 'peak_to_trough': Time from peak to trough (seconds)
        - 'instantaneous_frequencies': Instantaneous firing frequencies (Hz)
    """
    # Create current trace if not provided
    if current_trace is None:
        current_trace = create_current_trace(
            n_points=len(voltage_trace),
            baseline_points=1000,
            pulse_amplitude_pa=current_amplitude_pa,
            pulse_duration_points=10000,
            post_pulse_points=1000
        )
    
    # Extract spike features using IPFX
    features = extract_spike_features(
        voltage_trace,
        current_trace,
        sampling_rate=sampling_rate,
        start_time=start_time,
        end_time=end_time,
        filter_frequency=filter_frequency,
        dv_cutoff=dv_cutoff,
        min_peak_mv=min_peak_mv
    )
    
    # Extract waveforms
    if len(features['peak_t']) > 0:
        waveforms = extract_spike_waveforms(
            voltage_trace,
            features['peak_t'],
            sampling_rate=sampling_rate
        )
        
        # Calculate relative timing
        onset_to_peak, peak_to_trough = calculate_relative_times(
            features['peak_t'],
            features['threshold_t'],
            features['trough_t']
        )
        
        # Calculate instantaneous frequency
        inst_freq = calculate_instantaneous_frequency(features['peak_t'])
    else:
        waveforms = np.array([])
        onset_to_peak = np.array([])
        peak_to_trough = np.array([])
        inst_freq = np.array([])
    
    # Combine all results
    results = {
        **features,  # Include all IPFX features
        'n_spikes': len(features['peak_t']),
        'waveforms': waveforms,
        'onset_to_peak': onset_to_peak,
        'peak_to_trough': peak_to_trough,
        'instantaneous_frequencies': inst_freq
    }
    
    return results


def analyze_multiple_sweeps(
        voltage_sweeps: np.ndarray,
        current_trace: np.ndarray = None,
        sampling_rate: float = 10000.0,
        **kwargs
) -> list:
    """
    Analyze action potentials across multiple sweeps.
    
    Applies the complete analysis pipeline to each sweep in a 2D array.
    
    Parameters
    ----------
    voltage_sweeps : np.ndarray
        2D array of voltage recordings (n_sweeps × n_points).
    current_trace : np.ndarray, optional
        Current injection trace (1D array, in pA).
        If None, a standard square pulse is created for each sweep.
    sampling_rate : float, optional
        Sampling rate in Hz (default: 10000.0 Hz).
    **kwargs
        Additional arguments passed to analyze_action_potentials().
    
    Returns
    -------
    list
        List of dictionaries, one per sweep, each containing analysis results
        from analyze_action_potentials().
    """
    results = []
    
    for sweep in voltage_sweeps:
        sweep_results = analyze_action_potentials(
            sweep,
            current_trace=current_trace,
            sampling_rate=sampling_rate,
            **kwargs
        )
        results.append(sweep_results)
    
    return results


def convert_indices_to_times(
        indices: np.ndarray,
        sampling_rate: float = 10000.0
) -> np.ndarray:
    """
    Convert array indices to time values.
    
    Utility function to convert sample indices to time in seconds.
    
    Parameters
    ----------
    indices : np.ndarray
        Array of sample indices.
    sampling_rate : float, optional
        Sampling rate in Hz (default: 10000.0 Hz).
    
    Returns
    -------
    np.ndarray
        Array of time values in seconds.
    """
    return indices / sampling_rate


def convert_times_to_indices(
        times: np.ndarray,
        sampling_rate: float = 10000.0
) -> np.ndarray:
    """
    Convert time values to array indices.
    
    Utility function to convert time in seconds to sample indices.
    
    Parameters
    ----------
    times : np.ndarray
        Array of time values in seconds.
    sampling_rate : float, optional
        Sampling rate in Hz (default: 10000.0 Hz).
    
    Returns
    -------
    np.ndarray
        Array of sample indices (integers).
    """
    return (times * sampling_rate).astype(int)


def calculate_input_resistance(
        voltage_traces: np.ndarray,
        current_amplitudes: np.ndarray,
        baseline_window: tuple = (0, 4000),
        response_window: tuple = (6000, 11000)
) -> tuple:
    """
    Calculate input resistance (Rin) from voltage responses to current injections.
    
    Measures the voltage deflection in response to a series of hyperpolarizing
    and depolarizing current steps, then computes input resistance from the
    slope of the current-voltage (I-V) relationship using linear regression.
    
    Parameters
    ----------
    voltage_traces : np.ndarray
        2D array of voltage traces (n_sweeps × n_points), one trace per
        current step amplitude.
    current_amplitudes : np.ndarray
        1D array of current injection amplitudes in pA (one per sweep).
        Should be the same length as the first dimension of voltage_traces.
    baseline_window : tuple, optional
        (start, end) indices for baseline voltage measurement (default: (0, 4000)).
    response_window : tuple, optional
        (start, end) indices for steady-state response measurement (default: (6000, 11000)).
    
    Returns
    -------
    input_resistance : float
        Input resistance in MΩ (megaohms).
        Calculated as: slope × 1000 (to convert from mV/pA to MΩ).
    voltage_deflections : np.ndarray
        Array of voltage deflections (ΔV) for each current step (mV).
    slope : float
        Slope of the I-V curve (mV/pA).
    intercept : float
        Intercept of the I-V curve (mV).
    r_squared : float
        Coefficient of determination (R²) for the linear fit.
    
    Notes
    -----
    Input resistance is calculated from Ohm's law: R = V / I
    The slope of the I-V relationship gives ΔV/ΔI in mV/pA.
    Multiplying by 1000 converts to MΩ: (mV/pA) × 1000 = MΩ
    
    A linear fit is appropriate for subthreshold voltage deflections where
    the membrane behaves as a passive resistor.
    
    Examples
    --------
    >>> # Voltage traces from -40 to +30 pA in 10 pA steps
    >>> current_steps = np.arange(-40, 40, 10)  # pA
    >>> voltage_traces = split_trace(trace, 15000, 15000, 8, 40000)
    >>> rin = calculate_input_resistance(
    ...     voltage_traces, current_steps
    ... )
    >>> print(f"Input resistance: {rin:.1f} MΩ")
    """
    from scipy import stats
    
    baseline_start, baseline_end = baseline_window
    response_start, response_end = response_window
    
    # Calculate baseline voltage for each trace
    baselines = np.mean(voltage_traces[:, baseline_start:baseline_end], axis=1)
    
    # Calculate steady-state voltage during current injection
    steady_state = np.mean(voltage_traces[:, response_start:response_end], axis=1)
    
    # Calculate voltage deflections (ΔV)
    voltage_deflections = steady_state - baselines
    
    # Perform linear regression: ΔV = slope × I + intercept
    slope, _, _, _, _ = stats.linregress(
        current_amplitudes,
        voltage_deflections
    )
    
    # Calculate input resistance in MΩ
    # slope is in mV/pA, multiply by 1000 to get MΩ
    input_resistance = slope * 1000
    
    return input_resistance
