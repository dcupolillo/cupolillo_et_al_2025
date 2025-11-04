"""
Excitatory-Inhibitory (E-I) Ratio Analysis Module

This module contains functions for analyzing excitatory and inhibitory postsynaptic
currents (EPSCs and IPSCs) recorded during voltage clamp experiments. It provides
tools for trace segmentation, normalization, amplitude calculation, synaptic failure
detection, and decay kinetics analysis using monoexponential fitting.

The analysis workflow includes:
1. Trace segmentation from concatenated recordings
2. Baseline normalization of postsynaptic currents
3. Amplitude extraction for EPSCs and IPSCs
4. Weighted normalization for failure detection
5. E-I ratio and paired-pulse facilitation analysis

These functions are designed for paired recordings at different holding potentials
to isolate excitatory (-70 mV) and inhibitory (+5 mV) components.

Author: Dario Cupolillo
Publication: Cupolillo et al., 2025
"""

import numpy as np
import warnings


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
    stimulation trials recorded in a single continuous trace.
    
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
        
    Examples
    --------
    >>> trace = np.genfromtxt('recording.csv', delimiter=',')
    >>> # Extract 30 sweeps, 1000 points each, starting at index 0
    >>> sweeps = split_trace(trace, start=0, length=1000, sweeps=30, sweep_duration=1000)
    >>> print(sweeps.shape)  # (30, 1000)
    """
    # Generate start indices for each sweep
    sweep_starts = np.arange(
        start, start + (sweeps * sweep_duration), sweep_duration)
    
    # Extract each sweep
    segmented_sweeps = []
    for sweep_start in sweep_starts:
        segmented_sweeps.append(trace[sweep_start:sweep_start + length])
    
    return np.array(segmented_sweeps)


def normalize_psc(
        psc_array: np.ndarray,
        baseline_window: tuple = (0, 50)
) -> np.ndarray:
    """
    Normalize postsynaptic currents to their baseline values.
    
    Subtracts the mean baseline current from each sweep to normalize all events
    to zero baseline. This removes drift and allows comparison across sweeps.
    
    Parameters
    ----------
    psc_array : np.ndarray
        2D array of postsynaptic currents (sweeps × points).
    baseline_window : tuple, optional
        Start and end indices for baseline calculation (default: (0, 50)).
    
    Returns
    -------
    np.ndarray
        2D array of baseline-normalized currents (same shape as input).
        
    Examples
    --------
    >>> epscs = split_trace(trace_exc, 0, 1000, 30, 1000)
    >>> epscs_normalized = normalize_psc(epscs, baseline_window=(0, 50))
    """
    baseline_start, baseline_end = baseline_window
    
    # Calculate baseline for each sweep
    baselines = np.mean(psc_array[:, baseline_start:baseline_end], axis=1)
    
    # Normalize each sweep by subtracting its baseline
    psc_normalized = psc_array - baselines[:, np.newaxis]
    
    return psc_normalized


def extract_psc_amplitudes(
        psc_array: np.ndarray,
        polarity: str,
        peak_window: tuple,
        baseline_window: tuple = (0, 180)
) -> np.ndarray:
    """
    Extract amplitudes from postsynaptic currents.
    
    Calculates the peak amplitude for each sweep within a specified time window.
    For EPSCs (downward deflections), finds the minimum; for IPSCs (upward
    deflections), finds the maximum.
    
    Parameters
    ----------
    psc_array : np.ndarray
        2D array of normalized postsynaptic currents (sweeps × points).
    polarity : str
        Current type: 'EPSC' for excitatory (negative deflection) or
        'IPSC' for inhibitory (positive deflection).
    peak_window : tuple
        Start and end indices for peak detection (e.g., (220, 300) for EPSCs).
    baseline_window : tuple, optional
        Start and end indices for baseline reference (default: (0, 180)).
    
    Returns
    -------
    np.ndarray
        1D array of peak amplitudes (one per sweep), in pA.
        
    Examples
    --------
    >>> # Extract EPSC amplitudes
    >>> epsc_amps = extract_psc_amplitudes(
    ...     epscs_norm, polarity='EPSC', peak_window=(220, 300)
    ... )
    >>> 
    >>> # Extract IPSC amplitudes
    >>> ipsc_amps = extract_psc_amplitudes(
    ...     ipscs_norm, polarity='IPSC', peak_window=(300, 500)
    ... )
    """
    peak_start, peak_end = peak_window
    baseline_start, baseline_end = baseline_window
    
    amplitudes = []
    
    if polarity == 'EPSC':
        # For EPSCs, find minimum (negative peak)
        for sweep in psc_array:
            baseline = np.mean(sweep[baseline_start:baseline_end])
            peak = np.min(sweep[peak_start:peak_end])
            amplitudes.append(-(peak + baseline))
    
    elif polarity == 'IPSC':
        # For IPSCs, find maximum (positive peak)
        for sweep in psc_array:
            baseline = np.mean(sweep[baseline_start:baseline_end])
            peak = np.max(sweep[peak_start:peak_end])
            amplitudes.append(peak - baseline)
    
    else:
        raise ValueError("polarity must be 'EPSC' or 'IPSC'")
    
    return np.array(amplitudes)


def weight_current_normalization(
        psc_array: np.ndarray,
        analysis_window: tuple = (220, 400)
) -> np.ndarray:
    """
    Normalize currents using min-max scaling for failure detection.
    
    Applies weighted normalization to scale each sweep between -1 and 0 based on
    its own minimum and maximum values within the analysis window. This normalization
    is useful for detecting synaptic failures by standardizing response magnitudes
    across sweeps with different amplitudes.
    
    Parameters
    ----------
    psc_array : np.ndarray
        2D array of baseline-normalized currents (sweeps × points).
    analysis_window : tuple, optional
        Start and end indices for min-max calculation (default: (220, 400)).
    
    Returns
    -------
    np.ndarray
        2D array of weighted-normalized currents (sweeps × points).
        Values range from -1 (minimum) to 0 (maximum) within analysis window.
        
    Examples
    --------
    >>> epscs_weighted = weight_current_normalization(epscs_norm, (220, 400))
    >>> # Check variance in peak region to identify failures
    >>> variances = np.std(epscs_weighted[:, 220:300], axis=1)
    """
    window_start, window_end = analysis_window
    n_sweeps, n_points = psc_array.shape
    
    weighted_array = np.zeros_like(psc_array)
    
    for i in range(n_sweeps):
        # Find min and max in the analysis window
        ymin = np.min(psc_array[i, window_start:window_end])
        ymax = np.max(psc_array[i, window_start:window_end])
        
        # Apply weighted normalization: scale to [-1, 0]
        for k in range(n_points):
            weighted_array[i, k] = (psc_array[i, k] - ymin) / (ymax - ymin) - 1
    
    return weighted_array


def detect_synaptic_failures(
        weighted_psc: np.ndarray,
        peak_window: tuple = (220, 300),
        variance_threshold: float = 0.4
) -> np.ndarray:
    """
    Detect synaptic failures based on variance in weighted-normalized currents.
    
    Synaptic failures (trials with no postsynaptic response) can be identified by
    examining the variance in the peak region of weighted-normalized currents.
    Failures show low variance (flat trace), while successful events show higher
    variance due to the presence of a synaptic current.
    
    Parameters
    ----------
    weighted_psc : np.ndarray
        2D array of weighted-normalized currents from weight_current_normalization().
    peak_window : tuple, optional
        Start and end indices for variance calculation (default: (220, 300)).
    variance_threshold : float, optional
        Threshold for failure detection (default: 0.4).
        Sweeps with 2×std < threshold are marked as failures.
    
    Returns
    -------
    np.ndarray
        1D boolean array (length = n_sweeps).
        True = successful event, False = synaptic failure.
        
    Notes
    -----
    The variance_threshold should be adjusted based on your recording conditions
    and noise levels. Visual inspection of the variance distribution is recommended
    before setting the threshold.
        
    Examples
    --------
    >>> weighted_epscs = weight_current_normalization(epscs_norm)
    >>> success_mask = detect_synaptic_failures(weighted_epscs, (220, 300), 0.4)
    >>> print(f"Failure rate: {100 * np.sum(~success_mask) / len(success_mask):.1f}%")
    >>> 
    >>> # Filter out failures
    >>> successful_epscs = epscs_norm[success_mask]
    """
    peak_start, peak_end = peak_window
    
    success_mask = []
    
    for sweep in weighted_psc:
        # Calculate variance metric (2 × standard deviation)
        variance_metric = 2 * np.std(sweep[peak_start:peak_end])
        
        # Mark as success if variance exceeds threshold
        success_mask.append(variance_metric >= variance_threshold)
    
    return np.array(success_mask)


def calculate_ei_ratio(
        epsc_amplitudes: np.ndarray,
        ipsc_amplitudes: np.ndarray
) -> tuple:
    """
    Calculate excitatory-inhibitory (E-I) ratio statistics.
    
    Computes the ratio of excitatory to inhibitory current amplitudes,
    along with summary statistics.
    
    Parameters
    ----------
    epsc_amplitudes : np.ndarray
        Array of EPSC amplitudes (in pA).
    ipsc_amplitudes : np.ndarray
        Array of IPSC amplitudes (in pA).
    
    Returns
    -------
    ei_ratio_mean : float
        Mean E-I ratio across all pairs.
    ei_ratio_std : float
        Standard deviation of E-I ratio.
    ei_ratios : np.ndarray
        Array of individual E-I ratios (one per trial pair).
        
    Notes
    -----
    Arrays should have the same length. If they differ, the shorter length
    is used and a warning is issued.
        
    Examples
    --------
    >>> ei_mean, ei_std, ei_ratios = calculate_ei_ratio(epsc_amps, ipsc_amps)
    >>> print(f"E-I ratio: {ei_mean:.2f} ± {ei_std:.2f}")
    """
    # Ensure arrays have the same length
    min_length = min(len(epsc_amplitudes), len(ipsc_amplitudes))
    
    if len(epsc_amplitudes) != len(ipsc_amplitudes):
        warnings.warn(
            f"EPSC and IPSC arrays have different lengths "
            f"({len(epsc_amplitudes)} vs {len(ipsc_amplitudes)}). "
            f"Using first {min_length} values."
        )
    
    epsc_amps = epsc_amplitudes[:min_length]
    ipsc_amps = ipsc_amplitudes[:min_length]
    
    # Calculate E-I ratios for each trial
    ei_ratios = epsc_amps / ipsc_amps
    
    # Calculate statistics
    ei_ratio_mean = np.mean(ei_ratios)
    ei_ratio_std = np.std(ei_ratios)
    
    return ei_ratio_mean, ei_ratio_std, ei_ratios


def calculate_facilitation_ratio(
        baseline_amplitudes: np.ndarray,
        facilitated_amplitudes: np.ndarray
) -> tuple:
    """
    Calculate paired-pulse facilitation (PPF) ratio.
    
    Computes the ratio of facilitated response to baseline response,
    typically used to assess short-term synaptic plasticity.
    
    Parameters
    ----------
    baseline_amplitudes : np.ndarray
        Array of baseline EPSC amplitudes (e.g., at 0.1 Hz).
    facilitated_amplitudes : np.ndarray
        Array of facilitated EPSC amplitudes (e.g., at 1 Hz).
    
    Returns
    -------
    ppf_ratio_mean : float
        Mean facilitation ratio.
    ppf_ratio_std : float
        Standard deviation of facilitation ratio.
    ppf_ratios : np.ndarray
        Array of individual facilitation ratios.
        
    Notes
    -----
    PPF ratio = facilitated_amplitude / baseline_amplitude
    Values > 1 indicate facilitation, < 1 indicate depression.
        
    Examples
    --------
    >>> ppf_mean, ppf_std, ppf_ratios = calculate_facilitation_ratio(
    ...     epsc_amps_01hz, epsc_amps_1hz
    ... )
    >>> print(f"PPF ratio: {ppf_mean:.2f} ± {ppf_std:.2f}")
    """
    # Ensure arrays have the same length
    min_length = min(len(baseline_amplitudes), len(facilitated_amplitudes))
    
    if len(baseline_amplitudes) != len(facilitated_amplitudes):
        warnings.warn(
            f"Baseline and facilitated arrays have different lengths. "
            f"Using first {min_length} values."
        )
    
    baseline = baseline_amplitudes[:min_length]
    facilitated = facilitated_amplitudes[:min_length]
    
    # Calculate facilitation ratios
    ppf_ratios = facilitated / baseline
    
    # Calculate statistics
    ppf_ratio_mean = np.mean(ppf_ratios)
    ppf_ratio_std = np.std(ppf_ratios)
    
    return ppf_ratio_mean, ppf_ratio_std, ppf_ratios
