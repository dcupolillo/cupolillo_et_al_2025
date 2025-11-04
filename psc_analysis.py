"""
Postsynaptic Current (PSC) Analysis Module

This module provides functions for analyzing spontaneous and miniature 
excitatory and inhibitory postsynaptic currents (sEPSC/mEPSCs/mIPSCs)
recorded during voltage clamp experiments.

Functions include:
- Event extraction from continuous traces
- Baseline normalization and artifact removal
- Amplitude and kinetic property calculations
- Visualization utilities

Author: Dario Cupolillo
Reference: Cupolillo et al., 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def extract_events(
        trace: np.ndarray, 
        onset_indices: np.ndarray, 
        onset_offset: int, 
        event_length: int
) -> np.ndarray:
    """
    Extract individual synaptic events from a continuous trace.
    
    Parameters
    ----------
    trace : np.ndarray
        Continuous voltage clamp recording trace (in pA).
    onset_indices : np.ndarray
        Array of indices indicating event onset times.
    onset_offset : int
        Number of points before onset to include (for baseline).
    event_length : int
        Total length of each extracted event (in sampling points).
    
    Returns
    -------
    np.ndarray
        2D array of extracted events (n_events × event_length).
        
    Examples
    --------
    >>> trace = np.random.randn(100000)
    >>> onsets = np.array([1000, 5000, 10000])
    >>> events = extract_events(trace, onsets, onset_offset=50, event_length=800)
    >>> events.shape
    (3, 800)
    """
    events = []
    for onset_idx in onset_indices:
        start = int(onset_idx - onset_offset)
        end = int(onset_idx + event_length - onset_offset)
        
        # Only include events with complete length
        if start >= 0 and end <= len(trace):
            event = trace[start:end]
            if len(event) == event_length:
                events.append(event)
    
    return np.array(events)


def normalize_psc(
        psc_array: np.ndarray, 
        baseline_window: int = 50,
        sigma_threshold: float = 3.0
) -> tuple:
    """
    Normalize PSC events to baseline and remove artifacts.
    
    This function normalizes each event to its own baseline and removes
    artifacts using a statistical threshold based on the distribution of
    normalized values.
    
    Parameters
    ----------
    psc_array : np.ndarray
        2D array of PSC events (n_events × event_length).
    baseline_window : int, optional
        Number of initial points to use for baseline calculation (default: 50).
    sigma_threshold : float, optional
        Number of standard deviations for artifact detection threshold (default: 3.0).
    
    Returns
    -------
    psc_normalized : np.ndarray
        Clean normalized events.
    psc_excluded : np.ndarray
        Excluded events (artifacts).
        
    Examples
    --------
    >>> events = np.random.randn(100, 800)
    >>> clean_events, artifacts = normalize_psc(events)
    >>> print(f"Kept {len(clean_events)} events, removed {len(artifacts)} artifacts")
    """
    # Calculate baseline for each event
    baselines = np.mean(psc_array[:, :baseline_window], axis=1)
    
    # Normalize events to their baseline
    psc_normalized = psc_array - baselines[:, np.newaxis]
    
    # Fit Gaussian to the distribution to find noise level
    mu, sigma = norm.fit(psc_normalized.flatten())
    
    # Find histogram peak
    histogram, bins = np.histogram(psc_normalized.flatten(), bins=200)
    peak_idx = np.argmax(histogram)
    
    # Set upper cutoff threshold
    upper_cutoff = bins[peak_idx] + (sigma * sigma_threshold)
    
    # Identify artifacts (events with mean of last 200 points above threshold)
    event_means = np.mean(psc_normalized[:, -200:], axis=1)
    clean_mask = event_means <= upper_cutoff
    
    return psc_normalized[clean_mask], psc_normalized[~clean_mask]


def calculate_psc_amplitudes(
        psc_array: np.ndarray, 
        onset_offset: int,
        search_start: int = 50,
        search_end: int = 200
) -> np.ndarray:
    """
    Calculate peak amplitudes of PSC events.
    
    Parameters
    ----------
    psc_array : np.ndarray
        2D array of baseline-normalized PSC events.
    onset_offset : int
        Index of the event onset within each event array.
    search_start : int, optional
        Points after onset to start peak search (default: 50).
    search_end : int, optional
        Points after onset to end peak search (default: 200).
    
    Returns
    -------
    np.ndarray
        Array of peak amplitudes (in pA, absolute values).
        
    Notes
    -----
    For inhibitory currents (downward deflections), amplitudes are positive.
    For excitatory currents (also often downward in voltage clamp), 
    amplitudes are positive.
    """
    amplitudes = []
    
    for event in psc_array:
        baseline = np.mean(event[:onset_offset])
        search_window = event[onset_offset + search_start:onset_offset + search_end]
        peak_value = np.min(search_window)
        amplitude = -(peak_value + baseline)  # Make amplitude positive
        amplitudes.append(amplitude)
    
    return np.array(amplitudes)
