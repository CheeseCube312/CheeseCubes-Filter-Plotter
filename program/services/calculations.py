"""
Mathematical calculations for FS FilterLab.

This module provides all mathematical computation functions for:
- Transmission calculations and filtering
- Color processing and RGB response
- Channel mixing transformations
- Metrics computation and formatting
- White balance calculations
- Deviation analysis
"""
# Third-party imports
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Local imports
from models.constants import EPSILON, DEFAULT_WB_GAINS, INTERP_GRID
from models.core import FilterCollection, ChannelMixerSettings
from services.channel_mixer import apply_channel_mixing_to_responses, apply_channel_mixing_to_colors

# Constants for calculations
MIN_RGB_VALUE = 1/255  # Minimum RGB value to prevent complete black pixels


# ============================================================================
# TRANSMISSION CALCULATIONS
# ============================================================================

def _compute_combined_transmission(transmission_values: List[np.ndarray], combine: bool = True) -> np.ndarray:
    """
    Compute combined transmission from multiple filter transmissions.
    
    All transmission values are expected to be in fractional scale (0-1).
    Filter combination uses natural multiplication which preserves this scale.
    
    Args:
        transmission_values: List of transmission arrays (each 0-1 scale)
        combine: Whether to combine the transmissions (multiply them)

    Returns:
        Combined transmission in fractional scale (0-1)
    """
    if not transmission_values:
        return np.ones_like(INTERP_GRID)
        
    if combine and len(transmission_values) > 1:
        stack = np.array(transmission_values)
        combined = np.nanprod(stack, axis=0)
        combined[np.any(np.isnan(stack), axis=0)] = np.nan
        return combined
    
    return transmission_values[0]


def compute_filter_transmission(
    filter_indices: List[int], 
    filter_matrix: np.ndarray
) -> Tuple[np.ndarray, str, Optional[np.ndarray]]:
    """
    Compute filter transmission from filter indices.
    
    Args:
        filter_indices: List of filter indices
        filter_matrix: Matrix of filter transmissions
    
    Returns:
        Tuple of (transmission, label, combined_transmission)
    """
    if not filter_indices:
        return np.ones_like(INTERP_GRID), "No Filter", None
    
    if len(filter_indices) > 1:
        transmissions = [filter_matrix[idx] for idx in filter_indices]
        combined = _compute_combined_transmission(transmissions, combine=True)
        combined = np.clip(combined, EPSILON, 1.0)
        return combined, "Combined", combined
    
    transmission = filter_matrix[filter_indices[0]]
    return transmission, "Single", None


def compute_active_transmission(
    selected_filters: List[str],
    selected_indices: List[int],
    filter_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute the active transmission based on selected filters.
    
    Args:
        selected_filters: List of selected filter display names
        selected_indices: List of corresponding filter indices
        filter_matrix: Matrix of filter transmissions
    
    Returns:
        Active transmission array
    """
    if selected_filters and selected_indices and filter_matrix is not None:
        transmissions = [filter_matrix[idx] for idx in selected_indices]
        return _compute_combined_transmission(transmissions, combine=True)
    
    return np.ones_like(INTERP_GRID)  # Identity transmission (no filter effect)


def compute_selected_filter_indices(
    selected_filters: List[str],
    filter_multipliers: Dict[str, int],
    filter_collection: FilterCollection
) -> List[int]:
    """
    Compute indices of selected filters with their multipliers.
    
    Args:
        selected_filters: List of selected filter display names
        filter_multipliers: Dictionary mapping filter names to their multiplier counts
        filter_collection: Collection of available filters
    
    Returns:
        List of filter indices
    """
    if not selected_filters:
        return []
    
    display_to_index = filter_collection.get_display_to_index_map()
    selected_indices = []
    
    for name in selected_filters:
        if name not in display_to_index:
            continue
            
        idx = display_to_index[name]
        count = filter_multipliers.get(name, 1)
        selected_indices.extend([idx] * count)
        
    return selected_indices


def _is_valid_transmission(transmission: np.ndarray) -> bool:
    """
    Check if transmission array is valid for computation.
    
    Args:
        transmission: Transmission values to check
        
    Returns:
        True if transmission is valid, False otherwise
    """
    try:
        return (transmission is not None and 
                hasattr(transmission, '__len__') and
                len(transmission) > 0 and 
                np.any(np.isfinite(transmission)))
    except (TypeError, ValueError):
        # Handle cases where transmission is not array-like or has invalid shape
        return False


# ============================================================================
# METRICS CALCULATIONS
# ============================================================================

def compute_effective_stops(
    transmission: np.ndarray, 
    sensor_qe: np.ndarray,
    illuminant: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Compute effective stops from transmission, sensor QE, and illuminant.
    
    Args:
        transmission: Transmission values (0-1 fractional scale)
        sensor_qe: Sensor quantum efficiency values (0-1 fractional scale)
        illuminant: Illuminant spectrum (arbitrary units, optional)
    
    Returns:
        Tuple of (avg_transmission, effective_stops)
    """
    # Ensure inputs are numpy arrays
    transmission = np.asarray(transmission)
    sensor_qe = np.asarray(sensor_qe)
    
    # Default to uniform illuminant if not provided
    if illuminant is None:
        illuminant = np.ones_like(transmission)
    else:
        illuminant = np.asarray(illuminant)
    
    # Find valid indices where none are NaN
    valid = (~np.isnan(transmission) & ~np.isnan(sensor_qe) & ~np.isnan(illuminant))
    
    # If no valid data, return NaNs immediately
    if not np.any(valid):
        return np.nan, np.nan
    
    clipped_trans = np.clip(transmission[valid], EPSILON, 1.0)
    clipped_qe = sensor_qe[valid]
    clipped_illuminant = illuminant[valid]
    
    # Weight by actual photon flux (illuminant * QE)
    photometric_weights = clipped_illuminant * clipped_qe
    
    # If all weights are zero, cannot compute weighted average
    if np.all(photometric_weights == 0):
        return np.nan, np.nan
    
    # Defensive: Check if arrays are empty before averaging
    if clipped_trans.size == 0 or photometric_weights.size == 0:
        return np.nan, np.nan
    
    # Weighted average transmission by photon flux
    avg_trans = np.average(clipped_trans, weights=photometric_weights)
    
    # Prevent log2 of zero or negative (should be prevented by clipping but be safe)
    if avg_trans <= 0:
        return np.nan, np.nan
    
    effective_stops = -np.log2(avg_trans)
    
    return avg_trans, effective_stops


# ============================================================================
# COLOR PROCESSING AND RGB RESPONSE
# ============================================================================

def compute_rgb_response(
    transmission: np.ndarray,
    quantum_efficiency: Dict[str, np.ndarray],
    white_balance_gains: Dict[str, float],
    visible_channels: Dict[str, bool],
    channel_mixer: Optional[ChannelMixerSettings] = None
) -> Tuple[Dict[str, np.ndarray], np.ndarray, float]:
    """
    Compute RGB response from transmission and quantum efficiency.
    
    Args:
        transmission: Transmission values
        quantum_efficiency: Dictionary of quantum efficiency values by channel
        white_balance_gains: Dictionary of white balance gains by channel
        visible_channels: Dictionary of channel visibility flags
        channel_mixer: Optional channel mixer settings for RGB manipulation
    
    Returns:
        Tuple of (responses_by_channel, rgb_matrix, max_response)
    """
    # Create empty arrays for responses
    responses = {}
    rgb_stack = []
    
    # Check for valid transmission data - early exit if invalid
    if not _is_valid_transmission(transmission) or not quantum_efficiency:
        # Return zero arrays with correct dimensions
        sample_size = len(next(iter(quantum_efficiency.values()))) if quantum_efficiency else len(INTERP_GRID)
        zero_array = np.zeros(sample_size)
        for channel in ['R', 'G', 'B']:
            responses[channel] = zero_array
            rgb_stack.append(zero_array)
        return responses, np.stack(rgb_stack, axis=1) if rgb_stack else np.array([]), 0.0

    # Process each color channel
    max_response = 0.0
    for channel in ['R', 'G', 'B']:
        qe_curve = quantum_efficiency.get(channel)
        
        # Create zero response if no QE data or size mismatch
        if qe_curve is None or len(qe_curve) != len(transmission):
            responses[channel] = np.zeros_like(transmission)
        else:
            # Calculate weighted response with white balance
            gain = max(white_balance_gains.get(channel, 1.0), EPSILON)
            weighted = np.nan_to_num(transmission * qe_curve) / gain * 100
            max_response = max(max_response, np.nanmax(weighted))
            
            # Apply channel visibility
            responses[channel] = weighted if visible_channels.get(channel, True) else np.zeros_like(weighted)
            
        rgb_stack.append(responses[channel])
    
    # Apply channel mixing if enabled
    if channel_mixer is not None and channel_mixer.enabled:
        responses = apply_channel_mixing_to_responses(responses, channel_mixer)
        # Update rgb_stack with mixed responses
        rgb_stack = [responses['R'], responses['G'], responses['B']]

    # Create RGB matrix and normalize
    rgb_matrix = np.stack(rgb_stack, axis=1)
    max_val = np.nanmax(rgb_matrix)
    
    if max_val > 0:
        rgb_matrix = rgb_matrix / max_val
        
    # Clip to valid range
    rgb_matrix = np.clip(rgb_matrix, MIN_RGB_VALUE, 1.0)

    return responses, rgb_matrix, max_response


def compute_white_balance_gains(
    transmission: np.ndarray,
    quantum_efficiency: Dict[str, np.ndarray],
    illuminant: np.ndarray
) -> Dict[str, float]:
    """
    Compute white balance gains from transmission, QE, and illuminant.
    
    Args:
        transmission: Transmission values
        quantum_efficiency: Dictionary of quantum efficiency values by channel
        illuminant: Illuminant curve
    
    Returns:
        Dictionary of white balance gains by channel
    """
    # Early exit for invalid data
    if not _is_valid_transmission(transmission):
        return DEFAULT_WB_GAINS.copy()
        
    # Calculate response per channel
    rgb_resp = {}
    for ch in ['R', 'G', 'B']:
        qe_curve = quantum_efficiency.get(ch)
        if qe_curve is None:
            rgb_resp[ch] = np.nan
            continue
        
        # Find valid data points
        valid = ~np.isnan(transmission) & ~np.isnan(qe_curve) & ~np.isnan(illuminant)
        if not valid.any():
            rgb_resp[ch] = np.nan
            continue
        
        # Calculate total response for this channel
        rgb_resp[ch] = np.nansum(
            transmission[valid] * qe_curve[valid] * illuminant[valid]
        )

    # Normalize gains using green as reference
    g_response = rgb_resp.get('G', np.nan)
    if not np.isnan(g_response) and g_response > EPSILON:
        # Ensure all responses are valid before creating the ratio dictionary
        return {
            ch: rgb_resp.get(ch, 0.0) / g_response if not np.isnan(rgb_resp.get(ch, 0.0)) else 1.0
            for ch in ['R', 'G', 'B']
        }
    
    # Fall back to defaults if we can't normalize
    return DEFAULT_WB_GAINS.copy()


def compute_white_balance_gains_from_surface(
    reflector: np.ndarray,
    transmission: np.ndarray,
    quantum_efficiency: Dict[str, np.ndarray],
    illuminant: np.ndarray
) -> Dict[str, float]:
    """
    Compute white balance gains using a selected surface as reference.
    
    This simulates field white balancing where photographers white balance
    on a specific surface (e.g., foliage in IR photography).
    
    Args:
        reflector: Reflectance spectrum of the reference surface
        transmission: Combined filter transmission values
        quantum_efficiency: Dictionary of quantum efficiency values by channel
        illuminant: Illuminant curve
    
    Returns:
        Dictionary of white balance gains by channel
    """
    # Early exit for invalid data
    if not _is_valid_transmission(transmission):
        return DEFAULT_WB_GAINS.copy()
        
    # Calculate RGB response for this surface under current conditions
    rgb_resp = {}
    for ch in ['R', 'G', 'B']:
        qe_curve = quantum_efficiency.get(ch)
        if qe_curve is None:
            rgb_resp[ch] = np.nan
            continue
        
        # Find valid data points
        valid = (~np.isnan(reflector) & ~np.isnan(transmission) & 
                ~np.isnan(qe_curve) & ~np.isnan(illuminant))
        if not valid.any():
            rgb_resp[ch] = np.nan
            continue
        
        # Calculate channel response: reflector * transmission * QE * illuminant
        rgb_resp[ch] = np.nansum(
            reflector[valid] * 
            transmission[valid] * 
            qe_curve[valid] * 
            illuminant[valid]
        )

    # Normalize gains using green as reference
    g_response = rgb_resp.get('G', np.nan)
    if not np.isnan(g_response) and g_response > EPSILON:
        # Create white balance gains that will make this surface appear neutral
        gains = {
            ch: rgb_resp.get(ch, 0.0) / g_response if not np.isnan(rgb_resp.get(ch, 0.0)) else 1.0
            for ch in ['R', 'G', 'B']
        }
        return gains
    
    # Fall back to defaults if we can't normalize
    return DEFAULT_WB_GAINS.copy()



# ============================================================================
# REFLECTOR CALCULATIONS
# ============================================================================

def compute_reflector_color(
    reflector: np.ndarray,
    transmission: np.ndarray,
    quantum_efficiency: Dict[str, np.ndarray],
    illuminant: np.ndarray,
    channel_mixer: Optional[ChannelMixerSettings] = None,
    white_balance_gains: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Compute reflector color from reflector, transmission, QE, and illuminant.
    
    Args:
        reflector: Reflector spectrum
        transmission: Transmission values
        quantum_efficiency: Dictionary of quantum efficiency values by channel
        illuminant: Illuminant curve
        white_balance_gains: Pre-computed white balance gains (single source of truth)
        channel_mixer: Optional channel mixer settings for color manipulation
    
    Returns:
        RGB color as numpy array [R, G, B]
    """
    # Early exit for invalid data
    if not _is_valid_transmission(transmission) or reflector is None:
        return np.zeros(3)

    # Process each channel
    rgb_resp = {}
    for ch in ['R', 'G', 'B']:
        qe_curve = quantum_efficiency.get(ch)
        if qe_curve is None:
            rgb_resp[ch] = 0.0
            continue

        # Find valid data points
        valid = (~np.isnan(transmission) & ~np.isnan(qe_curve) & 
                ~np.isnan(illuminant) & ~np.isnan(reflector))
                
        if not valid.any():
            rgb_resp[ch] = 0.0
            continue

        # Calculate channel response
        rgb_resp[ch] = np.nansum(
            reflector[valid] * 
            transmission[valid] * 
            qe_curve[valid] * 
            illuminant[valid]
        )

    # Apply white balance with safety against division by zero
    rgb_values = np.zeros(3)
    for i, ch in enumerate(['R', 'G', 'B']):
        # Handle case where white_balance_gains is None
        if white_balance_gains is not None:
            wb_gain = white_balance_gains.get(ch, 1.0)
        else:
            wb_gain = 1.0
        
        if wb_gain > EPSILON:
            rgb_values[i] = rgb_resp.get(ch, 0.0) / wb_gain
        else:
            rgb_values[i] = rgb_resp.get(ch, 0.0)
    
    # Apply channel mixing if enabled
    if channel_mixer is not None and channel_mixer.enabled:
        rgb_values = apply_channel_mixing_to_colors(rgb_values, channel_mixer)
            
    return rgb_values


# ============================================================================
# FORMATTING FUNCTIONS
# ============================================================================

def format_transmission_metrics(
    trans: np.ndarray, 
    label: str, 
    avg_trans: float, 
    effective_stops: float
) -> Dict[str, str]:
    """
    Format transmission metrics for display.
    
    Args:
        trans: Transmission values
        label: Label for the transmission
        avg_trans: Average transmission (0-1)
        effective_stops: Effective light loss in stops
        
    Returns:
        Dictionary containing formatted metrics
    """
    return {
        "label": label,
        "effective_stops": f"{effective_stops:.2f}",
        "avg_transmission_pct": f"{avg_trans * 100:.1f}%"
    }


def format_white_balance_data(
    white_balance_gains: Dict[str, float],
    selected_filters: List[str]
) -> Dict[str, Any]:
    """
    Format white balance data for display.
    
    Args:
        white_balance_gains: Dictionary of white balance gains by channel
        selected_filters: List of selected filter names
        
    Returns:
        Dictionary with formatted white balance data
    """
    # Calculate relative channel intensities (inverted gains)
    intensities = {
        k: (1.0 / v if v != 0 else 0.0)
        for k, v in white_balance_gains.items()
    }
    
    # Add a note if no filters are selected
    has_filters = len(selected_filters) > 0
    
    return {
        "has_filters": has_filters,
        "intensities": {
            "R": f"{intensities['R']:.3f}",
            "G": f"{intensities['G']:.3f}",
            "B": f"{intensities['B']:.3f}"
        }
    }
