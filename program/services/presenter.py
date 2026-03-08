"""
Presenter service for FS FilterLab.

Bridges the gap between raw business logic (calculations, data) and the UI views.
Prepares ready-to-render data bundles so views contain zero computation logic.

This module is framework-agnostic — it depends only on models and services,
never on any UI toolkit.
"""
# Standard library imports
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import plotly.graph_objects as go

# Local imports
from models.constants import INTERP_GRID, UI_INFO_MESSAGES
from models.core import AppData, FilterCollection
from services.calculations import (
    compute_filter_transmission,
    compute_selected_filter_indices,
    compute_effective_stops,
    compute_rgb_response,
    compute_active_transmission,
    format_transmission_metrics,
    format_white_balance_data,
    compute_reflector_color,
)
from services.visualization import (
    create_filter_response_plot,
    create_sensor_response_plot,
    prepare_rgb_for_display,
)
from services.state_manager import StateManager

logger = logging.getLogger(__name__)


# =============================================================================
# VIEW MODEL DATA CLASSES
# =============================================================================

@dataclass
class TransmissionMetricsVM:
    """View model for transmission/light loss metrics display."""
    label: str
    avg_transmission_pct: str
    effective_stops: str
    has_valid_data: bool = True
    error_message: Optional[str] = None


@dataclass
class WhiteBalanceVM:
    """View model for white balance display."""
    gains: Dict[str, float]
    intensities: Dict[str, str]  # Pre-formatted strings
    has_filters: bool
    mode_note: str  # e.g. " (from surface reference)" or ""


@dataclass
class FilterAnalysisVM:
    """View model containing everything the filter analysis view needs to render."""
    transmission: np.ndarray
    label: str
    combined: Optional[np.ndarray]
    effective_transmission: np.ndarray
    metrics: Optional[TransmissionMetricsVM]
    filter_response_fig: go.Figure


@dataclass  
class SensorAnalysisVM:
    """View model containing everything the sensor analysis view needs to render."""
    sensor_response_fig: go.Figure
    wb_display: Optional[WhiteBalanceVM]
    has_qe_and_illuminant: bool
    info_message: Optional[str] = None  # Shown when QE+illuminant not available


# =============================================================================
# FILTER ANALYSIS PRESENTER
# =============================================================================

def prepare_filter_analysis(
    app_state: StateManager,
    filter_collection: FilterCollection,
    selected_indices: List[int]
) -> FilterAnalysisVM:
    """
    Prepare all data needed to render the filter analysis section.
    
    Computes transmission, updates state, calculates metrics,
    and creates the Plotly chart — returning everything as a view model.
    
    Args:
        app_state: Application state manager
        filter_collection: The loaded filter collection
        selected_indices: Indices of selected filters in the collection
        
    Returns:
        FilterAnalysisVM with all data needed for display
    """
    # Calculate transmission
    trans, label, combined = compute_filter_transmission(
        selected_indices,
        filter_collection.filter_matrix
    )
    effective_trans = combined if combined is not None else trans
    
    # Update state
    app_state.combined_transmission = effective_trans
    
    # Trigger RGB response computation (needed for downstream effects)
    if app_state.current_qe:
        compute_rgb_response(
            trans,
            app_state.current_qe,
            app_state.white_balance_gains,
            app_state.rgb_channels_visibility,
            app_state.channel_mixer
        )
    
    # Calculate transmission metrics
    metrics = _prepare_transmission_metrics(trans, label, app_state)
    
    # Create filter response chart
    filter_names = [f.name for f in filter_collection.filters]
    filter_hex_colors = [f.hex_color for f in filter_collection.filters]
    
    fig = create_filter_response_plot(
        interp_grid=INTERP_GRID,
        filter_matrix=filter_collection.filter_matrix,
        masks=filter_collection.extrapolated_masks,
        selected_indices=selected_indices,
        combined=combined,
        target_profile=app_state.target_profile,
        log_stops=app_state.log_view,
        filter_names=filter_names,
        filter_hex_colors=filter_hex_colors
    )
    
    return FilterAnalysisVM(
        transmission=trans,
        label=label,
        combined=combined,
        effective_transmission=effective_trans,
        metrics=metrics,
        filter_response_fig=fig
    )


def _prepare_transmission_metrics(
    trans: np.ndarray,
    label: str,
    app_state: StateManager
) -> Optional[TransmissionMetricsVM]:
    """Prepare transmission metrics view model."""
    valid = ~np.isnan(trans)
    if not valid.any():
        return TransmissionMetricsVM(
            label=label,
            avg_transmission_pct="N/A",
            effective_stops="N/A",
            has_valid_data=False,
            error_message=f"Cannot compute average transmission for {label}: insufficient data"
        )
    
    raw_qe = app_state.current_qe.get('G') if app_state.current_qe else None
    if raw_qe is None:
        return TransmissionMetricsVM(
            label=label,
            avg_transmission_pct="N/A",
            effective_stops="N/A",  
            has_valid_data=False,
            error_message=f"Cannot compute light loss for {label}: no sensor QE data"
        )
    
    avg_trans, effective_stops = compute_effective_stops(trans, raw_qe, app_state.illuminant)
    metrics = format_transmission_metrics(trans, label, avg_trans, effective_stops)
    
    return TransmissionMetricsVM(
        label=metrics['label'],
        avg_transmission_pct=metrics['avg_transmission_pct'],
        effective_stops=metrics['effective_stops'],
        has_valid_data=True
    )


# =============================================================================
# SENSOR ANALYSIS PRESENTER  
# =============================================================================

def prepare_sensor_analysis(
    app_state: StateManager,
    data: AppData,
    selected_indices: List[int]
) -> SensorAnalysisVM:
    """
    Prepare all data needed to render the sensor analysis section.
    
    Computes active transmission, white balance, creates sensor response 
    chart, and formats WB display data — returning everything as a view model.
    
    Args:
        app_state: Application state manager
        data: Application data dict with 'filter_collection' and 'reflector_collection'
        selected_indices: Indices of selected filters
        
    Returns:
        SensorAnalysisVM with all data needed for display
    """
    filter_collection = data.filter_collection
    reflector_collection = data.reflector_collection
    
    # Compute active transmission
    trans_interp = compute_active_transmission(
        app_state.selected_filters,
        selected_indices,
        filter_collection.filter_matrix
    )
    
    # Compute and update white balance
    wb_gains = app_state.recompute_white_balance(trans_interp, reflector_collection)
    
    # Create sensor response chart
    fig_response = create_sensor_response_plot(
        interp_grid=INTERP_GRID,
        transmission=trans_interp,
        qe_data=app_state.current_qe,
        visible_channels=app_state.rgb_channels_visibility,
        white_balance_gains=wb_gains,
        apply_white_balance=app_state.apply_white_balance,
        target_profile=app_state.target_profile,
        channel_mixer=app_state.channel_mixer
    )
    
    # Prepare white balance display
    has_qe_and_illuminant = (app_state.current_qe is not None 
                             and app_state.illuminant is not None)
    
    wb_display = None
    info_message = None
    
    if has_qe_and_illuminant:
        wb_data = format_white_balance_data(wb_gains, app_state.selected_filters)
        
        if app_state.wb_reference_surface:
            mode_note = " (from surface reference)"
        elif not wb_data["has_filters"]:
            mode_note = " (No filter selected)"
        else:
            mode_note = ""
        
        wb_display = WhiteBalanceVM(
            gains=wb_gains,
            intensities=wb_data['intensities'],
            has_filters=wb_data['has_filters'],
            mode_note=mode_note
        )
    else:
        info_message = UI_INFO_MESSAGES['qe_illuminant_required']
    
    return SensorAnalysisVM(
        sensor_response_fig=fig_response,
        wb_display=wb_display,
        has_qe_and_illuminant=has_qe_and_illuminant,
        info_message=info_message
    )
