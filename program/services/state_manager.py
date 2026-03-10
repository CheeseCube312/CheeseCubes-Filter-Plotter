"""
State Management for FS FilterLab.

Dict-backed reactive state with change callbacks. UI binds to state values,
and updates trigger targeted re-renders via registered listeners.
"""
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from models.constants import DEFAULT_WB_GAINS, MY_FILTERS_FILE
from models.core import ChannelMixerSettings, Filter

logger = logging.getLogger(__name__)


class StateManager:
    """
    Application state manager.

    Uses a plain dict as backing store with explicit property accessors.
    Change listeners can be registered to trigger UI updates when specific
    state keys change.
    """

    _DEFAULT_REFLECTORS_FILE = "program/data/reflectors/default_reflectors.json"

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._listeners: Dict[str, List[Callable]] = {}
        self._init_defaults()

    # ====================================================================
    # INITIALISATION
    # ====================================================================

    def _init_defaults(self) -> None:
        """Populate the backing store with sensible defaults."""
        defaults: Dict[str, Any] = {
            # Filter data
            "selected_filters": [],
            "filter_multipliers": {},
            # QE and illuminant
            "current_qe": None,
            "selected_camera": None,
            "illuminant": None,
            "illuminant_name": None,
            # Target profile
            "target_profile": None,
            "selected_target_name": None,
            # Computed results
            "combined_transmission": None,
            "white_balance_gains": DEFAULT_WB_GAINS.copy(),
            "wb_reference_surface": None,
            # Export
            "last_export": {},
            # Display toggles
            "log_view": False,
            "apply_white_balance": False,
            "show_advanced_search": False,
            "show_import_data": False,
            "show_reflector_search": False,
            "show_my_filters_manager": False,
            # My Filters toggle (search only owned)
            "my_filters_only": False,
            # Advanced filter search state (persists across UI rebuilds)
            "advanced_search_manufacturers": [],
            "advanced_search_wavelength_criteria": [{"wavelength": 550, "trans_min": 0, "trans_max": 100}],
            "advanced_search_sort": "Filter Number",
            # RGB channel visibility
            "show_R": True,
            "show_G": True,
            "show_B": True,
            # Channel mixer settings (identity matrix, disabled)
            "channel_mixer": ChannelMixerSettings(),
            # Import state
            "import_status": None,
            "import_error_message": None,
            # Sidebar expansion open/closed state (persists across rebuilds)
            "sidebar_expansions": {},
        }
        for key, value in defaults.items():
            if key not in self._store:
                self._store[key] = value

    # ====================================================================
    # CHANGE LISTENER MECHANISM
    # ====================================================================

    def on_change(self, key: str, callback: Callable) -> None:
        """Register *callback* to fire when *key* changes."""
        self._listeners.setdefault(key, []).append(callback)

    def _notify(self, key: str) -> None:
        for cb in self._listeners.get(key, []):
            try:
                cb()
            except Exception:
                logger.exception("Listener error for key %s", key)

    def _set(self, key: str, value: Any) -> None:
        old = self._store.get(key)
        self._store[key] = value
        # Avoid spurious notifications for identical primitive values
        if old is not value:
            self._notify(key)

    # ====================================================================
    # WIDGET-CONTROLLED PROPERTIES
    # ====================================================================

    @property
    def selected_filters(self) -> List[str]:
        return self._store.get("selected_filters", [])

    @selected_filters.setter
    def selected_filters(self, value: List[str]) -> None:
        self._set("selected_filters", list(value))

    @property
    def show_advanced_search(self) -> bool:
        return self._store.get("show_advanced_search", False)

    @show_advanced_search.setter
    def show_advanced_search(self, value: bool) -> None:
        self._set("show_advanced_search", value)

    # Advanced filter search state properties
    @property
    def advanced_search_manufacturers(self) -> List[str]:
        return self._store.get("advanced_search_manufacturers", [])

    @advanced_search_manufacturers.setter
    def advanced_search_manufacturers(self, value: List[str]) -> None:
        self._set("advanced_search_manufacturers", value)

    @property
    def advanced_search_wavelength_criteria(self) -> List[Dict]:
        """List of wavelength criteria, each with {wavelength, trans_min, trans_max}."""
        return self._store.get("advanced_search_wavelength_criteria", [{"wavelength": 550, "trans_min": 0, "trans_max": 100}])

    @advanced_search_wavelength_criteria.setter
    def advanced_search_wavelength_criteria(self, value: List[Dict]) -> None:
        self._set("advanced_search_wavelength_criteria", value)

    @property
    def advanced_search_sort(self) -> str:
        return self._store.get("advanced_search_sort", "Filter Number")

    @advanced_search_sort.setter
    def advanced_search_sort(self, value: str) -> None:
        self._set("advanced_search_sort", value)

    @property
    def show_import_data(self) -> bool:
        return self._store.get("show_import_data", False)

    @show_import_data.setter
    def show_import_data(self, value: bool) -> None:
        self._set("show_import_data", value)

    @property
    def show_channel_mixer(self) -> bool:
        return self.channel_mixer.enabled

    @show_channel_mixer.setter
    def show_channel_mixer(self, value: bool) -> None:
        mixer = self.channel_mixer
        if mixer.enabled != value:
            mixer.enabled = value
            self.channel_mixer = mixer

    @property
    def show_reflector_search(self) -> bool:
        return self._store.get("show_reflector_search", False)

    @show_reflector_search.setter
    def show_reflector_search(self, value: bool) -> None:
        self._set("show_reflector_search", value)

    @property
    def sidebar_expansions(self) -> Dict[str, bool]:
        """Open/closed state for each named sidebar expander."""
        return self._store.get("sidebar_expansions", {})

    @sidebar_expansions.setter
    def sidebar_expansions(self, value: Dict[str, bool]) -> None:
        self._store["sidebar_expansions"] = value  # direct write — no listener needed

    # ====================================================================
    # COMPUTED / DISPLAY PROPERTIES
    # ====================================================================

    @property
    def log_view(self) -> bool:
        return self._store.get("log_view", False)

    @log_view.setter
    def log_view(self, value: bool) -> None:
        self._set("log_view", value)

    @property
    def apply_white_balance(self) -> bool:
        return self._store.get("apply_white_balance", False)

    @apply_white_balance.setter
    def apply_white_balance(self, value: bool) -> None:
        self._set("apply_white_balance", value)

    @property
    def rgb_channels_visibility(self) -> Dict[str, bool]:
        return {
            "R": self._store.get("show_R", True),
            "G": self._store.get("show_G", True),
            "B": self._store.get("show_B", True),
        }

    @rgb_channels_visibility.setter
    def rgb_channels_visibility(self, value: Dict[str, bool]) -> None:
        for ch in ("R", "G", "B"):
            self._set(f"show_{ch}", value.get(ch, True))

    @property
    def channel_mixer(self) -> ChannelMixerSettings:
        return self._store.get("channel_mixer", ChannelMixerSettings())

    @channel_mixer.setter
    def channel_mixer(self, value: ChannelMixerSettings) -> None:
        self._set("channel_mixer", value)

    # ====================================================================
    # READ-WRITE STATE PROPERTIES
    # ====================================================================

    @property
    def filter_multipliers(self) -> Dict[str, int]:
        return self._store.get("filter_multipliers", {})

    @filter_multipliers.setter
    def filter_multipliers(self, value: Dict[str, int]) -> None:
        self._set("filter_multipliers", dict(value))

    @property
    def current_qe(self) -> Optional[Dict[str, np.ndarray]]:
        return self._store.get("current_qe")

    @current_qe.setter
    def current_qe(self, value: Optional[Dict[str, np.ndarray]]) -> None:
        self._set("current_qe", value)

    @property
    def selected_camera(self) -> Optional[str]:
        return self._store.get("selected_camera")

    @selected_camera.setter
    def selected_camera(self, value: Optional[str]) -> None:
        self._set("selected_camera", value)

    @property
    def illuminant(self) -> Optional[np.ndarray]:
        return self._store.get("illuminant")

    @illuminant.setter
    def illuminant(self, value: Optional[np.ndarray]) -> None:
        self._set("illuminant", value)

    @property
    def illuminant_name(self) -> Optional[str]:
        return self._store.get("illuminant_name")

    @illuminant_name.setter
    def illuminant_name(self, value: Optional[str]) -> None:
        self._set("illuminant_name", value)

    @property
    def target_profile(self) -> Optional[Filter]:
        return self._store.get("target_profile")

    @target_profile.setter
    def target_profile(self, value: Optional[Filter]) -> None:
        self._set("target_profile", value)

    @property
    def selected_target_name(self) -> Optional[str]:
        return self._store.get("selected_target_name")

    @selected_target_name.setter
    def selected_target_name(self, value: Optional[str]) -> None:
        self._set("selected_target_name", value)

    @property
    def combined_transmission(self) -> Optional[np.ndarray]:
        return self._store.get("combined_transmission")

    @combined_transmission.setter
    def combined_transmission(self, value: Optional[np.ndarray]) -> None:
        self._set("combined_transmission", value)

    @property
    def white_balance_gains(self) -> Dict[str, float]:
        g = self._store.get("white_balance_gains")
        return g.copy() if isinstance(g, dict) else DEFAULT_WB_GAINS.copy()

    @white_balance_gains.setter
    def white_balance_gains(self, value: Dict[str, float]) -> None:
        self._set("white_balance_gains", dict(value))

    @property
    def wb_reference_surface(self) -> Optional[str]:
        return self._store.get("wb_reference_surface")

    @wb_reference_surface.setter
    def wb_reference_surface(self, value: Optional[str]) -> None:
        self._set("wb_reference_surface", value)

    @property
    def last_export(self) -> Dict:
        return self._store.get("last_export", {})

    @last_export.setter
    def last_export(self, value: Dict) -> None:
        self._set("last_export", value)

    # ====================================================================
    # WHITE BALANCE METHODS
    # ====================================================================

    def set_white_balance_from_surface(
        self,
        reflector: np.ndarray,
        transmission: np.ndarray,
        source_file: Optional[str] = None,
    ) -> None:
        from services.calculations import compute_white_balance_gains_from_surface

        if self.current_qe and self.illuminant is not None:
            new_gains = compute_white_balance_gains_from_surface(
                reflector, transmission, self.current_qe, self.illuminant
            )
            self.white_balance_gains = new_gains
            self.wb_reference_surface = source_file

    def reset_white_balance(self) -> None:
        self.white_balance_gains = DEFAULT_WB_GAINS.copy()
        self.wb_reference_surface = None

    def recompute_white_balance(
        self,
        trans_interp: np.ndarray,
        reflector_collection: Any = None,
    ) -> Dict[str, float]:
        """Recompute and store white balance gains.

        If a reference surface is set, recalculates gains from that surface.
        Otherwise computes standard WB from the filter transmission, QE, and
        illuminant.  Returns the (possibly updated) gains dict.
        """
        from services.calculations import (
            compute_white_balance_gains,
            compute_white_balance_gains_from_surface,
        )

        wb_gains = self.white_balance_gains
        if not self.current_qe or self.illuminant is None:
            return wb_gains

        wb_ref = self.wb_reference_surface
        if wb_ref and reflector_collection:
            ref_reflector = None
            for r in reflector_collection.reflectors:
                if r.metadata.get("source_file", "") == wb_ref:
                    ref_reflector = r
                    break

            if ref_reflector is not None:
                wb_gains = compute_white_balance_gains_from_surface(
                    ref_reflector.values, trans_interp,
                    self.current_qe, self.illuminant,
                )
                self.white_balance_gains = wb_gains
            else:
                logger.warning("WB reference surface not found: %s", wb_ref)
                wb_gains = compute_white_balance_gains(
                    trans_interp, self.current_qe, self.illuminant,
                )
                self.white_balance_gains = wb_gains
                self.wb_reference_surface = None
        else:
            wb_gains = compute_white_balance_gains(
                trans_interp, self.current_qe, self.illuminant,
            )
            self.white_balance_gains = wb_gains

        return wb_gains

    # ====================================================================
    # MULTI-FIELD SELECTION HELPERS
    # ====================================================================

    def select_illuminant(self, name: str, illuminants: Dict[str, Any]) -> None:
        """Set the active illuminant by name."""
        self.illuminant_name = name
        self.illuminant = illuminants.get(name)

    def select_camera(self, name: Optional[str], qe_data: Dict[str, Any]) -> None:
        """Set the active camera/QE profile by name."""
        if name and name != "None":
            self.selected_camera = name
            self.current_qe = qe_data.get(name)
        else:
            self.selected_camera = None
            self.current_qe = None

    def select_target(self, name: Optional[str], filter_collection: Any) -> None:
        """Set the target profile from a named filter."""
        if not name or name == "None":
            self.target_profile = None
            self.selected_target_name = None
        else:
            display_to_index = filter_collection.get_display_to_index_map()
            idx = display_to_index[name]
            self.target_profile = filter_collection.filters[idx]
            self.selected_target_name = name

    # ====================================================================
    # DEFAULT REFLECTOR LIST MANAGEMENT
    # ====================================================================

    def _load_default_reflectors_from_file(self) -> List[str]:
        fp = Path(self._DEFAULT_REFLECTORS_FILE)
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                return data.get("default_reflectors", [])
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_default_reflectors_to_file(self, reflector_files: List[str]) -> None:
        fp = Path(self._DEFAULT_REFLECTORS_FILE)
        fp.parent.mkdir(parents=True, exist_ok=True)
        try:
            fp.write_text(
                json.dumps({"default_reflectors": reflector_files}, indent=2),
                encoding="utf-8",
            )
        except IOError:
            pass

    def get_default_reflector_files(self) -> List[str]:
        if "default_reflector_files" not in self._store:
            self._store["default_reflector_files"] = self._load_default_reflectors_from_file()
        return self._store.get("default_reflector_files", [])

    def is_default_reflector(self, source_file: str) -> bool:
        return source_file in self.get_default_reflector_files()

    def add_to_default_reflectors(self, source_file: str) -> None:
        current = self.get_default_reflector_files()
        if source_file not in current:
            current.append(source_file)
            self._store["default_reflector_files"] = current
            self._save_default_reflectors_to_file(current)
            self._notify("default_reflector_files")

    def remove_from_default_reflectors(self, source_file: str) -> None:
        current = self.get_default_reflector_files()
        if source_file in current:
            current.remove(source_file)
            self._store["default_reflector_files"] = current
            self._save_default_reflectors_to_file(current)
            self._notify("default_reflector_files")

    def seed_isdefault_reflectors(self, reflector_collection: Any) -> None:
        if not reflector_collection or not hasattr(reflector_collection, "reflectors"):
            return
        for reflector in reflector_collection.reflectors:
            is_default = reflector.metadata.get("IsDefault", "").strip()
            if is_default:
                source_file = reflector.metadata.get("source_file", "")
                if source_file and not self.is_default_reflector(source_file):
                    self.add_to_default_reflectors(source_file)

    # ====================================================================
    # MY FILTERS (OWNED FILTERS) MANAGEMENT
    # ====================================================================

    _MY_FILTERS_FILE = MY_FILTERS_FILE

    def _load_my_filters_from_file(self) -> List[str]:
        fp = Path(self._MY_FILTERS_FILE)
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                return data.get("my_filters", [])
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_my_filters_to_file(self, filter_names: List[str]) -> None:
        fp = Path(self._MY_FILTERS_FILE)
        fp.parent.mkdir(parents=True, exist_ok=True)
        try:
            fp.write_text(
                json.dumps({"my_filters": sorted(filter_names)}, indent=2),
                encoding="utf-8",
            )
        except IOError:
            pass

    def get_my_filters(self) -> List[str]:
        if "_my_filters_set" not in self._store:
            self._store["_my_filters_set"] = set(self._load_my_filters_from_file())
        return self._store["_my_filters_set"]

    def get_my_filters_count(self) -> int:
        return len(self.get_my_filters())

    def is_my_filter(self, display_name: str) -> bool:
        return display_name in self.get_my_filters()

    def add_to_my_filters(self, display_name: str) -> None:
        current = self.get_my_filters()
        if display_name not in current:
            current.add(display_name)
            self._save_my_filters_to_file(list(current))
            self._notify("my_filters")

    def remove_from_my_filters(self, display_name: str) -> None:
        current = self.get_my_filters()
        if display_name in current:
            current.discard(display_name)
            self._save_my_filters_to_file(list(current))
            self._notify("my_filters")

    def add_manufacturer_to_my_filters(self, manufacturer: str, filter_collection) -> int:
        """Add all filters from a manufacturer. Returns count added."""
        added = 0
        current = self.get_my_filters()
        for f in filter_collection.filters:
            name = str(f)
            if f.manufacturer == manufacturer and name not in current:
                current.add(name)
                added += 1
        if added:
            self._save_my_filters_to_file(list(current))
            self._notify("my_filters")
        return added

    def remove_manufacturer_from_my_filters(self, manufacturer: str, filter_collection) -> int:
        """Remove all filters from a manufacturer. Returns count removed."""
        removed = 0
        current = self.get_my_filters()
        for f in filter_collection.filters:
            name = str(f)
            if f.manufacturer == manufacturer and name in current:
                current.discard(name)
                removed += 1
        if removed:
            self._save_my_filters_to_file(list(current))
            self._notify("my_filters")
        return removed

    def clear_all_my_filters(self) -> None:
        self._store["_my_filters_set"] = set()
        self._save_my_filters_to_file([])
        self._notify("my_filters")

    @property
    def my_filters_only(self) -> bool:
        return self._store.get("my_filters_only", False)

    @my_filters_only.setter
    def my_filters_only(self, value: bool) -> None:
        self._set("my_filters_only", value)

    @property
    def show_my_filters_manager(self) -> bool:
        return self._store.get("show_my_filters_manager", False)

    @show_my_filters_manager.setter
    def show_my_filters_manager(self, value: bool) -> None:
        self._set("show_my_filters_manager", value)

    # ====================================================================
    # GENERIC STORE ACCESS (for cases where direct key access is needed)
    # ====================================================================

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the backing store."""
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the backing store and fire listeners."""
        self._set(key, value)


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get or create the global state manager singleton."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
