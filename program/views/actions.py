"""
Action handler for FS FilterLab.

Processes imperative user actions (report generation, cache rebuild)
triggered from sidebar buttons.
"""
import logging
from pathlib import Path
from typing import Any, Callable, Dict

from models.constants import (
    CACHE_DIR, UI_SUCCESS_MESSAGES, UI_WARNING_MESSAGES,
    UI_OPERATION_ERRORS, ACTION_TYPES,
)
from models.core import AppData
from services.app_operations import (
    generate_application_report, rebuild_application_cache,
)
from views.ui_utils import handle_error, show_success_message, try_operation

logger = logging.getLogger(__name__)

# Maps each action type to (operation_factory, success_key, fail_key, error_key).
# operation_factory(app_state, data, payload) -> bool
_REPORT_ACTIONS: Dict[str, Dict[str, str]] = {
    ACTION_TYPES["generate_report"]: {
        "success": "report_generated",
        "fail": "report_generation_failed",
        "error": "report_generation",
    },
    ACTION_TYPES["generate_full_report"]: {
        "success": "full_report_generated",
        "fail": "full_report_generation_failed",
        "error": "full_report_generation",
    },
}


def handle_app_actions(
    action_type: str,
    payload: Any,
    app_state,
    data: AppData,
    on_change: Callable,
) -> None:
    """
    Process a single user action.

    Args:
        action_type: One of the ACTION_TYPES values.
        payload: Action-specific data (e.g. selected camera name).
        app_state: StateManager instance.
        data: Application data dict.
        on_change: Callback to refresh UI after action completes.
    """
    keys = _REPORT_ACTIONS.get(action_type)
    if keys:
        def _run_report():
            result = generate_application_report(
                app_state=app_state,
                filter_collection=data.filter_collection,
                selected_camera=payload,
            )
            if result:
                app_state.last_export = result
                show_success_message(UI_SUCCESS_MESSAGES[keys["success"]])
                on_change()
            else:
                handle_error(UI_WARNING_MESSAGES.get(keys["fail"], "Report generation failed"))
        try_operation(_run_report, UI_OPERATION_ERRORS[keys["error"]])

    elif action_type == ACTION_TYPES["rebuild_cache"]:
        def _rebuild():
            cache_path = Path(CACHE_DIR)
            success = rebuild_application_cache(cache_path)
            if success:
                show_success_message(UI_SUCCESS_MESSAGES["cache_rebuilt"])
                on_change()
            else:
                handle_error(UI_WARNING_MESSAGES.get("cache_rebuild_failed", "Cache rebuild failed"))
        try_operation(_rebuild, UI_OPERATION_ERRORS["cache_rebuild"])

    else:
        logger.warning("Unknown action type: %s", action_type)
