"""
FS FilterLab - Entry Point
===================================
Launch with:  python program/app.py
   or from project root:  .venv/Scripts/python program/app.py

NiceGUI application providing full CSS / layout control via Vue + Quasar.
"""
import logging
import sys
from pathlib import Path

# Ensure the program/ directory is on the import path so that
# `from models.xxx import ...` etc. work regardless of working directory.
_program_dir = Path(__file__).resolve().parent
if str(_program_dir) not in sys.path:
    sys.path.insert(0, str(_program_dir))

from nicegui import app, ui

from models.constants import CACHE_DIR
from services.app_operations import initialize_application_data
from services.state_manager import get_state_manager
from views.sidebar import render_sidebar
from views.main_content import render_main_content
from views.actions import handle_app_actions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Build the NiceGUI application UI."""

    # Ensure cache directory exists
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Load data
    data = initialize_application_data()
    if not data:
        @ui.page("/")
        def error_page():
            ui.label("Failed to load application data. Check data files and the terminal for errors.").classes(
                "text-red-600 text-xl m-8"
            )
        ui.run(title="FS FilterLab — Error", host="127.0.0.1", port=8080, reload=False, show=True)
        return

    # 2. State manager
    app_state = get_state_manager()

    # 3. Seed IsDefault reflectors
    app_state.seed_isdefault_reflectors(data.reflector_collection)

    # ================================================================
    # UI STRUCTURE — built per-page via @ui.page decorator
    # ================================================================

    @ui.page("/")
    def index():
        # Mutable container so _refresh can reference areas after assignment
        page_state = {}

        def _refresh():
            """Rebuild both sidebar and main content from current state."""
            render_sidebar(page_state["sidebar_container"], app_state, data, _refresh, _on_action)
            render_main_content(page_state["main_area"], app_state, data, _refresh)

        def _on_action(action_type: str, payload=None):
            handle_app_actions(action_type, payload, app_state, data, _refresh)

        # Page-level styles
        ui.add_head_html('''
        <style>
            body { font-family: Inter, system-ui, sans-serif; }
            /* Fix dropdown width and truncate long filter names */
            .q-menu .q-item__label {
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                max-width: 400px;
            }
            .q-menu {
                max-width: 450px !important;
            }
        </style>
        ''')

        with ui.header().classes("items-center justify-between px-4 py-2 bg-blue-800"):
            ui.label("FS FilterLab").classes("text-white text-lg font-bold")

        drawer = ui.left_drawer(value=True, fixed=True).classes("bg-gray-50 p-2").style("width:380px")
        with drawer:
            page_state["sidebar_container"] = ui.column().classes("w-full gap-0")
        render_sidebar(page_state["sidebar_container"], app_state, data, _refresh, _on_action)

        # Main content area
        page_state["main_area"] = ui.column().classes("w-full p-4")
        render_main_content(page_state["main_area"], app_state, data, _refresh)

    # 4. Auto-shutdown when last client disconnects
    connected_clients = set()

    @app.on_connect
    def _on_connect(client):
        connected_clients.add(client.id)

    @app.on_disconnect
    def _on_disconnect(client):
        connected_clients.discard(client.id)
        if not connected_clients:
            # No clients left: gracefully shutdown server
            app.shutdown()

    ui.run(
        title="FS FilterLab",
        host="127.0.0.1",
        port=8080,
        reload=False,
        show=True,
    )


if __name__ == "__main__":
    main()
