"""
Forms for FS FilterLab.

Advanced filter search, advanced reflector search, and import dialogs.
"""
import functools
import io
import logging
import os
import tempfile
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from nicegui import ui, events

from models.constants import (
    INTERP_GRID, UI_BUTTONS, UI_LABELS, UI_SECTIONS, UI_WARNING_MESSAGES,
    OUTPUT_FOLDERS, IMPORT_TABS, IMPORT_DATA_TYPES, IMPORT_CATEGORIES,
    IMPORT_ECOSIS_MODES,
)
from views.ui_utils import (
    is_dark_color, is_valid_hex_color, handle_error,
    show_success_message,
)
from services.visualization import create_sparkline_plot

logger = logging.getLogger(__name__)


# ============================================================================
# HELPERS
# ============================================================================

@functools.lru_cache(maxsize=512)
def _cached_sparkline(wl_tuple, trans_tuple, color: str, height: int, width: int):
    """LRU-cached sparkline (tuples are hashable)."""
    return create_sparkline_plot(
        np.array(wl_tuple), np.array(trans_tuple), color=color,
        height=height, width=width,
    )


def _sparkline_fig(wavelengths, transmission, color, height=150, width=300):
    return _cached_sparkline(tuple(wavelengths), tuple(transmission), color, height, width)


def _filter_by_manufacturer(df: pd.DataFrame, manufacturers: List[str]) -> pd.DataFrame:
    return df if not manufacturers else df[df["Manufacturer"].isin(manufacturers)]


def _filter_by_trans_at_wavelength(
    df: pd.DataFrame, interp_grid: np.ndarray, matrix: np.ndarray,
    wavelength: int, min_t: float = 0.0, max_t: float = 1.0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    idx = np.where(interp_grid == wavelength)[0]
    if idx.size == 0:
        return df, np.zeros(len(df))
    si = idx[0]
    di = df.index.to_numpy()
    tv = matrix[di, si]
    mask = (tv >= min_t) & (tv <= max_t)
    return df.iloc[mask], tv[mask]


# ============================================================================
# ADVANCED FILTER SEARCH
# ============================================================================

def render_advanced_filter_search(
    df: pd.DataFrame,
    filter_matrix: np.ndarray,
    app_state,
    on_change: Callable,
) -> None:
    """Render the advanced filter search panel."""

    with ui.card().classes("w-full"):
        ui.label(UI_SECTIONS["advanced_filter_search"]).classes("text-lg font-bold")
        ui.label(UI_LABELS["search_by_manufacturer"]).classes("text-sm text-gray-500")

        # Mutable refs for closure-captured control values
        manufs = [app_state.advanced_search_manufacturers]
        wl = [app_state.advanced_search_wavelength]
        tmin = [app_state.advanced_search_trans_min]
        tmax = [app_state.advanced_search_trans_max]
        sort = [app_state.advanced_search_sort]

        with ui.row().classes("w-full items-end gap-2"):
            ui.select(
                options=sorted(df["Manufacturer"].unique().tolist()),
                label="Manufacturer",
                value=app_state.advanced_search_manufacturers,
                multiple=True,
                on_change=lambda e: (manufs.__setitem__(0, list(e.value) if e.value else []),
                                     setattr(app_state, "advanced_search_manufacturers", list(e.value) if e.value else [])),
            ).props("dense outlined use-chips").classes("flex-grow")

            ui.number(
                label="λ (nm)", value=app_state.advanced_search_wavelength, min=300, max=1100, step=5,
                on_change=lambda e: (wl.__setitem__(0, int(e.value)),
                                     setattr(app_state, "advanced_search_wavelength", int(e.value))),
            ).props("dense outlined").style("width:100px")

            ui.number(
                label="Trans min %", value=app_state.advanced_search_trans_min, min=0, max=100, step=1,
                on_change=lambda e: (tmin.__setitem__(0, int(e.value)),
                                     setattr(app_state, "advanced_search_trans_min", int(e.value))),
            ).props("dense outlined").style("width:100px")

            ui.number(
                label="Trans max %", value=app_state.advanced_search_trans_max, min=0, max=100, step=1,
                on_change=lambda e: (tmax.__setitem__(0, int(e.value)),
                                     setattr(app_state, "advanced_search_trans_max", int(e.value))),
            ).props("dense outlined").style("width:100px")

            ui.select(
                options=["Filter Number", "Filter Name", "Hex-Rainbow", "Trans @ λ"],
                label="Sort by", value=app_state.advanced_search_sort,
                on_change=lambda e: (sort.__setitem__(0, e.value),
                                     setattr(app_state, "advanced_search_sort", e.value)),
            ).props("dense outlined").style("width:160px")

            ui.button(UI_BUTTONS["apply"], on_click=lambda: _apply_filter_search(
                df, filter_matrix, app_state, on_change, results_container,
                manufs[0], wl[0], tmin[0], tmax[0], sort[0],
            )).props("dense")

        results_container = ui.column().classes("w-full mt-2")

        # Initial search - use saved state
        _apply_filter_search(
            df, filter_matrix, app_state, on_change, results_container,
            app_state.advanced_search_manufacturers, app_state.advanced_search_wavelength,
            app_state.advanced_search_trans_min, app_state.advanced_search_trans_max,
            app_state.advanced_search_sort,
        )

def _apply_filter_search(
    df, filter_matrix, app_state, on_change, results_container,
    manufs, wl, tmin, tmax, sort_choice,
):
    """Execute filter search and populate results container."""
    import colorsys

    filtered = _filter_by_manufacturer(df, manufs)
    filtered, tv = _filter_by_trans_at_wavelength(
        filtered, INTERP_GRID, filter_matrix, wl, tmin / 100, tmax / 100,
    )

    # Sort
    if sort_choice == "Hex-Rainbow":
        def _hsl(h):
            h = h.lstrip("#")
            try:
                r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                return colorsys.rgb_to_hls(r, g, b)
            except Exception:
                return (0, 0, 0)
        tmp = filtered.copy()
        tmp["_hue"] = tmp["Hex Color"].apply(lambda x: _hsl(x)[0])
        filtered = tmp.sort_values("_hue").drop(columns=["_hue"])
    elif sort_choice.startswith("Trans"):
        tmp = filtered.copy()
        tmp["_t"] = tv
        filtered = tmp.sort_values("_t", ascending=False).drop(columns=["_t"])
    elif sort_choice == "Filter Name":
        filtered = filtered.sort_values("Filter Name")
    else:
        filtered = filtered.sort_values("Filter Number")

    results_container.clear()
    with results_container:
        ui.label(f"{len(filtered)} filters found").classes("font-bold text-sm")
        ui.separator()

        selected_indices = set()

        for row_idx, row in filtered.iterrows():
            hex_c = row["Hex Color"]
            if not is_valid_hex_color(hex_c):
                hex_c = "#888888"
            name = row["Filter Name"]
            number = row["Filter Number"]
            brand = row["Manufacturer"]
            text_color = "#fff" if is_dark_color(hex_c) else "#111"

            # Mutable holder so the toggle closure can reference the detail
            # column that is created *after* the colored row element.
            dc_holder = [None]

            def _make_detail_toggle(dc_h=dc_holder, ri=row_idx, hc=hex_c):
                rendered = [False]
                def _toggle(e):
                    dc = dc_h[0]
                    if dc is None:
                        return
                    dc.set_visibility(e.value)
                    if e.value and not rendered[0]:
                        rendered[0] = True
                        with dc:
                            fig = _sparkline_fig(INTERP_GRID, filter_matrix[ri, :], hc, height=280, width=900)
                            ui.plotly(fig).classes("w-full")

                            def _make_sel(idx=ri):
                                def _t(ev):
                                    if ev.value:
                                        selected_indices.add(idx)
                                    else:
                                        selected_indices.discard(idx)
                                return _t
                            ui.checkbox("Select this filter", on_change=_make_sel(ri)).props("dense")
                return _toggle

            # Colored header row — full-width background matches filter hex color
            with ui.row().classes("w-full items-center gap-2 py-1 px-2").style(
                f"background-color:{hex_c};border-radius:6px"
            ):
                ui.label(f"{number} — {name} — {brand}").classes(
                    "flex-grow text-sm font-semibold"
                ).style(f"color:{text_color}")
                ui.switch("Details", on_change=_make_detail_toggle()).props("dense").style(
                    f"color:{text_color}"
                )

            # Detail container lives outside (below) the colored row
            dc = ui.column().classes("w-full gap-1")
            dc.set_visibility(False)
            dc_holder[0] = dc

        ui.separator()
        with ui.row().classes("gap-2"):
            def _done():
                sel_display = [
                    f"{filtered.loc[i, 'Filter Name']} "
                    f"({filtered.loc[i, 'Filter Number']}, "
                    f"{filtered.loc[i, 'Manufacturer']})"
                    for i in selected_indices if i in filtered.index
                ]
                existing = list(app_state.selected_filters)
                app_state.selected_filters = list(set(existing + sel_display))
                app_state.show_advanced_search = False
                on_change()

            ui.button(UI_BUTTONS["done"], on_click=_done).props("dense")
            ui.button(
                UI_BUTTONS["cancel"],
                on_click=lambda: (setattr(app_state, "show_advanced_search", False), on_change()),
            ).props("dense flat")


# ============================================================================
# ADVANCED REFLECTOR SEARCH
# ============================================================================

def render_advanced_reflector_search(
    df: pd.DataFrame,
    reflector_matrix: np.ndarray,
    app_state,
    on_change: Callable,
) -> None:
    """Render the advanced reflector search panel."""

    with ui.card().classes("w-full"):
        ui.label(UI_SECTIONS["advanced_reflector_search"]).classes("text-lg font-bold")
        ui.label(UI_LABELS["filter_reflectors"]).classes("text-sm text-gray-500")

        orgs = sorted([x for x in df["Organization"].unique() if x])
        packages = sorted([x for x in df["Package Title"].unique() if x])
        targets = sorted([x for x in df["Target Type"].unique() if x])

        sel_orgs = [[]];  sel_pkgs = [[]];  sel_tgts = [[]]

        with ui.row().classes("w-full items-end gap-2"):
            ui.select(
                options=orgs, label="Organization", multiple=True,
                on_change=lambda e: sel_orgs.__setitem__(0, list(e.value) if e.value else []),
            ).props("dense outlined use-chips").classes("flex-grow")
            ui.select(
                options=packages, label="Package/Dataset", multiple=True,
                on_change=lambda e: sel_pkgs.__setitem__(0, list(e.value) if e.value else []),
            ).props("dense outlined use-chips").classes("flex-grow")
            ui.select(
                options=targets, label="Target Type", multiple=True,
                on_change=lambda e: sel_tgts.__setitem__(0, list(e.value) if e.value else []),
            ).props("dense outlined use-chips").classes("flex-grow")

            ui.button(UI_BUTTONS["apply"], on_click=lambda: _apply_reflector_search(
                df, reflector_matrix, app_state, on_change, refl_results,
                sel_orgs[0], sel_pkgs[0], sel_tgts[0],
            )).props("dense")

        refl_results = ui.column().classes("w-full mt-2")

        _apply_reflector_search(df, reflector_matrix, app_state, on_change, refl_results, [], [], [])

        ui.separator()
        ui.button(
            UI_BUTTONS["done"],
            on_click=lambda: (setattr(app_state, "show_reflector_search", False), on_change()),
        ).props("dense")


def _apply_reflector_search(df, reflector_matrix, app_state, on_change, container, orgs, pkgs, tgts):
    filtered = df.copy()
    if orgs:
        filtered = filtered[filtered["Organization"].isin(orgs)]
    if pkgs:
        filtered = filtered[filtered["Package Title"].isin(pkgs)]
    if tgts:
        filtered = filtered[filtered["Target Type"].isin(tgts)]

    container.clear()
    with container:
        ui.label(f"{len(filtered)} reflectors found").classes("font-bold text-sm")
        ui.separator()

        for row_idx, row in filtered.iterrows():
            sf = row["Source File"]
            is_def = app_state.is_default_reflector(sf)
            name = row["Name"]
            org = row.get("Organization", "") or row.get("Source Folder", "")
            tt = row.get("Target Type", "")

            with ui.row().classes("w-full items-center gap-2 py-1"):
                with ui.column().classes("flex-grow gap-0"):
                    ui.label(name).classes("font-medium text-sm")
                    meta_parts = [x for x in [org, tt] if x]
                    if meta_parts:
                        ui.label(" | ".join(meta_parts)).classes("text-xs text-gray-500")

                # Details toggle — lazy-renders sparkline on first open
                detail_container = ui.column().classes("w-full gap-1")
                detail_container.set_visibility(False)

                def _make_detail_toggle(dc=detail_container, ri=row_idx):
                    rendered = [False]
                    def _toggle(e):
                        dc.set_visibility(e.value)
                        if e.value and not rendered[0]:
                            rendered[0] = True
                            with dc:
                                fig = _sparkline_fig(INTERP_GRID, reflector_matrix[ri, :], "#4CAF50")
                                ui.plotly(fig).classes("w-full").style("height:120px")
                    return _toggle

                ui.switch("Details", on_change=_make_detail_toggle()).props("dense")

                if is_def:
                    def _make_rem(s=sf):
                        def _r():
                            app_state.remove_from_default_reflectors(s)
                            on_change()
                        return _r
                    ui.button("X", on_click=_make_rem()).props("dense flat size=sm color=negative").tooltip(
                        "Remove from defaults"
                    )
                else:
                    def _make_add(s=sf):
                        def _a():
                            app_state.add_to_default_reflectors(s)
                            on_change()
                        return _a
                    ui.button("+", on_click=_make_add()).props("dense flat size=sm").tooltip(
                        "Add to defaults"
                    )


# ============================================================================
# IMPORT DIALOG
# ============================================================================

def render_import_dialog(app_state, on_change: Callable) -> None:
    """Render the data import tabs."""

    with ui.card().classes("w-full"):
        ui.label(UI_SECTIONS["import_data"]).classes("text-lg font-bold")

        with ui.tabs().classes("w-full") as tabs:
            tab_f = ui.tab(IMPORT_TABS["filters"])
            tab_i = ui.tab(IMPORT_TABS["illuminants"])
            tab_q = ui.tab(IMPORT_TABS["camera_qe"])
            tab_r = ui.tab(IMPORT_TABS["reflectance_ecosis"])

        with ui.tab_panels(tabs).classes("w-full"):
            with ui.tab_panel(tab_f):
                _import_filter_panel(on_change)
            with ui.tab_panel(tab_i):
                _import_illuminant_panel(on_change)
            with ui.tab_panel(tab_q):
                _import_qe_panel(on_change)
            with ui.tab_panel(tab_r):
                _import_reflectance_panel(on_change)


# -- Reusable import helpers --

def _file_upload(label: str) -> List:
    """Render a CSV upload widget and return mutable [bytes|None, filename] state."""
    file_state: List = [None, ""]

    def _on_upload(e: events.UploadEventArguments):
        file_state[0] = e.content.read()
        file_state[1] = e.name

    ui.upload(label=label, on_upload=_on_upload, auto_upload=True).props(
        "accept=.csv dense"
    ).classes("w-full")

    return file_state

def _make_import_button(
    file_state: List,
    import_fn: Callable,
    button_label: str,
    on_change: Callable,
    success_message: str,
    build_args: Callable,
) -> None:
    """Create the import button that reads file_state and calls import_fn."""
    def _import():
        if file_state[0] is None:
            handle_error(UI_LABELS["upload_file_first"], "warning")
            return
        buf = io.BytesIO(file_state[0])
        buf.name = file_state[1]
        args = build_args(buf)
        success, msg = import_fn(*args)
        if success:
            show_success_message(success_message)
            on_change()
        else:
            handle_error(f"Import failed: {msg}")

    ui.button(button_label, on_click=_import).props("dense color=primary")


# -- Individual import panels --

def _import_filter_panel(on_change: Callable) -> None:
    from services.importing import import_filter_from_csv

    file_state = _file_upload(UI_LABELS["upload_csv_wl_trans"])

    manufacturer = ui.input("Manufacturer", value="Custom").props("dense outlined")
    filter_name = ui.input("Filter Name", value="Custom Filter").props("dense outlined")
    filter_number = ui.input("Filter Number", value="001").props("dense outlined")
    hex_color = ui.color_input("Color", value="#808080").props("dense outlined")

    def _build_args(buf):
        meta = {
            "manufacturer": manufacturer.value.strip(),
            "filter_name": filter_name.value.strip(),
            "filter_number": filter_number.value.strip(),
            "hex_color": hex_color.value or "#808080",
        }
        return (buf, meta, True, True)

    _make_import_button(
        file_state, import_filter_from_csv,
        UI_BUTTONS["import_filter"], on_change,
        "Filter imported successfully!", _build_args,
    )


def _import_illuminant_panel(on_change: Callable) -> None:
    from services.importing import import_illuminant_from_csv

    file_state = _file_upload(UI_LABELS["upload_csv_wl_power"])

    name_input = ui.input("Illuminant Name", value="Custom Illuminant").props("dense outlined")

    _make_import_button(
        file_state, import_illuminant_from_csv,
        UI_BUTTONS["import_illuminant"], on_change,
        "Illuminant imported!",
        lambda buf: (buf, name_input.value.strip()),
    )


def _import_qe_panel(on_change: Callable) -> None:
    from services.importing import import_qe_from_csv

    file_state = _file_upload(UI_LABELS["upload_csv_wl_rgb"])

    brand = ui.input("Brand", value="Custom").props("dense outlined")
    model = ui.input("Model", value="Custom Model").props("dense outlined")

    _make_import_button(
        file_state, import_qe_from_csv,
        UI_BUTTONS["import_camera_qe"], on_change,
        "Camera QE imported!",
        lambda buf: (buf, brand.value.strip(), model.value.strip()),
    )


def _import_reflectance_panel(on_change: Callable) -> None:
    from services.importing import (
        import_reflectance_absorption_from_csv,
        import_ecosis_csv,
        get_ecosis_csv_metadata_columns,
    )

    ui.label(UI_SECTIONS["import_reflectance_absorption"]).classes("font-bold text-sm")

    # -- Mode containers (only one visible at a time) --
    single_section = ui.column().classes("w-full gap-2")
    ecosis_section = ui.column().classes("w-full gap-2")
    ecosis_section.set_visibility(False)

    def _on_mode_change(e):
        is_single = "Single" in e.value
        single_section.set_visibility(is_single)
        ecosis_section.set_visibility(not is_single)

    ui.radio(
        IMPORT_ECOSIS_MODES,
        value=IMPORT_ECOSIS_MODES[0],
        on_change=_on_mode_change,
    ).props("dense inline")

    # -- Single spectrum --
    with single_section:
        file_state = _file_upload(UI_LABELS["upload_csv"])

        sname = ui.input("Spectrum Name", value="Custom Spectrum").props("dense outlined")
        stype = ui.select(IMPORT_DATA_TYPES, label="Data Type", value="Reflectance").props("dense outlined")
        scat = ui.select(IMPORT_CATEGORIES, label="Category", value="Plant").props("dense outlined")

        _make_import_button(
            file_state, import_reflectance_absorption_from_csv,
            UI_BUTTONS["import_single_spectrum"], on_change,
            "Spectrum imported!",
            lambda buf: (buf, {"name": sname.value.strip(), "data_type": stype.value,
                               "category": scat.value, "description": ""}, True, True),
        )

    # -- ECOSIS multi-spectrum --
    with ecosis_section:
        ecosis_state: Dict = {"content": None, "name": "", "tmp_path": None, "meta_cols": []}
        meta_cols_container = ui.column().classes("w-full")
        name_col = [None]
        relevant_cols: List = [[]]

        def _up_e(e: events.UploadEventArguments):
            ecosis_state["content"] = e.content.read()
            ecosis_state["name"] = e.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(ecosis_state["content"])
                ecosis_state["tmp_path"] = tmp.name
            ecosis_state["meta_cols"] = get_ecosis_csv_metadata_columns(ecosis_state["tmp_path"])
            meta_cols_container.clear()
            with meta_cols_container:
                if ecosis_state["meta_cols"]:
                    ui.label(UI_SECTIONS["column_selection"]).classes("font-bold text-sm")
                    ui.select(
                        options=["None"] + ecosis_state["meta_cols"],
                        label=UI_LABELS["choose_name_column"],
                        value="None",
                        on_change=lambda ev: name_col.__setitem__(0, ev.value if ev.value != "None" else None),
                    ).props("dense outlined").classes("w-full")
                    ui.select(
                        options=ecosis_state["meta_cols"],
                        label=UI_LABELS["relevant_metadata"],
                        multiple=True,
                        on_change=lambda ev: relevant_cols.__setitem__(0, list(ev.value) if ev.value else []),
                    ).props("dense outlined use-chips").classes("w-full")

        ui.upload(label=UI_LABELS["upload_ecosis_csv"], on_upload=_up_e, auto_upload=True).props(
            "accept=.csv dense"
        ).classes("w-full")
        api_url = ui.input(
            UI_LABELS["ecosis_api_url"], placeholder="https://ecosis.org/api/package/..."
        ).props("dense outlined").classes("w-full")

        def _import_ecosis():
            if ecosis_state["tmp_path"] is None:
                handle_error(UI_LABELS["upload_file_first"], "warning")
                return
            csv_fn = os.path.splitext(ecosis_state["name"])[0]
            csv_fn = "".join(c for c in csv_fn if c.isalnum() or c in " _-").strip().replace(" ", "_")
            out_dir = os.path.join(OUTPUT_FOLDERS["ecosis"], csv_fn)
            os.makedirs(out_dir, exist_ok=True)
            try:
                files = import_ecosis_csv(
                    ecosis_state["tmp_path"], out_dir,
                    api_url.value.strip() or None,
                    name_col[0],
                    relevant_cols[0],
                )
                os.unlink(ecosis_state["tmp_path"])
                show_success_message(f"Imported {len(files)} spectra!")
                on_change()
            except Exception as exc:
                handle_error(f"ECOSIS import failed: {exc}")

        ui.button(UI_BUTTONS["import_ecosis_file"], on_click=_import_ecosis).props("dense color=primary")
