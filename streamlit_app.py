import calendar
import html
from pathlib import Path

import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional, Set

from lunar_calendar_manager import LunarCalendarManager


def _phase_short(phase_name: Optional[str]) -> str:
    if not phase_name:
        return ""

    # Keep it compact so it fits inside a small day cell.
    if "Full" in phase_name:
        return "Full"
    if "New" in phase_name:
        return "New"
    if "First Quarter" in phase_name:
        return "1st Q"
    if "Third Quarter" in phase_name:
        return "3rd Q"
    return phase_name.split()[0]


def _clean_record(record: dict) -> dict:
    cleaned = {}
    for k, v in record.items():
        if pd.isna(v):
            cleaned[k] = None
        elif isinstance(v, pd.Timestamp):
            cleaned[k] = v.strftime("%Y-%m-%d")
        else:
            cleaned[k] = v
    return cleaned


def _build_year_day_info(manager: LunarCalendarManager) -> Dict[str, Dict[str, Any]]:
    # Phase + illumination for each day.
    phase_map: Dict[str, Dict[str, Any]] = {}
    for row in manager.lunar.to_dict(orient="records"):
        key = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        illum = row.get("illumination")
        phase_map[key] = {
            "phase_name": row.get("phase_name"),
            "illumination": None if pd.isna(illum) else float(illum),
        }

    # Special events, grouped by day.
    special_map: Dict[str, List[Dict[str, Any]]] = {}
    for row in manager.special_events.to_dict(orient="records"):
        key = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        special_map.setdefault(key, []).append(_clean_record(row))

    # User events, grouped by day.
    user_map: Dict[str, List[Dict[str, Any]]] = {}
    for row in manager.user_events.to_dict(orient="records"):
        key = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        user_map.setdefault(key, []).append(_clean_record(row))

    # Dark-sky: expand the window ranges to all dates.
    dark_dates: Set[str] = set()
    for row in manager.dark_sky.to_dict(orient="records"):
        start = pd.Timestamp(row["window_start"]).normalize()
        end = pd.Timestamp(row["window_end"]).normalize()
        for d in pd.date_range(start, end, freq="D"):
            dark_dates.add(pd.Timestamp(d).strftime("%Y-%m-%d"))

    # Final per-day info used by the HTML renderer.
    year_day_info: Dict[str, Dict[str, Any]] = {}
    for month in range(1, 13):
        for day in range(1, 32):
            try:
                ts = pd.Timestamp(manager.year, month, day)
            except ValueError:
                continue

            key = ts.strftime("%Y-%m-%d")
            phase = phase_map.get(key, {})
            year_day_info[key] = {
                "phase_name": phase.get("phase_name"),
                "illumination": phase.get("illumination"),
                "special_events": special_map.get(key, []),
                "user_events": user_map.get(key, []),
                "is_dark_sky": key in dark_dates,
            }

    return year_day_info


def _available_years(data_folder: str) -> List[int]:
    p = Path(data_folder)
    years: List[int] = []
    for match in p.glob("lunar_data_*.csv"):
        suffix = match.stem.split("_")[-1]
        if suffix.isdigit():
            years.append(int(suffix))
    years = sorted(set(years))
    return years


def _build_month_table_html(year_day_info: Dict[str, Dict[str, Any]], year: int, month: int) -> str:
    # Month matrix (0 means "blank day" padding).
    weeks = calendar.monthcalendar(year, month)
    month_title = calendar.month_name[month] + f" {year}"
    dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def cell_for_day(day: int) -> str:
        if day == 0:
            return "<td class='daycell empty'></td>"

        key = f"{year:04d}-{month:02d}-{day:02d}"
        info = year_day_info.get(key, {})
        phase_name = info.get("phase_name")
        illumination = info.get("illumination")
        is_dark = bool(info.get("is_dark_sky"))
        special_events = info.get("special_events") or []
        user_events = info.get("user_events") or []

        phase_short = _phase_short(phase_name)
        special_count = len(special_events)
        user_count = len(user_events)

        # Choose a class by the main phase label (so it stands out).
        phase_class = "phase-other"
        if isinstance(phase_name, str):
            if "Full" in phase_name:
                phase_class = "phase-full"
            elif "New" in phase_name:
                phase_class = "phase-new"
            elif "First Quarter" in phase_name:
                phase_class = "phase-firstq"
            elif "Third Quarter" in phase_name:
                phase_class = "phase-thirdq"

        icons = []
        if is_dark:
            icons.append("D")  # dark-sky
        if special_count:
            icons.append(f"S{special_count}")
        if user_count:
            icons.append(f"U{user_count}")
        icons_str = " ".join(icons)

        # Tooltip text for "clickless" per-date details.
        illum_text = "Unknown"
        if isinstance(illumination, (int, float)) and not pd.isna(illumination):
            illum_text = f"{illumination:.2f}%"
        special_text = (
            ", ".join(e.get("event_type") or "Special" for e in special_events) or "None"
        )
        user_text = (
            ", ".join(e.get("event_name") or "Event" for e in user_events) or "None"
        )
        tooltip = (
            f"Date: {key}\n"
            f"Phase: {phase_name or 'N/A'}\n"
            f"Illumination: {illum_text}\n"
            f"Special: {special_text}\n"
            f"User: {user_text}\n"
            f"Dark-sky: {'Yes' if is_dark else 'No'}"
        )
        tooltip_attr = html.escape(tooltip)

        return (
            f"<td class='daycell {phase_class}{' dark' if is_dark else ''}' title='{tooltip_attr}'>"
            f"<div class='daynum'>{day}</div>"
            f"<div class='phase'>{html.escape(phase_short)}</div>"
            f"<div class='events'>{html.escape(icons_str)}</div>"
            f"</td>"
        )

    rows_html = []
    for week in weeks:
        cells = "".join(cell_for_day(day) for day in week)
        rows_html.append(f"<tr>{cells}</tr>")

    return f"""
      <div class='monthbox'>
        <div class='monthtitle'>{html.escape(month_title)}</div>
        <table class='monthtable'>
          <thead>
            <tr>{"".join(f"<th>{d}</th>" for d in dow)}</tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
      </div>
    """.strip()


@st.cache_data(show_spinner=False)
def _load_year_day_info(data_folder: str, year: int) -> Dict[str, Dict[str, Any]]:
    manager = LunarCalendarManager(data_folder=data_folder, year=year)
    return _build_year_day_info(manager)


def main() -> None:
    st.set_page_config(page_title="Astro Calendar", layout="wide")
    st.title("Lunar Phase + Events Calendar")

    default_data_folder = str(Path(__file__).parent / "data")
    if "data_folder" not in st.session_state:
        st.session_state["data_folder"] = default_data_folder
    if "year" not in st.session_state:
        st.session_state["year"] = 2026

    with st.sidebar:
        st.header("Data settings")
        data_folder = st.text_input("data_folder", value=st.session_state["data_folder"])
        st.caption("Must contain CSV files like `lunar_data_<year>.csv`.")

        years = _available_years(data_folder)
        if years:
            min_year, max_year = min(years), max(years)
            if min_year == max_year:
                # Streamlit requires min_value < max_value for slider.
                year = min_year
                st.write(f"Year: {year}")
            else:
                year = st.slider("Year", min_value=min_year, max_value=max_year, value=min_year)
        else:
            year = st.number_input("Year", min_value=1900, max_value=2100, value=st.session_state["year"])

        st.session_state["data_folder"] = data_folder
        st.session_state["year"] = int(year)

    # Load the year once, then render all 12 months quickly.
    year_day_info = _load_year_day_info(data_folder=data_folder, year=int(year))

    st.markdown(
        """
        <style>
          .monthbox { border: 1px solid #ddd; border-radius: 8px; padding: 6px; margin: 6px; background: #fff; }
          .monthtitle { text-align: center; font-weight: 600; font-size: 14px; margin-bottom: 4px; }
          .monthtable { width: 100%; border-collapse: collapse; table-layout: fixed; }
          .monthtable th { font-size: 10px; color: #666; padding: 2px; border-bottom: 1px solid #eee; }
          .monthtable td.daycell { border: 1px solid #f0f0f0; padding: 2px; vertical-align: top; height: 46px; }
          .daycell.empty { background: #fafafa; border-color: #fafafa; }
          .daynum { font-size: 12px; font-weight: 600; }
          .phase { font-size: 10px; color: #333; margin-top: 2px; line-height: 1.0; }
          .events { font-size: 10px; color: #666; margin-top: 2px; line-height: 1.0; }

          /* Phase colors */
          .phase-full { background: #fff0f0; }
          .phase-new { background: #f0fff3; }
          .phase-firstq { background: #eff6ff; }
          .phase-thirdq { background: #f2f2f2; }
          .phase-other { background: #ffffff; }

          /* Dark-sky highlight */
          td.daycell.dark { outline: 2px solid #ffe08a; outline-offset: -2px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render 12 months as a 3x4 grid.
    for row_start in range(1, 13, 3):
        cols = st.columns(3)
        for idx, col in enumerate(cols):
            month = row_start + idx
            if month > 12:
                continue
            col.markdown(
                _build_month_table_html(year_day_info, int(year), month),
                unsafe_allow_html=True,
            )

    # Below the calendar: pick a date for full detail.
    st.divider()
    picked = st.date_input(
        "Pick a date for details",
        value=pd.Timestamp(f"{int(year)}-01-01").date(),
        min_value=pd.Timestamp(f"{int(year)}-01-01").date(),
        max_value=pd.Timestamp(f"{int(year)}-12-31").date(),
    )
    picked_key = pd.Timestamp(picked).strftime("%Y-%m-%d")
    picked_info = year_day_info.get(picked_key)
    if not picked_info:
        st.info("No data found for that date in your CSV files.")
        return

    st.subheader(f"Details for {picked_key}")
    st.write(
        {
            "phase_name": picked_info.get("phase_name"),
            "illumination": picked_info.get("illumination"),
            "is_dark_sky": picked_info.get("is_dark_sky"),
        }
    )

    if picked_info.get("special_events"):
        st.markdown("**Special events**")
        st.write(picked_info["special_events"])
    if picked_info.get("user_events"):
        st.markdown("**User events**")
        st.write(picked_info["user_events"])


if __name__ == "__main__":
    main()

