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
    # calendar.month_name is a list where index 1 = "January", 2 = "February", etc.
    # We only show the month name here because the year is displayed as a large heading above the whole calendar grid.
    month_title = calendar.month_name[month]
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

        # We build a small list of text badges to show inside each day cell.
        # S followed by a number means that many special astronomical events fall on this day.
        # U followed by a number means that many user-created events fall on this day.
        # We join them with a space so they appear side by side, for example "S2 U1".
        icons = []
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

    # st.markdown lets us write raw HTML and CSS directly into the Streamlit page.
    # The unsafe_allow_html=True argument is required whenever we pass real HTML, otherwise Streamlit would display it as plain text.
    # Everything inside the <style> tags is CSS, which controls how things look — colors, sizes, spacing, and so on.
    # CSS uses curly braces to group rules for a selector, but because we are inside a Python f-string we must write {{ and }} instead of { and } so Python does not get confused.
    # A CSS selector like .monthbox targets any HTML element that has class="monthbox".
    # The @media (prefers-color-scheme: dark) block is a special CSS rule that only applies when the user's device or browser is set to dark mode.
    # This means we can write separate color rules for light mode and dark mode inside the same stylesheet, was having an issue with white text on a white background in dark mode.
    # The f-string inserts the actual year value into the HTML so the heading always shows the correct year.
    # The legend div sits between the year heading and the calendar grid and uses the CSS classes defined in the style block above.
    st.markdown(
        f"""
        <style>
          /* ── Year heading ── */
          /* This styles the large year number that sits above the calendar grid. */
          .cal-year-heading {{
            text-align: center;
            font-size: 28px;
            font-weight: 700;
            letter-spacing: 2px;
            margin: 0 0 12px 0;
            color: #1a1a2e;
          }}
          /* In dark mode we switch to a light colour so the heading stays readable. */
          @media (prefers-color-scheme: dark) {{
            .cal-year-heading {{ color: #e8e8f0; }}
          }}

          /* ── Month box ── */
          /* Each month lives inside a rounded box with a subtle border. */
          .monthbox {{
            border: 1px solid #d0d0e0;
            border-radius: 10px;
            padding: 8px;
            margin: 6px;
            background: #ffffff;
          }}
          @media (prefers-color-scheme: dark) {{
            .monthbox {{ background: #1e1e2e; border-color: #3a3a5c; }}
          }}

          /* The month name title inside each box, shown in uppercase small text. */
          .monthtitle {{
            text-align: center;
            font-weight: 700;
            font-size: 13px;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #2d2d4e;
          }}
          @media (prefers-color-scheme: dark) {{
            .monthtitle {{ color: #c8c8e8; }}
          }}

          /* The table itself stretches to fill the full width of its container. */
          .monthtable {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
          /* These are the day-of-week header cells at the top of each month table (Mon Tue Wed etc). */
          .monthtable th {{
            font-size: 10px;
            color: #7070a0;
            padding: 2px;
            border-bottom: 2px solid #e0e0f0;
            text-align: center;
          }}
          @media (prefers-color-scheme: dark) {{
            .monthtable th {{ color: #9090c0; border-bottom-color: #3a3a5c; }}
          }}

          /* Each individual day is a table cell (td) with the class daycell. */
          .monthtable td.daycell {{
            border: 1px solid #ebebf5;
            padding: 3px;
            vertical-align: top;
            height: 48px;
          }}
          @media (prefers-color-scheme: dark) {{
            .monthtable td.daycell {{ border-color: #2e2e4a; }}
          }}

          /* Empty cells are the blank padding days at the start or end of a month. */
          .daycell.empty {{ background: #f8f8fc; border-color: #f0f0f8; }}
          @media (prefers-color-scheme: dark) {{
            .daycell.empty {{ background: #16162a; border-color: #1e1e32; }}
          }}

          /* The day number shown in the top-left of each cell, for example 1, 2, 3.
             We must explicitly set the color here because in dark mode Streamlit can make the background dark too,
             which would cause black text on a dark background and make the numbers invisible. */
          .daynum {{
            font-size: 12px;
            font-weight: 700;
            color: #1a1a3a;
          }}
          @media (prefers-color-scheme: dark) {{
            .daynum {{ color: #dcdcf0; }}
          }}

          /* The small moon phase label shown below the day number, for example "Full" or "Waning". */
          .phase {{ font-size: 10px; color: #4a4a7a; margin-top: 2px; line-height: 1.1; }}
          /* The even smaller icon badges shown below the phase label, for example "S1 U2". */
          .events {{ font-size: 10px; color: #7070a0; margin-top: 2px; line-height: 1.1; }}
          @media (prefers-color-scheme: dark) {{
            .phase {{ color: #a0a0cc; }}
            .events {{ color: #8080b0; }}
          }}

          /* ── Vibrant phase colors ── */
          /* Each moon phase gets a distinct background color so it stands out at a glance.
             These are the light-mode versions; each has a darker equivalent below for dark mode. */
          .phase-full  {{ background: #ffe4e8; }}
          .phase-new   {{ background: #d4f5e0; }}
          .phase-firstq {{ background: #dceeff; }}
          .phase-thirdq {{ background: #ede0ff; }}
          /* Waxing and waning days get no special background color. */
          .phase-other  {{ background: #ffffff; }}

          @media (prefers-color-scheme: dark) {{
            .phase-full   {{ background: #4a1e28; }}
            .phase-new    {{ background: #1a3d28; }}
            .phase-firstq {{ background: #1a2e4a; }}
            .phase-thirdq {{ background: #2e1e4a; }}
            .phase-other  {{ background: #1e1e2e; }}
            .daycell.empty {{ background: #16162a; }}
          }}

          /* ── Dark-sky highlight ── */
          /* Days inside a dark-sky observation window get an orange outline so they are easy to spot.
             outline is like border but it sits outside the element without affecting its size or layout. */
          td.daycell.dark {{ outline: 2px solid #f5a623; outline-offset: -2px; }}

          /* ── Legend ── */
          /* The legend is a horizontal strip of labels that explains what the colors and icons mean.
             display: flex makes its children sit side by side in a row.
             flex-wrap: wrap means they will drop to a new line automatically if there is not enough room. */
          .cal-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px 20px;
            margin: 10px 6px 18px 6px;
            font-size: 12px;
            align-items: center;
          }}
          /* Each legend item is also a small flex row so the swatch and its label sit side by side. */
          .leg-item {{ display: flex; align-items: center; gap: 6px; }}
          /* The small colored square that represents a phase color in the legend. */
          .leg-swatch {{
            width: 14px; height: 14px;
            border-radius: 3px;
            border: 1px solid rgba(0,0,0,0.15);
            flex-shrink: 0;
          }}
          .leg-label {{ color: #3a3a5a; }}
          @media (prefers-color-scheme: dark) {{
            .leg-label {{ color: #b0b0d0; }}
            .leg-swatch {{ border-color: rgba(255,255,255,0.15); }}
          }}
          /* This is the small orange-bordered square in the legend that represents a dark-sky window.
             It has no fill so the border alone conveys the meaning, matching the orange outline on day cells. */
          .leg-dark-swatch {{
            width: 14px; height: 14px;
            border-radius: 3px;
            border: 2px solid #f5a623;
            flex-shrink: 0;
            background: transparent;
          }}
        </style>

        <div class="cal-year-heading">{year}</div>

        <div class="cal-legend">
          <strong style="color:#3a3a5a;margin-right:4px;">Phase colors:</strong>
          <div class="leg-item">
            <div class="leg-swatch" style="background:#ffe4e8;"></div>
            <span class="leg-label">🌕 Full Moon</span>
          </div>
          <div class="leg-item">
            <div class="leg-swatch" style="background:#d4f5e0;"></div>
            <span class="leg-label">🌑 New Moon</span>
          </div>
          <div class="leg-item">
            <div class="leg-swatch" style="background:#dceeff;"></div>
            <span class="leg-label">🌓 First Quarter</span>
          </div>
          <div class="leg-item">
            <div class="leg-swatch" style="background:#ede0ff;"></div>
            <span class="leg-label">🌗 Third Quarter</span>
          </div>
          <div class="leg-item">
            <span class="leg-label">(no highlight) = 🌔🌒 Waxing / Waning</span>
          </div>
          <span style="margin-left:8px;color:#3a3a5a;font-weight:600;">Icons:</span>
          <div class="leg-item"><span class="leg-label"><b>S#</b> = # Special events</span></div>
          <div class="leg-item"><span class="leg-label"><b>U#</b> = # User events</span></div>
          <div class="leg-item">
            <div class="leg-dark-swatch"></div>
            <span class="leg-label">Orange border = Dark-sky window</span>
          </div>
        </div>
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

