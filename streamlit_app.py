import calendar
import html
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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


# --- Moon phase diagram (matplotlib) ------------------------------------------
#
# The moon drawings are meant to match northern-hemisphere views:
#   - Waxing: light grows from the right toward the left.
#   - Waning: light shrinks from the right, so the bright part sits on the left.
#
# For exact quarters, full, and new we ignore small illumination noise in the CSV
# and draw a simple "icon" shape. For Waxing/Waning *Crescent* and *Gibbous* we
# use the illumination percentage to place the curved terminator between those icons.


MoonVisualKind = Literal["full", "new", "first_quarter", "third_quarter", "waxing_waning"]


def _moon_visual_kind(phase_name: Optional[str]) -> MoonVisualKind:
    """
    Decide whether we should draw a fixed icon (full/new/quarters) or a phase that
    follows illumination (waxing/waning crescent and gibbous).
    """
    if not phase_name:
        return "waxing_waning"

    if "Full" in phase_name:
        return "full"
    if "New" in phase_name:
        return "new"
    if "First Quarter" in phase_name:
        return "first_quarter"
    if "Third Quarter" in phase_name:
        return "third_quarter"

    # Everything else in this project is Waxing/Waning Crescent or Gibbous.
    return "waxing_waning"


def _illumination_fraction(illumination: Optional[float]) -> float:
    """Turn the CSV percentage into a 0-1 fraction, clipped so the geometry stays stable."""
    if illumination is None or (isinstance(illumination, float) and pd.isna(illumination)):
        return 0.0
    return float(np.clip(float(illumination) / 100.0, 0.0, 1.0))


def _terminator_ellipse_x(y_vals: np.ndarray, ellipse_a: float) -> np.ndarray:
    """
    Given an array of y-coordinates and a horizontal semi-axis *ellipse_a*, return the
    matching x-coordinates on the terminator. We use the same shape as an ellipse
    aligned with the moon:

        (x / ellipse_a)² + (y / radius)² = 1

    So at the equator (y = 0) you get x = ellipse_a.

      - ellipse_a > 0  → midpoint of the terminator is on the **eastern (right)** half
      - ellipse_a < 0  → midpoint is on the **western (left)** half
      - ellipse_a = 0  → straight vertical line through the center (quarter moon)

    The waxing and waning helpers choose *ellipse_a* so the **small** illumination
    cases (crescents) put the terminator on the correct side — see those functions.

    We only need x at each y to stitch the terminator into a closed polygon with the
    visible lunar limb (the circular arc on either the right or left side).
    """
    # The ellipse equation is (x/ellipse_a)^2 + (y/radius)^2 = 1
    # Solving for x: x = ellipse_a * sqrt(1 - (y/radius)^2)
    # np.clip keeps the argument of sqrt non-negative despite floating-point rounding.
    return ellipse_a * np.sqrt(np.clip(1.0 - (y_vals / 1.0) ** 2, 0.0, 1.0))


def _waxing_lit_vertices(fraction: float, radius: float = 1.0) -> Optional[np.ndarray]:
    """
    Build a filled polygon for the lit part of a *waxing* moon.

    *fraction* is how much of the disk is illuminated (0 = new moon, 1 = full moon).
    For a northern-hemisphere schematic, waxing means the bright part grows from the
    **right (east)** limb toward the left — so a thin waxing crescent is a bright
    sliver on the **right**, not a big disk with a shadow nibble on the left.

    How the terminator works
    -------------------------
    The terminator (boundary between sunlit and night sides) is modeled with a
    horizontal semi-axis *ellipse_a*:

        ellipse_a = radius * (1 - 2 * fraction)

    Why this formula?
    ------------------------------------
    - Right after **new moon**, only a tiny sliver on the **right** should be lit
      (small *fraction*). The terminator must sit just **inboard** of that lit
      sliver — still on the **eastern** side of the disk — so at the equator the
      terminator’s x-coordinate must be **positive** and fairly large.
    - Plugging a small *fraction* into ``(1 - 2 * fraction)`` gives a **positive**
      *ellipse_a*, which moves the terminator to the correct (eastern) side.

    Edge cases this produces (for the math geometry; new/full are handled separately):
      fraction → 0   →  ellipse_a → +radius   (terminator hugging the east; crescent)
      fraction = 0.5 →  ellipse_a = 0        (straight line — first quarter)
      fraction → 1   →  ellipse_a → -radius  (terminator crosses the west; gibbous)

    The terminator always meets the moon’s north and south poles (y = ±radius) at
    x = 0 for every *ellipse_a*, so it joins cleanly with whichever limb arc we use.

    Crescent vs gibbous (same loop of code)
    ---------------------------------------
    We always trace:
      1. The **right** circular limb from north pole → south pole (the sunlit “outer”
         edge for waxing phases).
      2. The terminator from south pole → north pole back to the start.

    Whether *fraction* is below or above 50%, the sign of *ellipse_a* above picks
    the correct bulge so the filled region is the **small bright crescent** when
    illumination is low, and the **almost-full gibbous** when illumination is high —
    without separate if/else shape logic.
    """
    if fraction <= 0.0:
        return None  # New moon: nothing lit, draw nothing.
    if fraction >= 1.0:
        # Full moon: the whole disk is lit.
        theta = np.linspace(0.0, 2.0 * np.pi, 180)
        return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

    # ── Step 1: terminator semi-axis ──────────────────────────────────────────
    # Positive for waxing crescent (terminator on the eastern half), negative for
    # waxing gibbous (small dark bite on the west). Zero at first quarter.
    ellipse_a = radius * (1.0 - 2.0 * fraction)

    n_pts = 90  # More points = smoother curves; 90 is plenty for this size.

    # ── Step 2: right rim arc (top → bottom, passing through the rightmost point) ──
    # Angles: π/2 at the top of the circle, 0 at the right, -π/2 at the bottom.
    theta_rim = np.linspace(np.pi / 2.0, -np.pi / 2.0, n_pts)
    rim_x = radius * np.cos(theta_rim)
    rim_y = radius * np.sin(theta_rim)

    # ── Step 3: terminator curve (bottom → top) ───────────────────────────────
    # y runs from -radius to +radius.  x = ellipse_a * sqrt(1 - (y/radius)^2).
    # For waxing *crescent* (small fraction), ellipse_a > 0 so the terminator sits
    # in the eastern half and the lit patch stays a thin blade on the **right**.
    # For waxing *gibbous* (large fraction), ellipse_a < 0 and the terminator moves
    # west so only a small unlit region remains on the **left**.
    y_term = np.linspace(-radius, radius, n_pts)
    x_term = _terminator_ellipse_x(y_term, ellipse_a)

    # ── Step 4: concatenate into a closed polygon ──────────────────────────────
    # Rim goes top→bottom; terminator goes bottom→top.  Together they enclose the
    # lit region and matplotlib's fill() closes the shape automatically.
    xs = np.concatenate([rim_x, x_term])
    ys = np.concatenate([rim_y, y_term])
    return np.column_stack([xs, ys])


def _waning_lit_vertices(fraction: float, radius: float = 1.0) -> Optional[np.ndarray]:
    """
    Build a filled polygon for the lit part of a *waning* moon.

    *fraction* is how much of the disk is illuminated (0 = new moon, 1 = full moon).
    Waning is the mirror of waxing: the bright cap hugs the **left (west)** limb,
    so a thin **waning** crescent is a bright sliver on the **left**.

    Mirror recipe (why the formula differs from waxing)
    ---------------------------------------------------
    Waxing uses the **right** limb arc plus a terminator whose semi-axis is
    ``radius * (1 - 2 * fraction)``. For waning we keep the same ellipse *family* but
    flip the bow direction relative to that eastern limb:

        ellipse_a = radius * (2 * fraction - 1)

    Check intuition for a **waning crescent** (small *fraction*): ``(2f - 1)`` is
    **negative**, so the terminator’s midpoint slides to the **western** half and the
    lit flake appears on the **left** — exactly what we want.

    We trace the **left** circular limb (north → south through longitude π) and close
    along the terminator, so the same “one arc + one terminator” pattern covers both
    crescent and gibbous waning phases automatically.
    """
    if fraction <= 0.0:
        return None  # New moon: nothing lit.
    if fraction >= 1.0:
        theta = np.linspace(0.0, 2.0 * np.pi, 180)
        return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

    # ── Step 1: terminator semi-axis (mirror of the waxing formula) ─────────────
    # Waxing uses (1 - 2f); waning uses (2f - 1) so crescent/gibbous swap correctly
    # to the western limb. At third quarter both formulas give 0, same as waxing.
    ellipse_a = radius * (2.0 * fraction - 1.0)

    n_pts = 90

    # ── Step 2: left rim arc (top → bottom, passing through the leftmost point) ──
    # Angles: π/2 at the top, π at the far left, 3π/2 (= -π/2) at the bottom.
    theta_rim = np.linspace(np.pi / 2.0, 3.0 * np.pi / 2.0, n_pts)
    rim_x = radius * np.cos(theta_rim)
    rim_y = radius * np.sin(theta_rim)

    # ── Step 3: terminator curve (bottom → top) ───────────────────────────────
    # Same sqrt-based ellipse as waxing; only *ellipse_a* (and the left rim) differ.
    y_term = np.linspace(-radius, radius, n_pts)
    x_term = _terminator_ellipse_x(y_term, ellipse_a)

    # ── Step 4: concatenate into a closed polygon ──────────────────────────────
    xs = np.concatenate([rim_x, x_term])
    ys = np.concatenate([rim_y, y_term])
    return np.column_stack([xs, ys])


def _guess_waxing_from_name(phase_name: Optional[str]) -> bool:
    """Return True if the label says waxing, False if it says waning, default waxing."""
    if not phase_name:
        return True
    if "Waning" in phase_name:
        return False
    return True


def build_moon_phase_figure(
    phase_name: Optional[str],
    illumination: Optional[float],
    *,
    figsize: tuple[float, float] = (3.2, 3.2),
) -> plt.Figure:
    """
    Create a simple matplotlib Figure showing the moon for the picked date.

    This function only draws; Streamlit displays it with ``st.pyplot``.
    """
    kind = _moon_visual_kind(phase_name)
    fraction = _illumination_fraction(illumination)
    is_waxing = _guess_waxing_from_name(phase_name)

    # Cosmetic colors (feel free to tweak).
    sky = "#0b1020"
    shadow = "#1f1f1f"
    light = "#f2e6b9"
    edge = "#444444"

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.set_facecolor(sky)
    fig.patch.set_facecolor(sky)

    radius = 1.0
    disk_theta = np.linspace(0.0, 2.0 * np.pi, 200)
    disk_xy = np.column_stack([radius * np.cos(disk_theta), radius * np.sin(disk_theta)])

    lit_patch: Optional[np.ndarray] = None

    if kind == "full":
        lit_patch = disk_xy
    elif kind == "new":
        lit_patch = None
    elif kind == "first_quarter":
        lit_patch = _waxing_lit_vertices(0.5, radius)
    elif kind == "third_quarter":
        lit_patch = _waning_lit_vertices(0.5, radius)
    else:
        lit_patch = _waxing_lit_vertices(fraction, radius) if is_waxing else _waning_lit_vertices(fraction, radius)

    # Dark disk (the whole moon), then draw light on top where needed.
    ax.fill(disk_xy[:, 0], disk_xy[:, 1], color=shadow, zorder=1)
    if lit_patch is not None:
        ax.fill(lit_patch[:, 0], lit_patch[:, 1], color=light, zorder=2)

    ax.plot(disk_xy[:, 0], disk_xy[:, 1], color=edge, linewidth=1.0, zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    lim = radius * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Build the subtitle shown just below the moon disk.
    # For the fixed-icon phases (full, new, quarters) we just show the phase name.
    # For waxing/waning phases we add the illumination percentage so the viewer
    # can see exactly how far through the cycle the moon currently is.
    subtitle = phase_name or "Unknown phase"
    if kind == "waxing_waning":
        subtitle = f"{subtitle}\nIllumination: {fraction * 100:.1f}%"

    # Only one title line — no "schematic" label above the figure.
    ax.set_title(subtitle, color="#c8c8d8", fontsize=9, pad=6)

    return fig


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

    # Two columns: facts on the left, moon sketch on the right (easy to scan on wide layouts).
    details_col, moon_col = st.columns([1.1, 0.9], gap="large")

    with details_col:
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

    with moon_col:
        st.subheader("Moon Phase View")
        fig = build_moon_phase_figure(
            picked_info.get("phase_name"),
            picked_info.get("illumination"),
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


if __name__ == "__main__":
    main()

