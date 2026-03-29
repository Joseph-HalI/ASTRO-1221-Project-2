# Lunar Phase and Dark-Sky Planning Calendar

## Project Goals

**Project 5: Moon Phase Calendar with Events**

This project creates a lunar calendar that correlates moon phases with personal events and pinpoints optimal dark-sky observing windows. It leverages Skyfield for astronomy calculations and Pandas for data management, enabling pattern discovery between lunar phases and user activities.

- Generate daily moon phase, illumination percentage, and rise/set times for an entire year using Skyfield, storing everything in a Pandas DataFrame.
- Import user-defined events (birthdays, holidays, observations) from a CSV file and merge them with lunar data dates.
- Identify dark-sky periods, nights when the moon’s illumination falls below 20%, ideal for astronomers.
- Detect and highlight key astronomical events: new moons, full moons, quarter moons, supermoons, and blue moons.
- Visualize moon phase progression, mark special events, and display events in a Streamlit calendar interface.

## Methodology

### 1) Data Collection and Astronomical Computation

The data pipeline is implemented in `collect_lunar_data.py` and uses Skyfield ephemerides (`de421.bsp`) to generate year-level data.

For each day in the selected year, the script computes:

- Phase angle and illumination percentage.
- Phase name (New Moon, Waxing Crescent, First Quarter, etc.).
- Moon rise and set times (UTC).
- Moon-Earth distance (km) for supermoon detection.

It then detects special events and saves multiple CSV outputs.

### 2) Data Wrangling and Integration

`lunar_calendar_manager.py` loads multiple CSVs and normalizes date columns into pandas datetime objects:

- `lunar_data_<year>.csv`
- `special_events_<year>.csv`
- `dark_sky_windows_<year>.csv`
- `sample_user_events.csv` (+ optional `local_user_events.csv`)

The manager performs date-based filtering and joins results logically by day to return:

- phase and illumination
- matching special events
- matching user events
- dark-sky boolean status

### 3) Reporting and Visualization

- `dark_sky_month_report.py` creates monthly dark-sky text reports for CLI use.
- `streamlit_app.py` renders:
  - a 12-month calendar grid with visual phase/event markers
  - dark-sky highlights
  - a drop down menu for specific dates
  - a custom moon-phase diagram using Matplotlib and NumPy geometry

## Data Sources

- **Primary astronomy source:** NASA JPL DE421 ephemeris accessed through Skyfield.
- **Generated local datasets:** yearly CSV outputs in `data/`.
- **User input dataset:** `data/sample_user_events.csv` and optional local overrides in `data/local_user_events.csv`.

## Preprocessing Steps

The preprocessing pipeline includes:

1. **Date parsing and normalization**
   - Convert text dates to `datetime64` and normalize to day-level timestamps.
2. **Missing-value cleanup**
   - Convert NaN-style values to `None` where records are converted to dictionaries for app rendering.
3. **Daily-to-window transformation**
   - Filter by illumination threshold and group contiguous dark dates into start/end windows with aggregate metrics.
4. **Event aggregation**
   - Group special events and user events by day for calendar display and tooltips.
5. **Year inference and validation**
   - Infer available year(s) from CSV naming patterns and expose selectable years in the app.

## Requirements

The following Python packages are required:

- `pandas`
- `numpy`
- `matplotlib`
- `streamlit`
- `skyfield`

You can install them with:

```bash
pip install pandas numpy matplotlib streamlit skyfield
```


## Usage Instructions

### 1) Generate yearly astronomy data

```bash
python collect_lunar_data.py --year 2026 --lat 39.9612 --lon -82.9988 --elev 288 --dark-threshold 20
```

Outputs are written to `data/`:

- `lunar_data_2026.csv`
- `special_events_2026.csv`
- `dark_sky_windows_2026.csv`
- `sample_user_events.csv` (template)

### 2) Add custom local events

```bash
python add_user_event.py
```

This writes to `data/local_user_events.csv` (kept local by `.gitignore` if configured).

### 3) Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

In the sidebar, select the data folder and year. Then use the calendar and date picker to inspect daily moon conditions and events.

### 4) Generate a monthly dark-sky report (optional)

```bash
python dark_sky_month_report.py
```

## Potential Improvements

- Add timezone-aware local rise/set times instead of UTC-only display.
- Expand event detection to include major meteor shower peaks and planetary conjunctions.
- Include monthly dark-sky reports in Streamlit.
- Ability to add user events in Streamlit.
- Add export options in Streamlit (filtered CSV download and report generation buttons).

## Repository Structure

- `collect_lunar_data.py` - astronomy computation + CSV pipeline
- `lunar_calendar_manager.py` - data loading, cleaning, and date queries
- `streamlit_app.py` - interactive web UI
- `dark_sky_month_report.py` - CLI month report
- `add_user_event.py` - local user event utility
- `data/` - generated datasets and event templates

## AI Usage

### 1) AI tools used
- Claude
- Cursor

### 2) What AI helped with
- Data collection pipeline: Mostly written by Claude, with minor debugging for simpler errors.
- Moon phase visualization: Claude helped redesign the terminator, so it appears as a crescent rather than a straight line.
- General: AI was used for brainstorming, drafting documentation, occasional code cleanup, or providing suggestions.
