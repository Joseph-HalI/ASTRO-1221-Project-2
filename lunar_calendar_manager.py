from pathlib import Path
from typing import Optional
import pandas as pd


def rows_to_records(df):
    """
    Converts a filtered DataFrame into a list of plain Python dictionaries.

    """
    if df.empty:
        return []

    # to_dict("records") turns each row into a dict like {"col1": val1, "col2": val2, ...}
    records = df.to_dict(orient="records")
    cleaned_records = []

    for record in records:
        cleaned = {}
        for key, value in record.items():
            if pd.isna(value):
                # Empty cell in the CSV — use None instead of NaN
                cleaned[key] = None
            elif isinstance(value, pd.Timestamp):
                # Date object — convert to a readable string
                cleaned[key] = value.strftime("%Y-%m-%d")
            else:
                cleaned[key] = value
        cleaned_records.append(cleaned)

    return cleaned_records


class LunarCalendarManager:
    """
    Manages lunar calendar data loaded from CSV files.
    Once created, you can ask it what's happening on any given date.

    All four CSV files are loaded once when the manager is created and held
    in memory as DataFrames. Lookups with get_date_info() then just filter
    those in-memory tables, so repeated queries are fast.
    """

    def __init__(self, data_folder=None, year: Optional[int] = None):
        """

        Loads all four CSV files into memory and prepares their date columns.

        """
        # __file__ is the path to this script. .parent is the folder it lives in.
        # So if no folder is given, we look for a 'data' subfolder right next to this file.
        if data_folder is None:
            data_folder = Path(__file__).parent / "data"
        else:
            data_folder = Path(data_folder)

        if year is None:
            year = self._infer_year(data_folder)
        self.year = int(year)

        # Load each CSV into a DataFrame.
        self.lunar = pd.read_csv(data_folder / f"lunar_data_{self.year}.csv")
        self.special_events = pd.read_csv(data_folder / f"special_events_{self.year}.csv")
        self.user_events = pd.read_csv(data_folder / "sample_user_events.csv")
        self.dark_sky = pd.read_csv(data_folder / f"dark_sky_windows_{self.year}.csv")

        # CSV files store dates as plain text like "2026-06-21".
        # pd.to_datetime() converts those strings into proper date objects pandas can compare.
        # .dt.normalize() strips any time-of-day component (sets it to midnight),
        # so "2026-06-21 14:30" and "2026-06-21 00:00" both match "2026-06-21".
        self.lunar["date"] = pd.to_datetime(self.lunar["date"]).dt.normalize()
        self.special_events["date"] = pd.to_datetime(self.special_events["date"]).dt.normalize()
        self.user_events["date"] = pd.to_datetime(self.user_events["date"]).dt.normalize()
        self.dark_sky["window_start"] = pd.to_datetime(self.dark_sky["window_start"]).dt.normalize()
        self.dark_sky["window_end"] = pd.to_datetime(self.dark_sky["window_end"]).dt.normalize()

    def get_date_info(self, query_date):
        """
        Look up everything the manager knows about a specific calendar date.

        Args:
            query_date: The date to look up. Can be a string ("2026-06-21"),
                        or a pandas Timestamp. Time of day is ignored.

        Returns:
            A dictionary with five keys:
                phase_name     (str or None)  - e.g. "Full Moon", "First Quarter"
                illumination   (float or None) - percentage of the moon's face that is lit, 0–100
                special_events (list of dicts) - astronomical events on this date, may be empty
                user_events    (list of dicts) - user-defined events on this date, may be empty
                is_dark_sky    (bool)          - True if this date falls within a dark-sky window

        """
        # Normalize for the same reason as in __init__ — strip time so date comparisons work
        date = pd.to_datetime(query_date).normalize()

        # Filter the lunar DataFrame down to only rows where the date column matches.
        # The result is a smaller DataFrame (often just one row, or zero if date not found).
        lunar_row = self.lunar[self.lunar["date"] == date]

        if lunar_row.empty:
            # Date not found in the lunar data — return None for both fields
            phase_name = None
            illumination = None
        else:
            # .iloc[0] gets the first (and normally only) matching row as a Series.
            phase_name = str(lunar_row.iloc[0]["phase_name"])
            illumination = float(lunar_row.iloc[0]["illumination"])

        # Same filtering pattern for special events and user events.
        # rows_to_records() then converts those filtered rows to plain dicts.
        matching_special = self.special_events[self.special_events["date"] == date]
        special_list = rows_to_records(matching_special)

        matching_user = self.user_events[self.user_events["date"] == date]
        user_list = rows_to_records(matching_user)

        # Dark-sky windows are date ranges, not single dates, so we check
        # whether our date falls between window_start and window_end (inclusive).
        # This produces a boolean Series (one True/False per row in dark_sky).
        in_window = (self.dark_sky["window_start"] <= date) & (date <= self.dark_sky["window_end"])
        is_dark_sky = bool(in_window.any())

        return {
            "phase_name": phase_name,
            "illumination": illumination,
            "special_events": special_list,
            "user_events": user_list,
            "is_dark_sky": is_dark_sky,
        }

    @staticmethod
    def _infer_year(data_folder: Path) -> int:
        """
        Infer the year from available `lunar_data_<year>.csv` files.
        Falls back to 2026 if no match is found.
        """
        matches = list(Path(data_folder).glob("lunar_data_*.csv"))
        years = []
        for p in matches:
            suffix = p.stem.split("_")[-1]
            if suffix.isdigit():
                years.append(int(suffix))
        if not years:
            return 2026
        return sorted(years)[-1]