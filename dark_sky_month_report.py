import argparse

import pandas as pd

from lunar_calendar_manager import LunarCalendarManager


def _prompt_for_int(prompt_text: str, min_value: int, max_value: int) -> int:
    while True:
        raw = input(prompt_text).strip()
        try:
            value = int(raw)
        except ValueError:
            print(f"Enter a whole number between {min_value} and {max_value}.")
            continue

        if min_value <= value <= max_value:
            return value

        print(f"Value must be between {min_value} and {max_value}.")


def build_dark_sky_report(manager: LunarCalendarManager, year: int, month: int) -> str:
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end = month_start + pd.offsets.MonthEnd(1)

    # Keep only dark-sky windows that overlap this month.
    dark_windows = manager.dark_sky[
        (manager.dark_sky["window_end"] >= month_start)
        & (manager.dark_sky["window_start"] <= month_end)
    ].copy()

    dark_days = []
    for _, row in dark_windows.iterrows():
        start = max(row["window_start"], month_start)
        end = min(row["window_end"], month_end)
        dark_days.extend(pd.date_range(start, end, freq="D"))

    # Remove duplicates and sort.
    dark_days = sorted({d.normalize() for d in dark_days})

    month_title = month_start.strftime("%B %Y")
    lines = []
    lines.append("=" * 62)
    lines.append(f"Dark Sky Report: {month_title}")
    lines.append("=" * 62)

    if dark_windows.empty:
        lines.append("No dark-sky windows overlap this month.")
        lines.append("=" * 62)
        return "\n".join(lines)

    lines.append("Dark-sky windows:")
    for _, row in dark_windows.sort_values("window_start").iterrows():
        start = max(row["window_start"], month_start).strftime("%Y-%m-%d")
        end = min(row["window_end"], month_end).strftime("%Y-%m-%d")
        lines.append(
            f"- {start} to {end} "
            f"(min illumination: {row['min_illumination']:.2f}%, "
            f"avg illumination: {row['avg_illumination']:.1f}%)"
        )

    lines.append("")
    lines.append("Dark-sky days with lunar/event details:")
    for day in dark_days:
        info = manager.get_date_info(day)
        phase = info["phase_name"] or "Unknown"
        illum = (
            f"{info['illumination']:.2f}%"
            if info["illumination"] is not None
            else "Unknown"
        )

        lines.append(f"- {day.strftime('%Y-%m-%d')}: phase={phase}, illumination={illum}")

        if info["special_events"]:
            for event in info["special_events"]:
                time = event.get("time_utc") or "Time unavailable"
                event_type = event.get("event_type") or "Special event"
                lines.append(f"    special: {event_type} at {time}")

        if info["user_events"]:
            for event in info["user_events"]:
                event_name = event.get("event_name") or "User event"
                event_type = event.get("event_type") or "General"
                lines.append(f"    user: {event_name} ({event_type})")

    lines.append("")
    lines.append(f"Total dark-sky days: {len(dark_days)}")
    lines.append("=" * 62)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Print a dark-sky report for a chosen month using LunarCalendarManager data."
    )
    parser.add_argument("--year", type=int, help="Year of the report (for example: 2026)")
    parser.add_argument("--month", type=int, choices=range(1, 13), help="Month number (1-12)")
    parser.add_argument(
        "--data-folder",
        default=None,
        help="Optional path to the folder containing CSV files.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional output text file path for saving the report.",
    )
    args = parser.parse_args()

    manager = LunarCalendarManager(data_folder=args.data_folder)

    available_years = manager.lunar["date"].dt.year.unique()
    min_year = int(available_years.min())
    max_year = int(available_years.max())

    year = args.year if args.year is not None else _prompt_for_int(
        f"Enter year ({min_year}-{max_year}): ", min_year, max_year
    )
    month = args.month if args.month is not None else _prompt_for_int("Enter month (1-12): ", 1, 12)

    report = build_dark_sky_report(manager, year, month)
    print(report)

    output_path = args.output_file
    if output_path is None:
        month_token = f"{month:02d}"
        output_path = f"dark_sky_report_{year}_{month_token}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
