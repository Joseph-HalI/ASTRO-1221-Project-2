from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["date", "event_name", "event_type"]


def _validate_event_date(event_date: str) -> str:
    """Validate and normalize date input to YYYY-MM-DD."""
    return pd.to_datetime(event_date).strftime("%Y-%m-%d")


def add_user_event(event_name: str, event_date: str, event_type: str = "Custom", output_csv: Path | None = None) -> Path:
    """
    Add one user event to a local-only CSV file.

    This file is intended to stay on the local machine and be ignored by git.
    """
    if output_csv is None:
        output_csv = Path(__file__).parent / "data" / "local_user_events.csv"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    normalized_date = _validate_event_date(event_date)

    if output_csv.exists():
        events = pd.read_csv(output_csv)
        for col in REQUIRED_COLUMNS:
            if col not in events.columns:
                events[col] = None
        events = events[REQUIRED_COLUMNS]
    else:
        events = pd.DataFrame(columns=REQUIRED_COLUMNS)

    new_event = pd.DataFrame(
        [{"date": normalized_date, "event_name": event_name.strip(), "event_type": event_type.strip() or "Custom"}]
    )
    events = pd.concat([events, new_event], ignore_index=True)
    events.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    print("Add a local user event (saved to data/local_user_events.csv)")
    event_name = input("Event name: ").strip()
    event_date = input("Event date (YYYY-MM-DD): ").strip()
    event_type = input("Event type (optional, press enter for 'Custom'): ").strip()

    if not event_name:
        raise ValueError("Event name cannot be empty.")

    saved_path = add_user_event(event_name=event_name, event_date=event_date, event_type=event_type or "Custom")
    print(f"Saved event to: {saved_path}")


if __name__ == "__main__":
    main()