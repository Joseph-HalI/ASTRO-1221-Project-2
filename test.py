from lunar_calendar_manager import LunarCalendarManager

manager = LunarCalendarManager(data_folder="data")

result = manager.get_date_info("2026-06-21")

print(result)
print(result["phase_name"])
print(result["illumination"])
print(result["is_dark_sky"])
#Events are just lists, we have to say which list first then the field we want
print(result["special_events"][0]["event_type"])
print(result["user_events"][0]["event_name"])