# utils/time_utils.py
# CHANGE 5: Time arithmetic utilities (HH:MM safe ops)

from datetime import datetime, timedelta

TIME_FMT = "%H:%M"


def time_str_to_minutes(t):
    h, m = map(int, t.split(":"))
    return h * 60 + m


def minutes_to_time_str(minutes):
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def add_minutes(time_str, minutes):
    base = datetime.strptime(time_str, TIME_FMT)
    new_time = base + timedelta(minutes=minutes)
    return new_time.strftime(TIME_FMT)
