# utils/interval_utils.py
# CHANGE 6: Interval merge utility for faculty lunch unions

from utils.time_utils import time_str_to_minutes, minutes_to_time_str


def merge_intervals(intervals):
    """
    intervals: list of {'start': 'HH:MM', 'end': 'HH:MM'}
    returns merged list in same format
    """
    if not intervals:
        return []

    normalized = sorted(
        [(time_str_to_minutes(i['start']), time_str_to_minutes(i['end'])) for i in intervals],
        key=lambda x: x[0]
    )

    merged = [normalized[0]]

    for current in normalized[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # overlap or touch
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return [
        {"start": minutes_to_time_str(s), "end": minutes_to_time_str(e)}
        for s, e in merged
    ]
