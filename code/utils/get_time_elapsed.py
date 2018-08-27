from time import time

from .format_time_str import format_time_str

def get_time_elapsed(event_start):
    event_elapsed = time() - event_start
    event_elapsed_str = format_time_str(event_elapsed)
    return event_elapsed, event_elapsed_str
