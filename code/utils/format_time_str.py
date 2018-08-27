from datetime import timedelta

def format_time_str(num_seconds):
    return str(timedelta(seconds=round(num_seconds)))
