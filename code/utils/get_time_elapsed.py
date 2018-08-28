from time import time

from .format_time_str import format_time_str

def get_time_elapsed(event_start):
    """Retrieve the elapsed time of an event as the number of seconds
    and in the string format hh:mm:ss.

    Parameters
    ----------
    event_start : float
        The start of the event in seconds since the Epoch.

    Returns
    -------
    results : tuple
        - event_elapsed : float
            The elapsed time in seconds.
        - event_elapsed_str : string
            The elapsed time as the string hh:mm:ss.
    """

    event_elapsed = time() - event_start
    event_elapsed_str = format_time_str(event_elapsed)
    return event_elapsed, event_elapsed_str
