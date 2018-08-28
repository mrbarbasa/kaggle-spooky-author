from datetime import timedelta

def format_time_str(num_seconds):
    """Format the elapsed time into the string hh:mm:ss.

    Parameters
    ----------
    num_seconds : float
        The elapsed time in seconds.

    Returns
    -------
    time_str : string
        The elapsed time as the string hh:mm:ss.
    """

    return str(timedelta(seconds=round(num_seconds)))
