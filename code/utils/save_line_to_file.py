from contextlib import redirect_stdout

def save_line_to_file(line, file_path, mode):
    """Save a string line(s) to a file.

    Parameters
    ----------
    line : string
        The line(s) to save to a file.
    file_path : string
        The file path to save the line(s) to.
    mode : string
        Whether to write 'w' or append 'a' to the output file.

    Returns
    -------
    None
    """

    with open(file_path, mode) as f:
        with redirect_stdout(f):
            print(line)
